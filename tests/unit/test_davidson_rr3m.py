"""Exact fp64 Rayleigh-Ritz 3M/Karatsuba complex GEMM (``GRADWAVE_RR3M``).

PyTorch dispatches complex128 matmul to cuBLAS's 4-multiply ZGEMM; the RR
subspace algebra (BUILD ``s = v.conj()@hv.mT`` and the two COMBINE contractions)
dominates the batched Davidson wall on fp64-crippled cards. Gauss's 3-multiply
complex product does the same contraction with 3 real fp64 GEMMs — a lossless
25% fp64-FLOP cut, gated per-signature by a measured A/B trial.

These tests pin: (a) the two 3M helpers reproduce native complex matmul to fp64
round-off; (b) the gated solver converges to the SAME eigenpairs AND the SAME
iteration count as native (the property that separated this from the struck fp32
path, which floored at 1e-5 and ADDED iterations); (c) the eligibility/verdict
gate logic (CPU untouched under "auto", forced under "on", off disables).
"""

import importlib

import pytest
import torch

from gradwave.solvers.davidson import (
    _gemm3,
    _gemm3_conjl,
    _rr3m_eligible,
    _rr3m_verdict,
    davidson_batched,
)

davidson = importlib.import_module("gradwave.solvers.davidson")


def _make_gapped_operator(nk, npw, seed=0, device="cpu"):
    """Evenly-spaced diagonal + small perturbation: every band sits in a clean
    gap so Davidson converges each band cleanly and the iteration count is a
    stable property (isolates "does 3M change the physics" from band slowness)."""
    torch.manual_seed(seed)
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128, device=device)
    diag = torch.linspace(0.0, 50.0, npw, dtype=torch.float64, device=device)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128, device=device)
        herm = 0.05 * 0.5 * (a + a.conj().T)
        h[k] = herm + torch.diag(diag.to(torch.complex128))
    mask = torch.ones(nk, npw, dtype=torch.bool, device=device)

    def apply(c):
        return torch.einsum("kij,kbj->kbi", h.to(c.dtype), c)

    return apply, mask


# --- 3M helpers: numerical equivalence to native complex matmul ---------------


def test_gemm3_matches_native_matmul():
    """(A @ B) via 3 real GEMMs agrees with native ZGEMM to fp64 round-off."""
    torch.manual_seed(0)
    a = torch.randn(4, 12, 30, dtype=torch.complex128)
    b = torch.randn(4, 30, 9, dtype=torch.complex128)
    ar, ai = a.real.contiguous(), a.imag.contiguous()
    br, bi = b.real.contiguous(), b.imag.contiguous()
    got = _gemm3(ar, ai, br, bi)
    ref = torch.matmul(a, b)
    rel = float((got - ref).abs().max() / ref.abs().max())
    assert rel <= 1e-13


def test_gemm3_conjl_matches_conj_build():
    """conj(A) @ B via 3 real GEMMs — the BUILD contraction — matches native."""
    torch.manual_seed(1)
    v = torch.randn(3, 10, 40, dtype=torch.complex128)  # (nk, m, npw)
    hv = torch.randn(3, 10, 40, dtype=torch.complex128)
    vr, vi = v.real.contiguous(), v.imag.contiguous()
    hvr, hvi = hv.real.contiguous(), hv.imag.contiguous()
    got = _gemm3_conjl(vr, vi, hvr.transpose(-1, -2), hvi.transpose(-1, -2))
    ref = torch.matmul(v.conj(), hv.mT)  # (nk, m, m)
    rel = float((got - ref).abs().max() / ref.abs().max())
    assert rel <= 1e-13


def test_gemm3_combine_matches_einsum():
    """The COMBINE contraction einsum("kja,kjg->kag", u, v) == _gemm3(u.mT, v)."""
    torch.manual_seed(2)
    u = torch.randn(3, 16, 6, dtype=torch.complex128)  # (nk, m, nb)
    v = torch.randn(3, 16, 40, dtype=torch.complex128)  # (nk, m, npw)
    ur, ui = u.real.contiguous(), u.imag.contiguous()
    vr, vi = v.real.contiguous(), v.imag.contiguous()
    got = _gemm3(ur.transpose(-1, -2), ui.transpose(-1, -2), vr, vi)
    ref = torch.einsum("kja,kjg->kag", u, v)
    rel = float((got - ref).abs().max() / ref.abs().max())
    assert rel <= 1e-13


# --- gated solver: same eigenpairs AND same iteration count -------------------


def test_rr3m_on_matches_native_eigenpairs_and_iterations(monkeypatch):
    """Forcing 3M ("on") on CPU must reproduce the native solver's converged
    eigenvalues to <= 1e-9 AND the exact iteration count — the fp64-exactness
    contract that distinguishes 3M from the struck fp32 path."""
    nk, npw, nb = 3, 70, 7
    apply, mask = _make_gapped_operator(nk, npw, seed=4)
    t = torch.zeros(nk, npw, dtype=torch.float64)
    torch.manual_seed(11)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)

    monkeypatch.setenv("GRADWAVE_RR3M", "off")
    r_nat = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9, max_iter=200)
    monkeypatch.setenv("GRADWAVE_RR3M", "on")
    r_3m = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9, max_iter=200)

    assert float(r_nat.residual_norms.max()) < 1e-8
    assert float(r_3m.residual_norms.max()) < 1e-8
    assert r_3m.n_iter == r_nat.n_iter  # iteration count UNCHANGED
    assert float((r_3m.eigenvalues - r_nat.eigenvalues).abs().max()) <= 1e-9
    # eigenpairs are genuine, not just equal eigenvalues by coincidence
    hx = apply(r_3m.eigenvectors)
    resid = hx - r_3m.eigenvalues[..., None] * r_3m.eigenvectors
    assert float(torch.linalg.norm(resid, dim=-1).max()) < 1e-6


def test_rr3m_survives_restart(monkeypatch):
    """A tight max_dim forces several restarts; the plane cache is rebuilt there
    (not incrementally), so this exercises the restart plane path. 3M must still
    match native to 1e-9 with identical iteration count."""
    nk, npw, nb = 2, 90, 10
    apply, mask = _make_gapped_operator(nk, npw, seed=6)
    t = torch.zeros(nk, npw, dtype=torch.float64)
    torch.manual_seed(5)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)

    monkeypatch.setenv("GRADWAVE_RR3M", "off")
    r_nat = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9,
                             max_iter=200, max_dim_factor=3)
    monkeypatch.setenv("GRADWAVE_RR3M", "on")
    r_3m = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9,
                            max_iter=200, max_dim_factor=3)
    assert float(r_3m.residual_norms.max()) < 1e-8
    assert r_3m.n_iter == r_nat.n_iter
    assert float((r_3m.eigenvalues - r_nat.eigenvalues).abs().max()) <= 1e-9


# --- gate logic ---------------------------------------------------------------


def test_rr3m_eligible_complex64_never(monkeypatch):
    """A complex64 block is a mixed-precision draft on the fast fp32 path — 3M
    (a fp64-FLOP saver) never applies, in any mode."""
    monkeypatch.setattr(davidson, "_RR3M_ENV", "on")
    assert _rr3m_eligible(torch.zeros(1, 1, 1, dtype=torch.complex64)) is False


def test_rr3m_eligible_cpu_untouched_under_auto(monkeypatch):
    """Under "auto" the CPU native path keeps zero overhead — eligibility is
    False so no plane cache is maintained. "on" forces it (for tests)."""
    z = torch.zeros(1, 1, 1, dtype=torch.complex128)
    monkeypatch.setattr(davidson, "_RR3M_ENV", "auto")
    assert _rr3m_eligible(z) is False
    monkeypatch.setattr(davidson, "_RR3M_ENV", "on")
    assert _rr3m_eligible(z) is True
    monkeypatch.setattr(davidson, "_RR3M_ENV", "off")
    assert _rr3m_eligible(z) is False


def test_rr3m_verdict_on_forces_true(monkeypatch):
    """"on" short-circuits the timed trial to True without touching the cache."""
    monkeypatch.setattr(davidson, "_RR3M_ENV", "on")
    v = torch.randn(2, 4, 20, dtype=torch.complex128)
    hv = torch.randn(2, 4, 20, dtype=torch.complex128)
    vr, vi = v.real.contiguous(), v.imag.contiguous()
    hvr, hvi = hv.real.contiguous(), hv.imag.contiguous()
    assert _rr3m_verdict(v, hv, vr, vi, hvr, hvi) is True


def test_rr3m_verdict_trial_caches_per_signature(monkeypatch):
    """"auto" runs the timed A/B once per (device, nk, m, npw, dtype) and caches
    the verdict; the cached value is returned on the next call for that key."""
    monkeypatch.setattr(davidson, "_RR3M_ENV", "auto")
    monkeypatch.setattr(davidson, "_RR3M_VERDICT", {})
    v = torch.randn(2, 4, 20, dtype=torch.complex128)
    hv = torch.randn(2, 4, 20, dtype=torch.complex128)
    vr, vi = v.real.contiguous(), v.imag.contiguous()
    hvr, hvi = hv.real.contiguous(), hv.imag.contiguous()
    verdict = _rr3m_verdict(v, hv, vr, vi, hvr, hvi)
    key = ("cpu", 2, 4, 20, torch.complex128)
    assert key in davidson._RR3M_VERDICT
    assert davidson._RR3M_VERDICT[key] is verdict
    # a poisoned cache entry is honored (proves the cache is consulted)
    davidson._RR3M_VERDICT[key] = not verdict
    assert _rr3m_verdict(v, hv, vr, vi, hvr, hvi) is (not verdict)


def test_rr3m_env_off_uses_native(monkeypatch):
    """"off" disables the whole path: eligibility False, verdict never True."""
    monkeypatch.setattr(davidson, "_RR3M_ENV", "off")
    assert _rr3m_eligible(torch.zeros(1, 1, 1, dtype=torch.complex128)) is False


@pytest.mark.gpu
def test_rr3m_matches_native_on_gpu(monkeypatch):
    """On CUDA the converged eigenpairs must be identical whether 3M or native
    ZGEMM ran the RR GEMMs, with unchanged iteration count."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    nk, npw, nb = 3, 80, 8
    apply, mask = _make_gapped_operator(nk, npw, seed=8, device="cuda")
    t = torch.zeros(nk, npw, dtype=torch.float64, device="cuda")
    torch.manual_seed(3)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128, device="cuda")

    monkeypatch.setenv("GRADWAVE_RR3M", "off")
    r_nat = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9, max_iter=200)
    monkeypatch.setenv("GRADWAVE_RR3M", "on")
    r_3m = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9, max_iter=200)
    assert r_3m.n_iter == r_nat.n_iter
    assert float((r_3m.eigenvalues - r_nat.eigenvalues).abs().max().cpu()) <= 1e-9
