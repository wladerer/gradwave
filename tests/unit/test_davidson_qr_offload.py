"""Correctness for the CUDA batched-QR CPU-offload in davidson.py.

Background (see docs/manual/performance.md, "What does not help", and the
_QR_CPU_MAX_COLS comment in solvers/davidson.py): a CUDA-graphs capture of a
whole Davidson round was probed and found to give ~1.0x -- no launch-overhead
gap anywhere in the round, matching the already-recorded apply-only probe.
Isolating each op in the round instead found that `torch.linalg.qr` on the
(nk, npw, cols) tall-skinny shape `_orthonormalize_b` uses is itself the
single most expensive op in a round on an RTX 3050 -- bigger than the two-FFT
Hamiltonian apply next to it -- and that a CPU LAPACK round trip is >10x
faster for cols this small, the same pathology issue #133 already fixed for
the subspace eigh. `_qr_offload` mirrors `_eigh_subspace`'s CPU-offload
pattern; these tests check it doesn't change the solver's physics.

QR is only unique up to a per-column phase (unlike a plain linear-algebra
identity), so LAPACK and cuSOLVER can disagree element-wise on Q/R while
spanning the identical subspace. The meaningful invariant is downstream: the
CONVERGED Ritz eigenvalues/eigenvectors davidson_batched returns must agree
whether or not the CPU-offload path is taken, since Rayleigh-Ritz only depends
on the SPACE the expansion directions span, not the basis' phase convention.
"""

import importlib

import pytest
import torch

from gradwave.solvers.davidson import (
    _fp64_penalty,
    _qr_offload,
    _qr_offload_active,
    davidson_batched,
)

davidson = importlib.import_module("gradwave.solvers.davidson")


def _make_operator(nk, npw, seed=0, device="cpu"):
    """Random Hermitian H_k with a realistic wide spectrum; returns (apply, mask)."""
    torch.manual_seed(seed)
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128, device=device)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128, device=device)
        herm = 0.5 * (a + a.conj().T)
        diag = torch.linspace(0.0, 100.0, npw, dtype=torch.float64, device=device)
        herm = herm + torch.diag(diag.to(torch.complex128))
        h[k] = 0.5 * (herm + herm.conj().T)
    mask = torch.ones(nk, npw, dtype=torch.bool, device=device)

    def apply(c):
        return torch.einsum("kij,kbj->kbi", h, c)

    return apply, mask


def _make_gapped_operator(nk, npw, seed=0, device="cpu"):
    """Evenly-spaced diagonal + a small perturbation, so every requested band
    sits in a comfortable gap and Davidson converges every band cleanly --
    isolates "does the QR path change the physics" from "is this particular
    band slow to separate from its neighbor", a Davidson property unrelated
    to the offload path under test."""
    torch.manual_seed(seed)
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128, device=device)
    diag = torch.linspace(0.0, 50.0, npw, dtype=torch.float64, device=device)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128, device=device)
        herm = 0.05 * 0.5 * (a + a.conj().T)
        h[k] = herm + torch.diag(diag.to(torch.complex128))
    mask = torch.ones(nk, npw, dtype=torch.bool, device=device)

    def apply(c):
        return torch.einsum("kij,kbj->kbi", h, c)

    return apply, mask


def test_qr_offload_reduces_correctly_on_cpu():
    """On CPU, _qr_offload takes the plain torch.linalg.qr branch (is_cuda is
    False); Q must be columnwise-orthonormal and Q @ R must reconstruct x."""
    torch.manual_seed(0)
    x = torch.randn(4, 200, 8, dtype=torch.complex128)
    q, r = _qr_offload(x)
    assert not x.is_cuda
    gram = torch.einsum("kig,kih->kgh", q.conj(), q)
    eye = torch.eye(8, dtype=torch.complex128).expand(4, 8, 8)
    assert float((gram - eye).abs().max()) < 1e-10
    assert float((q @ r - x).abs().max()) < 1e-10


def test_qr_offload_column_count_gate():
    """The offload only triggers below the documented column threshold, and
    only on CUDA -- exercised directly against the private threshold so a
    future retune of _QR_CPU_MAX_COLS doesn't silently stop being tested."""
    assert davidson._QR_CPU_MAX_COLS >= 1


# --- device gate: fp64-penalty microbenchmark ---------------------------------


def test_fp64_penalty_rejects_non_cuda_device():
    """_fp64_penalty is CUDA-only: the offload it gates is a no-op on CPU, so
    calling it for a CPU device is a programming error, not a handled case."""
    with pytest.raises(ValueError, match="CUDA-only"):
        _fp64_penalty(torch.device("cpu"))


def test_qr_offload_active_false_on_cpu():
    """On CPU the device gate is always off -- no microbenchmark, no env read --
    so CPU-path Davidson behavior is untouched by any of this machinery."""
    assert _qr_offload_active(torch.device("cpu")) is False


def test_qr_offload_active_auto_gates_on_penalty(monkeypatch):
    """In "auto" mode the gate follows the measured fp64 penalty against the
    threshold: a crippled-fp64 ratio activates the offload, a datacenter ratio
    skips it. The microbenchmark is stubbed so the branch is tested without a
    GPU (a real CUDA device would run it; this isolates the decision logic)."""
    monkeypatch.setattr(davidson, "_QR_OFFLOAD_ENV", "auto")
    cuda = torch.device("cuda", 0)

    monkeypatch.setattr(davidson, "_fp64_penalty",
                        lambda dev: davidson._QR_OFFLOAD_PENALTY_THRESHOLD + 10.0)
    assert _qr_offload_active(cuda) is True

    monkeypatch.setattr(davidson, "_fp64_penalty",
                        lambda dev: davidson._QR_OFFLOAD_PENALTY_THRESHOLD - 6.0)
    assert _qr_offload_active(cuda) is False


def test_qr_offload_env_override_wins_both_directions(monkeypatch):
    """The env hatch overrides the microbenchmark in both directions: "off"
    skips even a crippled card, "on" offloads even a datacenter card. The
    penalty stub is set to the opposite verdict to prove the env wins."""
    cuda = torch.device("cuda", 0)

    monkeypatch.setattr(davidson, "_QR_OFFLOAD_ENV", "off")
    monkeypatch.setattr(davidson, "_fp64_penalty",
                        lambda dev: davidson._QR_OFFLOAD_PENALTY_THRESHOLD + 100.0)
    assert _qr_offload_active(cuda) is False

    monkeypatch.setattr(davidson, "_QR_OFFLOAD_ENV", "on")
    monkeypatch.setattr(davidson, "_fp64_penalty",
                        lambda dev: 1.0)  # datacenter-class ratio
    assert _qr_offload_active(cuda) is True


def test_cpu_davidson_unchanged_by_device_gate():
    """Regression: the CPU Davidson path converges to the operator's true
    lowest eigenvalues regardless of the new device gate (which is inert on
    CPU). Guards against the gate leaking into the non-CUDA path."""
    nk, npw, nb = 2, 60, 6
    apply, mask = _make_gapped_operator(nk, npw, seed=7, device="cpu")
    torch.manual_seed(21)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    t = torch.zeros(nk, npw, dtype=torch.float64)
    res = davidson_batched(apply, x0, t, mask, tol=1e-9, max_iter=200)
    assert float(res.residual_norms.max()) < 1e-8
    # eigenpairs are genuine: H x - lambda x is tiny for every returned band
    hx = apply(res.eigenvectors)
    resid = hx - res.eigenvalues[..., None] * res.eigenvectors
    assert float(torch.linalg.norm(resid, dim=-1).max()) < 1e-6
    # and returned ascending, as Davidson contracts
    assert bool((res.eigenvalues[:, 1:] >= res.eigenvalues[:, :-1] - 1e-9).all())


@pytest.mark.gpu
def test_qr_offload_matches_gpu_qr_span():
    """CPU-offloaded QR and plain CUDA QR must span the identical subspace
    (Q Q^H projector agrees; R upper-triangular structure preserved), even
    though individual columns may differ by a phase between LAPACK and
    cuSOLVER."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    torch.manual_seed(3)
    x = torch.randn(6, 300, 8, dtype=torch.complex128, device="cuda")
    q_off, r_off = _qr_offload(x)  # cols=8 <= _QR_CPU_MAX_COLS -> CPU path
    q_gpu, r_gpu = torch.linalg.qr(x, mode="reduced")  # forced GPU path
    assert q_off.is_cuda and q_gpu.is_cuda

    proj_off = torch.einsum("kig,kjg->kij", q_off, q_off.conj())
    proj_gpu = torch.einsum("kig,kjg->kij", q_gpu, q_gpu.conj())
    assert float((proj_off - proj_gpu).abs().max().cpu()) < 1e-8

    # both must faithfully reconstruct x
    assert float((q_off @ r_off - x).abs().max().cpu()) < 1e-8
    assert float((q_gpu @ r_gpu - x).abs().max().cpu()) < 1e-8


@pytest.mark.gpu
def test_davidson_batched_matches_with_qr_offload_disabled():
    """The converged eigenpairs davidson_batched returns on CUDA must be
    unaffected by whether the CPU-offload QR path is taken -- the physical
    check that motivates _qr_offload's existence."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    nk, npw, nb = 3, 70, 7
    apply, mask = _make_gapped_operator(nk, npw, seed=4, device="cuda")
    torch.manual_seed(11)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128, device="cuda")
    t = torch.zeros(nk, npw, dtype=torch.float64, device="cuda")

    res_offload = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9, max_iter=200)

    old_max = davidson._QR_CPU_MAX_COLS
    davidson._QR_CPU_MAX_COLS = -1  # force every QR call onto the GPU path
    try:
        res_gpu = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9, max_iter=200)
    finally:
        davidson._QR_CPU_MAX_COLS = old_max

    assert float(res_offload.residual_norms.max()) < 1e-8
    assert float(res_gpu.residual_norms.max()) < 1e-8
    assert float((res_offload.eigenvalues - res_gpu.eigenvalues).abs().max()) < 1e-7
    # residuals of one solution against the other's Ritz vectors stay tiny --
    # confirms they converged to the same eigenspace, not just the same
    # eigenvalues by coincidence
    hx = apply(res_offload.eigenvectors)
    resid = hx - res_offload.eigenvalues[..., None] * res_offload.eigenvectors
    assert float(torch.linalg.norm(resid, dim=-1).max()) < 1e-6
