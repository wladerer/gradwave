"""fp64-certified fp32-expansion Davidson (``GRADWAVE_FP32_EXPANSION``).

The mode runs expansion-vector H-applies in complex64 while keeping every
convergence-deciding quantity fp64: the subspace reduction stays on the
issue-#136 fp64 contract, and no band is ever declared converged by an
fp32-sourced residual — before any return the retained Ritz block is refreshed
and re-measured against the full-precision operator (``_certify_fp64``).

These tests pin the certification contract on the same synthetic batched
Hermitian operator the CheFSI/LOBPCG tests use (dtype-polymorphic apply, as
``BatchedHamiltonian.apply`` is), plus a tiny metallic SCF (fcc Al, smeared)
where the existing whole-solve mixed-precision scheme is known to cost outer
iterations — the certified mode must not.
"""

import numpy as np
import torch

from gradwave.solvers.davidson import davidson_batched

TOL = 1e-9


def _make_operator(nk, npw, seed=0):
    """Random Hermitian H_k with a wide kinetic-like spectrum (same synthetic
    as the CheFSI/LOBPCG tests), exposed through a dtype-POLYMORPHIC apply —
    the fp32-expansion contract, satisfied by the real BatchedHamiltonian."""
    torch.manual_seed(seed)
    diag = torch.linspace(1.0, 100.0, npw, dtype=torch.float64)
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128)
        herm = 0.5 * (a + a.conj().T)
        h[k] = herm + torch.diag(diag.to(torch.complex128))
        h[k] = 0.5 * (h[k] + h[k].conj().T)
    mask = torch.ones(nk, npw, dtype=torch.bool)
    t = diag.expand(nk, npw).contiguous()

    def apply(c):  # recompute in c's dtype -> genuine fp32 during expansion
        return torch.einsum("kij,kbj->kbi", h.to(c.dtype), c)

    return h, apply, mask, t


def test_certified_matches_pure_fp64(monkeypatch):
    """Converged eigenvalues agree with the pure-fp64 solver to <= 1e-9, and
    the fp32/full apply split is populated (fp32 must carry the bulk)."""
    nk, npw, nb = 4, 80, 8
    h, apply, mask, t = _make_operator(nk, npw, seed=2)
    ref = torch.linalg.eigvalsh(h)[:, :nb]
    torch.manual_seed(99)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)

    monkeypatch.delenv("GRADWAVE_FP32_EXPANSION", raising=False)
    r64 = davidson_batched(apply, x0.clone(), t, mask, tol=TOL, max_iter=100)
    monkeypatch.setenv("GRADWAVE_FP32_EXPANSION", "on")
    r32 = davidson_batched(apply, x0.clone(), t, mask, tol=TOL, max_iter=100)

    assert float(r64.residual_norms.max()) < TOL
    assert float(r32.residual_norms.max()) < TOL
    assert float((r32.eigenvalues - r64.eigenvalues).abs().max()) <= 1e-9
    assert float((r32.eigenvalues - ref).abs().max()) < 1e-7
    # bookkeeping: baseline is all-full; certified mode routes expansion applies
    # through fp32 and pays full-precision applies only for certification
    assert r64.n_apply_low == 0 and r64.n_apply_full > 0
    assert r32.n_apply_low > 0 and r32.n_apply_full > 0
    assert r32.n_apply_low > r32.n_apply_full


def test_returned_residuals_are_fp64_certified(monkeypatch):
    """Every return path re-measures against the full-precision operator: the
    reported residual norms must match an independent fp64 recompute, including
    on the max_iter exit (the final refresh pass)."""
    nk, npw, nb = 2, 60, 6
    h, apply, mask, t = _make_operator(nk, npw, seed=3)
    torch.manual_seed(7)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    monkeypatch.setenv("GRADWAVE_FP32_EXPANSION", "on")

    for max_iter in (3, 100):  # 3 forces the max_iter exit; 100 converges
        res = davidson_batched(apply, x0.clone(), t, mask, tol=TOL,
                               max_iter=max_iter)
        hx = apply(res.eigenvectors)  # complex128 -> exact operator
        rn = torch.linalg.norm(
            hx - res.eigenvalues[..., None] * res.eigenvectors, dim=-1).real
        assert float((rn - res.residual_norms).abs().max()) < 1e-12
        gram = torch.einsum("kag,kbg->kab", res.eigenvectors.conj(),
                            res.eigenvectors)
        eye = torch.eye(nb, dtype=torch.complex128).expand(nk, nb, nb)
        assert float((gram - eye).abs().max()) < 1e-10


def test_inactive_on_complex64_block(monkeypatch):
    """A complex64 block is already a mixed-precision draft — the mode must not
    stack a second downcast (nothing to certify against)."""
    nk, npw, nb = 2, 50, 5
    h, apply, mask, t = _make_operator(nk, npw, seed=4)
    torch.manual_seed(11)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex64)
    monkeypatch.setenv("GRADWAVE_FP32_EXPANSION", "on")
    res = davidson_batched(apply, x0, t.to(torch.float32), mask, tol=1e-4,
                           max_iter=100)
    assert res.n_apply_low == 0
    assert res.n_apply_full > 0


def test_env_value_validated(monkeypatch):
    import pytest

    from gradwave.solvers.davidson import _fp32_expansion_active

    monkeypatch.setenv("GRADWAVE_FP32_EXPANSION", "yes")
    with pytest.raises(ValueError, match="GRADWAVE_FP32_EXPANSION"):
        _fp32_expansion_active(torch.zeros(1, 1, 1, dtype=torch.complex128))


def test_al_scf_energy_and_iteration_parity(monkeypatch):
    """Tiny metallic SCF (fcc Al, gaussian smearing): the certified mode must
    reproduce the fp64 free energy/eigenvalues AND the outer-iteration count —
    the failure mode of the whole-solve fp32 draft on metals (extra outer
    iterations from unconditioned fp32 noise) must not reappear."""
    from gradwave.core.xc.lda_pw92 import LDA_PW92
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system
    from tests.helpers import PSEUDOS, RY

    torch.set_num_threads(2)
    a = 4.05
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0]])
    al = parse_upf(str(PSEUDOS / "Al_ONCV_PBE-1.2.upf"))

    def run(mode):
        monkeypatch.setenv("GRADWAVE_FP32_EXPANSION", mode)
        system = setup_system(cell, pos, [0], [al], ecut=12 * RY, kmesh=(2, 2, 2))
        return scf(system, LDA_PW92(), smearing="gaussian", width=0.1,
                   etol=1e-9, rhotol=1e-8, verbose=False)

    r64 = run("off")
    r32 = run("on")
    assert r64.converged and r32.converged
    assert r32.n_iter == r64.n_iter  # SCF outer-iteration parity
    assert abs(float(r64.energies.free_energy)
               - float(r32.energies.free_energy)) < 1e-7
    assert torch.allclose(r64.eigenvalues, r32.eigenvalues, atol=1e-6)
