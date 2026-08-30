"""Hermitian half-build rider (Piece 4, round-off exact).

Under the incremental RR build the two new cross blocks are C = conj(v_old)@hd.mT
and C' = conj(d)@hv_old.mT; by Hermiticity of the true subspace matrix
(hv = H v, H Hermitian) C' = C.conj().mT exactly. So compute ONE cross GEMM and
mirror it, halving the incremental cross-block FLOPs. Measured GO on the asus RTX
3050 (one GEMM + mirror ~1.95x the two-GEMM path; >= 1.6x threshold).

These tests pin: (a) the mirror reproduces the separately-computed cross block to
fp64 round-off on a Hermitian operator; (b) the end-to-end A/B (half-build vs the
two-GEMM path) gives the same converged eigenvalues to round-off.
"""

import importlib

import torch

from gradwave.solvers.davidson import davidson_batched

davidson = importlib.import_module("gradwave.solvers.davidson")


def _make_gapped_operator(nk, npw, seed=0):
    torch.manual_seed(seed)
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128)
    diag = torch.linspace(0.0, 50.0, npw, dtype=torch.float64)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128)
        herm = 0.05 * 0.5 * (a + a.conj().T)
        h[k] = herm + torch.diag(diag.to(torch.complex128))
    mask = torch.ones(nk, npw, dtype=torch.bool)

    def apply(c):
        return torch.einsum("kij,kbj->kbi", h.to(c.dtype), c)

    return apply, mask


def test_mirror_matches_second_gemm_on_hermitian_operator():
    """C' = C.conj().mT reproduces the separately-computed conj(d)@hv_old.mT to
    fp64 round-off when hv = H v for a Hermitian H."""
    torch.manual_seed(0)
    nk, npw, m_old, n_add = 3, 70, 16, 5
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128)
        h[k] = 0.5 * (a + a.conj().T)

    v_old = torch.randn(nk, m_old, npw, dtype=torch.complex128)
    d = torch.randn(nk, n_add, npw, dtype=torch.complex128)
    hv_old = torch.einsum("kij,kbj->kbi", h, v_old)
    hd = torch.einsum("kij,kbj->kbi", h, d)

    c = torch.matmul(v_old.conj(), hd.mT)          # top-right
    c_mirror = c.conj().transpose(-1, -2)          # Piece 4
    c_direct = torch.matmul(d.conj(), hv_old.mT)   # bottom-left, second GEMM

    rel = float((c_mirror - c_direct).abs().max() / c_direct.abs().max())
    assert rel < 1e-12


def test_halfbuild_matches_twogemm_end_to_end(monkeypatch):
    """A/B: half-build (default) vs the two-GEMM cross-block path must give the
    same converged eigenvalues to round-off. Native ZGEMM so the only difference
    is the mirror-vs-second-GEMM round-off."""
    nk, npw, nb = 2, 85, 8
    apply, mask = _make_gapped_operator(nk, npw, seed=6)
    t = torch.zeros(nk, npw, dtype=torch.float64)
    torch.manual_seed(9)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    monkeypatch.setenv("GRADWAVE_RR3M", "off")

    monkeypatch.setattr(davidson, "_RR_HALFBUILD", False)
    r_two = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9,
                             max_iter=300, max_dim_factor=4)
    monkeypatch.setattr(davidson, "_RR_HALFBUILD", True)
    r_half = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9,
                              max_iter=300, max_dim_factor=4)

    assert float(r_half.residual_norms.max()) < 1e-9
    assert float(r_two.residual_norms.max()) < 1e-9
    assert float((r_half.eigenvalues - r_two.eigenvalues).abs().max()) < 1e-11
    # n_iter is NOT asserted: the mirror-vs-second-GEMM round-off can shift the
    # exact iteration where a flat convergence tail crosses tol by a handful,
    # and the size of that shift is BLAS-dependent. Convergence + eigenvalue
    # agreement to round-off is the invariant; a bug that inflated iterations
    # would fail the convergence assert (both cap at max_iter).


def test_halfbuild_composes_with_all_pieces(monkeypatch):
    """All four pieces on together (3M forced, incremental + restart-carry +
    half-build) converge to the operator's true lowest eigenvalues to 1e-7."""
    nk, npw, nb = 2, 75, 8
    apply, mask = _make_gapped_operator(nk, npw, seed=12)
    t = torch.zeros(nk, npw, dtype=torch.float64)
    torch.manual_seed(21)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    monkeypatch.setenv("GRADWAVE_RR3M", "on")
    res = davidson_batched(apply, x0, t, mask, tol=1e-9, max_iter=400,
                           max_dim_factor=3)
    # true reference from a dense eigensolve of the operator
    torch.manual_seed(12)
    h = torch.zeros(nk, npw, npw, dtype=torch.complex128)
    diag = torch.linspace(0.0, 50.0, npw, dtype=torch.float64)
    for k in range(nk):
        a = torch.randn(npw, npw, dtype=torch.complex128)
        h[k] = 0.05 * 0.5 * (a + a.conj().T) + torch.diag(diag.to(torch.complex128))
    ref = torch.linalg.eigvalsh(h)[:, :nb]
    assert float(res.residual_norms.max()) < 1e-9
    assert float((res.eigenvalues - ref).abs().max()) < 1e-7
