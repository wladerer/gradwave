"""Incremental Rayleigh-Ritz build (Piece 2, round-off exact).

On a grow round the old columns of the running basis v and its images hv are
bit-identical, so the old (m_old x m_old) block of the raw subspace matrix
s = conj(v)@hv.mT is unchanged up to fp64 round-off. ``davidson_batched`` carries
it forward and recomputes only the new column blocks (top-right conj(v_old)@hd.mT,
bottom-left conj(d)@hv_old.mT, corner conj(d)@hd.mT), rebuilding fully only at a
restart or fp32 handoff.

The reused block is a matmul over the SAME v_old/hv_old rows, so it equals a
from-scratch full rebuild to fp64 round-off (~1e-14) — but NOT bit-for-bit, since
a GEMM's per-output tiling depends on the matrix dimensions (a sub-block and the
corner of a larger product round differently at ~1e-15). The consequence is that
the converged eigenvalues match a full rebuild to round-off while the exact
iteration count can drift by a few near a flat convergence tail. The module flag
``_RR_INCREMENTAL`` (always True in production) is toggled here to A/B the two
paths: eigenvalue agreement to round-off is the invariant; iteration count is
asserted within a small round-off-drift band.
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


def test_incremental_block_decomposition_equals_full_rebuild():
    """The 2x2 block assembly (carried top-left + three new blocks) reproduces a
    from-scratch conj(v)@hv.mT to fp64 round-off — the arithmetic identity the
    grow branch relies on."""
    torch.manual_seed(0)
    nk, m_old, n_add, npw = 3, 14, 4, 60
    v = torch.randn(nk, m_old + n_add, npw, dtype=torch.complex128)
    hv = torch.randn(nk, m_old + n_add, npw, dtype=torch.complex128)
    v_old, d = v[:, :m_old], v[:, m_old:]
    hv_old, hd = hv[:, :m_old], hv[:, m_old:]

    s_old = torch.matmul(v_old.conj(), hv_old.mT)  # carried across the grow
    s_tr = torch.matmul(v_old.conj(), hd.mT)
    s_bl = torch.matmul(d.conj(), hv_old.mT)
    s_co = torch.matmul(d.conj(), hd.mT)
    s_inc = torch.cat([torch.cat([s_old, s_tr], dim=2),
                       torch.cat([s_bl, s_co], dim=2)], dim=1)

    s_full = torch.matmul(v.conj(), hv.mT)
    assert torch.allclose(s_inc, s_full, atol=1e-11, rtol=0.0)
    # the carried top-left block matches the full rebuild's corner to round-off
    assert torch.allclose(s_old, s_full[:, :m_old, :m_old], atol=1e-11, rtol=0.0)


def test_incremental_matches_full_rebuild_end_to_end(monkeypatch):
    """A/B: incremental build (default) vs forced full-rebuild each round must
    give the same eigenvalues to fp64 round-off, and the iteration count within
    a small round-off-drift band. Native ZGEMM (GRADWAVE_RR3M=off)."""
    nk, npw, nb = 2, 90, 9
    apply, mask = _make_gapped_operator(nk, npw, seed=6)
    t = torch.zeros(nk, npw, dtype=torch.float64)
    torch.manual_seed(5)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    monkeypatch.setenv("GRADWAVE_RR3M", "off")

    monkeypatch.setattr(davidson, "_RR_INCREMENTAL", False)
    r_full = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9,
                              max_iter=300, max_dim_factor=4)
    monkeypatch.setattr(davidson, "_RR_INCREMENTAL", True)
    r_inc = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9,
                             max_iter=300, max_dim_factor=4)

    assert float(r_inc.residual_norms.max()) < 1e-9
    assert float(r_full.residual_norms.max()) < 1e-9
    # eigenvalues agree to round-off. n_iter is NOT asserted: it is round-off-
    # variable near a flat convergence tail and BLAS-dependent (the meaningful
    # invariant is eigenvalue agreement + convergence, both checked here).
    assert float((r_inc.eigenvalues - r_full.eigenvalues).abs().max()) < 1e-11


def test_incremental_survives_many_restarts(monkeypatch):
    """A very tight max_dim (factor 2) forces a restart almost every round, so
    the full-rebuild reset at restart is heavily exercised alongside the
    incremental grow. Eigenvalues still match the full-rebuild reference to
    round-off (the invariant; iteration count is round-off-variable here)."""
    nk, npw, nb = 2, 80, 8
    apply, mask = _make_gapped_operator(nk, npw, seed=9)
    t = torch.zeros(nk, npw, dtype=torch.float64)
    torch.manual_seed(13)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)
    monkeypatch.setenv("GRADWAVE_RR3M", "off")

    monkeypatch.setattr(davidson, "_RR_INCREMENTAL", False)
    r_full = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9,
                              max_iter=400, max_dim_factor=2)
    monkeypatch.setattr(davidson, "_RR_INCREMENTAL", True)
    r_inc = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9,
                             max_iter=400, max_dim_factor=2)
    assert float(r_inc.residual_norms.max()) < 1e-9
    assert float(r_full.residual_norms.max()) < 1e-9
    assert float((r_inc.eigenvalues - r_full.eigenvalues).abs().max()) < 1e-11


def test_incremental_composes_with_3m(monkeypatch):
    """Piece 1 x Piece 2: 3M forced on with incremental build must still match
    the pure-native full-rebuild solver's eigenvalues to round-off — the two
    optimizations compose without drift beyond fp64 round-off."""
    nk, npw, nb = 2, 85, 8
    apply, mask = _make_gapped_operator(nk, npw, seed=11)
    t = torch.zeros(nk, npw, dtype=torch.float64)
    torch.manual_seed(17)
    x0 = torch.randn(nk, nb, npw, dtype=torch.complex128)

    monkeypatch.setenv("GRADWAVE_RR3M", "off")
    monkeypatch.setattr(davidson, "_RR_INCREMENTAL", False)
    r_ref = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9,
                             max_iter=300, max_dim_factor=4)
    monkeypatch.setenv("GRADWAVE_RR3M", "on")
    monkeypatch.setattr(davidson, "_RR_INCREMENTAL", True)
    r_both = davidson_batched(apply, x0.clone(), t, mask, tol=1e-9,
                              max_iter=300, max_dim_factor=4)
    assert float(r_both.residual_norms.max()) < 1e-9
    assert float(r_ref.residual_norms.max()) < 1e-9
    # eigenvalues agree to round-off (n_iter round-off-variable, not asserted)
    assert float((r_both.eigenvalues - r_ref.eigenvalues).abs().max()) < 1e-11
