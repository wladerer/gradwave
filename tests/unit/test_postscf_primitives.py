"""Property-based contracts for the shared post-SCF primitives.

These pin the numerical behaviour of the helpers extracted during the post-SCF
dedup so a future edit to one call site cannot silently change the physics:

- pdos.o_inv_sqrt          — the single Löwdin O^{-1/2}
- scf.implicit.projected_cg — the shared band-frozen preconditioned CG
- pdos.spectral_grid       — the DOS/COHP energy window + grid
- pdos.split_spinor        — the spinor up/down PW split

All are fast (synthetic tensors, no SCF), so they live in the fast tier.
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf.pdos import o_inv_sqrt, spectral_grid, split_spinor
from gradwave.scf.implicit import projected_cg


def _hermitian_pd(n: int, seed: int, floor: float = 0.5) -> torch.Tensor:
    """A well-conditioned Hermitian positive-definite (n, n) complex matrix."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=RDTYPE) + 1j * torch.randn(
        n, n, generator=g, dtype=RDTYPE)
    return a @ a.conj().T + floor * torch.eye(n, dtype=CDTYPE)


@settings(max_examples=40, deadline=None)
@given(n=st.integers(2, 12), seed=st.integers(0, 10_000))
def test_o_inv_sqrt_is_inverse_square_root(n, seed):
    """M = O^{-1/2} is Hermitian and satisfies M O M = I."""
    o = _hermitian_pd(n, seed)
    m = o_inv_sqrt(o)
    eye = torch.eye(n, dtype=CDTYPE)
    assert torch.allclose(m, m.conj().T, atol=1e-8)          # Hermitian
    assert torch.allclose(m @ o @ m, eye, atol=1e-6)          # M O M = I


@settings(max_examples=40, deadline=None)
@given(n=st.integers(2, 16), nb=st.integers(1, 6), seed=st.integers(0, 10_000))
def test_projected_cg_solves_spd_system(n, nb, seed):
    """projected_cg drives the residual of A x = rhs below tol for an SPD A.

    The operator acts per band row as x @ A with A Hermitian PD, matching the
    (nbands, ngrid) convention of the real callers; the identity preconditioner
    isolates the CG core itself.
    """
    a = _hermitian_pd(n, seed, floor=1.0)
    g = torch.Generator().manual_seed(seed + 1)
    rhs = torch.randn(nb, n, generator=g, dtype=RDTYPE).to(CDTYPE)

    def a_apply(x):
        return x @ a

    tol = 1e-10
    x0 = torch.zeros_like(rhs)
    r0 = rhs - a_apply(x0)
    x = projected_cg(a_apply, lambda r: r, x0, r0, tol=tol, max_iter=4 * n)
    resid = torch.linalg.norm(rhs - a_apply(x), dim=1).max()
    # CG on an n-dim SPD system is exact in <= n steps in exact arithmetic;
    # allow a generous float64 slack above the requested tol.
    assert float(resid) < 1e-6


@settings(max_examples=50, deadline=None)
@given(
    lo=st.floats(-30, 0), span=st.floats(0.1, 40),
    width=st.floats(0.01, 1.0), npoints=st.integers(8, 400),
)
def test_spectral_grid_brackets_eigenvalues(lo, span, width, npoints):
    """Default window pads the eigenvalue range by 10*width; grid is sorted."""
    import numpy as np
    all_e = np.array([lo, lo + span])
    window, grid = spectral_grid(all_e, width, npoints)
    assert window[0] <= all_e.min() - 10 * width + 1e-9
    assert window[1] >= all_e.max() + 10 * width - 1e-9
    assert grid.shape == (npoints,)
    assert np.all(np.diff(grid) >= 0)                         # non-decreasing
    # an explicit window is passed through verbatim
    w2, _ = spectral_grid(all_e, width, npoints, window=(-1.0, 2.0))
    assert w2 == (-1.0, 2.0)


@settings(max_examples=40, deadline=None)
@given(npw=st.integers(1, 50), pad=st.integers(0, 20), nb=st.integers(1, 8))
def test_split_spinor_recovers_components(npw, pad, nb):
    """split_spinor slices the up/down PW blocks at the fixed padding offset."""
    m_pw = npw + pad
    g = torch.Generator().manual_seed(npw * 1000 + pad * 10 + nb)
    c = torch.randn(nb, 2 * m_pw, generator=g, dtype=CDTYPE)
    cu, cd = split_spinor(c, npw, m_pw)
    assert cu.shape == (nb, npw) and cd.shape == (nb, npw)
    assert torch.equal(cu, c[:, :npw])
    assert torch.equal(cd, c[:, m_pw:m_pw + npw])
