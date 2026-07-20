"""ISDF orbital-pair factorization and compressed Fock exchange (Layer C).

Synthetic orbitals on a small Γ box, so the algebra is exercised without an SCF:
the ISDF-compressed exchange must equal the direct O(N²) plane-wave Fock build
to machine precision once the interpolation points saturate the pair rank, and
must converge to it monotonically as the rank grows below saturation.
"""

import numpy as np
import torch

from gradwave.dtypes import CDTYPE
from gradwave.grids import build_fft_grid, build_gsphere
from gradwave.postscf import isdf
from tests.helpers import RY


def _synthetic_orbitals(n_orb, ecut, a=6.0, seed=0):
    """Orthonormal random orbitals at Γ: (phi_r, shape, g2, volume)."""
    cell = a * np.eye(3)
    grid = build_fft_grid(cell, ecut)
    sphere = build_gsphere(grid, ecut, k_frac=(0.0, 0.0, 0.0))
    gen = torch.Generator().manual_seed(seed)
    c = torch.randn(sphere.npw, n_orb, dtype=CDTYPE, generator=gen)
    q, _ = torch.linalg.qr(c)  # (npw, n_orb), columns orthonormal in G
    phi_r = isdf.orbitals_on_grid(q.transpose(0, 1), sphere.flat_idx, grid.shape)
    return phi_r, grid.shape, grid.g2, grid.volume


def test_isdf_exact_at_saturated_rank():
    """With n_mu ≥ the pair rank, ISDF exchange ≡ the direct build."""
    phi_r, shape, g2, volume = _synthetic_orbitals(n_orb=5, ecut=8 * RY)
    e_direct = isdf.exchange_energy_direct(phi_r, shape, g2, volume)
    gen = torch.Generator().manual_seed(1)
    ex = isdf.build_exchange(phi_r, shape, g2, volume, n_mu=64, generator=gen)
    # complex orbitals: pair space is n_orb², well under 64 points
    assert ex.n_mu <= 25
    assert abs(float(ex.energy()) - float(e_direct)) < 1e-9 * abs(float(e_direct))


def test_isdf_rank_convergence():
    """Below saturation the exchange error shrinks as the rank grows; at the
    saturated rank (n_orb² for complex orbitals) it reaches machine precision."""
    phi_r, shape, g2, volume = _synthetic_orbitals(n_orb=10, ecut=10 * RY)
    e_direct = float(isdf.exchange_energy_direct(phi_r, shape, g2, volume))
    gen = torch.Generator().manual_seed(2)
    errs = []
    for n_mu in (20, 40, 80):
        ex = isdf.build_exchange(phi_r, shape, g2, volume, n_mu, generator=gen)
        errs.append(abs(float(ex.energy()) - e_direct))
    assert errs[0] > errs[1] > errs[2]
    sat = isdf.build_exchange(phi_r, shape, g2, volume, n_mu=200, generator=gen)
    assert sat.n_mu <= 100  # pair rank saturated below the requested count
    assert abs(float(sat.energy()) - e_direct) < 1e-8 * abs(e_direct)


def test_isdf_reproduces_pair_products_at_saturation():
    """The fit ζ reconstructs the pair products on the whole grid, not only at
    the interpolation points, once the rank saturates."""
    phi_r, shape, g2, volume = _synthetic_orbitals(n_orb=4, ecut=8 * RY)
    points = isdf.select_interpolation_points(phi_r, 64)
    zeta = isdf.build_isdf(phi_r, points)  # (N_r, n_mu)
    phi_mu = phi_r[:, points]
    # reconstruct rho_02(r) = phi_0*(r) phi_2(r) from the fit
    p02 = (phi_mu[0].conj() * phi_mu[2])          # (n_mu,)
    recon = zeta.to(p02.dtype) @ p02               # (N_r,)
    exact = phi_r[0].conj() * phi_r[2]
    assert torch.linalg.vector_norm(recon - exact) < 1e-9 * torch.linalg.vector_norm(exact)


def test_pivoted_columns_picks_independent_directions():
    """The pivoted-QR selector returns distinct columns and stops at the rank."""
    torch.manual_seed(0)
    # rank-3 matrix embedded in 8 columns
    basis = torch.randn(12, 3, dtype=CDTYPE)
    coeff = torch.randn(3, 8, dtype=CDTYPE)
    a = basis @ coeff
    piv = isdf._pivoted_columns(a, k=8)
    assert piv.numel() == 3                       # stops at numerical rank
    assert len(set(piv.tolist())) == piv.numel()  # no repeats
