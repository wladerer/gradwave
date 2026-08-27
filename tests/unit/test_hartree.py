import math

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.core.energies.hartree import (
    hartree_energy,
    hartree_potential_g,
    hartree_potential_r,
)
from gradwave.grids import build_fft_grid


def test_neutral_double_gaussian_vs_analytic():
    # ρ(r) = q[g_{α1}(r) − g_{α2}(r)] with normalized Gaussians: neutral, so
    # periodic-image corrections decay exponentially in L and
    # E_H = (E2/2) q² [I(α1,α1) − 2I(α1,α2) + I(α2,α2)],
    # I(a,b) = (2/√π) √(ab/(a+b))  (two unit Gaussians, same center).
    L, q, a1, a2 = 14.0, 2.0, 1.1, 0.4
    cell = L * np.eye(3)
    grid = build_fft_grid(cell, ecut=250.0)
    vol = grid.volume

    # analytic ρ(G) (Fourier-series coefficients of periodized Gaussians)
    g2 = grid.g2
    rho_g = (q / vol) * (torch.exp(-g2 / (4 * a1)) - torch.exp(-g2 / (4 * a2)))
    rho_g = rho_g.to(torch.complex128)

    e = hartree_energy(rho_g, grid.g2, vol).item()

    def pair(a, b):
        return (2.0 / math.sqrt(math.pi)) * math.sqrt(a * b / (a + b))

    ref = 0.5 * E2 * q**2 * (pair(a1, a1) - 2 * pair(a1, a2) + pair(a2, a2))
    assert abs(e - ref) / abs(ref) < 1e-10


def test_potential_consistency():
    # E_H must equal (Ω/2) Σ_G ρ*(G) v_H(G)
    L = 10.0
    grid = build_fft_grid(L * np.eye(3), ecut=120.0)
    gen = torch.Generator().manual_seed(2)
    raw = torch.randn(*grid.shape, generator=gen, dtype=torch.float64)
    rho_g = torch.fft.fftn(raw).to(torch.complex128) / raw.numel()
    rho_g = torch.where(grid.dens_mask, rho_g, torch.zeros_like(rho_g))

    e = hartree_energy(rho_g, grid.g2, grid.volume)
    v = hartree_potential_g(rho_g, grid.g2)
    e2 = 0.5 * grid.volume * (rho_g.conj() * v).sum().real
    assert torch.allclose(e, e2, rtol=1e-12)
    assert v.reshape(-1)[0].abs() == 0  # v_H(G=0) ≡ 0


def test_real_space_potential_matches_full_g_path_triclinic():
    # Pins hartree_potential_r (the rfft path with the Nyquist-plane inv_g2
    # symmetrization) against the full complex-FFT reference built from the
    # already-pinned hartree_potential_g. A NON-orthogonal cell is deliberate:
    # on an orthogonal cell inv_g2 is already symmetric so the symmetrization is
    # a no-op — the triclinic cell is what makes it load-bearing.
    #
    # This closes a real unit-level gap the mutation probe surfaced: the 4π e²
    # prefactor (hartree.py:70) and the symmetrization (hartree.py:65-67) were
    # exercised only by the integration/SCF tier, not by any unit test — mutating
    # either survived the fast suite.
    cell = np.array([[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 1.5, 8.0]])
    grid = build_fft_grid(cell, ecut=120.0)
    gen = torch.Generator().manual_seed(7)
    rho_r = torch.randn(*grid.shape, generator=gen, dtype=torch.float64)

    v_r = hartree_potential_r(rho_r, grid.g2)

    # full complex-FFT reference: v_H(r) = IFFT[ 4π e² ρ(G)/|G|² ]
    rho_g = torch.fft.fftn(rho_r).to(torch.complex128)
    v_g = hartree_potential_g(rho_g, grid.g2)
    v_r_ref = torch.fft.ifftn(v_g).real

    # the rfft/symmetrized path reproduces the full path to machine precision
    assert torch.allclose(v_r, v_r_ref, atol=1e-12, rtol=0)
