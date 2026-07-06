import math

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.core.energies.hartree import hartree_energy, hartree_potential_g
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
