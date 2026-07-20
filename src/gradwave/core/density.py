"""Density-gradient invariant σ = |∇ρ|² for GGA XC (Layer A).

σ = |∇ρ|² is computed spectrally (iG·ρ(G) chain) INSIDE the autograd graph,
so autograd's GGA v_xc automatically carries the −∇·(∂e/∂∇ρ) term with
spectral accuracy.

The electron density itself,

    ρ(r_j) = (1/Ω) Σ_k w_k Σ_n f_nk |ψ_nk(r_j)|²,

is built on the batched SCF path by `core.batch.density_b` (the fftbox
convention gives |g_to_r(c)|²/Ω integrating to Σf when Σ|c|² = 1 per band;
a time-reversal-reduced k-mesh already carries the doubled weights).
"""

from __future__ import annotations

import torch

from gradwave.core.fftbox import r_to_g


def sigma_from_rho(rho: torch.Tensor, g_cart: torch.Tensor) -> torch.Tensor:
    """|∇ρ|²(r) on the dense grid, computed spectrally (differentiable).

    g_cart: (n1,n2,n3,3) Cartesian G of the box (grids.FFTGrid.g_cart).
    """
    rho_g = r_to_g(rho.to(torch.complex128))
    grad = torch.fft.ifftn(
        1j * g_cart.permute(3, 0, 1, 2) * rho_g[None], dim=(-3, -2, -1)
    ) * rho.numel()
    # ifftn already includes 1/N; multiply back the N our r_to_g removed
    grad_r = grad.real
    return (grad_r**2).sum(dim=0)
