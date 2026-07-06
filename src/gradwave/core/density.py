"""Electron density from orbital coefficients (Layer A).

ρ(r_j) = (1/Ω) Σ_k w_k Σ_n f_nk |ψ_nk(r_j)|²,  ψ from the fftbox convention
(so |g_to_r(c)|²/Ω integrates to Σf when Σ|c|² = 1 per band).

With a time-reversal-reduced k-mesh, the −k contribution equals |ψ_k*|² =
|ψ_k|², so doubled weights (from kpoints.monkhorst_pack) are already correct.
ρ is real by construction; we take .real and assert the imaginary residue
in debug mode only (callers may run under torch.func where asserts on data
are not allowed).

Also here: σ = |∇ρ|² computed spectrally (iG·ρ(G) chain) INSIDE the autograd
graph, so GGA v_xc from autograd is complete.
"""

from __future__ import annotations

import torch

from gradwave.core.fftbox import g_to_r, r_to_g


def density_from_orbitals(
    coeffs_per_k: list[torch.Tensor],  # [(nb, npw_k) complex]
    occ: torch.Tensor,  # (nk, nb), values in [0, 2]
    kweights: torch.Tensor,  # (nk,), sums to 1
    spheres: list,  # [GSphere]
    shape: tuple[int, int, int],
    volume: float,
) -> torch.Tensor:
    """ρ(r) on the dense grid [e/Å³], real tensor."""
    rho = None
    for ik, c in enumerate(coeffs_per_k):
        psi = g_to_r(c, spheres[ik].flat_idx, shape)  # (nb, n1,n2,n3)
        w = kweights[ik] * occ[ik, : c.shape[0]]  # (nb,)
        contrib = torch.einsum("b,bxyz->xyz", w.to(psi.real.dtype), psi.real**2 + psi.imag**2)
        rho = contrib if rho is None else rho + contrib
    return rho / volume


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
