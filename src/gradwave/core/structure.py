"""Structure factors — differentiable in atomic positions (Layer A).

S_a(G) = exp(−i G·τ_a), with τ in Cartesian Å and G in Cartesian Å⁻¹.
The 2π lives inside the reciprocal vectors (grids.reciprocal_cell); it must
never be applied again here.

This is one of only three places atomic positions enter the total energy
(the others: Ewald and the nonlocal projector phases) — all force
contributions flow through these.
"""

from __future__ import annotations

import torch


def structure_factors(positions: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """S_a(G) = e^{−iG·τ_a}.

    positions: (na, 3) Cartesian Å (may require grad); g: (..., 3) Å⁻¹.
    Returns complex (na, ...).
    """
    phase = torch.tensordot(positions, g, dims=([1], [g.dim() - 1]))  # (na, ...)
    return torch.exp(torch.complex(torch.zeros_like(phase), -phase))
