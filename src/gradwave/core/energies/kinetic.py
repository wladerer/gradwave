"""Kinetic energy — diagonal in the plane-wave basis (Layer A)."""

from __future__ import annotations

import torch

from gradwave.constants import HBAR2_2M


def kinetic_energy(
    coeffs_per_k: list[torch.Tensor],  # [(nb, npw_k) complex]
    occ: torch.Tensor,  # (nk, nb) in [0, 2]
    kweights: torch.Tensor,  # (nk,)
    spheres: list,  # [GSphere]
) -> torch.Tensor:
    """E_kin = Σ_k w_k Σ_n f_nk Σ_G (ħ²/2m)|k+G|² |c_nk(G)|²  [eV]."""
    e = None
    for ik, c in enumerate(coeffs_per_k):
        t = HBAR2_2M * spheres[ik].kpg2  # (npw,)
        band = torch.einsum("bg,g->b", (c.real**2 + c.imag**2), t)
        term = (kweights[ik] * occ[ik, : c.shape[0]] * band).sum()
        e = term if e is None else e + term
    return e
