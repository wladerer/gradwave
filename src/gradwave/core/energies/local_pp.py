"""Local pseudopotential energy and potential (Layer A).

Given per-species form-factor tables v_s(|G|) [eV·Å³] evaluated on the dense
box (setup layer, pseudo/local.py — G=0 entry set to the alpha-Z value α_s),

    V_loc(G) = (1/Ω) Σ_a S_a(G) · v_{s(a)}(|G|)          [eV]
    E_loc    = Ω Σ_{G ∈ dens sphere} ρ*(G) V_loc(G)       [eV, real]

The G=0 term is INCLUDED here through α: ρ(0)·Σ_a α_a = (N_e/Ω)Σα_a — the
finite short-range moment that survives the Coulomb-tail cancellation
(ownership table in energies/total.py). Positions enter via S_a(G):
this term carries local-potential forces.
"""

from __future__ import annotations

import torch

from gradwave.core.structure import structure_factors


def local_potential_g(
    positions: torch.Tensor,  # (na, 3) Å, may require grad
    species_index: torch.Tensor,  # (na,) int — row of each atom in vloc_tables
    vloc_tables: torch.Tensor,  # (nspecies, n1, n2, n3) [eV·Å³], G=0 entry = alpha-Z
    g_cart: torch.Tensor,  # (n1, n2, n3, 3)
    volume: float,
) -> torch.Tensor:
    """V_loc(G) on the dense box [eV], complex."""
    s = structure_factors(positions, g_cart)  # (na, n1,n2,n3)
    v = torch.einsum("axyz,axyz->xyz", s, vloc_tables[species_index].to(s.dtype))
    return v / volume


def local_energy(rho_g: torch.Tensor, vloc_g: torch.Tensor, volume: float) -> torch.Tensor:
    """E_loc = Ω Σ_G ρ*(G) V_loc(G) [eV]."""
    return volume * (rho_g.conj() * vloc_g).sum().real
