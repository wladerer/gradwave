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
    vloc_atom: torch.Tensor | None = None,  # (na, n1,n2,n3) per-atom override
) -> torch.Tensor:
    """V_loc(G) on the dense box [eV], complex.

    vloc_atom overrides the per-species gather with a per-atom table. The
    alchemical composition channel passes a lambda-blended table there, so
    V_loc stays differentiable in composition (scf/alchemical.py)."""
    if positions.requires_grad or torch.is_grad_enabled():
        # Differentiable path (forces / alchemical composition rebuild this with
        # grad): one batched structure-factor block + a single einsum keeps a
        # clean autograd node. The SCF's V_loc build runs under no_grad.
        s = structure_factors(positions, g_cart)  # (na, n1,n2,n3)
        tab = vloc_atom if vloc_atom is not None else vloc_tables[species_index]
        v = torch.einsum("axyz,axyz->xyz", s, tab.to(s.dtype))
        return v / volume
    # Memory-light no-grad path: accumulate per atom to avoid the full
    # (na, n1,n2,n3) structure-factor block (measured 0.36 GiB at Si-64/30 Ry).
    # Same accumulation order as the einsum over `a`, so agreement is at the
    # ~1e-16 ulp level (the sequential add vs the einsum's internal reduction).
    v = None
    for a in range(positions.shape[0]):
        sa = structure_factors(positions[a : a + 1], g_cart)[0]
        ta = vloc_atom[a] if vloc_atom is not None else vloc_tables[species_index[a]]
        va = sa * ta.to(sa.dtype)
        v = va if v is None else v + va
    assert v is not None  # >=1 atom always
    return v / volume


def local_energy(rho_g: torch.Tensor, vloc_g: torch.Tensor, volume: float) -> torch.Tensor:
    """E_loc = Ω Σ_G ρ*(G) V_loc(G) [eV]."""
    return volume * (rho_g.conj() * vloc_g).sum().real
