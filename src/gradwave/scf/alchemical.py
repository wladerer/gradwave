"""Alchemical composition channel, phase 1 (local potential and ionic charge).

A per-atom weight lambda in [0, 1] blends two endpoint pseudopotentials so the
ionic charge and the local potential become differentiable in composition. At
lambda=0 the atom is species A, at lambda=1 it is species B, and dE/dlambda is
the exact alchemical derivative of the local and Ewald terms by Hellmann-Feynman.

This never averages a potential into a fictitious crystal, so it is not the
virtual crystal approximation. The blended endpoints are real species and the
derivative is taken at those real endpoints. The nonlocal (KB) projectors are
not blended here, so full-SCF endpoint exactness is a later phase. What phase 1
establishes is the differentiable local-potential and charge channel, verified
against finite difference.
"""

from __future__ import annotations

import torch

from gradwave.dtypes import RDTYPE


def alchemical_charges(z_a: float, z_b: float, lam: torch.Tensor) -> torch.Tensor:
    """Per-atom ionic charge (1-lambda) Z_A + lambda Z_B [e], differentiable in
    lambda. lam is a per-atom weight tensor, so a cell can mix untouched atoms
    (weight held at 0 or 1) with alchemical ones."""
    lam = torch.as_tensor(lam, dtype=RDTYPE)
    return (1.0 - lam) * float(z_a) + lam * float(z_b)


def endpoint_local_tables(upf_a, upf_b, uniq, inverse, shape):
    """The two endpoint local form-factor tables [eV.Ang^3] on the dense box,
    with the G=0 entry set to alpha-Z. Built through the same path as
    setup_common.build_vloc_tables, so the endpoints match a plain setup."""
    from gradwave.scf.setup_common import build_vloc_tables

    tabs = build_vloc_tables([upf_a, upf_b], uniq, inverse, shape,
                             guard_single_shell=True)
    return tabs[0], tabs[1]


def blend_local_table(tab_a: torch.Tensor, tab_b: torch.Tensor,
                      lam: torch.Tensor) -> torch.Tensor:
    """One atom's blended local table (1-lambda) v_A + lambda v_B, differentiable
    in the scalar weight lambda. tab_a and tab_b are (n1, n2, n3) endpoint
    tables."""
    lam = torch.as_tensor(lam, dtype=RDTYPE)
    return (1.0 - lam) * tab_a + lam * tab_b


def blend_projector_data(pd_a, pd_b, lam: torch.Tensor):
    """Blend two single-atom ProjectorData for the same atom into one alchemical
    atom (phase 2, the nonlocal channel).

    The nonlocal operator V_nl = sum_i |beta_i> D_i <beta_i| is linear in the KB
    energies D, so (1-lambda) V_nl^A + lambda V_nl^B is realized by carrying both
    endpoints' projector columns and the block-diagonal matrix
    diag((1-lambda) D_A, lambda D_B). The projector columns are the fixed
    endpoint form factors, so lambda enters through D alone. At lambda=0 the B
    block is zero and only A acts, at lambda=1 only B acts, and the existing
    apply consumes the result unchanged. The two must share the k-sphere and
    refer to the same atom, and no angular-momentum channels need to match.
    """
    from gradwave.core.hamiltonian import ProjectorData

    lam = torch.as_tensor(lam, dtype=RDTYPE)
    f = torch.cat([pd_a.f_ylm_phase_free, pd_b.f_ylm_phase_free], dim=0)
    atom_index = torch.cat([pd_a.atom_index, pd_b.atom_index])
    dij = torch.block_diag((1.0 - lam) * pd_a.dij_full, lam * pd_b.dij_full)
    return ProjectorData(atom_index=atom_index, f_ylm_phase_free=f,
                         kpg=pd_a.kpg, dij_full=dij)


def per_atom_local_tables(base_tables: torch.Tensor, species_index: torch.Tensor,
                          alchemical: dict) -> torch.Tensor:
    """Assemble the full (na, n1, n2, n3) per-atom local table that
    local_potential_g consumes through its vloc_atom argument.

    Untouched atoms gather their species row from base_tables. Each alchemical
    atom, keyed by its atom index in `alchemical`, carries a
    (tab_a, tab_b, lam) triple and contributes the blended table, so the result
    stays differentiable in every lambda that appears.
    """
    per_atom = base_tables[species_index].clone()
    rows = list(per_atom)  # keep autograd through the alchemical rows
    for a, (tab_a, tab_b, lam) in alchemical.items():
        rows[a] = blend_local_table(tab_a, tab_b, lam)
    return torch.stack(rows, dim=0)
