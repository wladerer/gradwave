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


def setup_alchemical_system(cell, positions, upf_a, upf_b, lam, ecut,
                            kmesh=(1, 1, 1), nbands=None, **setup_kw):
    """A System whose atoms uniformly blend two endpoint species A and B with a
    scalar weight lam, phase 3. It runs a full SCF at any lam. lam=0 is species
    A, lam=1 is species B. The ionic charge and local potential blend per atom
    and the nonlocal operator blends per k, so the total energy is differentiable
    in lam through charges, the local table, and dij_full.

    The blend carries both endpoints' projectors, so at lam=0 the B projectors
    are zero-coupled (inert) and the SCF reproduces pure A, and at lam=1 the A
    projectors are inert and it reproduces pure B. This is not the virtual
    crystal approximation, since the endpoints are real and the derivative is
    taken there.
    """
    import dataclasses

    import numpy as np

    from gradwave.core.batch import build_batched
    from gradwave.scf.loop import setup_system
    from gradwave.scf.setup_common import _unique_shells, default_nbands

    na = len(positions)
    lam = torch.as_tensor(lam, dtype=RDTYPE)
    species = [0] * na
    sys_a = setup_system(cell, positions, species, [upf_a], ecut, kmesh, **setup_kw)
    sys_b = setup_system(cell, positions, species, [upf_b], ecut, kmesh, **setup_kw)
    grid = sys_a.grid

    z_a, z_b = float(upf_a.z_valence), float(upf_b.z_valence)
    charges = ((1.0 - lam) * z_a + lam * z_b).reshape(()).repeat(na)  # (na,), grad-carrying
    n_electrons = float(charges.detach().sum())

    g_flat = np.sqrt(grid.g2.reshape(-1).numpy())
    uniq, inverse = _unique_shells(g_flat)
    tab_a, tab_b = endpoint_local_tables(upf_a, upf_b, uniq, inverse, grid.shape)
    blended = blend_local_table(tab_a, tab_b, lam)  # (n1,n2,n3)
    vloc_atom = blended.unsqueeze(0).expand(na, *grid.shape)

    proj_data = [blend_projector_data(sys_a.proj_data[k], sys_b.proj_data[k], lam)
                 for k in range(len(sys_a.spheres))]
    batch = build_batched(sys_a.spheres, proj_data)

    if nbands is None:
        nbands = default_nbands(max(na * z_a, na * z_b))
    return dataclasses.replace(sys_a, charges=charges, n_electrons=n_electrons,
                               vloc_atom=vloc_atom, proj_data=proj_data,
                               batch=batch, nbands=nbands)


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
