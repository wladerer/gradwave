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

from typing import TYPE_CHECKING, TypedDict

import torch

from gradwave.dtypes import RDTYPE

if TYPE_CHECKING:
    from gradwave.core.hamiltonian import ProjectorData


class AlchemicalSpec(TypedDict):
    """Endpoint spec stashed on ``System.alchemical`` by
    ``setup_alchemical_system``, and read back by ``alchemical_energy_gradient``
    (see both below). Both endpoints' per-k projector data, valence charges,
    and local-potential tables, plus the optional NLCC core-density form
    factors carried only when an endpoint has a core charge."""

    pd_a: list[ProjectorData]  # per-k, endpoint A
    pd_b: list[ProjectorData]  # per-k, endpoint B
    z_a: float
    z_b: float
    tab_a: torch.Tensor  # (n1,n2,n3) endpoint A local table [eV·Å³]
    tab_b: torch.Tensor  # (n1,n2,n3) endpoint B local table [eV·Å³]
    core_a: torch.Tensor | None  # endpoint A NLCC |G| shells, or None
    core_b: torch.Tensor | None  # endpoint B NLCC |G| shells, or None


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

    lam may be a scalar (whole cell blends A->B together) or a per-atom vector
    (na,), where each atom transmutes with its own weight. Per-atom scaling uses
    the block-diagonal structure of D, so scaling each atom's KB block by its
    weight is a row scale keyed on the projector's atom index.
    """
    from gradwave.core.hamiltonian import ProjectorData

    lam = torch.as_tensor(lam, dtype=RDTYPE)
    dij_a = _scale_dij_blocks(pd_a.dij_full, 1.0 - lam, pd_a.atom_index)
    dij_b = _scale_dij_blocks(pd_b.dij_full, lam, pd_b.atom_index)
    f = torch.cat([pd_a.f_ylm_phase_free, pd_b.f_ylm_phase_free], dim=0)
    atom_index = torch.cat([pd_a.atom_index, pd_b.atom_index])
    dij = torch.block_diag(dij_a, dij_b)
    return ProjectorData(atom_index=atom_index, f_ylm_phase_free=f,
                         kpg=pd_a.kpg, dij_full=dij)


def _scale_dij_blocks(dij: torch.Tensor, w: torch.Tensor,
                      atom_index: torch.Tensor) -> torch.Tensor:
    """Scale each atom's block of a block-diagonal KB matrix by its weight. w is
    a scalar or a per-atom (na,) vector. Because dij is block-diagonal over
    atoms, scaling the rows by w[atom_index] scales each whole block, since the
    only nonzero entries have matching atom indices."""
    if w.dim() == 0:
        return dij * w
    return dij * w[atom_index].unsqueeze(1)


def _alchemical_ionic_terms(lam, na, z_a, z_b, tab_a, tab_b, shape, pd_a, pd_b):
    """Assemble the lambda-dependent ionic pieces shared by the system builder
    and the gradient. lam is a scalar (uniform transmutation) or a per-atom (na,)
    vector. Returns per-atom charges, the per-atom local table, and the blended
    per-k projector data."""
    lam = torch.as_tensor(lam, dtype=RDTYPE)
    if lam.dim() == 0:
        charges = ((1.0 - lam) * z_a + lam * z_b).reshape(()).repeat(na)
        vloc_atom = ((1.0 - lam) * tab_a + lam * tab_b).unsqueeze(0).expand(na, *shape)
    else:
        charges = (1.0 - lam) * z_a + lam * z_b
        w = lam.reshape(na, *([1] * tab_a.dim()))
        vloc_atom = (1.0 - w) * tab_a + w * tab_b
    proj = [blend_projector_data(pd_a[k], pd_b[k], lam) for k in range(len(pd_a))]
    return charges, vloc_atom, proj


def _alchemical_core_density(lam, na, pos_t, grid, core_a, core_b):
    """Blended NLCC core density rho_core(r), or None if neither endpoint carries
    one. Per atom (1-lam_a) n_core^A + lam_a n_core^B, so E_xc(rho + rho_core)
    tracks the semicore through the transmutation. Differentiable in lam. A pseudo
    with a semicore core (for example Ge) needs this, and dropping it leaves E_xc
    wrong by hundreds of eV."""
    if core_a is None and core_b is None:
        return None
    from gradwave.scf.setup_common import assemble_core_density

    lam = torch.as_tensor(lam, dtype=RDTYPE)
    ref = core_a if core_a is not None else core_b
    ca = core_a if core_a is not None else torch.zeros_like(ref)
    cb = core_b if core_b is not None else torch.zeros_like(ref)
    w = lam.repeat(na) if lam.dim() == 0 else lam
    shells = [(1.0 - w[a]) * ca + w[a] * cb for a in range(na)]
    return assemble_core_density(shells, list(range(na)), pos_t, grid)


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
    g_flat = np.sqrt(grid.g2.reshape(-1).numpy())
    uniq, inverse = _unique_shells(g_flat)
    tab_a, tab_b = endpoint_local_tables(upf_a, upf_b, uniq, inverse, grid.shape)
    charges, vloc_atom, proj_data = _alchemical_ionic_terms(
        lam, na, z_a, z_b, tab_a, tab_b, grid.shape,
        sys_a.proj_data, sys_b.proj_data)
    n_electrons = float(charges.detach().sum())
    batch = build_batched(sys_a.spheres, proj_data)

    from gradwave.scf.setup_common import core_shell_tables
    core_a = core_shell_tables([upf_a], uniq, inverse)[0]
    core_b = core_shell_tables([upf_b], uniq, inverse)[0]
    pos_t = torch.as_tensor(np.asarray(positions), dtype=RDTYPE)
    rho_core = _alchemical_core_density(lam, na, pos_t, grid, core_a, core_b)

    if nbands is None:
        nbands = default_nbands(max(na * z_a, na * z_b))
    spec: AlchemicalSpec = {"pd_a": sys_a.proj_data, "pd_b": sys_b.proj_data,
            "z_a": z_a, "z_b": z_b, "tab_a": tab_a, "tab_b": tab_b,
            "core_a": core_a, "core_b": core_b}
    return dataclasses.replace(sys_a, charges=charges, n_electrons=n_electrons,
                               vloc_atom=vloc_atom, proj_data=proj_data,
                               batch=batch, nbands=nbands, rho_core=rho_core,
                               alchemical=spec)


def alchemical_energy_gradient(res, lam, xc=None):
    """dE/dlambda through the converged SCF by Hellmann-Feynman (phases 3b, 4).

    At the self-consistent density only the ionic terms depend on lambda, so
    dE/dlambda = d(E_local + E_nl + E_ewald)/dlambda evaluated at the converged
    (detached) density and orbitals. The XC, Hartree, and kinetic terms are
    stationary in the density, so their lambda-derivative vanishes at convergence
    (the envelope theorem), and the derivative is the transmutation energy. Needs
    a system built by setup_alchemical_system, which stashes the endpoint spec.

    xc is required only when an endpoint carries an NLCC core charge. The core
    density depends on lambda, so E_xc(rho + rho_core(lambda)) contributes a
    core-correction term, the composition analogue of the NLCC force. lam matches
    the shape used to build the system, so a scalar returns the uniform
    dE/dlambda and a per-atom (na,) vector returns the per-site gradient.
    """
    from gradwave.core.energies.ewald import ewald_energy
    from gradwave.core.energies.local_pp import local_energy, local_potential_g
    from gradwave.core.energies.nl_pp import nonlocal_energy
    from gradwave.core.fftbox import r_to_g
    from gradwave.core.hamiltonian import becp, projectors

    system = res.system
    spec = system.alchemical
    if spec is None:
        raise ValueError("res.system was not built by setup_alchemical_system")
    grid = system.grid
    nspin = getattr(res, "nspin", 1)
    na = len(system.positions)
    lam = torch.as_tensor(lam, dtype=RDTYPE).detach().clone().requires_grad_(True)

    charges, vloc_atom, proj_lam = _alchemical_ionic_terms(
        lam, na, spec["z_a"], spec["z_b"], spec["tab_a"], spec["tab_b"],
        grid.shape, spec["pd_a"], spec["pd_b"])

    rho_g = r_to_g(res.rho.detach().to(torch.complex128))
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume,
                               vloc_atom=vloc_atom)
    e_local = local_energy(rho_g, vloc_g, grid.volume)
    e_ewald = ewald_energy(system.positions, charges, grid.cell)

    # the blended KB matrix is k-independent (D_A, D_B do not depend on k), so one
    # dij serves every k, exactly as the force path stacks a single dij
    projs = [projectors(pd, system.positions) for pd in proj_lam]
    dij_lam = proj_lam[0].dij_full
    coeffs_s = res.coeffs if nspin == 2 else [res.coeffs]
    occ_s = res.occupations.detach()
    occ_s = occ_s if nspin == 2 else occ_s[None]
    e_nl = torch.zeros((), dtype=RDTYPE)
    for sp in range(nspin):
        cs = [c.detach() for c in coeffs_s[sp]]
        becps = [becp(projs[ik], cs[ik]) for ik in range(len(cs))]
        e_nl = e_nl + nonlocal_energy(becps, dij_lam, occ_s[sp], system.kweights)

    e_ion = e_local + e_nl + e_ewald

    # Heterovalent (charge-changing) transmutation: the electron count follows the
    # ionic charge, N(lambda) = sum_i Z_i(lambda), so the free energy gains the
    # Janak chemical-potential term mu * dN/dlambda with mu the Fermi level. This
    # vanishes for isovalent pairs where N is constant.
    if getattr(res, "fermi", None) is not None:
        e_ion = e_ion + float(res.fermi) * charges.sum()

    # NLCC core-correction: rho_core depends on lambda, so E_xc(rho + rho_core)
    # carries a lambda derivative at the (detached) valence density
    has_core = spec.get("core_a") is not None or spec.get("core_b") is not None
    if has_core:
        if xc is None:
            raise ValueError("an endpoint has an NLCC core charge; pass xc so the "
                             "core-correction term can be evaluated")
        from gradwave.core.density import sigma_from_rho
        core_lam = _alchemical_core_density(lam, na, system.positions, grid,
                                            spec["core_a"], spec["core_b"])
        rho_xc = res.rho.detach() + core_lam
        sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
        e_ion = e_ion + xc.energy(rho_xc, grid.volume, sigma, None)

    (grad,) = torch.autograd.grad(e_ion, lam)
    return grad.detach()


def per_atom_local_tables(
    base_tables: torch.Tensor, species_index: torch.Tensor,
    alchemical: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
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
