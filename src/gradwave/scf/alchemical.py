"""Alchemical composition channel: differentiable substitution and design gradients.

A per-atom weight lambda in [0, 1] blends two endpoint pseudopotentials so
composition becomes a differentiable coordinate. Every ionic piece blends: the
ionic charge, the local potential, the KB nonlocal projectors (both endpoints'
columns carried with the inactive one zero-coupled), and the NLCC core density.
At lambda=0 the atom is species A, at lambda=1 species B. This is not the virtual
crystal approximation -- the endpoints are real species and derivatives are taken
there, so lambda=0/1 reproduce the pure cells to SCF noise.

Two builders: ``setup_alchemical_system`` blends a whole cell between one A->B
pair; ``setup_alchemical_substitution`` transmutes a chosen subset of sites in a
real multi-species cell (the perovskite / defect / doping case). Two gradients:
``alchemical_energy_gradient`` gives dE/dlambda for free by Hellmann-Feynman (the
energy is variational, so the density response drops by the envelope theorem);
``alchemical_gap_gradient`` gives the relaxed d(E_gap)/dlambda by a composition
DFPT -- an eigenvalue is not variational, so it carries the self-consistent
density response to a local+nonlocal bare perturbation (Sternheimer with an
operator RHS, then a forward Dyson dressing). Both are checked against finite
difference. nspin=1 insulator for the gap DFPT; the energy gradient is general.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from gradwave.dtypes import RDTYPE

if TYPE_CHECKING:
    from gradwave.core.hamiltonian import HamiltonianK, ProjectorData


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


class SubstitutionSpec(TypedDict):
    """Endpoint spec stashed on ``System.alchemical`` by
    ``setup_alchemical_substitution`` and read back by the relaxed alchemical
    gradient (``alchemical_gap_gradient`` / ``alchemical_eigenvalue_gradient``).

    Unlike :class:`AlchemicalSpec` (a single A→B pair over the whole cell), this
    carries a heterogeneous multi-species cell in which only ``sub_atoms``
    transmute. Everything here is a detached λ-derivative building block: the
    perturbation ∂V_ion/∂λ is linear in λ (a blend), so its derivative is the
    fixed endpoint difference these fields encode. ``sub_mask`` is 1.0 on the
    substituted sites and 0.0 elsewhere, keyed by atom index.

    Local: per-atom ∂v_loc/∂λ = tab_target − tab_base, rebuilt from the two
    ``vloc_tables`` and species maps. Nonlocal: ∂D/∂λ = block_diag(−mask·D_a,
    +mask·D_b) over the stacked [A cols, B cols] layout ``blend_projector_data``
    produced, so ``dij_a``/``dij_b`` (k-independent) and the two atom-index maps
    reconstruct it. Core (NLCC): per-atom ∂n_core/∂λ = core_target − core_base.
    """

    sub_atoms: list[int]
    sub_mask: torch.Tensor  # (na,) 1.0 on substituted sites, else 0.0
    vloc_tables_a: torch.Tensor  # base per-species local tables [eV·Å³]
    species_a: list[int]  # per-atom base species row into vloc_tables_a
    vloc_tables_b: torch.Tensor  # target twin per-species tables
    species_b: list[int]  # per-atom target species row into vloc_tables_b
    dij_a: torch.Tensor  # (nproj_a, nproj_a) base KB matrix [eV], k-independent
    dij_b: torch.Tensor  # (nproj_b, nproj_b) target KB matrix [eV]
    atom_index_a: torch.Tensor  # (nproj_a,) atom of each base projector column
    atom_index_b: torch.Tensor  # (nproj_b,) atom of each target projector column
    core_base: list[torch.Tensor | None]  # per base-species NLCC |G| shells / None
    tgt_cores: dict[int, torch.Tensor | None]  # substituted atom → target shells / None
    z_base: torch.Tensor  # (na,) per-atom base valence charge [e]
    z_target: torch.Tensor  # (na,) per-atom target valence charge [e]


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


def _substitution_energy_gradient(res, lam, xc=None, grand_canonical=False):
    """dE/dλ for a heterogeneous substitution System (rung 2 of the aliovalent
    ladder), by Hellmann-Feynman at the converged density.

    Same envelope-theorem structure as the binary
    :func:`alchemical_energy_gradient`, but the ionic terms are rebuilt per atom
    from the ``SubstitutionSpec``. For an ALIOVALENT (charge-changing)
    transmutation the electron count follows the ionic charge, N(λ)=ΣZ_i(λ), so
    the free energy carries the Janak chemical-potential term μ·dN/dλ (μ the
    Fermi level) — the same term the binary path already has. ``lam`` is a scalar
    driving all substituted sites together.

    ``grand_canonical`` (rung 4) drops the Janak term to return the grand-potential
    derivative dΩ/dλ = dF/dλ − μ·dN/dλ instead of the canonical dF/dλ. At fixed
    chemical potential the μ·dN/dλ bookkeeping cancels against the −μN Legendre
    term, so the grand-canonical alchemical gradient is exactly the bare
    Hellmann-Feynman ionic derivative — the reservoir absorbs the electron-count
    change (the standard indirect Legendre route to grand-canonical properties).
    """
    from gradwave.core.energies.ewald import ewald_energy
    from gradwave.core.energies.local_pp import local_energy, local_potential_g
    from gradwave.core.energies.nl_pp import nonlocal_energy
    from gradwave.core.fftbox import r_to_g
    from gradwave.core.hamiltonian import becp, projectors
    from gradwave.scf.setup_common import assemble_core_density

    system = res.system
    spec = system.alchemical
    grid = system.grid
    nspin = getattr(res, "nspin", 1)
    na = len(system.positions)
    lam = torch.as_tensor(lam, dtype=RDTYPE).detach().clone().requires_grad_(True)
    lam_vec = lam * spec["sub_mask"]  # scalar λ → per-atom rate (0 on fixed sites)

    charges = (1.0 - lam_vec) * spec["z_base"] + lam_vec * spec["z_target"]

    # local: per-atom (1-λ)tab_base + λ tab_target
    ta = spec["vloc_tables_a"][torch.as_tensor(spec["species_a"])]
    tb = spec["vloc_tables_b"][torch.as_tensor(spec["species_b"])]
    w = lam_vec.reshape(na, *([1] * (ta.dim() - 1)))
    vloc_atom = (1.0 - w) * ta + w * tb
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume,
                               vloc_atom=vloc_atom)
    rho_g = r_to_g(res.rho.detach().to(torch.complex128))
    e_local = local_energy(rho_g, vloc_g, grid.volume)
    e_ewald = ewald_energy(system.positions, charges, grid.cell)

    # nonlocal: dij(λ) over the stacked [A cols, B cols] layout already on
    # system.proj_data (columns are λ-independent; only the KB energies scale).
    dij_a = _scale_dij_blocks(spec["dij_a"], 1.0 - lam_vec, spec["atom_index_a"])
    dij_b = _scale_dij_blocks(spec["dij_b"], lam_vec, spec["atom_index_b"])
    dij_lam = torch.block_diag(dij_a, dij_b)
    projs = [projectors(pd, system.positions) for pd in system.proj_data]
    coeffs_s = res.coeffs if nspin == 2 else [res.coeffs]
    occ_s = res.occupations.detach()
    occ_s = occ_s if nspin == 2 else occ_s[None]
    e_nl = torch.zeros((), dtype=RDTYPE)
    for sp in range(nspin):
        cs = [c.detach() for c in coeffs_s[sp]]
        becps = [becp(projs[ik], cs[ik]) for ik in range(len(cs))]
        e_nl = e_nl + nonlocal_energy(becps, dij_lam, occ_s[sp], system.kweights)

    e_ion = e_local + e_nl + e_ewald

    # Janak term: N(λ) follows the ionic charge for an aliovalent transmutation.
    # Omitted in the grand-canonical ensemble, where it cancels the −μN Legendre
    # term (rung 4): dΩ/dλ is then the bare Hellmann-Feynman ionic derivative.
    if not grand_canonical and getattr(res, "fermi", None) is not None:
        e_ion = e_ion + float(res.fermi) * charges.sum()

    core_base = spec["core_base"]
    tgt_cores = spec["tgt_cores"]
    has_core = any(c is not None for c in core_base) or any(
        c is not None for c in tgt_cores.values())
    if has_core:
        if xc is None:
            raise ValueError("an endpoint has an NLCC core charge; pass xc")
        from gradwave.core.density import sigma_from_rho
        ref = next(c for c in list(core_base) + list(tgt_cores.values())
                   if c is not None)
        shells = []
        for i in range(na):
            base_c = core_base[spec["species_a"][i]]
            base_c = base_c if base_c is not None else torch.zeros_like(ref)
            if i in tgt_cores:
                tc = tgt_cores[i]
                tc = tc if tc is not None else torch.zeros_like(ref)
                shells.append((1.0 - lam_vec[i]) * base_c + lam_vec[i] * tc)
            else:
                shells.append(base_c)
        core_lam = assemble_core_density(shells, list(range(na)),
                                         system.positions, grid)
        rho_xc = res.rho.detach() + core_lam
        sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
        e_ion = e_ion + xc.energy(rho_xc, grid.volume, sigma, None)

    (grad,) = torch.autograd.grad(e_ion, lam)
    return grad.detach()


def alchemical_energy_gradient(res, lam, xc=None, grand_canonical=False):
    """dE/dlambda through the converged SCF by Hellmann-Feynman (phases 3b, 4).

    At the self-consistent density only the ionic terms depend on lambda, so
    dE/dlambda = d(E_local + E_nl + E_ewald)/dlambda evaluated at the converged
    (detached) density and orbitals. The XC, Hartree, and kinetic terms are
    stationary in the density, so their lambda-derivative vanishes at convergence
    (the envelope theorem), and the derivative is the transmutation energy. Works
    for both the whole-cell binary (setup_alchemical_system) and the heterogeneous
    substitution (setup_alchemical_substitution) endpoint specs.

    xc is required only when an endpoint carries an NLCC core charge. The core
    density depends on lambda, so E_xc(rho + rho_core(lambda)) contributes a
    core-correction term, the composition analogue of the NLCC force. For an
    aliovalent (charge-changing) transmutation the electron count follows the
    ionic charge and the free energy gains the Janak term mu*dN/dlambda (rung 2).

    ``grand_canonical=True`` (rung 4) drops the Janak term and returns the grand
    potential derivative dOmega/dlambda = dF/dlambda - mu*dN/dlambda instead of
    the canonical dF/dlambda. At fixed chemical potential the two are linked by a
    Legendre transform, and the grand-canonical alchemical gradient reduces to the
    bare Hellmann-Feynman ionic derivative (no chemical-potential bookkeeping).
    lam matches the shape used to build the system, so a scalar returns the uniform
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
        raise ValueError(
            "res.system was not built by setup_alchemical_system/substitution")
    if "sub_atoms" in spec:  # heterogeneous substitution spec
        return _substitution_energy_gradient(res, lam, xc, grand_canonical)
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
    # vanishes for isovalent pairs where N is constant, and is dropped in the
    # grand-canonical ensemble (rung 4), leaving the bare ionic derivative.
    if not grand_canonical and getattr(res, "fermi", None) is not None:
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


def setup_alchemical_substitution(cell, positions, pseudos, species_index,
                                  substitutions, lam, ecut, kmesh=(1, 1, 1),
                                  nbands=None, **setup_kw):
    """Heterogeneous alchemical System: a real multi-species cell in which a chosen
    subset of sites transmutes toward a target species with weight ``lam``, while
    every other site stays its own fixed species.

    This is the perovskite/defect/doping generalization of
    :func:`setup_alchemical_system` (which blends the WHOLE cell between a single
    endpoint pair). ``pseudos`` is the base per-species UPF list and
    ``species_index`` the per-atom species (as passed to ``setup_system``).
    ``substitutions`` maps an atom index to the target parsed UPF it transmutes
    toward (e.g. every X-site of an ABX3 mapped to a different halide). ``lam`` is
    a scalar (all substituted sites move together) or a vector over the substituted
    sites in ``substitutions`` order. At ``lam=0`` this is the untouched base cell;
    at ``lam=1`` the substituted sites are fully the target species. Only the ionic
    terms (local table, KB projectors, ionic charge, NLCC core) depend on ``lam``.

    The build mirrors :func:`setup_alchemical_system`: it forms the base cell and a
    target twin (identical except at the substituted sites), then blends per atom
    with a weight vector that is 0 on the fixed sites and ``lam`` on the substituted
    ones. The blended KB operator carries both endpoints' projectors with the
    inactive block zero-coupled, so ``lam=0``/``lam=1`` reproduce the pure cells.
    """
    import dataclasses

    import numpy as np

    from gradwave.core.batch import build_batched
    from gradwave.scf.loop import setup_system
    from gradwave.scf.setup_common import (
        _unique_shells,
        assemble_core_density,
        core_shell_tables,
        default_nbands,
    )

    na = len(positions)
    species_index = list(species_index)
    sub_atoms = list(substitutions.keys())

    # base cell (lam=0) and a target twin (lam=1) that differs only at the
    # substituted sites -- each gets a fresh species slot for its target pseudo.
    sys_a = setup_system(cell, positions, species_index, list(pseudos),
                         ecut, kmesh, **setup_kw)
    tgt_species = list(species_index)
    tgt_pseudos = list(pseudos)
    for a, tgt_upf in substitutions.items():
        tgt_pseudos.append(tgt_upf)
        tgt_species[a] = len(tgt_pseudos) - 1
    sys_b = setup_system(cell, positions, tgt_species, tgt_pseudos, ecut, kmesh,
                         **setup_kw)

    # per-atom transmutation weight: 0 on fixed sites, lam on substituted ones
    lam_t = torch.as_tensor(lam, dtype=RDTYPE)
    lam_vals = lam_t.expand(len(sub_atoms)) if lam_t.dim() == 0 else lam_t
    lam_vec = torch.zeros(na, dtype=RDTYPE)
    lam_vec[torch.tensor(sub_atoms, dtype=torch.long)] = lam_vals.to(RDTYPE)

    # local table: base per atom, substituted rows blended toward the target
    alch = {a: (sys_a.vloc_tables[species_index[a]],
                sys_b.vloc_tables[tgt_species[a]], lam_vec[a])
            for a in sub_atoms}
    vloc_atom = per_atom_local_tables(
        sys_a.vloc_tables, torch.as_tensor(species_index), alch)

    # KB projectors: blend the two full multi-species projector sets per k
    proj_data = [blend_projector_data(sys_a.proj_data[k], sys_b.proj_data[k], lam_vec)
                 for k in range(len(sys_a.proj_data))]

    # ionic charge per atom (constant across lam for an isovalent substitution)
    z_a = torch.tensor([float(pseudos[species_index[i]].z_valence)
                        for i in range(na)], dtype=RDTYPE)
    z_b = torch.tensor([float(tgt_pseudos[tgt_species[i]].z_valence)
                        for i in range(na)], dtype=RDTYPE)
    charges = (1.0 - lam_vec) * z_a + lam_vec * z_b
    n_electrons = float(charges.sum())

    # NLCC core density, blended per atom where an endpoint carries one
    grid = sys_a.grid
    uniq, inverse = _unique_shells(np.sqrt(grid.g2.reshape(-1).numpy()))
    core_base = core_shell_tables(list(pseudos), uniq, inverse)
    tgt_cores = {a: core_shell_tables([substitutions[a]], uniq, inverse)[0]
                 for a in sub_atoms}
    has_core = any(c is not None for c in core_base) or any(
        c is not None for c in tgt_cores.values())
    if has_core:
        ref = next(c for c in list(core_base) + list(tgt_cores.values())
                   if c is not None)
        shells = []
        for i in range(na):
            base_c = core_base[species_index[i]]
            base_c = base_c if base_c is not None else torch.zeros_like(ref)
            if i in substitutions:
                tc = tgt_cores[i]
                tc = tc if tc is not None else torch.zeros_like(ref)
                shells.append((1.0 - lam_vec[i]) * base_c + lam_vec[i] * tc)
            else:
                shells.append(base_c)
        pos_t = torch.as_tensor(np.asarray(positions), dtype=RDTYPE)
        rho_core = assemble_core_density(shells, list(range(na)), pos_t, grid)
    else:
        rho_core = None

    batch = build_batched(sys_a.spheres, proj_data)
    if nbands is None:
        nbands = default_nbands(max(n_electrons, float(sys_b.n_electrons)))

    # stash the λ-derivative building blocks so the relaxed alchemical gradient
    # (alchemical_gap_gradient) can reconstruct the bare perturbation ∂V_ion/∂λ.
    sub_mask = torch.zeros(na, dtype=RDTYPE)
    sub_mask[torch.tensor(sub_atoms, dtype=torch.long)] = 1.0
    spec: SubstitutionSpec = {
        "sub_atoms": sub_atoms,
        "sub_mask": sub_mask,
        "vloc_tables_a": sys_a.vloc_tables.detach(),
        "species_a": list(species_index),
        "vloc_tables_b": sys_b.vloc_tables.detach(),
        "species_b": list(tgt_species),
        "dij_a": sys_a.proj_data[0].dij_full.detach(),
        "dij_b": sys_b.proj_data[0].dij_full.detach(),
        "atom_index_a": sys_a.proj_data[0].atom_index,
        "atom_index_b": sys_b.proj_data[0].atom_index,
        "core_base": list(core_base),
        "tgt_cores": tgt_cores,
        "z_base": z_a.detach(),
        "z_target": z_b.detach(),
    }
    return dataclasses.replace(
        sys_a, charges=charges, n_electrons=n_electrons, vloc_atom=vloc_atom,
        proj_data=proj_data, batch=batch, nbands=nbands, rho_core=rho_core,
        alchemical=spec)


# --------------------------------------------------------------------------- #
# Relaxed alchemical eigenvalue / band-gap gradient (composition DFPT)
#
# alchemical_energy_gradient above is free: at the SCF stationary point the
# energy is variational in ρ, so dE/dλ is the bare Hellmann-Feynman ionic term
# and the density response drops out (envelope theorem). An *eigenvalue* is NOT
# variational, so dε_i/dλ carries the self-consistent density response and is a
# genuine DFPT problem:
#
#     dε_i/dλ = ⟨ψ_i| dV_KS/dλ |ψ_i⟩,   dV_KS/dλ = dV_ion/dλ + K_Hxc · dρ/dλ
#
# The bare perturbation dV_ion/dλ is a *local + nonlocal* operator: the local
# form-factor change ∂v_loc/∂λ (and, for NLCC pseudos, the explicit core-XC
# term f_xc·∂ρ_core/∂λ), plus the KB change ∂D/∂λ acting through the fixed
# projectors. The density response solves the Dyson equation
#
#     dρ/dλ = χ₀[ dV_ion/dλ + K_Hxc dρ/dλ ]  =>  (1 − χ₀K_Hxc) dρ/dλ = χ₀[dV_ion/dλ]
#
# χ₀[dV_ion/dλ] needs a Sternheimer solve with an OPERATOR right-hand side
# (scf.implicit.apply_chi0 only accepts a local field), which _chi0_operator
# below supplies by adding the nonlocal ∂D/∂λ piece to the usual local RHS.
# The self-consistent dressing then reuses the local χ₀ and K_Hxc kernels.
# nspin=1 insulator (smearing="none"), use_symmetry=False — same regime as the
# rest of scf.implicit. --------------------------------------------------------


@dataclass
class AlchemicalGapGradient:
    """Result of :func:`alchemical_gap_gradient`.

    ``dgap`` is the physically-meaningful relaxed d(E_gap)/dλ [eV] — the number
    a finite difference of re-converged SCF gaps returns. ``dgap_frozen`` is the
    sudden (frozen-density) estimate ⟨ψ|dV_ion/dλ|ψ⟩ with the same edges; the
    difference ``dgap − dgap_frozen`` is exactly the self-consistent density
    response, so reporting both makes the relaxation contribution explicit.
    """

    dgap: float
    dvbm: float
    dcbm: float
    dgap_frozen: float
    dvbm_frozen: float
    dcbm_frozen: float
    vbm_index: tuple[int, int]
    cbm_index: tuple[int, int]


def _substitution_bare_perturbation(
    res, xc, mask=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bare alchemical perturbation ∂V_ion/∂λ from a substitution System.

    Returns ``(dvloc_r, ddij)``: the local field ∂v_loc/∂λ [eV] on the real-space
    grid (including the explicit NLCC core-XC term f_xc·∂ρ_core/∂λ when an
    endpoint carries a core), and the nonlocal KB derivative ∂D/∂λ over the
    stacked [A cols, B cols] projector layout that ``blend_projector_data`` and
    hence ``system.proj_data`` already use. Everything is an exact λ-derivative
    (the blend is linear), evaluated from the endpoint differences stashed on
    ``system.alchemical`` by :func:`setup_alchemical_substitution`.

    ``mask`` is the per-atom transmutation-rate vector (na,): the derivative
    direction in composition space. Default (None) is the stashed ``sub_mask``
    (1.0 on every substituted site) — the "all substituted sites move together"
    scalar λ. A one-hot mask selects a SINGLE site, giving that site's ∂V_ion/∂λ_k
    for the per-site Jacobian (:func:`alchemical_gap_gradient_per_site`). Because
    the perturbation is linear and additive over sites, the full-mask result is
    the sum of the one-hot results.
    """
    from gradwave.core.energies.local_pp import local_potential_g
    from gradwave.core.fftbox import g_to_r_box
    from gradwave.postscf._response import fxc_hvp
    from gradwave.scf.setup_common import assemble_core_density

    system = res.system
    spec = system.alchemical
    if spec is None or "sub_atoms" not in spec:
        raise ValueError(
            "res.system was not built by setup_alchemical_substitution "
            "(no substitution spec on system.alchemical)")
    grid = system.grid
    na = len(system.positions)
    m = spec["sub_mask"] if mask is None else torch.as_tensor(mask, dtype=RDTYPE)

    # local: per-atom ∂v_loc/∂λ = tab_target − tab_base, scaled by the per-atom
    # rate m (0 on fixed sites, whose base and target rows are the same pseudo).
    ta = spec["vloc_tables_a"][torch.as_tensor(spec["species_a"])]
    tb = spec["vloc_tables_b"][torch.as_tensor(spec["species_b"])]
    dtable = (tb - ta) * m.reshape(na, *([1] * (tb.dim() - 1)))  # (na, n1,n2,n3)
    dvloc_g = local_potential_g(system.positions, system.species_index,
                                system.vloc_tables, grid.g_cart, grid.volume,
                                vloc_atom=dtable)
    dvloc_r = g_to_r_box(dvloc_g).real  # (n1,n2,n3) [eV]

    # explicit NLCC core-XC term: v_xc is built at ρ + ρ_core in the SCF, so a
    # λ-dependent core contributes f_xc[ρ+ρ_core]·∂ρ_core/∂λ to the bare
    # perturbation, exactly as the NLCC force carries ∫v_xc ∂ρ_core/∂τ.
    core_base = spec["core_base"]
    tgt_cores = spec["tgt_cores"]
    has_core = any(c is not None for c in core_base) or any(
        c is not None for c in tgt_cores.values())
    if has_core:
        ref = next(c for c in list(core_base) + list(tgt_cores.values())
                   if c is not None)
        shells = []
        for i in range(na):
            base_c = core_base[spec["species_a"][i]]
            base_c = base_c if base_c is not None else torch.zeros_like(ref)
            if i in tgt_cores:
                tc = tgt_cores[i]
                tc = tc if tc is not None else torch.zeros_like(ref)
                shells.append((tc - base_c) * m[i])  # ∂n_core/∂λ on a sub site
            else:
                shells.append(torch.zeros_like(ref))
        dcore_r = assemble_core_density(shells, list(range(na)),
                                        system.positions, grid)
        rho_xc = res.rho if system.rho_core is None else res.rho + system.rho_core
        dvloc_r = dvloc_r + fxc_hvp(xc, rho_xc, grid, dcore_r)

    # nonlocal: ∂D/∂λ = block_diag(−m·D_a, +m·D_b), row-scaling each atom's block
    # by its transmutation rate. Block-diagonal over atoms, so the row scale is
    # the whole-block scale.
    da = _scale_dij_blocks(spec["dij_a"], -m, spec["atom_index_a"])
    db = _scale_dij_blocks(spec["dij_b"], m, spec["atom_index_b"])
    ddij = torch.block_diag(da, db)
    return dvloc_r, ddij


def _sternheimer_op(h, c_occ, eps_occ, dvloc_r, ddij, alpha, tol, max_iter):
    """(H − ε_n + α P_occ) δψ_n = −P_c (dV_ion/dλ) ψ_n for all occupied n at one k.

    The operator counterpart of ``scf.implicit._sternheimer``: identical LHS, but
    the RHS applies the full bare perturbation — the local field dvloc_r as a
    real-space multiply plus the nonlocal ∂D/∂λ through the fixed projectors h.p.
    """
    from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g
    from gradwave.core.hamiltonian import becp
    from gradwave.scf.implicit import projected_cg
    from gradwave.solvers.precond import teter

    def p_c(x):
        return x - (x @ c_occ.conj().T) @ c_occ

    psi_r = g_to_r(c_occ, h.sphere.flat_idx, h.shape)
    dv_psi = box_to_sphere(r_to_g(psi_r * dvloc_r), h.sphere.flat_idx)
    if h.p.shape[0]:
        b = becp(h.p, c_occ)
        dv_psi = dv_psi + (b.to(h.p.dtype) @ ddij.to(h.p.dtype)) @ h.p
    rhs = -p_c(dv_psi)

    def a_apply(x):
        hx = h.apply(x) - eps_occ[:, None] * x
        return p_c(hx) + alpha * ((x @ c_occ.conj().T) @ c_occ)

    x = torch.zeros_like(rhs)
    r = rhs - a_apply(x)
    t_g = h.t
    t_band = torch.clamp(
        torch.einsum("bg,g,bg->b", c_occ.conj(), t_g.to(c_occ.dtype), c_occ).real,
        min=1e-6)
    x = projected_cg(a_apply, lambda rr: teter(rr, t_g, t_band), x, r, tol, max_iter)
    return p_c(x)


def _chi0_operator(res, dvloc_r, ddij, tol=1e-8, max_iter=200) -> torch.Tensor:
    """δρ⁰ = χ₀[dV_ion/dλ] for the operator-valued bare perturbation (nspin=1).

    Mirrors ``scf.implicit._chi0_channel`` — same occupied-band Sternheimer sum,
    factor 2·f_full·w_k per band — but with the operator RHS of _sternheimer_op.
    """
    from gradwave.core.fftbox import g_to_r
    from gradwave.postscf._response import sternheimer_shift
    from gradwave.scf.implicit import _hamiltonians, _occupied

    system = res.system
    grid = system.grid
    # nspin=1 (guarded by the caller alchemical_density_response), so the union
    # return of _hamiltonians is the flat per-k list.
    hs = cast("list[HamiltonianK]", _hamiltonians(res))
    dr = torch.zeros(grid.shape, dtype=RDTYPE, device=grid.g2.device)
    for ik, h in enumerate(hs):
        c_occ, eps_occ = _occupied(res, 0, ik)
        dpsi = _sternheimer_op(h, c_occ, eps_occ, dvloc_r, ddij,
                               sternheimer_shift(eps_occ), tol, max_iter)
        psi_r = g_to_r(c_occ, h.sphere.flat_idx, grid.shape)
        dpsi_r = g_to_r(dpsi, h.sphere.flat_idx, grid.shape)
        dr += (4.0 * float(system.kweights[ik])
               * (psi_r.conj() * dpsi_r).real.sum(dim=0))
    return dr / grid.volume


def _solve_forward_response(res, xc, drho0, beta=0.4, tol=1e-9, max_iter=100,
                            history=8) -> torch.Tensor:
    """Forward Dyson solve (1 − χ₀K_Hxc) δρ = δρ⁰ by Anderson fixed-point.

    δρ = δρ⁰ + χ₀[K_Hxc δρ]; the dressing term is a local field, so it reuses
    scf.implicit.apply_chi0 / apply_k_hxc. This is the forward companion of
    scf.implicit.solve_adjoint (which solves the transpose u = v̄ + K_Hxc[χ₀u]).
    """
    from gradwave.core._anderson import AndersonMixer
    from gradwave.scf.implicit import apply_chi0, apply_k_hxc

    def g(u):
        return drho0 + apply_chi0(res, apply_k_hxc(res, xc, u))

    u = drho0.clone()
    mixer = AndersonMixer(history, beta)
    step = float("inf")
    for _ in range(max_iter):
        r = g(u) - u
        step = float(torch.linalg.norm(r)) / max(1.0, float(torch.linalg.norm(u)))
        if step < tol:
            return u
        u = mixer.step(u.reshape(-1), r.reshape(-1)).reshape(u.shape)
    raise RuntimeError(
        f"alchemical forward response not converged ({step:.2e} after {max_iter})")


def alchemical_density_response(res, xc, tol=1e-8, max_iter=200, mask=None):
    """Relaxed first-order density response dρ/dλ to the alchemical perturbation.

    Returns ``(drho, dvloc_r, ddij)``: the self-consistent dρ/dλ on the grid and
    the bare-perturbation pieces (reused by the eigenvalue expectations). nspin=1
    insulator with use_symmetry=False, matching scf.implicit's regime. ``mask``
    is the per-atom transmutation-rate direction (None = all substituted sites
    together); see :func:`_substitution_bare_perturbation`.
    """
    from gradwave.scf.implicit import _check_no_symmetry, _res_is_insulating

    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError(
            "relaxed alchemical gradient: nspin=1 only for now (the nspin=2 "
            "χ₀/K_Hxc kernels exist; threading the spin channel is a follow-up)")
    _check_no_symmetry(res)
    if not _res_is_insulating(res):
        raise NotImplementedError(
            "relaxed alchemical gradient: insulator (smearing='none') only")
    dvloc_r, ddij = _substitution_bare_perturbation(res, xc, mask=mask)
    drho0 = _chi0_operator(res, dvloc_r, ddij, tol, max_iter)
    drho = _solve_forward_response(res, xc, drho0)
    return drho, dvloc_r, ddij


def _vbm_cbm(res) -> tuple[tuple[int, int], tuple[int, int]]:
    """(k, band) of the valence-band max (highest occupied) and conduction-band
    min (lowest empty), from an insulating nspin=1 result."""
    eig = res.eigenvalues  # (nk, nb)
    occ_mask = res.occupations > 1e-6
    if not bool(occ_mask.any()) or not bool((~occ_mask).any()):
        raise ValueError("no clean occupied/empty split (metal or full/empty)")
    vbm = eig[occ_mask].max()
    cbm = eig[~occ_mask].min()
    vk, vb = (int(i) for i in (eig == vbm).nonzero()[0])
    ck, cb = (int(i) for i in (eig == cbm).nonzero()[0])
    return (vk, vb), (ck, cb)


def _band_deps_dlam(res, k, band, dv_local_r, ddij) -> float:
    """dε_{k,band}/dλ = ⟨ψ|dv_local|ψ⟩ + ⟨ψ|∂D/∂λ|ψ⟩ for a single frozen orbital.

    ``dv_local_r`` is the local field the orbital sees (the bare ∂v_loc/∂λ for the
    sudden term, or the full dV_KS local part dv_loc + K_Hxc dρ for the relaxed
    one). The nonlocal ∂D/∂λ expectation uses the same becp†·D·becp form as the
    nonlocal energy.
    """
    from gradwave.core.fftbox import g_to_r
    from gradwave.core.hamiltonian import becp, projectors

    system = res.system
    grid = system.grid
    c = res.coeffs[k][band]
    psi_box = g_to_r(c, system.spheres[k].flat_idx, grid.shape)
    # ⟨ψ|V|ψ⟩ = ∫|ψ|²V dr = (1/N) Σ_grid |ψ_box|² V(r)  (ψ normalized: Σ_G|c|²=1)
    local = ((psi_box.real ** 2 + psi_box.imag ** 2) * dv_local_r).mean()
    p = projectors(system.proj_data[k], system.positions)
    b = becp(p, c.unsqueeze(0))[0]  # (nproj,)
    nl = torch.einsum("i,ij,j->", b.conj(), ddij.to(b.dtype), b).real
    return float(local + nl)


def _gap_gradient_for_mask(res, xc, mask, edges, tol, max_iter):
    """d(E_gap)/dλ along one composition direction ``mask`` (per-atom rate), for
    fixed band edges ``edges = ((vk,vb),(ck,cb))``. The DFPT response is solved
    for this direction; the edges are shared across directions so a per-site
    Jacobian reuses the reference VBM/CBM."""
    from gradwave.scf.implicit import apply_k_hxc

    (vk, vb), (ck, cb) = edges
    drho, dvloc_r, ddij = alchemical_density_response(res, xc, tol, max_iter,
                                                      mask=mask)
    dv_ks_local_r = dvloc_r + apply_k_hxc(res, xc, drho)
    dvbm = _band_deps_dlam(res, vk, vb, dv_ks_local_r, ddij)
    dcbm = _band_deps_dlam(res, ck, cb, dv_ks_local_r, ddij)
    dvbm_f = _band_deps_dlam(res, vk, vb, dvloc_r, ddij)
    dcbm_f = _band_deps_dlam(res, ck, cb, dvloc_r, ddij)
    return AlchemicalGapGradient(
        dgap=dcbm - dvbm, dvbm=dvbm, dcbm=dcbm,
        dgap_frozen=dcbm_f - dvbm_f, dvbm_frozen=dvbm_f, dcbm_frozen=dcbm_f,
        vbm_index=(vk, vb), cbm_index=(ck, cb))


def alchemical_gap_gradient(res, xc, tol=1e-8, max_iter=200) -> AlchemicalGapGradient:
    """Relaxed d(E_gap)/dλ for a substitution System, by composition DFPT.

    The fundamental gap's sensitivity to the alchemical weight with all
    substituted sites moving together (scalar λ), with the full self-consistent
    density response — the analytic counterpart of a finite difference of
    re-converged SCF gaps. Also returns the sudden (frozen-density) estimate so
    the relaxation contribution is explicit. Needs a System from
    :func:`setup_alchemical_substitution`; nspin=1 insulator, use_symmetry=False.
    """
    return _gap_gradient_for_mask(res, xc, None, _vbm_cbm(res), tol, max_iter)


def alchemical_gap_gradient_per_site(
    res, xc, tol=1e-8, max_iter=200,
) -> dict[int, AlchemicalGapGradient]:
    """Per-site Jacobian ∂(E_gap)/∂λ_k for each substituted site k independently.

    Returns ``{atom_index: AlchemicalGapGradient}`` — one composition DFPT solve
    per substituted site, all sharing the reference VBM/CBM edges. Because the
    alchemical perturbation is linear and additive over sites, the scalar
    all-sites-together gradient (:func:`alchemical_gap_gradient`) equals the sum
    of the per-site ``dgap`` values (a built-in consistency check).

    This is the inverse-design gradient over composition: which site, transmuted
    which way, moves the gap fastest. For an ISOVALENT substitution every per-site
    component is individually finite-difference-verifiable (the site stays
    charge-neutral). For a charge-conserving CO-substitution (a donor+acceptor
    pair whose total valence is invariant), only the coupled sum is directly
    FD-verifiable — the individual aliovalent components are the fixed-N linear
    responses that sum to it (see docs/ideas.md, the aliovalent ladder).

    Degenerate-edge caveat: a single-site perturbation breaks the crystal
    symmetry, so if the band edge is degenerate (e.g. the cubic-perovskite p-like
    VBM triplet) the single-state expectation ⟨ψ_edge|∂V/∂λ_k|ψ_edge⟩ is
    gauge-dependent and a per-site component need not match its own single-site
    FD to meV (the FD tracks the least-shifted degenerate partner). The exact
    sum, the coupled (symmetry-preserving) gradient, and any non-degenerate edge
    are unaffected; a degenerate-subspace first-order form is the fix (open).
    """
    system = res.system
    spec = system.alchemical
    if spec is None or "sub_atoms" not in spec:
        raise ValueError("res.system was not built by setup_alchemical_substitution")
    na = len(system.positions)
    edges = _vbm_cbm(res)
    out: dict[int, AlchemicalGapGradient] = {}
    for k in spec["sub_atoms"]:
        onehot = torch.zeros(na, dtype=RDTYPE)
        onehot[k] = 1.0
        out[int(k)] = _gap_gradient_for_mask(res, xc, onehot, edges, tol, max_iter)
    return out
