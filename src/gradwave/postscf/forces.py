"""Hellmann–Feynman forces (Layer A entry point).

At SCF convergence the energy is stationary in the orbitals/density, so
dE/dτ is the PARTIAL derivative at fixed (detached) ψ, ρ — no response
needed. Positions enter the energy in exactly three places: Ewald,
structure factors in E_loc, and the projector phases in E_NL; only those
three terms are rebuilt on the autograd graph (kinetic/Hartree/XC carry no
τ-dependence at fixed ρ and would only add FFT cost). Plane waves ⇒ no
Pulay forces at fixed cell.

``forces()`` dispatches on ``res.formalism``: a plain collinear ``SCFResult``
("nc") runs the code below; a noncollinear/SOC ``NCResult`` ("noncollinear")
runs ``_forces_noncollinear`` — the same three-term (Ewald / E_loc structure
factor / E_NL projector-phase) argument, generalized onto the spinor
Hamiltonian the way ``postscf.stress``'s ``is_fr`` branch already generalized
stress. That branch further splits on ``system.is_fr``: fully-relativistic
(spin-orbit) pseudos use the j-resolved spinor projectors
(``core.spinor_proj.build_so_projectors``, DOUBLED plane-wave axis), while a
scalar-relativistic pseudo run through the noncollinear code path uses plain
KB projectors acting on the up/down spinor halves separately (mirroring
``scf.noncollinear``'s ``SpinorHamiltonian`` split and
``spinor_scalar_nonlocal_energy``) — so it reduces to the ordinary collinear
force in the z-collinear (no-canting) limit. ``hubbard_force_noncollinear`` is
the +U analogue of ``hubbard_force`` on the noncollinear occupation-matrix
machinery (``core.hubbard.occupation_matrices_noncollinear``, PR #159).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch

from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.fftbox import r_to_g
from gradwave.core.hamiltonian import becp, projectors
from gradwave.core.hubbard import HubbardManifold
from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.grids import FFTGrid
from gradwave.postscf._response import pad_coeffs
from gradwave.scf.loop import SCFResult, System, _stack_dij
from gradwave.scf.noncollinear import NCResult


def _core_correction_tau(
    res: SCFResult, system: System, grid: FFTGrid, nspin: int
) -> list[torch.Tensor]:
    """Valence τ_σ(r) rebuilt from the converged orbitals, DETACHED.

    τ is a fixed orbital field at the SCF point — the only position dependence in
    the core-correction energy flows through ρ_core's structure factor, so τ
    carries no pos gradient (the HF stationarity argument: dE/dτ_pos is taken at
    fixed ψ). The core charge has no orbitals, so τ_core = 0 and the meta-GGA XC
    energy is E_xc(ρ+ρ_core, σ(ρ+ρ_core), τ_valence) — exactly the SCF assembly.
    Rebuilt via the same pad_coeffs → tau_b path ELF/stress use; returns a
    per-spin list (length nspin)."""
    from gradwave.core.metagga import tau_b
    from gradwave.postscf._response import pad_coeffs

    bk = getattr(system, "batch", None)
    if bk is None:
        raise NotImplementedError(
            "meta-GGA NLCC force needs the batched-k geometry (system.batch); "
            "only collinear norm-conserving results are supported"
        )
    shape, vol = grid.shape, grid.volume
    occ = res.occupations
    if nspin == 2:
        coeffs_sp = cast("list[list[torch.Tensor]]", res.coeffs)
        return [
            tau_b(
                pad_coeffs(coeffs_sp[sp], bk.npw_max), occ[sp],
                system.kweights, bk, shape, vol,
            ).detach()
            for sp in range(2)
        ]
    coeffs_flat = cast("list[torch.Tensor]", res.coeffs)
    return [
        tau_b(
            pad_coeffs(coeffs_flat, bk.npw_max), occ,
            system.kweights, bk, shape, vol,
        ).detach()
    ]


def _core_correction_energy(
    res: SCFResult, xc: XCFunctional | SpinXC, pos: torch.Tensor
) -> torch.Tensor:
    """E_xc(ρ + ρ_core(pos)) with the SCF density detached — the ONLY position
    dependence is the NLCC core charge's structure factor, so autograd of this
    over pos gives −F_cc = ∫ v_xc ∂ρ_core/∂τ (v_xc = δE_xc/δρ_xc, so the full
    LDA/GGA gradient correction is carried automatically). For meta-GGA the
    valence τ is rebuilt (detached) and passed alongside the core-augmented ρ/σ,
    so E_xc picks up the τ-dependent XC energy without adding any τ→pos path.
    Returns a scalar to add to the position-dependent energy assembled in
    forces()."""
    from gradwave.core.density import sigma_from_rho
    from gradwave.scf.common import spin_xc_energy
    from gradwave.scf.setup_common import (
        _unique_shells,
        assemble_core_density,
        core_shell_tables,
    )

    system = res.system
    grid = system.grid
    nspin = getattr(res, "nspin", 1)
    # |G| shells reproduced from the same g2 the frozen build used, so the
    # differentiable ρ_core matches system.rho_core at the converged geometry.
    g_flat = np.sqrt(grid.g2.detach().cpu().reshape(-1).numpy())
    uniq, inverse = _unique_shells(g_flat)
    shells = core_shell_tables(system.upfs, uniq, inverse)
    core = assemble_core_density(shells, system.species_of_atom, pos, grid)

    # Meta-GGA (needs_tau): valence τ_σ rebuilt from the converged orbitals and
    # detached (τ carries no position dependence — see _core_correction_tau).
    tau_s = _core_correction_tau(res, system, grid, nspin) if xc.needs_tau else None

    if nspin == 2:
        assert res.rho_spin is not None  # scf() always sets rho_spin at nspin=2
        rho_s = [r.detach() for r in res.rho_spin]
        return spin_xc_energy(
            xc, rho_s, core, grid.volume, grid.g_cart, tau_s=tau_s
        )
    # nspin=1 is always paired with a collinear XCFunctional (same convention
    # as scf/uspp_loop.py's own xc dispatch).
    assert isinstance(xc, XCFunctional)
    rho_xc = res.rho.detach() + core
    sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
    tau = tau_s[0] if tau_s is not None else None
    return xc.energy(rho_xc, grid.volume, sigma, tau)


def forces(
    res: SCFResult | NCResult, remove_net: bool = True,
    xc: XCFunctional | SpinXC | None = None,
) -> torch.Tensor:
    """F_a = −dE/dτ_a, (na, 3) [eV/Å], at the converged SCF point.

    remove_net subtracts the mean force (default). The net component is
    unphysical XC-grid egg-box noise — large for semicore species (~0.01
    eV/Å for Al at 20 Ry vs ~1e-5 for valence-only Si) — and QE/VASP remove
    it the same way; with it removed, forces match QE to ~1e-5 eV/Å.

    xc is required only when the system carries an NLCC core charge: the
    core-correction force −∫ v_xc ∂ρ_core/∂τ is the autograd gradient of
    E_xc(ρ + ρ_core(τ)) and needs the functional to evaluate v_xc. Pass the
    same XCFunctional the SCF ran with; it is ignored for valence-only species.

    A noncollinear/SOC ``NCResult`` (``res.formalism == "noncollinear"``) is
    dispatched to ``_forces_noncollinear`` — see the module docstring.
    """
    if isinstance(res, NCResult):
        return _forces_noncollinear(res, remove_net=remove_net, xc=xc)
    system = res.system
    grid = system.grid
    nspin = getattr(res, "nspin", 1)
    has_core = getattr(system, "rho_core", None) is not None
    if has_core and xc is None:
        raise ValueError(
            "system has an NLCC core charge; pass the XCFunctional to forces() "
            "so the core-correction force term can be evaluated"
        )
    pos = system.positions.detach().clone().requires_grad_(True)

    # Local (total density) and Ewald terms are spin-agnostic; only E_NL sees
    # per-spin orbitals/occupations. Normalize coeffs to [spin][k] and occ to a
    # leading spin axis so the single loop below covers both nspin values —
    # exactly the spin sum assemble_pw_energies uses for the SCF energy, so the
    # analytic force matches that energy's derivative by construction.
    rho_g = r_to_g(res.rho.detach().to(torch.complex128))  # total ρ↑+ρ↓
    projs = [projectors(pd, pos) for pd in system.proj_data]  # spin-independent
    dij, kw = _stack_dij(system), system.kweights

    if nspin == 2:
        coeffs_s = cast("list[list[torch.Tensor]]", res.coeffs)
    else:
        coeffs_s = [cast("list[torch.Tensor]", res.coeffs)]
    occ_s = res.occupations.detach()
    occ_s = occ_s if nspin == 2 else occ_s[None]
    e_nl = 0.0
    for sp in range(nspin):
        cs = [c.detach() for c in coeffs_s[sp]]
        becps = [becp(projs[ik], cs[ik]) for ik in range(len(cs))]
        e_nl = e_nl + nonlocal_energy(becps, dij, occ_s[sp], kw)

    vloc_g = local_potential_g(
        pos, system.species_index, system.vloc_tables, grid.g_cart, grid.volume
    )
    e_pos = (
        local_energy(rho_g, vloc_g, grid.volume)
        + e_nl
        + ewald_energy(pos, system.charges, grid.cell)
    )
    from gradwave.core.energies.esm import esm_energy, esm_mode_of

    _esm_mode = esm_mode_of(getattr(res, "boundary", "periodic"))
    if _esm_mode is not None:
        # Open-boundary (ESM): the correction's only τ-dependence is the Gaussian
        # ion charge, so its autograd gradient is the ESM force — added to the
        # same fixed-density Hellmann-Feynman graph as the local/Ewald terms.
        e_pos = e_pos + esm_energy(res.rho.detach(), pos, system.charges, grid,
                                   mode=_esm_mode, bias=getattr(res, "esm_bias", 0.0))
    if has_core:
        # XC carries no τ-dependence at fixed ρ EXCEPT through the core charge;
        # the valence-only part of E_xc has zero gradient here (ρ detached), so
        # this contributes exactly the NLCC force term. The `has_core and xc
        # is None` guard above already raised if xc were missing here.
        assert xc is not None
        e_pos = e_pos + _core_correction_energy(res, xc, pos)
    (grad,) = torch.autograd.grad(e_pos, pos)
    f = -grad
    if remove_net:
        f = f - f.mean(dim=0, keepdim=True)
    if system.sym is not None:
        from gradwave.symmetry import symmetrize_forces

        f = symmetrize_forces(f, system.sym, grid.cell)
    return f


def _forces_noncollinear(
    res: NCResult, remove_net: bool = True, xc: XCFunctional | SpinXC | None = None
) -> torch.Tensor:
    """F_a = −dE/dτ_a for a noncollinear/SOC (spinor) ``NCResult``, (na, 3) [eV/Å].

    Positions enter through exactly the same three terms as the collinear
    path (module docstring): Ewald, the local-potential structure factor (on
    the TOTAL spinor density ρ↑+ρ↓ — Hartree/XC/kinetic are frozen at fixed,
    detached ρ/m⃗/coeffs and carry no τ-dependence), and the nonlocal
    projector phases. Only the nonlocal term differs by pseudopotential kind:

    - fully-relativistic (``system.is_fr``): j-resolved spinor projectors
      built on the position-autograd graph (``core.spinor_proj.
      build_so_projectors(..., positions=pos)``), contracted against the
      spinor coefficients on the DOUBLED plane-wave axis — mirroring
      ``scf.noncollinear``'s SCF assembly (``build_so_projectors`` /
      ``nonlocal_energy``) and ``postscf.stress``'s ``_energy_strained_fr``
      strain analogue.
    - scalar-relativistic (plain noncollinear, no SOC): ordinary KB
      projectors (``core.batch.projectors_b``) act on the up/down spinor
      halves SEPARATELY and are summed, mirroring
      ``scf.spinor_common.spinor_scalar_nonlocal_energy`` — so this branch
      reduces exactly to the collinear force in the z-collinear (no-canting)
      limit (the reduction oracle in ``tests/integration/
      test_forces_noncollinear.py``).

    NLCC is not yet supported here (see docs/gate_inventory.md): the core
    charge's structure-factor force term would need the noncollinear (ρ, m⃗)
    generalization of ``_core_correction_energy``, left as a follow-up.
    """
    from gradwave.core.batch import becp_b, projectors_b

    system = res.system
    if getattr(system, "rho_core", None) is not None:
        raise NotImplementedError(
            "NLCC core-correction force is not yet available on the "
            "noncollinear/SOC path (system carries an NLCC core charge); "
            "this needs the (ρ, m⃗) generalization of the collinear "
            "_core_correction_energy — left as a follow-up"
        )
    grid = system.grid
    bk = system.batch
    assert bk is not None
    pos = system.positions.detach().clone().requires_grad_(True)

    rho_g = r_to_g(res.rho.detach().to(torch.complex128))  # total ρ↑+ρ↓
    vloc_g = local_potential_g(
        pos, system.species_index, system.vloc_tables, grid.g_cart, grid.volume
    )
    e_pos = local_energy(rho_g, vloc_g, grid.volume) + ewald_energy(
        pos, system.charges, grid.cell
    )

    assert res.coeffs is not None, "res carries no spinor coefficients"
    assert res.occupations is not None
    coeffs = res.coeffs.detach()  # (nk, nb, 2·npw_max)
    occ = res.occupations.detach()  # (nk, nb), spinor degeneracy g=1
    kw = system.kweights
    nk = bk.nk
    m_pw = coeffs.shape[-1] // 2

    if system.is_fr:
        from gradwave.core.spinor_proj import build_so_projectors

        q_so, dij_so = build_so_projectors(bk, system, positions=pos)
        b_so = torch.einsum("kpg,kbg->kbp", q_so.conj(), coeffs)
        e_nl = nonlocal_energy([b_so[ik] for ik in range(nk)], dij_so, occ, kw)
    else:
        p = projectors_b(bk, pos)  # (nk, nproj, npw_max)
        if p.shape[1]:
            bu = becp_b(p, coeffs[..., :m_pw])
            bd = becp_b(p, coeffs[..., m_pw:])
            dij = bk.dij_full
            e_nl = (
                nonlocal_energy([bu[ik] for ik in range(nk)], dij, occ, kw)
                + nonlocal_energy([bd[ik] for ik in range(nk)], dij, occ, kw)
            )
        else:
            e_nl = torch.zeros((), dtype=e_pos.dtype, device=e_pos.device)
    e_pos = e_pos + e_nl

    (grad,) = torch.autograd.grad(e_pos, pos)
    f = -grad
    if remove_net:
        f = f - f.mean(dim=0, keepdim=True)
    if system.sym is not None:
        from gradwave.symmetry import symmetrize_forces

        f = symmetrize_forces(f, system.sym, grid.cell)
    return f


def hubbard_force_noncollinear(
    res: NCResult, manifolds: list[HubbardManifold]
) -> torch.Tensor:
    """+U contribution to the noncollinear/SOC Hellmann–Feynman force,
    −dE_U/dτ, (na, 3) [eV/Å].

    The noncollinear generalization of ``hubbard_force``: E_U enters through
    the same atomic-orbital projector phases e^{−i(k+G)·τ}
    (``core.hubbard.hubbard_projectors``), now feeding the 2×2 spin-block
    occupation matrix N^I (``occupation_matrices_noncollinear``, PR #159) that
    the noncollinear +U SCF/energy path already uses — reduces exactly to the
    collinear ``hubbard_force`` in the z-collinear (no down component) limit
    since the occupation matrix itself does (``test_occupation_matrix_
    noncollinear_reduces_to_collinear_limit``). Unlike the collinear nspin=1
    branch of ``hubbard_force``, no ×2 factor is needed: spinor occupations
    already carry the full electron count at g=1 degeneracy. Additive to
    ``forces()``'s noncollinear KB/local/Ewald force, same convention as the
    collinear ``hubbard_force``.
    """
    from gradwave.core.hubbard import (
        build_hubbard_projectors,
        hubbard_energy,
        hubbard_projectors,
        occupation_matrices_noncollinear,
    )

    system = res.system
    pos = system.positions.detach().clone().requires_grad_(True)
    hub = build_hubbard_projectors(system, manifolds)
    q = hubbard_projectors(hub, pos)  # differentiable in pos

    assert res.coeffs is not None, "res carries no spinor coefficients"
    assert res.occupations is not None
    coeffs = res.coeffs.detach()  # (nk, nb, 2·npw_max)
    occ = res.occupations.detach()  # (nk, nb), spinor degeneracy g=1
    m_pw = coeffs.shape[-1] // 2
    mats = occupation_matrices_noncollinear(
        q, coeffs[..., :m_pw], coeffs[..., m_pw:], occ, system.kweights, hub.sites
    )
    e_u = hubbard_energy(mats, hub.sites)

    (grad,) = torch.autograd.grad(e_u, pos)
    return -grad


def hubbard_force(res: SCFResult, manifolds: list[HubbardManifold]) -> torch.Tensor:
    """+U contribution to the Hellmann–Feynman force, −dE_U/dτ, (na, 3) [eV/Å].

    E_U enters through the atomic-orbital projector phases e^{−i(k+G)·τ}; with
    the orbitals and occupations detached (stationary at convergence), autograd
    of E_U over the differentiable projectors gives the force. Works for both
    nspin=1 and 2, and is additive to the (KB/local/Ewald) force above — kept
    separate because the full nspin=2/NLCC force path is not yet assembled.
    """
    from gradwave.core.hubbard import (
        build_hubbard_projectors,
        hubbard_energy,
        hubbard_projectors,
        occupation_matrices,
    )

    system = res.system
    nspin = getattr(res, "nspin", 1)
    pos = system.positions.detach().clone().requires_grad_(True)
    hub = build_hubbard_projectors(system, manifolds)
    q = hubbard_projectors(hub, pos)  # differentiable in pos

    kw = system.kweights
    if nspin == 2:
        occ = res.occupations.detach()  # (2, nk, nb) in [0,1]
        # res.coeffs is [spin][k] ragged; rebuild padded (nk, nb, npw_max)
        coeffs_sp = cast("list[list[torch.Tensor]]", res.coeffs)
        e_u = 0.0
        for sp in range(2):
            cpad = pad_coeffs(coeffs_sp[sp], hub.q_free.shape[-1], q.device)
            mats = occupation_matrices(q, cpad, occ[sp], kw, hub.sites)
            e_u = e_u + hubbard_energy(mats, hub.sites)
    else:
        occ = res.occupations.detach()
        coeffs_flat = cast("list[torch.Tensor]", res.coeffs)
        cpad = pad_coeffs(coeffs_flat, hub.q_free.shape[-1], q.device)
        mats = occupation_matrices(q, cpad, 0.5 * occ, kw, hub.sites)
        e_u = 2.0 * hubbard_energy(mats, hub.sites)

    (grad,) = torch.autograd.grad(e_u, pos)
    return -grad
