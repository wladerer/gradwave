"""Hellmann–Feynman forces (Layer A entry point).

At SCF convergence the energy is stationary in the orbitals/density, so
dE/dτ is the PARTIAL derivative at fixed (detached) ψ, ρ — no response
needed. Positions enter the energy in exactly three places: Ewald,
structure factors in E_loc, and the projector phases in E_NL; only those
three terms are rebuilt on the autograd graph (kinetic/Hartree/XC carry no
τ-dependence at fixed ρ and would only add FFT cost). Plane waves ⇒ no
Pulay forces at fixed cell.
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.fftbox import r_to_g
from gradwave.core.hamiltonian import becp, projectors
from gradwave.core.xc.base import XCFunctional
from gradwave.postscf._response import pad_coeffs
from gradwave.scf.loop import SCFResult, _stack_dij


def _core_correction_energy(res, xc, pos):
    """E_xc(ρ + ρ_core(pos)) with the SCF density detached — the ONLY position
    dependence is the NLCC core charge's structure factor, so autograd of this
    over pos gives −F_cc = ∫ v_xc ∂ρ_core/∂τ (v_xc = δE_xc/δρ_xc, so the full
    LDA/GGA gradient correction is carried automatically). Returns a scalar to
    add to the position-dependent energy assembled in forces()."""
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
    if xc.needs_tau:
        # τ (kinetic-energy density) is not stored on SCFResult; the meta-GGA
        # NLCC-force path would need to rebuild it. Not assembled yet.
        raise NotImplementedError(
            "NLCC force for meta-GGA (needs_tau) functionals not implemented"
        )
    # |G| shells reproduced from the same g2 the frozen build used, so the
    # differentiable ρ_core matches system.rho_core at the converged geometry.
    g_flat = np.sqrt(grid.g2.detach().cpu().reshape(-1).numpy())
    uniq, inverse = _unique_shells(g_flat)
    shells = core_shell_tables(system.upfs, uniq, inverse)
    core = assemble_core_density(shells, system.species_of_atom, pos, grid)

    if nspin == 2:
        rho_s = [r.detach() for r in res.rho_spin]
        return spin_xc_energy(xc, rho_s, core, grid.volume, grid.g_cart)
    rho_xc = res.rho.detach() + core
    sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
    return xc.energy(rho_xc, grid.volume, sigma)


def forces(
    res: SCFResult, remove_net: bool = True, xc: XCFunctional | None = None
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
    """
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

    coeffs_s = res.coeffs if nspin == 2 else [res.coeffs]
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
    if has_core:
        # XC carries no τ-dependence at fixed ρ EXCEPT through the core charge;
        # the valence-only part of E_xc has zero gradient here (ρ detached), so
        # this contributes exactly the NLCC force term.
        e_pos = e_pos + _core_correction_energy(res, xc, pos)
    (grad,) = torch.autograd.grad(e_pos, pos)
    f = -grad
    if remove_net:
        f = f - f.mean(dim=0, keepdim=True)
    if system.sym is not None:
        from gradwave.symmetry import symmetrize_forces

        f = symmetrize_forces(f, system.sym, grid.cell)
    return f


def hubbard_force(res: SCFResult, manifolds) -> torch.Tensor:
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
        e_u = 0.0
        for sp in range(2):
            cpad = pad_coeffs(res.coeffs[sp], hub.q_free.shape[-1], q.device)
            mats = occupation_matrices(q, cpad, occ[sp], kw, hub.sites)
            e_u = e_u + hubbard_energy(mats, hub.sites)
    else:
        occ = res.occupations.detach()
        cpad = pad_coeffs(res.coeffs, hub.q_free.shape[-1], q.device)
        mats = occupation_matrices(q, cpad, 0.5 * occ, kw, hub.sites)
        e_u = 2.0 * hubbard_energy(mats, hub.sites)

    (grad,) = torch.autograd.grad(e_u, pos)
    return -grad
