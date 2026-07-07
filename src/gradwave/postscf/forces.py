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

import torch

from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.fftbox import r_to_g
from gradwave.core.hamiltonian import becp, projectors
from gradwave.scf.loop import SCFResult, _stack_dij


def forces(res: SCFResult, remove_net: bool = True) -> torch.Tensor:
    """F_a = −dE/dτ_a, (na, 3) [eV/Å], at the converged SCF point.

    remove_net subtracts the mean force (default). The net component is
    unphysical XC-grid egg-box noise — large for semicore species (~0.01
    eV/Å for Al at 20 Ry vs ~1e-5 for valence-only Si) — and QE/VASP remove
    it the same way; with it removed, forces match QE to ~1e-5 eV/Å.
    """
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("forces for nspin=2 land next — SCF/magnetization only for now")
    if getattr(res.system, "rho_core", None) is not None:
        raise NotImplementedError("NLCC force term (force_cc) not implemented yet")
    system = res.system
    grid = system.grid
    pos = system.positions.detach().clone().requires_grad_(True)

    coeffs = [c.detach() for c in res.coeffs]
    occ = res.occupations.detach()
    rho_g = r_to_g(res.rho.detach().to(torch.complex128))

    projs = [projectors(pd, pos) for pd in system.proj_data]
    becps = [becp(projs[ik], coeffs[ik]) for ik in range(len(coeffs))]

    vloc_g = local_potential_g(
        pos, system.species_index, system.vloc_tables, grid.g_cart, grid.volume
    )
    e_pos = (
        local_energy(rho_g, vloc_g, grid.volume)
        + nonlocal_energy(becps, _stack_dij(system), occ, system.kweights)
        + ewald_energy(pos, system.charges, grid.cell)
    )
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
            cpad = _pad_coeffs(res.coeffs[sp], hub.q_free.shape[-1], q.device)
            mats = occupation_matrices(q, cpad, occ[sp], kw, hub.sites)
            e_u = e_u + hubbard_energy(mats, hub.sites)
    else:
        occ = res.occupations.detach()
        cpad = _pad_coeffs(res.coeffs, hub.q_free.shape[-1], q.device)
        mats = occupation_matrices(q, cpad, 0.5 * occ, kw, hub.sites)
        e_u = 2.0 * hubbard_energy(mats, hub.sites)

    (grad,) = torch.autograd.grad(e_u, pos)
    return -grad


def _pad_coeffs(coeffs_per_k, npw_max, device):
    """[(nb, npw_k)] per k → padded (nk, nb, npw_max), detached."""
    nk = len(coeffs_per_k)
    nb = coeffs_per_k[0].shape[0]
    out = torch.zeros(nk, nb, npw_max, dtype=torch.complex128, device=device)
    for ik, c in enumerate(coeffs_per_k):
        out[ik, :, : c.shape[1]] = c.detach().to(device)
    return out
