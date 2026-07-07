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


def forces(res: SCFResult) -> torch.Tensor:
    """F_a = −dE/dτ_a, (na, 3) [eV/Å], at the converged SCF point."""
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
    if system.sym is not None:
        from gradwave.symmetry import symmetrize_forces

        f = symmetrize_forces(f, system.sym, grid.cell)
    return f
