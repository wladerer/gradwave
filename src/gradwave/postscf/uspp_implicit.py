"""Differentiability through the USPP/PAW SCF — stage 1: dE/dθ by
stationarity (task #58 milestone 1).

At the converged generalized SCF point, E_total is stationary w.r.t. the
S-orthonormal orbitals and the occupations, and becsum is a function of the
orbitals — so the total derivative w.r.t. an XC-functional parameter θ is
the PARTIAL derivative at fixed state:

    dE/dθ = ∂E_xc^grid[ρ_tot + ρ_core; θ]/∂θ + Σ_a ∂E_1c^a[becsum; θ]/∂θ

The one-center piece is what norm-conserving never had; OneCenter's
energy_theta keeps the functional parameters on the autograd graph through
the same angular quadrature the SCF energy uses (exactness heritage of the
exact-ddd rewrite). nspin=1 for now — spin-resolved learnable functionals
don't exist yet.

Stage 2 (the adjoint/Sternheimer backward for density-dependent losses on
the (ρ, becsum) composite response vector) lives here next.
"""

from __future__ import annotations

import torch

from gradwave.core.density import sigma_from_rho


def uspp_energy_param_grads(res: dict, xc) -> dict[str, torch.Tensor]:
    """dE_total/dθ for every parameter of `xc` at a converged scf_uspp point.

    res: scf_uspp result (nspin=1). Includes the one-center term.
    """
    if res.get("nspin", 1) != 1:
        raise NotImplementedError("dE/dθ for nspin=2 USPP not implemented")
    system = res["system"]
    grid = system.grid

    rho = res["rho"].detach()
    rho_xc = rho if system.rho_core is None else rho + system.rho_core
    sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
    e_theta = xc.energy(rho_xc, grid.volume, sigma)

    if any(p.is_paw for p in system.paws):
        from gradwave.scf.paw_onsite import OneCenter

        onec = {sp: OneCenter(system.paws[sp], xc)
                for sp in set(system.species_of_atom)}
        for a, sp in enumerate(system.species_of_atom):
            e_theta = e_theta + onec[sp].energy_theta(res["rho_ij_atoms"][a])

    grads = torch.autograd.grad(e_theta, list(xc.parameters()),
                                allow_unused=True)
    return {name: g for (name, _), g in
            zip(xc.named_parameters(), grads, strict=True)}
