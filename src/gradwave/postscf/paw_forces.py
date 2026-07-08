"""Forces for ultrasoft/PAW — autograd over a τ-differentiable energy (Layer A).

Beyond the norm-conserving Hellmann–Feynman terms, USPP/PAW forces carry:

- the augmentation-density term ∫ v_Hxc+loc · ∂ρ_aug/∂τ (ρ_aug moves with the
  projector AND Q-function phases),
- the S-orthogonality term −Σ w f ε ⟨ψ|∂S/∂τ|ψ⟩ (the constraint
  ⟨ψ|S(τ)|ψ⟩ = 1 makes fixed-coefficient states denormalize under τ),
- the one-center term Σ_ij ddd_ij ∂ρ^a_ij/∂τ (exact chain rule — ddd is
  ∂E_1c/∂ρ_ij at the converged becsum),
- the NLCC core force (∂ρ_core/∂τ through the XC argument).

All four come out of ONE autograd backward over the energy expression

    E(τ) = E_H[ρ(τ)] + E_xc[ρ(τ)+ρc(τ)] + E_loc[ρ(τ),τ] + E_NL(becp(τ))
         + E_ewald(τ) + Σ_a ddd_a·ρ^a(τ) − Σ w f ε ⟨ψ|S(τ)|ψ⟩

at fixed plane-wave coefficients, occupations, eigenvalues, smooth density,
and ddd. E_kin is τ-independent and omitted.
"""

from __future__ import annotations

import torch

from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.hartree import hartree_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.fftbox import r_to_g
from gradwave.core.hamiltonian import becp, projectors
from gradwave.dtypes import CDTYPE


def forces_uspp(res: dict, xc, remove_net: bool = True) -> torch.Tensor:
    """F_a = −dE/dτ_a (na, 3) [eV/Å] for a converged scf_uspp result."""
    system = res["system"]
    grid = system.grid
    vol = grid.volume
    shape = grid.shape
    pos = system.positions.detach().clone().requires_grad_(True)

    coeffs = [c.detach() for c in res["coeffs"]]
    occ = res["occupations"].detach()
    eigs = res["eigenvalues"].detach()
    kw = system.kweights

    # ddd at the converged becsum (detached — chain rule is exact at the point)
    is_paw = any(p.is_paw for p in system.paws)
    ddd_atoms = []
    if is_paw:
        from gradwave.scf.paw_onsite import OneCenter

        onec = {sp: OneCenter(system.paws[sp], xc)
                for sp in set(system.species_of_atom)}
        for a, sp in enumerate(system.species_of_atom):
            _, ddd = onec[sp].energy_and_ddd(res["rho_ij_atoms"][a])
            ddd_atoms.append(ddd)

    # projectors, becp, becsum on the τ-graph
    projs = [projectors(pd, pos) for pd in system.proj_data]
    becps = [becp(projs[ik], coeffs[ik]) for ik in range(len(coeffs))]
    rho_ij = [torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE)
              for (s0, s1) in system.atom_slices]
    for ik, b in enumerate(becps):
        w = (kw[ik] * occ[ik]).to(CDTYPE)
        for a, (s0, s1) in enumerate(system.atom_slices):
            ba = b[:, s0:s1]
            rho_ij[a] = rho_ij[a] + torch.einsum("b,bi,bj->ij", w, ba.conj(), ba)
    rho_ij = [0.5 * (m + m.conj().T) for m in rho_ij]

    # augmentation density on the graph (Q̃ tables fixed; phases move)
    phase_arg = system.g_sphere @ pos.T  # (nGm, na)
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
    aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE)
    for a, sp in enumerate(system.species_of_atom):
        aug_sph = aug_sph + phases[:, a].conj() * torch.einsum(
            "ij,ijg->g", rho_ij[a], system.aug[sp].q_g
        )
    aug_box = torch.zeros(grid.n_points, dtype=CDTYPE)
    aug_box[system.sphere_idx] = aug_sph / vol
    rho_aug = torch.fft.ifftn(aug_box.reshape(shape) * grid.n_points,
                              dim=(-3, -2, -1)).real

    rho_s = (res["rho"].detach() - _aug_at_fixed(res, system)).detach()
    rho = rho_s + rho_aug
    rho_g = r_to_g(rho.to(CDTYPE))

    # NLCC core on the graph
    rho_xc = rho
    if system.rho_core is not None:
        from gradwave.pseudo.radial_torch import RadialTables

        q_sph = torch.linalg.norm(system.g_sphere, dim=1)
        core = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE)
        for sp in set(system.species_of_atom):
            paw = system.paws[sp]
            if paw.core_rho is None:
                continue
            tab = RadialTables(paw)
            with torch.no_grad():
                f_core = tab.core_of_g(q_sph)
            atoms = [a for a, sa in enumerate(system.species_of_atom) if sa == sp]
            core = core + phases[:, atoms].conj().sum(dim=1) * f_core.to(CDTYPE) / vol
        core_box = torch.zeros(grid.n_points, dtype=CDTYPE)
        core_box[system.sphere_idx] = core
        rho_core = torch.fft.ifftn(core_box.reshape(shape) * grid.n_points,
                                   dim=(-3, -2, -1)).real
        rho_xc = rho + rho_core

    from gradwave.core.density import sigma_from_rho

    sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
    species_index = torch.tensor(system.species_of_atom, dtype=torch.int64)
    vloc_g = local_potential_g(pos, species_index, system.vloc_tables,
                               grid.g_cart, vol)

    e = (
        hartree_energy(rho_g, grid.g2, vol)
        + xc.energy(rho_xc, vol, sigma)
        + local_energy(rho_g, vloc_g, vol)
        + nonlocal_energy(becps, system.proj_data[0].dij_full, occ, kw)
        + ewald_energy(pos, system.charges, grid.cell)
    )
    # one-center chain term
    if is_paw:
        for a in range(len(system.atom_slices)):
            e = e + (ddd_atoms[a].to(CDTYPE) * rho_ij[a]).sum().real
    # S-orthogonality term: −Σ w f ε (b† q b)  (the Σ|c|² part is constant)
    q = system.q_full.to(CDTYPE)
    for ik, b in enumerate(becps):
        quad = torch.einsum("bi,ij,bj->b", b.conj(), q, b).real
        e = e - (kw[ik] * occ[ik] * eigs[ik] * quad).sum()

    (grad,) = torch.autograd.grad(e, pos)
    f = -grad
    if remove_net:
        f = f - f.mean(dim=0, keepdim=True)
    return f


def _aug_at_fixed(res: dict, system) -> torch.Tensor:
    """ρ_aug at the converged positions/becsum (to isolate the smooth part)."""
    with torch.no_grad():
        grid = system.grid
        phase_arg = system.g_sphere @ system.positions.T
        phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
        aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE)
        for a, sp in enumerate(system.species_of_atom):
            aug_sph = aug_sph + phases[:, a].conj() * torch.einsum(
                "ij,ijg->g", res["rho_ij_atoms"][a], system.aug[sp].q_g
            )
        aug_box = torch.zeros(grid.n_points, dtype=CDTYPE)
        aug_box[system.sphere_idx] = aug_sph / grid.volume
        return torch.fft.ifftn(aug_box.reshape(grid.shape) * grid.n_points,
                               dim=(-3, -2, -1)).real
