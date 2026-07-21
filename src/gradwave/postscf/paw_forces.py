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

    E(τ) = E_H[ρ(τ)] + E_xc[ρ_σ(τ)+ρc(τ)/nspin] + E_loc[ρ(τ),τ]
         + Σ_σ E_NL(becp_σ(τ)) + E_ewald(τ) + Σ_aσ ddd_aσ·ρ^aσ(τ)
         − Σ_σ w f ε ⟨ψ|S(τ)|ψ⟩

at fixed plane-wave coefficients, occupations, eigenvalues, smooth densities,
and ddd. E_kin is τ-independent and omitted. nspin ∈ {1, 2} (spin uses the
per-spin becsum and the SpinXC functional).
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
from gradwave.postscf._response import spin_sigma_triple
from gradwave.postscf.uspp_frozen import aug_density_from_becsum


def _normalize_spin(res: dict):
    """Uniform per-spin lists regardless of nspin."""
    nspin = res.get("nspin", 1)
    if nspin == 1:
        return (1, [res["coeffs"]], res["occupations"][None], res["eigenvalues"][None],
                [res["rho_ij_atoms"]], [res["rho"]])
    return (2, res["coeffs"], res["occupations"], res["eigenvalues"],
            res["rho_ij_atoms"], res["rho_spin"])


def _aug_from_becsum(system, rho_ij, phases):
    """ρ_aug(r) from one spin channel's becsum with given e^{+iGτ} phases."""
    return aug_density_from_becsum(system, rho_ij, phases)


def rho_core_on_graph(system, phases) -> torch.Tensor | None:
    """NLCC core density on the τ-graph, or None when the system has no core.

    The core rides the same e^{+iGτ} phases as the augmentation (pass the
    in-graph ``phases`` built from a positions leaf), so its τ-derivative is
    the NLCC core force once it enters the XC argument. Shared by
    ``forces_uspp``, ``uspp_position.hessian_column`` and the USPP
    discretization force error.
    """
    if system.rho_core is None:
        return None
    from gradwave.pseudo.radial_torch import RadialTables

    grid = system.grid
    vol = grid.volume
    dev = phases.device
    q_sph = torch.linalg.norm(system.g_sphere, dim=1)
    core = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE, device=dev)
    for sp in set(system.species_of_atom):
        paw = system.paws[sp]
        if paw.core_rho is None:
            continue
        tab = RadialTables(paw, device=dev)
        with torch.no_grad():
            f_core = tab.core_of_g(q_sph)
        atoms = [a for a, sa in enumerate(system.species_of_atom) if sa == sp]
        core = core + phases[:, atoms].conj().sum(dim=1) * f_core.to(CDTYPE) / vol
    core_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=dev)
    core_box[system.sphere_idx] = core
    return torch.fft.ifftn(core_box.reshape(grid.shape) * grid.n_points,
                           dim=(-3, -2, -1)).real


def _aug_at_fixed(res: dict, system, isp: int | None = None) -> torch.Tensor:
    """ρ_aug at the converged positions/becsum (isolates the smooth part).
    isp selects one spin channel; None sums all."""
    with torch.no_grad():
        phase_arg = system.g_sphere @ system.positions.T
        phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
        nspin = res.get("nspin", 1)
        chans = [res["rho_ij_atoms"]] if nspin == 1 else res["rho_ij_atoms"]
        sel = range(nspin) if isp is None else [isp]
        out = 0.0
        for s in sel:
            out = out + _aug_from_becsum(system, chans[s], phases)
        return out


def forces_uspp(res: dict, xc, remove_net: bool = True) -> torch.Tensor:
    """F_a = −dE/dτ_a (na, 3) [eV/Å] for a converged scf_uspp result."""
    system = res["system"]
    grid = system.grid
    vol = grid.volume
    nspin, coeffs_s, occ_s, eigs_s, becsum_s, rho_sp = _normalize_spin(res)
    pos = system.positions.detach().clone().requires_grad_(True)
    kw = system.kweights

    # ddd at the converged becsum (detached — chain rule is exact at the point)
    is_paw = any(p.is_paw for p in system.paws)
    ddd_atoms = []
    if is_paw:
        from gradwave.scf.paw_onsite import OneCenter

        onec = {sp: OneCenter(system.paws[sp], xc)
                for sp in set(system.species_of_atom)}
        dev0 = system.positions.device
        for a, sp in enumerate(system.species_of_atom):
            bec = (becsum_s[0][a] if nspin == 1
                   else [becsum_s[0][a], becsum_s[1][a]])
            _, ddd = onec[sp].energy_and_ddd(bec)  # one-center is CPU-side
            ddd_atoms.append([ddd.to(dev0)] if nspin == 1
                             else [d.to(dev0) for d in ddd])

    projs = [projectors(pd, pos) for pd in system.proj_data]
    phase_arg = system.g_sphere @ pos.T
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))

    # DFT+U: E_U(τ) joins the τ-differentiable energy as the explicit
    # in-graph n(τ) expression — autograd carries the φ phases AND the β
    # phases inside the S-dressing
    hub_sites = res.get("hub_sites")
    hub_phi_free = None
    if hub_sites is not None:
        from gradwave.scf.uspp_hubbard import hubbard_e_channel, phi_free_per_k

        hub_phi_free = phi_free_per_k(system, hub_sites)

    e = ewald_energy(pos, system.charges, grid.cell)
    q = system.q_full.to(CDTYPE)
    rho_chans = []
    for isp in range(nspin):
        coeffs = [c.detach() for c in coeffs_s[isp]]
        occ = occ_s[isp].detach()
        eigs = eigs_s[isp].detach()
        becps = [becp(projs[ik], coeffs[ik]) for ik in range(len(coeffs))]
        rho_ij = [torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=pos.device)
                  for (s0, s1) in system.atom_slices]
        for ik, b in enumerate(becps):
            w = (kw[ik] * occ[ik]).to(CDTYPE)
            for a, (s0, s1) in enumerate(system.atom_slices):
                ba = b[:, s0:s1]
                rho_ij[a] = rho_ij[a] + torch.einsum("b,bi,bj->ij", w, ba.conj(), ba)
        rho_ij = [0.5 * (m + m.conj().T) for m in rho_ij]

        rho_aug = _aug_from_becsum(system, rho_ij, phases)
        rho_s_fixed = (rho_sp[isp].detach() - _aug_at_fixed(res, system, isp)).detach()
        rho_chans.append(rho_s_fixed + rho_aug)

        e = e + nonlocal_energy(becps, system.proj_data[0].dij_full, occ, kw)
        for ik, b in enumerate(becps):
            quad = torch.einsum("bi,ij,bj->b", b.conj(), q, b).real
            e = e - (kw[ik] * occ[ik] * eigs[ik] * quad).sum()
        if is_paw:
            for a in range(len(system.atom_slices)):
                e = e + (ddd_atoms[a][isp].to(CDTYPE) * rho_ij[a]).sum().real
        if hub_sites is not None:
            mult = 2.0 if nspin == 1 else 1.0
            e = e + mult * hubbard_e_channel(
                hub_sites, hub_phi_free, system.q_full, pos, system.spheres,
                projs, coeffs, becps, occ, kw,
                occ_scale=(0.5 if nspin == 1 else 1.0))

    rho_tot = sum(rho_chans)
    rho_g = r_to_g(rho_tot.to(CDTYPE))

    # NLCC core on the graph
    rho_core = rho_core_on_graph(system, phases)

    from gradwave.core.density import sigma_from_rho

    if nspin == 1:
        rho_xc = rho_tot if rho_core is None else rho_tot + rho_core
        sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
        e = e + xc.energy(rho_xc, vol, sigma)
    else:
        c2 = 0.0 if rho_core is None else 0.5 * rho_core
        r_u, r_d = rho_chans[0] + c2, rho_chans[1] + c2
        s_uu, s_dd, s_tt = spin_sigma_triple(xc, r_u, r_d, grid.g_cart)
        e = e + xc.energy(r_u, r_d, vol, s_uu, s_dd, s_tt)

    species_index = torch.tensor(system.species_of_atom, dtype=torch.int64,
                                 device=pos.device)
    vloc_g = local_potential_g(pos, species_index, system.vloc_tables,
                               grid.g_cart, vol)
    e = e + hartree_energy(rho_g, grid.g2, vol) + local_energy(rho_g, vloc_g, vol)

    (grad,) = torch.autograd.grad(e, pos)
    f = -grad
    if remove_net:
        f = f - f.mean(dim=0, keepdim=True)
    if getattr(system, "sym", None) is not None:
        from gradwave.symmetry import symmetrize_forces

        f = symmetrize_forces(f, system.sym, grid.cell)
    return f
