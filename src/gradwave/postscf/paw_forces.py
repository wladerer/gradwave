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
from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.core.hamiltonian import becp, projectors
from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.dtypes import CDTYPE
from gradwave.postscf._response import spin_sigma_triple
from gradwave.postscf.uspp_frozen import aug_density_from_becsum
from gradwave.scf.results import USPPResult
from gradwave.scf.uspp_setup import USPPSystem


def _normalize_spin(
    res: USPPResult,
) -> tuple[
    int,
    list[list[torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
    list[list[torch.Tensor]],
    list[torch.Tensor],
]:
    """Uniform per-spin lists regardless of nspin."""
    nspin = res.get("nspin", 1)
    if nspin == 1:
        return (1, [res["coeffs"]], res["occupations"][None], res["eigenvalues"][None],
                [res["rho_ij_atoms"]], [res["rho"]])
    return (2, res["coeffs"], res["occupations"], res["eigenvalues"],
            res["rho_ij_atoms"], res["rho_spin"])


def _aug_from_becsum(
    system: USPPSystem, rho_ij: list[torch.Tensor], phases: torch.Tensor
) -> torch.Tensor:
    """ρ_aug(r) from one spin channel's becsum with given e^{+iGτ} phases."""
    return aug_density_from_becsum(system, rho_ij, phases)


def rho_core_on_graph(system: USPPSystem, phases: torch.Tensor) -> torch.Tensor | None:
    """NLCC core density on the τ-graph, or None when the system has no core.

    The core rides the same e^{+iGτ} phases as the augmentation (pass the
    in-graph ``phases`` built from a positions leaf), so its τ-derivative is
    the NLCC core force once it enters the XC argument. Shared by
    ``forces_uspp``, ``uspp_position.hessian_column`` and the USPP
    discretization force error.
    """
    if system.rho_core is None:
        return None
    from gradwave.pseudo.radial_torch import radial_tables

    grid = system.grid
    vol = grid.volume
    dev = phases.device
    q_sph = torch.linalg.norm(system.g_sphere, dim=1)
    core = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE, device=dev)
    tabs = radial_tables(system, device=dev)  # cached on the system
    for sp in set(system.species_of_atom):
        paw = system.paws[sp]
        if paw.core_rho is None:
            continue
        tab = tabs[sp]
        with torch.no_grad():
            f_core = tab.core_of_g(q_sph)
        atoms = [a for a, sa in enumerate(system.species_of_atom) if sa == sp]
        core = core + phases[:, atoms].conj().sum(dim=1) * f_core.to(CDTYPE) / vol
    core_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=dev)
    core_box[system.sphere_idx] = core
    return g_to_r_box(core_box.reshape(grid.shape), real=True)


def _aug_at_fixed(res: USPPResult, system: USPPSystem, isp: int | None = None) -> torch.Tensor:
    """ρ_aug at the converged positions/becsum (isolates the smooth part).
    isp selects one spin channel; None sums all."""
    with torch.no_grad():
        phase_arg = system.g_sphere @ system.positions.T
        phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
        nspin = res.get("nspin", 1)
        chans = [res["rho_ij_atoms"]] if nspin == 1 else res["rho_ij_atoms"]
        sel = range(nspin) if isp is None else [isp]
        out: torch.Tensor | None = None
        for s in sel:
            contrib = _aug_from_becsum(system, chans[s], phases)
            out = contrib if out is None else out + contrib
        # sel is never empty: isp is None -> range(nspin) with nspin in {1, 2},
        # or isp given -> the singleton [isp].
        assert out is not None
        return out


def _smooth_tau_fixed(
    system: USPPSystem,
    xc: XCFunctional | SpinXC,
    coeffs_s: list[list[torch.Tensor]],
    occ_s: torch.Tensor,
    nspin: int,
) -> list[torch.Tensor]:
    """Per-spin smooth τ̃ = ½Σf|∇ψ̃|² [e/Å⁵] from the converged orbitals, detached
    (a fixed constant on the force/stress energy graph). Pads the per-k coeffs to
    the batched (nk, nb, npw_max) layout `core.metagga.tau_b` consumes."""
    from gradwave.core.batch import build_batched
    from gradwave.core.metagga import tau_b

    grid, vol, dev = system.grid, system.grid.volume, system.positions.device
    bk = build_batched(system.spheres, system.proj_data, device=dev)
    tau_s = []
    for isp in range(nspin):
        nb = occ_s[isp].shape[1]
        cb = torch.zeros(len(system.spheres), nb, bk.npw_max, dtype=CDTYPE, device=dev)
        for ik, sph in enumerate(system.spheres):
            cb[ik, :, : sph.npw] = coeffs_s[isp][ik].detach()
        tau_s.append(tau_b(cb, occ_s[isp].detach(), system.kweights, bk, grid.shape, vol))
    return tau_s


def forces_uspp(
    res: USPPResult, xc: XCFunctional | SpinXC, remove_net: bool = True
) -> torch.Tensor:
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
        from gradwave.scf.paw_onsite import onecenters

        dev0 = system.positions.device
        onec = onecenters(system, xc, device=dev0)  # cached from the SCF
        for a, sp in enumerate(system.species_of_atom):
            bec = (becsum_s[0][a] if nspin == 1
                   else [becsum_s[0][a], becsum_s[1][a]])
            _, ddd = onec[sp].energy_and_ddd(bec)
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
            from gradwave.scf.uspp_hubbard import hubbard_e_channel

            mult = 2.0 if nspin == 1 else 1.0
            e = e + mult * hubbard_e_channel(
                hub_sites, hub_phi_free, system.q_full, pos, system.spheres,
                projs, coeffs, becps, occ, kw,
                occ_scale=(0.5 if nspin == 1 else 1.0))

    # Not `sum(rho_chans)`: its int `0` default start makes the inferred type
    # `Tensor | int` even though rho_chans always has nspin (>= 1) entries.
    rho_tot = rho_chans[0]
    for rc in rho_chans[1:]:
        rho_tot = rho_tot + rc
    rho_g = r_to_g(rho_tot.to(CDTYPE))

    # NLCC core on the graph
    rho_core = rho_core_on_graph(system, phases)

    # meta-GGA: the smooth τ̃ = ½Σf|∇ψ̃|² is a functional of the (position-free)
    # plane-wave coefficients only, so at fixed coefficients it has NO explicit
    # position dependence and enters the force energy as a DETACHED constant —
    # it carries no force term of its own (the one-center τ force rides ddd, the
    # smooth-τ Hellmann-Feynman term vanishes at the stationary point) but is
    # needed so the meta-GGA v_xc = ∂e/∂ρ|_τ is evaluated at the correct τ̃.
    tau_fixed_s = _smooth_tau_fixed(system, xc, coeffs_s, occ_s, nspin) \
        if xc.needs_tau else None

    from gradwave.core.density import sigma_from_rho

    if nspin == 1:
        # nspin=1 is always dispatched with a collinear XCFunctional (the
        # SpinXC/nspin=2 pairing is enforced by forces_uspp's own callers,
        # same convention as scf/uspp_loop.py's xc dispatch).
        assert isinstance(xc, XCFunctional)
        rho_xc = rho_tot if rho_core is None else rho_tot + rho_core
        sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
        tau_xc = tau_fixed_s[0] if tau_fixed_s is not None else None
        e = e + xc.energy(rho_xc, vol, sigma, tau_xc)
    else:
        assert isinstance(xc, SpinXC)
        c2 = 0.0 if rho_core is None else 0.5 * rho_core
        r_u, r_d = rho_chans[0] + c2, rho_chans[1] + c2
        s_uu, s_dd, s_tt = spin_sigma_triple(xc, r_u, r_d, grid.g_cart)
        tu, td = (None, None) if tau_fixed_s is None else (tau_fixed_s[0], tau_fixed_s[1])
        e = e + xc.energy(r_u, r_d, vol, s_uu, s_dd, s_tt, tu, td)

    species_index = torch.tensor(system.species_of_atom, dtype=torch.int64,
                                 device=pos.device)
    vloc_g = local_potential_g(pos, species_index, system.vloc_tables,
                               grid.g_cart, vol)
    e = e + hartree_energy(rho_g, grid.g2, vol) + local_energy(rho_g, vloc_g, vol)
    from gradwave.core.energies.esm import esm_energy, esm_mode_of

    _esm_mode = esm_mode_of(getattr(res, "boundary", "periodic"))
    if _esm_mode is not None:
        # open-boundary (ESM) force: −∂ΔE/∂R via the Gaussian ion charge, on the
        # FULL (smooth + aug) total density (same term the NC forces() adds).
        e = e + esm_energy(res.rho.detach(), pos, system.charges, grid,
                           mode=_esm_mode, bias=getattr(res, "esm_bias", 0.0))

    (grad,) = torch.autograd.grad(e, pos)
    f = -grad
    if remove_net:
        f = f - f.mean(dim=0, keepdim=True)
    if getattr(system, "sym", None) is not None:
        from gradwave.symmetry import symmetrize_forces

        f = symmetrize_forces(f, system.sym, grid.cell)
    return f
