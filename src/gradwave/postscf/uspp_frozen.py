"""Frozen v_eff and screened D of a converged USPP/PAW state.

The post-SCF band Hamiltonian H_σ(k) c = ε S(k) c freezes the converged
effective potential and the D matrix screened by it (plus the PAW one-center
ddd). uspp_bands, discretization_error, and uspp_implicit each rebuilt this from
the converged density and becsum; the shared pieces live here.

`res` is the converged-state dict: res["system"], res["rho"] (total density),
res["rho_spin"] (per-spin, nspin=2), res["rho_ij_atoms"] (becsum), res["nspin"].
"""

from __future__ import annotations

import torch

from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.dtypes import CDTYPE


def screen_phase(system) -> torch.Tensor:
    """e^{+i G·τ_a} on the density sphere, (nGm, na)."""
    phase_arg = system.g_sphere @ system.positions.T
    return torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))


def aug_dmat(system, w_r: torch.Tensor, phase_pos: torch.Tensor) -> torch.Tensor:
    """Block-diagonal ∫ w(r) Q_ij(r−τ_a) d³r: how a real grid field screens D.

    Returns the (nproj, nproj) real, Hermitized augmentation contribution only —
    callers add dij_full and the PAW one-center ddd. This is the exact pairing
    the SCF uses to fold v_eff into D, so it also serves grid perturbations.
    The single-field wrapper over the shared species-batched screening kernel
    (stays autograd-safe for the force/stress graph).
    """
    from gradwave.scf.uspp_loop import aug_dmat_batched

    mask_flat = system.grid.dens_mask.reshape(-1)
    w_g = r_to_g(w_r.to(CDTYPE)).reshape(-1)[mask_flat]
    return aug_dmat_batched(system, w_g[None], phase_pos)[0]


def aug_density_from_becsum(system, becsum, phases) -> torch.Tensor:
    """ρ_aug(r) on the real grid from a per-atom becsum list and e^{+iG·τ} phases.

    Σ_a e^{-iG·τ_a} Σ_ij becsum_a[ij] Q_ij(G), scattered onto the dense sphere,
    scaled by 1/Ω, and iFFT'd to real space.
    """
    grid = system.grid
    dev = phases.device
    aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE, device=dev)
    for a, sp in enumerate(system.species_of_atom):
        aug_sph = aug_sph + phases[:, a].conj() * torch.einsum(
            "ij,ijg->g", becsum[a], system.aug[sp].q_g)
    aug_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=dev)
    aug_box[system.sphere_idx] = aug_sph / grid.volume
    return g_to_r_box(aug_box.reshape(grid.shape), real=True)


def frozen_veff(res: dict, xc) -> list[torch.Tensor]:
    """Per-spin v_eff = v_H + v_loc + v_xc [eV] of a converged state.

    Length-nspin list; the NLCC core density is folded into v_xc.
    """
    system = res["system"]
    grid = system.grid
    dev = system.positions.device
    nspin = int(res.get("nspin", 1))

    vloc_g = local_potential_g(
        system.positions, torch.as_tensor(system.species_of_atom, device=dev),
        system.vloc_tables, grid.g_cart, grid.volume)
    vloc_r = g_to_r_box(vloc_g, real=True)

    # same per-spin v_eff the SCF iterates (loop.effective_potentials), on the
    # detached converged densities.
    from gradwave.scf.loop import effective_potentials

    rho_s = ([res["rho"].detach()] if nspin == 1
             else [r.detach() for r in res["rho_spin"]])
    return effective_potentials(system, xc, rho_s, vloc_r)


def screened_dscr(res: dict, xc, veff_s: list[torch.Tensor]) -> list[torch.Tensor]:
    """dij_full + ∫ v_eff^σ Q + PAW one-center ddd, per spin. len == len(veff_s)."""
    system = res["system"]
    dev = system.positions.device
    nspin = len(veff_s)
    phase_pos = screen_phase(system)
    dscr_s = [system.proj_data[0].dij_full + aug_dmat(system, v, phase_pos)
              for v in veff_s]

    if any(p.is_paw for p in system.paws):
        from gradwave.scf.paw_onsite import OneCenter

        onec = {sp: OneCenter(system.paws[sp], xc)
                for sp in set(system.species_of_atom)}
        dscr_s = [d.clone() for d in dscr_s]
        bec = res["rho_ij_atoms"]
        for a, sp in enumerate(system.species_of_atom):
            s0, s1 = system.atom_slices[a]
            if nspin == 1:
                _, ddd = onec[sp].energy_and_ddd(bec[a])
                dscr_s[0][s0:s1, s0:s1] += ddd.to(dev)
            else:
                _, ddd = onec[sp].energy_and_ddd([bec[0][a], bec[1][a]])
                for isp in range(nspin):
                    dscr_s[isp][s0:s1, s0:s1] += ddd[isp].to(dev)
    return dscr_s
