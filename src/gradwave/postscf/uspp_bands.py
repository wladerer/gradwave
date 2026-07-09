"""Band eigenvalues for USPP/PAW at arbitrary k (frozen-potential, Layer B).

Rebuilds v_eff and the screened D (including the one-center ddd at the
converged becsum) from a converged scf_uspp result, then solves the
generalized problem H(k)c = εS(k)c at each requested k with the same block
Davidson. nspin=1; spin bands follow the same pattern per channel.
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.energies.hartree import hartree_potential_g
from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.fftbox import r_to_g
from gradwave.core.hamiltonian import build_projector_data, projectors
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_gsphere
from gradwave.pseudo.kb import beta_form_factors
from gradwave.scf.loop import vxc_potential
from gradwave.scf.uspp import _HkS, davidson_gen


def bands_uspp(res: dict, xc, k_frac_list, nbands: int | None = None,
               tol: float = 1e-9) -> torch.Tensor:
    """Eigenvalues (nk, nbands) [eV] at the given fractional k-points."""
    if res.get("nspin", 1) != 1:
        raise NotImplementedError("USPP bands for nspin=2 not implemented yet")
    system = res["system"]
    grid = system.grid
    vol = grid.volume
    nbands = nbands or system.nbands
    mask_flat = grid.dens_mask.reshape(-1)

    # frozen v_eff from the converged density
    rho = res["rho"].detach()
    rho_g_box = r_to_g(rho.to(CDTYPE))
    v_h = (torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2),
                           dim=(-3, -2, -1)) * grid.n_points).real
    rho_xc = rho if system.rho_core is None else rho + system.rho_core
    v_xc, _ = vxc_potential(xc, rho_xc, grid)
    vloc_g = local_potential_g(system.positions,
                               torch.tensor(system.species_of_atom),
                               system.vloc_tables, grid.g_cart, vol)
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real
    v_eff = v_h + v_xc + vloc_r

    # frozen screened D: ∫v_eff Q + bare + one-center ddd at converged becsum
    v_eff_g = r_to_g(v_eff.to(CDTYPE)).reshape(-1)[mask_flat]
    phase_arg = system.g_sphere @ system.positions.T
    phase_pos = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
    dscr = torch.zeros_like(system.q_full)
    for a, sp in enumerate(system.species_of_atom):
        s0, s1 = system.atom_slices[a]
        contr = torch.einsum("ijg,g->ij", system.aug[sp].q_g.conj(),
                             v_eff_g * phase_pos[:, a])
        dscr[s0:s1, s0:s1] = (0.5 * (contr + contr.conj().T)).real
    dscr_full = dscr + system.proj_data[0].dij_full
    if any(p.is_paw for p in system.paws):
        from gradwave.scf.paw_onsite import OneCenter

        onec = {sp: OneCenter(system.paws[sp], xc)
                for sp in set(system.species_of_atom)}
        dscr_full = dscr_full.clone()
        for a, sp in enumerate(system.species_of_atom):
            _, ddd = onec[sp].energy_and_ddd(res["rho_ij_atoms"][a])
            s0, s1 = system.atom_slices[a]
            dscr_full[s0:s1, s0:s1] += ddd

    dij_species = [torch.as_tensor(p.dij, dtype=RDTYPE) for p in system.paws]
    beta_ls = [[b.l for b in p.betas] for p in system.paws]
    out = []
    for k in k_frac_list:
        sph = build_gsphere(grid, system.ecut, np.asarray(k, dtype=float))
        q_of_k = np.sqrt(sph.kpg2.numpy())
        beta_tables = [torch.as_tensor(beta_form_factors(p, q_of_k), dtype=RDTYPE)
                       for p in system.paws]
        pd = build_projector_data(sph, system.species_of_atom, beta_tables,
                                  beta_ls, dij_species, vol)
        p = projectors(pd, system.positions)
        hs = _HkS(sph, grid.shape, v_eff, pd, p, dscr_full, system.q_full)
        gen = torch.Generator().manual_seed(4321)
        x0 = (torch.randn(nbands + 4, sph.npw, generator=gen, dtype=torch.float64)
              + 1j * torch.randn(nbands + 4, sph.npw, generator=gen,
                                 dtype=torch.float64))
        x0 = (x0 * torch.exp(-0.5 * sph.kpg2 / system.ecut * 4.0)).to(CDTYPE)
        eps, _ = davidson_gen(hs, x0, nbands, tol=tol, max_iter=120)
        out.append(eps)
    return torch.stack(out)
