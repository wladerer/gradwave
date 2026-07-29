"""Band eigenvalues for USPP/PAW at arbitrary k (frozen-potential, Layer B).

Rebuilds v_eff and the screened D (including the one-center ddd at the
converged becsum) from a converged scf_uspp result, then solves the
generalized problem H(k)c = εS(k)c at each requested k with the same block
Davidson. For nspin=2 the solve runs once per spin channel (frozen_veff and
screened_dscr already return per-spin lists), gaining a leading spin axis.

DFT+U: ``_HkS`` already applies a Hubbard term ``V_U = Σ|Sφ_m⟩D_{mm'}⟨Sφ_m'|``
alongside the bare/augmentation D whenever it is handed ``hub_sphi``/``hub_d``
(the SCF loop's own precedent — see ``scf.uspp_loop._solve_bands_uspp``); this
module builds that pair at each requested k from the converged ``hub_occ`` /
``hub_sites`` the SCF result carries, mirroring ``scf.uspp_hubbard.
build_uspp_hubbard`` but at an arbitrary band k instead of the fixed SCF mesh.
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.hamiltonian import projectors
from gradwave.core.hubbard import hubbard_dmatrix
from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.dtypes import CDTYPE
from gradwave.grids import GSphere, build_gsphere
from gradwave.postscf._kb import projector_data_at_k, species_projector_tables
from gradwave.postscf.uspp_frozen import frozen_veff, screened_dscr
from gradwave.scf.results import USPPResult
from gradwave.scf.uspp import _HkS, davidson_gen
from gradwave.scf.uspp_hubbard import (
    _hubbard_radial_setup,
    atom_of_col,
    phi_free_at_sphere,
)
from gradwave.scf.uspp_setup import USPPSystem


def _hubbard_sphi_at_k(
    system: USPPSystem, sites: list[dict[str, int | float]],
    radial: tuple[dict[int, np.ndarray], dict[int, int], int, int], sph: GSphere, p: torch.Tensor,
    dev: torch.device,
) -> torch.Tensor:
    """S-dressed Hubbard atomic-orbital projectors Sφ = φ + Σ|β⟩q⟨β|φ⟩ at one
    band k, mirroring ``scf.uspp_hubbard.build_uspp_hubbard`` (batched over the
    fixed SCF k-mesh) with the k axis dropped."""
    acol = atom_of_col(sites).to(dev)
    phi_free = phi_free_at_sphere(system, sites, sph, radial=radial)
    phase_arg = sph.kpg @ system.positions.T  # (npw, na)
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))
    phi = phi_free * phases[:, acol].T
    bphi = torch.einsum("jg,mg->mj", p.conj(), phi)  # <β_j|φ_m>
    q = system.q_full.to(CDTYPE)
    return phi + torch.einsum("mj,ij,ig->mg", bphi, q, p)


def bands_uspp(
    res: USPPResult, xc: XCFunctional | SpinXC,
    k_frac_list: list[list[int | float]] | np.ndarray | list[list[float]],
    nbands: int | None = None, tol: float = 1e-9,
) -> torch.Tensor:
    """Eigenvalues at the given fractional k-points: (nk, nbands) [eV] for
    nspin=1, (2, nk, nbands) for nspin=2."""
    system = res["system"]
    grid = system.grid
    vol = grid.volume
    dev = system.positions.device
    nbands = nbands or system.nbands
    nspin = res.get("nspin", 1)

    # frozen v_eff and screened D (∫v_eff Q + bare + one-center ddd) at the
    # converged density/becsum — both are per-spin lists
    veff_s = frozen_veff(res, xc)
    dscr_s = screened_dscr(res, xc, veff_s)

    hub_sites = res.get("hub_sites")
    hub_occ = res.get("hub_occ")
    hub_radial = nproj_u = hub_d_s = None
    if hub_sites is not None:
        nproj_u = sum(s["dim"] for s in hub_sites)
        hub_radial = _hubbard_radial_setup(system, hub_sites)
        # D^scr_U = (U−J)(½δ−n) at the frozen converged occupation, one per
        # spin; apply wants D^T = conj(D) for Hermitian D (same convention as
        # the SCF loop's _solve_bands_uspp)
        hub_d_s = [hubbard_dmatrix(hub_occ[sp], hub_sites, nproj_u, dev).conj()
                  for sp in range(nspin)]

    beta_ls, dij_species = species_projector_tables(system.paws, dev)
    out = [[] for _ in range(nspin)]
    for k in k_frac_list:
        # sphere/projectors are spin-independent — build once per k, reuse
        sph = build_gsphere(grid, system.ecut, np.asarray(k, dtype=float),
                            device=dev)
        pd = projector_data_at_k(sph, system.species_of_atom, system.paws,
                                 beta_ls, dij_species, vol, dev)
        p = projectors(pd, system.positions)
        hub_sphi = None
        if hub_sites is not None:
            # hub_radial is set in lockstep with hub_sites above (same guard).
            assert hub_radial is not None
            hub_sphi = _hubbard_sphi_at_k(system, hub_sites, hub_radial, sph, p, dev)
        for sp in range(nspin):
            hd = hub_d_s[sp] if hub_d_s is not None else None
            hs = _HkS(sph, grid.shape, veff_s[sp], pd, p, dscr_s[sp], system.q_full,
                     hub_sphi=hub_sphi, hub_d=hd)
            # seed on CPU (device-independent determinism), then move
            gen = torch.Generator().manual_seed(4321)
            x0 = (torch.randn(nbands + 4, sph.npw, generator=gen, dtype=torch.float64)
                  + 1j * torch.randn(nbands + 4, sph.npw, generator=gen,
                                     dtype=torch.float64))
            x0 = (x0.to(dev) * torch.exp(-0.5 * sph.kpg2 / system.ecut * 4.0)).to(CDTYPE)
            eps, _ = davidson_gen(hs, x0, nbands, tol=tol, max_iter=120)
            out[sp].append(eps)
    per_spin = [torch.stack(e) for e in out]  # each (nk, nbands)
    return per_spin[0] if nspin == 1 else torch.stack(per_spin)
