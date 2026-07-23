"""Band eigenvalues for USPP/PAW at arbitrary k (frozen-potential, Layer B).

Rebuilds v_eff and the screened D (including the one-center ddd at the
converged becsum) from a converged scf_uspp result, then solves the
generalized problem H(k)c = εS(k)c at each requested k with the same block
Davidson. nspin=1; spin bands follow the same pattern per channel.
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.hamiltonian import projectors
from gradwave.dtypes import CDTYPE
from gradwave.grids import build_gsphere
from gradwave.postscf._kb import projector_data_at_k, species_projector_tables
from gradwave.postscf.uspp_frozen import frozen_veff, screened_dscr
from gradwave.scf.uspp import _HkS, davidson_gen


def bands_uspp(res: dict, xc, k_frac_list, nbands: int | None = None,
               tol: float = 1e-9) -> torch.Tensor:
    """Eigenvalues (nk, nbands) [eV] at the given fractional k-points."""
    if res.get("nspin", 1) != 1:
        raise NotImplementedError("USPP bands for nspin=2 not implemented yet")
    if res.get("hub_sites") is not None:
        raise NotImplementedError(
            "USPP bands with DFT+U not implemented (V_U missing from the "
            "frozen band Hamiltonian)")
    system = res["system"]
    grid = system.grid
    vol = grid.volume
    dev = system.positions.device
    nbands = nbands or system.nbands

    # frozen v_eff and screened D (∫v_eff Q + bare + one-center ddd) at the
    # converged density/becsum
    v_eff = frozen_veff(res, xc)[0]
    dscr_full = screened_dscr(res, xc, [v_eff])[0]

    beta_ls, dij_species = species_projector_tables(system.paws, dev)
    out = []
    for k in k_frac_list:
        sph = build_gsphere(grid, system.ecut, np.asarray(k, dtype=float),
                            device=dev)
        pd = projector_data_at_k(sph, system.species_of_atom, system.paws,
                                 beta_ls, dij_species, vol, dev)
        p = projectors(pd, system.positions)
        hs = _HkS(sph, grid.shape, v_eff, pd, p, dscr_full, system.q_full)
        # seed on CPU (device-independent determinism), then move
        gen = torch.Generator().manual_seed(4321)
        x0 = (torch.randn(nbands + 4, sph.npw, generator=gen, dtype=torch.float64)
              + 1j * torch.randn(nbands + 4, sph.npw, generator=gen,
                                 dtype=torch.float64))
        x0 = (x0.to(dev) * torch.exp(-0.5 * sph.kpg2 / system.ecut * 4.0)).to(CDTYPE)
        eps, _ = davidson_gen(hs, x0, nbands, tol=tol, max_iter=120)
        out.append(eps)
    return torch.stack(out)
