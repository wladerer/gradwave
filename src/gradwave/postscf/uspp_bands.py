"""Band eigenvalues for USPP/PAW at arbitrary k (frozen-potential, Layer B).

Rebuilds v_eff and the screened D (including the one-center ddd at the
converged becsum) from a converged scf_uspp result, then solves the
generalized problem H(k)c = εS(k)c at each requested k with the same block
Davidson. For nspin=2 the solve runs once per spin channel (frozen_veff and
screened_dscr already return per-spin lists), gaining a leading spin axis.
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
    """Eigenvalues at the given fractional k-points: (nk, nbands) [eV] for
    nspin=1, (2, nk, nbands) for nspin=2."""
    if res.get("hub_sites") is not None:
        raise NotImplementedError(
            "USPP bands with DFT+U not implemented (V_U missing from the "
            "frozen band Hamiltonian)")
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

    beta_ls, dij_species = species_projector_tables(system.paws, dev)
    out = [[] for _ in range(nspin)]
    for k in k_frac_list:
        # sphere/projectors are spin-independent — build once per k, reuse
        sph = build_gsphere(grid, system.ecut, np.asarray(k, dtype=float),
                            device=dev)
        pd = projector_data_at_k(sph, system.species_of_atom, system.paws,
                                 beta_ls, dij_species, vol, dev)
        p = projectors(pd, system.positions)
        for sp in range(nspin):
            hs = _HkS(sph, grid.shape, veff_s[sp], pd, p, dscr_s[sp], system.q_full)
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
