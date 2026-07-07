"""Non-SCF band structure at fixed converged potential (M3).

Freezes V_eff(r) and the D_ij/projector machinery from a converged SCF and
runs Davidson-only diagonalizations at arbitrary k (typically an ASE
bandpath). Eigenvalues are referenced to the Fermi level (metals) or the
valence-band maximum (fixed occupations) by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gradwave.core.hamiltonian import build_projector_data
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_gsphere
from gradwave.pseudo.kb import beta_form_factors
from gradwave.scf.loop import SCFResult


@dataclass
class BandStructure:
    kpts_frac: np.ndarray  # (nkpath, 3)
    eigenvalues: np.ndarray  # (nkpath, nbands) [eV], NOT referenced
    reference: float  # Fermi level or VBM [eV]
    labels: list | None = None  # (index, label) special-point markers
    x: np.ndarray | None = None  # path coordinate for plotting


@torch.no_grad()
def band_structure(
    res: SCFResult,
    kpts_frac: np.ndarray,
    nbands: int | None = None,
    diago_tol: float = 1e-9,
    verbose: bool = False,
) -> BandStructure:
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("spin-resolved band structures land next")
    system = res.system
    grid = system.grid
    nbands = nbands or system.nbands
    v_eff = res.v_eff
    device = v_eff.device

    beta_ls = [[b.l for b in upf.betas] for upf in system.upfs]
    dij_species = [torch.as_tensor(upf.dij, dtype=RDTYPE, device=device) for upf in system.upfs]

    kpts = np.asarray(kpts_frac, dtype=float)
    eigs = np.empty((len(kpts), nbands))

    # batch path points through the k-batched solver (the v0 per-k loop was
    # ~10x slower); chunk count bounded by dense-box memory (~1.5 GB budget)
    n_box = grid.shape[0] * grid.shape[1] * grid.shape[2]
    chunk = max(1, min(24, int(1.5e9 / (nbands * n_box * 16))))
    from gradwave.core.batch import BatchedHamiltonian, build_batched, projectors_b
    from gradwave.solvers.davidson import davidson_batched_ms

    mixed_precision = False  # opt-in only (fp32 draft is situational — see scf())

    for lo in range(0, len(kpts), chunk):
        hi = min(lo + chunk, len(kpts))
        spheres = [build_gsphere(grid, system.ecut, k, device=device) for k in kpts[lo:hi]]
        pd_list = []
        for sph in spheres:
            q = np.sqrt(sph.kpg2.cpu().numpy())
            beta_tables = [
                torch.as_tensor(beta_form_factors(upf, q), dtype=RDTYPE, device=device)
                for upf in system.upfs
            ]
            pd_list.append(build_projector_data(
                sph, system.species_of_atom, beta_tables, beta_ls, dij_species,
                grid.volume))
        bk = build_batched(spheres, pd_list, device=device)
        p_b = projectors_b(bk, system.positions)
        h = BatchedHamiltonian(bk, grid.shape, v_eff, p_b)
        c0 = torch.zeros(hi - lo, nbands, bk.npw_max, dtype=CDTYPE, device=device)
        c0[:, torch.arange(nbands), torch.arange(nbands)] = 1.0
        out = davidson_batched_ms(h.apply, c0, bk.t, bk.mask, tol=diago_tol,
                                  max_iter=80, mixed_precision=mixed_precision)
        eigs[lo:hi] = out.eigenvalues.cpu().numpy()
        if verbose:
            print(f"  band chunk {lo}-{hi - 1}/{len(kpts) - 1}  "
                  f"max|res| = {float(out.residual_norms.max()):.1e}", flush=True)

    # reference energy: Fermi (smeared) or VBM (highest eigenvalue with occ > 0)
    if float(res.occupations.min()) < 1e-12 and float(res.occupations.max()) > 1.999999:
        occ_mask = res.occupations > 1e-6
        reference = float(res.eigenvalues[occ_mask].max())
    else:
        reference = res.fermi
    return BandStructure(kpts_frac=np.asarray(kpts_frac), eigenvalues=eigs, reference=reference)


def bands_along_ase_path(res: SCFResult, atoms, path: str = "", npoints: int = 120,
                         nbands: int | None = None, verbose: bool = False) -> BandStructure:
    """Band structure along an ASE bandpath (special-point string or lattice default)."""
    bp = atoms.cell.bandpath(path=path or None, npoints=npoints)
    bs = band_structure(res, bp.kpts, nbands=nbands, verbose=verbose)
    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    bs.x = x
    bs.labels = list(zip(xticks.tolist(), list(xlabels), strict=True))
    return bs
