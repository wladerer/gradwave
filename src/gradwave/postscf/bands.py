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

from gradwave.dtypes import CDTYPE
from gradwave.grids import build_gsphere
from gradwave.postscf._kb import projector_data_at_k, species_projector_tables
from gradwave.scf.loop import SCFResult

# A state counts as partially occupied (⇒ metal) when its occupation lands
# meaningfully inside (0, 2). One consistent tolerance for both the metal gate
# and the VBM mask; smeared metals keep states far from E_F pinned at 0/2 to
# machine precision, so a tighter gate misreads them as insulating.
_OCC_TOL = 1e-4


@dataclass
class BandStructure:
    kpts_frac: np.ndarray  # (nkpath, 3)
    eigenvalues: np.ndarray  # (nkpath, nbands) [eV]; leading spin axis if nspin=2; NOT referenced
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
    system = res.system
    grid = system.grid
    nbands = nbands or system.nbands
    # Collinear nspin=2 shares projectors/D_ij across channels; only the local
    # potential splits. Band structure is then the frozen-potential solve run
    # once per spin with that channel's v_eff. Normalize to a leading spin axis
    # so the nspin=1 (v_eff shape (n1,n2,n3)) and nspin=2 ((2,n1,n2,n3)) paths
    # share one loop body.
    nspin = getattr(res, "nspin", 1)
    v_eff_s = res.v_eff if nspin == 2 else res.v_eff[None]
    device = v_eff_s.device

    beta_ls, dij_species = species_projector_tables(system.upfs, device)

    kpts = np.asarray(kpts_frac, dtype=float)
    eigs = np.empty((nspin, len(kpts), nbands))

    # batch path points through the k-batched solver (the v0 per-k loop was
    # ~10x slower); chunk count bounded by dense-box memory (~1.5 GB budget)
    n_box = grid.shape[0] * grid.shape[1] * grid.shape[2]
    chunk = max(1, min(24, int(1.5e9 / (nbands * n_box * 16))))
    from gradwave.core.batch import BatchedHamiltonian, build_batched, projectors_b
    from gradwave.solvers.davidson import davidson_batched_ms

    mixed_precision = False  # opt-in only (fp32 draft is situational — see scf())

    for lo in range(0, len(kpts), chunk):
        hi = min(lo + chunk, len(kpts))
        # spheres/projectors are spin-independent — build once and reuse across
        # channels so nspin=2 does not double the projector-table cost.
        spheres = [build_gsphere(grid, system.ecut, k, device=device) for k in kpts[lo:hi]]
        pd_list = [
            projector_data_at_k(sph, system.species_of_atom, system.upfs,
                                beta_ls, dij_species, grid.volume, device)
            for sph in spheres
        ]
        bk = build_batched(spheres, pd_list, device=device)
        p_b = projectors_b(bk, system.positions)
        for sp in range(nspin):
            h = BatchedHamiltonian(bk, grid.shape, v_eff_s[sp], p_b)
            c0 = torch.zeros(hi - lo, nbands, bk.npw_max, dtype=CDTYPE, device=device)
            c0[:, torch.arange(nbands), torch.arange(nbands)] = 1.0
            out = davidson_batched_ms(h.apply, c0, bk.t, bk.mask, tol=diago_tol,
                                      max_iter=80, mixed_precision=mixed_precision)
            eigs[sp, lo:hi] = out.eigenvalues.cpu().numpy()
            if verbose:
                tag = f" spin {sp}" if nspin == 2 else ""
                print(f"  band chunk {lo}-{hi - 1}/{len(kpts) - 1}{tag}  "
                      f"max|res| = {float(out.residual_norms.max()):.1e}", flush=True)

    # reference energy: Fermi (metal) or VBM (fixed/insulating occupations).
    # SCFResult carries no smearing scheme/width, so decide from the
    # occupations themselves: a metal has at least one partially-filled state.
    # Full occupancy per state is 2 for nspin=1 (spin-paired) but 1 for nspin=2
    # (one electron per channel), so the metal gate scales with nspin.
    occ = res.occupations
    g = 2.0 if nspin == 1 else 1.0
    is_metal = bool(((occ > _OCC_TOL) & (occ < g - _OCC_TOL)).any())
    if is_metal:
        reference = res.fermi
    else:
        reference = float(res.eigenvalues[occ > _OCC_TOL].max())
    eigenvalues = eigs[0] if nspin == 1 else eigs
    return BandStructure(
        kpts_frac=np.asarray(kpts_frac), eigenvalues=eigenvalues, reference=reference)


def bands_along_ase_path(res: SCFResult, atoms, path: str = "", npoints: int = 120,
                         nbands: int | None = None, verbose: bool = False) -> BandStructure:
    """Band structure along an ASE bandpath (special-point string or lattice default)."""
    bp = atoms.cell.bandpath(path=path or None, npoints=npoints)
    bs = band_structure(res, bp.kpts, nbands=nbands, verbose=verbose)
    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    bs.x = x
    bs.labels = list(zip(xticks.tolist(), list(xlabels), strict=True))
    return bs
