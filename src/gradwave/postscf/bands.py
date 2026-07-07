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

from gradwave.constants import HBAR2_2M
from gradwave.core.hamiltonian import HamiltonianK, build_projector_data, projectors
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
    system = res.system
    grid = system.grid
    nbands = nbands or system.nbands
    v_eff = res.v_eff
    device = v_eff.device

    beta_ls = [[b.l for b in upf.betas] for upf in system.upfs]
    dij_species = [torch.as_tensor(upf.dij, dtype=RDTYPE, device=device) for upf in system.upfs]

    eigs = np.empty((len(kpts_frac), nbands))
    for i, kf in enumerate(np.asarray(kpts_frac, dtype=float)):
        sph = build_gsphere(grid, system.ecut, kf, device=device)
        q = np.sqrt(sph.kpg2.cpu().numpy())
        beta_tables = [
            torch.as_tensor(beta_form_factors(upf, q), dtype=RDTYPE, device=device)
            for upf in system.upfs
        ]
        pd = build_projector_data(
            sph, system.species_of_atom, beta_tables, beta_ls, dij_species, grid.volume
        )
        p = projectors(pd, system.positions)
        h = HamiltonianK(sph, grid.shape, v_eff, pd, p)

        c0 = torch.zeros(nbands, sph.npw, dtype=CDTYPE, device=device)
        c0[torch.arange(nbands), torch.arange(nbands)] = 1.0
        from gradwave.solvers.davidson import davidson

        out = davidson(h.apply, c0, HBAR2_2M * sph.kpg2, tol=diago_tol, max_iter=80)
        eigs[i] = out.eigenvalues.cpu().numpy()
        if verbose:
            print(f"  band k {i+1}/{len(kpts_frac)}  max|res| = {out.residual_norms.max():.1e}")

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
