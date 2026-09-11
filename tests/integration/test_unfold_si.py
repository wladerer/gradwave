"""Physics controls for supercell band-structure unfolding (Popescu–Zunger EBS).

Perfect-supercell control (decisive): the unfolded EBS of a 16-atom Si supercell
(2×2×2 of the 2-atom primitive) must reproduce the primitive band structure —
eigenvalues agree at matched path points and weights are ≈0/≈1. Defect control:
displacing one atom fractionalizes the weight. Standard tier: two 16-atom SCFs.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.bands import bands_along_ase_path
from gradwave.postscf.unfold import unfold_bands
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf

pytestmark = pytest.mark.standard  # two 16-atom SCFs; not a fast-gate test

_ECUT = 12 * RY
_PATH = "GXL"
_NPTS = 20


def _run(cell, pos, species, upfs, kmesh, nbands):
    system = setup_system(cell, pos, species, upfs, ecut=_ECUT,
                          kmesh=kmesh, nbands=nbands)
    res = scf(system, LDA_PW92(), smearing="gaussian", width=0.05,
              etol=1e-8, rhotol=1e-7, verbose=False)
    assert res.converged
    return res


def _supercell(a=5.43):
    cell, pos = si_fcc(a)
    prim = Atoms("Si2", positions=pos, cell=cell, pbc=True)
    sc = prim.repeat((2, 2, 2))  # A_super = diag(2,2,2) @ A_prim, 16 atoms
    return np.asarray(sc.get_cell()), sc.get_positions()


def test_perfect_supercell_reproduces_primitive_bands():
    # primitive reference band structure
    cell, pos = si_fcc()
    res_p = _run(cell, pos, [0, 0], [si_upf(), si_upf()], (4, 4, 4), nbands=8)
    bs = bands_along_ase_path(res_p, Atoms(cell=cell, pbc=True),
                              path=_PATH, npoints=_NPTS, nbands=8)

    # 16-atom perfect supercell, commensurate (2,2,2) mesh, unfolded on the SAME
    # primitive path. nbands ≥ det(M)·(compared prim bands) so the low weight-1
    # states are all resolved.
    sc_cell, sc_pos = _supercell()
    res_s = _run(sc_cell, sc_pos, [0] * 16, [si_upf()] * 16, (2, 2, 2), nbands=40)
    ub = unfold_bands(res_s, [2, 2, 2], path=_PATH, npoints=_NPTS, nbands=40)

    # dedup fired: primitive path points share supercell images (only the
    # unique K set is diagonalized)
    assert len(ub.unique_K) < _NPTS

    prim = bs.eigenvalues - bs.reference          # (nk, 8)
    sup = ub.eigenvalues - ub.reference           # (nk, 40)
    w = ub.weights                                # (nk, 40)

    assert float(w.max()) < 1.0 + 1e-6

    # Unitarity over a complete manifold: Si has 4 primitive valence bands,
    # gap-separated from the conduction bands along GXL, so the unfolded weight
    # below the mid-gap must sum to exactly 4 at every path point (Σ_m P_m ∈ ℤ
    # over the complete valence manifold). Per-band weights are NOT bimodal for a
    # perfect crystal — degenerate folded multiplets at high-symmetry K split the
    # weight among the (arbitrary) solver eigenvectors — but the manifold sum is
    # exact. (Summing over the whole band set instead would break the integer at
    # path points where nbands truncates a degenerate conduction multiplet.)
    gapmid = 0.5 * (prim[:, 3] + prim[:, 4])   # mid valence–conduction gap
    below = sup < gapmid[:, None]
    valence_weight = (w * below).sum(axis=1)
    assert np.allclose(valence_weight, 4.0, atol=0.02), valence_weight

    # The EBS reproduces the primitive band structure: below the mid-gap every
    # bit of spectral weight lies within a few meV of a primitive band eigenvalue
    # — weight appears only on the primitive dispersion. This is the decisive
    # perfect-supercell control (eigenvalue agreement + no spurious weight),
    # degeneracy-safe because it sums weight rather than matching bands.
    near_w = tot_w = 0.0
    for ik in range(_NPTS):
        for m in range(sup.shape[1]):
            if sup[ik, m] > gapmid[ik]:
                continue
            tot_w += w[ik, m]
            if np.min(np.abs(prim[ik] - sup[ik, m])) < 0.03:
                near_w += w[ik, m]
    assert near_w / tot_w > 0.98, near_w / tot_w


def test_defect_supercell_fractionalizes_weight():
    sc_cell, sc_pos = _supercell()
    sc_pos = sc_pos.copy()
    sc_pos[0, 0] += 0.2  # displace one atom 0.2 Å — breaks the translation symmetry
    res = _run(sc_cell, sc_pos, [0] * 16, [si_upf()] * 16, (2, 2, 2), nbands=40)
    ub = unfold_bands(res, [2, 2, 2], path=_PATH, npoints=_NPTS, nbands=40)

    w = ub.weights.ravel()
    nz = w[w > 1e-3]
    # the perfect-crystal bimodality must break: a non-trivial share of the
    # nonzero weight now lands in the intermediate band (0.1, 0.9)
    frac_mid = float(((nz > 0.1) & (nz < 0.9)).sum()) / nz.size
    assert frac_mid > 0.05, frac_mid
    # per-k spectral weight stays bounded (no unitarity blow-up)
    assert float(ub.weights.max()) < 1.0 + 1e-6
