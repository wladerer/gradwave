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

    # dedup fired: many primitive path points share a supercell image
    assert len(ub.unique_K) < _NPTS

    # weights are bimodal (0 or 1) for a perfect crystal
    w = ub.weights.ravel()
    nz = w[w > 1e-3]
    frac_high = float((nz > 0.9).sum()) / nz.size
    assert frac_high > 0.8, frac_high

    # the lowest 4 primitive (valence) bands are reproduced by a weight≈1
    # supercell state within a few meV at every path point
    prim = bs.eigenvalues - bs.reference          # (nk, 8)
    sup = ub.eigenvalues - ub.reference           # (nk, 40)
    for ik in range(_NPTS):
        heavy = sup[ik][ub.weights[ik] > 0.9]
        heavy.sort()
        for b in range(4):
            assert np.min(np.abs(heavy - prim[ik, b])) < 0.02, (ik, b)


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
