"""M3 acceptance: non-SCF band structure vs QE bands.x-style run.

Same UPF/ecut/k-path; occupied bands within 10 meV everywhere on the path,
indirect gap within 20 meV. Eigenvalues are compared ABSOLUTELY (both codes
place the alpha-Z average potential in v_loc(G=0), so no alignment shift).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.bands import bands_along_ase_path
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
CELL, POS = si_fcc()


pytestmark = pytest.mark.standard  # full SCF vs QE; not a fast-gate test


@pytest.fixture(scope="module")
def bands_pair():
    torch.set_num_threads(4)
    ref = json.loads((FIX / "si_bands_ci" / "reference.json").read_text())["bands"]
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(CELL, POS, [0, 0], [upf], ecut=15 * RY, kmesh=(2, 2, 2))
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-10, rhotol=1e-9, verbose=False)
    assert res.converged
    atoms = Atoms("Si2", positions=POS, cell=CELL, pbc=True)
    bs = bands_along_ase_path(res, atoms, path="LGXUG", npoints=16, nbands=8)
    return bs, np.array(ref["eigenvalues_eV"])


def test_occupied_bands_match_qe(bands_pair):
    bs, qe = bands_pair
    diff = np.abs(bs.eigenvalues[:, :4] - qe[:, :4])
    assert diff.max() < 0.010, f"max occupied-band deviation {diff.max()*1000:.2f} meV"


def test_lowest_conduction_bands_match_qe(bands_pair):
    # first two conduction bands (higher ones may brush the Davidson buffer)
    bs, qe = bands_pair
    diff = np.abs(bs.eigenvalues[:, 4:6] - qe[:, 4:6])
    assert diff.max() < 0.015, f"max conduction-band deviation {diff.max()*1000:.2f} meV"


def test_indirect_gap_matches_qe(bands_pair):
    bs, qe = bands_pair
    gap_us = bs.eigenvalues[:, 4].min() - bs.eigenvalues[:, 3].max()
    gap_qe = qe[:, 4].min() - qe[:, 3].max()
    assert abs(gap_us - gap_qe) < 0.020


def test_gamma_degeneracies(bands_pair):
    bs, _ = bands_pair
    gamma = bs.eigenvalues[3]  # Γ is the 4th path point
    assert np.ptp(gamma[1:4]) < 1e-6  # 3-fold valence top
    assert np.ptp(gamma[4:7]) < 1e-6  # 3-fold conduction
