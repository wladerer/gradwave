"""Hellmann–Feynman forces vs QE and vs finite differences (M2 acceptance).

The force sum rule is violated at the XC-grid egg-box level (~5e-5 eV/Å at
15 Ry, decaying with cutoff: 8.6e-6 at 35 Ry) — same order as QE itself at
matched grids; the threshold here reflects that, not an idealized zero.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
A = 5.43
CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
FRAC = np.array([[0.0, 0.0, 0.0], [0.24, 0.26, 0.255]])  # matches si_forces_ci/pw.in
POS = FRAC @ CELL


pytestmark = pytest.mark.standard  # full SCF vs QE; not a fast-gate test


@pytest.fixture(scope="module")
def si_result():
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(CELL, POS, [0, 0], [upf], ecut=15 * RY, kmesh=(2, 2, 2))
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-10, rhotol=1e-9, verbose=False)
    assert res.converged
    return res


def test_forces_match_qe(si_result):
    ref = json.loads((FIX / "si_forces_ci" / "reference.json").read_text())
    f_qe = np.array(ref["forces_eV_ang"])
    f_us = forces(si_result).cpu().numpy()
    assert np.abs(f_us - f_qe).max() < 5e-3, f"\nqe:\n{f_qe}\nus:\n{f_us}"


def test_force_sum_rule_egg_box_level(si_result):
    f = forces(si_result)
    assert float(f.sum(dim=0).abs().max()) < 1e-4


def test_forces_match_finite_difference(si_result):
    # one component; FD of our own total energy (independent of QE)
    f = forces(si_result)
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    h = 1e-4
    vals = []
    for sign in (+1, -1):
        pos = POS.copy()
        pos[1, 0] += sign * h
        system = setup_system(CELL, pos, [0, 0], [upf], ecut=15 * RY, kmesh=(2, 2, 2))
        r = scf(system, LDA_PW92(), smearing="none", etol=1e-11, rhotol=1e-10, verbose=False)
        vals.append(float(r.energies.total))
    fd = -(vals[0] - vals[1]) / (2 * h)
    assert abs(fd - float(f[1, 0])) < 1e-4
