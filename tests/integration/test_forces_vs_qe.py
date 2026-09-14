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


@pytest.fixture(scope="module")
def si_result():
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(CELL, POS, [0, 0], [upf], ecut=15 * RY, kmesh=(2, 2, 2))
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-10, rhotol=1e-9, verbose=False)
    assert res.converged
    return res


@pytest.mark.standard  # full SCF vs QE anchor; not a fast-gate test
def test_forces_match_qe(si_result):
    ref = json.loads((FIX / "si_forces_ci" / "reference.json").read_text())
    f_qe = np.array(ref["forces_eV_ang"])
    f_us = forces(si_result).cpu().numpy()
    assert np.abs(f_us - f_qe).max() < 5e-3, f"\nqe:\n{f_qe}\nus:\n{f_us}"


@pytest.mark.standard  # rides the module-scoped QE-parameter SCF fixture
def test_force_sum_rule_egg_box_level(si_result):
    f = forces(si_result)
    assert float(f.sum(dim=0).abs().max()) < 1e-4


@pytest.mark.standard  # rides the module-scoped QE-parameter SCF fixture
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


def test_forces_autograd_vs_fd():
    """Analytic forces vs central finite differences of the SCF total energy,
    for ALL 3 Cartesian components on BOTH atoms — the convention-free
    self-oracle F = −dE/dR, independent of QE. Fast tier (unmarked): the
    identity holds at any basis/grid, so a coarse Γ-point cell runs in a few
    seconds. Mirrors ``test_stress_vs_qe.py::test_stress_autograd_vs_fd``.

    ``remove_net=False``: FD of a single atom's displacement measures the raw
    force −dE/dr_{a,i}; the net-removal shift would break that per-atom identity
    (it is an egg-box cleanup, not part of the energy derivative)."""
    torch.set_num_threads(6)
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")

    def run(pos):
        # coarse ecut/k: the F = −dE/dR identity is basis-independent
        system = setup_system(CELL, pos, [0, 0], [upf], ecut=12 * RY, kmesh=(1, 1, 1))
        return scf(system, LDA_PW92(), smearing="none",
                   etol=1e-11, rhotol=1e-10, verbose=False)

    res = run(POS)
    assert res.converged
    f = forces(res, remove_net=False).cpu().numpy()  # (2, 3) eV/Å

    h = 1e-4
    worst = 0.0
    for a in range(2):
        for i in range(3):
            vals = []
            for sign in (+1, -1):
                pos = POS.copy()
                pos[a, i] += sign * h
                vals.append(float(run(pos).energies.total))
            fd = -(vals[0] - vals[1]) / (2 * h)
            err = abs(fd - float(f[a, i]))
            worst = max(worst, err)
            assert err < 1e-5, (a, i, f[a, i], fd, err)
    print(f"\nmax |analytic − FD| force = {worst:.2e} eV/Å")
