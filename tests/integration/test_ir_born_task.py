"""Berry-phase Born charges + IR on MgO (rocksalt, ONCV NC, PBE).

Slow tier: 1 + 12 primitive-cell SCFs on the full (TR-unfolded) 4x4x4 mesh.
Validates the Stage-1/2 physics end to end on real wavefunctions:

- Z* is isotropic with the rocksalt signs/magnitudes (literature ~ +/-1.96-2.0;
  a 4^3 Berry-phase mesh at 35 Ry gives ~ +/-2.17 -- see PR #395 for the k-mesh
  refinement step, 6^3 -> +/-2.12),
- the acoustic sum rule holds to the k-mesh convergence level,
- a centrosymmetric-path sanity: the undisplaced reduced polarization sits on
  the lattice (P = 0 mod the quantum) for this centrosymmetric crystal.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.born import born_effective_charges
from gradwave.postscf.polarization import berry_phase_polarization
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.slow

A = 4.24  # PBE-ish MgO lattice constant [Angstrom]
CELL = A / 2.0 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL  # Mg, O
MESH = (4, 4, 4)


@pytest.fixture(scope="module")
def scf_fn():
    torch.set_num_threads(8)
    upfs = [parse_upf(str(Path(PSEUDOS) / "Mg_ONCV_PBE-1.2.upf")),
            parse_upf(str(Path(PSEUDOS) / "O_ONCV_PBE-1.2.upf"))]

    def _fn(positions):
        system = setup_system(CELL, positions, [0, 1], upfs, ecut=35 * RY,
                              kmesh=MESH, kshift=(0, 0, 0), use_symmetry=False,
                              time_reversal=False)
        return scf(system, PBE(), nspin=1, smearing="none",
                   etol=1e-7, rhotol=1e-6, diago_tol=1e-8, verbose=False)

    return _fn


@pytest.fixture(scope="module")
def mgo_born(scf_fn):
    return born_effective_charges(scf_fn, POS, MESH, step=2e-3)


def test_centrosymmetric_polarization_on_lattice(scf_fn):
    # MgO rocksalt is centrosymmetric: the reduced total polarization must sit
    # on the lattice of quanta (integer or half-integer per direction is the
    # symmetry-allowed set; for this cell it lands on integers/halves).
    res = scf_fn(POS)
    pol = berry_phase_polarization(res, MESH)
    p = pol.reduced_total.numpy()
    # distance to the nearest half-integer lattice point (0 mod 1/2)
    frac = np.abs(p * 2.0 - np.rint(p * 2.0)) / 2.0
    assert frac.max() < 5e-3


def test_born_charges_rocksalt(mgo_born):
    z = mgo_born["born"].numpy()
    z_mg = np.trace(z[0]) / 3.0
    z_o = np.trace(z[1]) / 3.0
    # rocksalt: isotropic, opposite, |Z*| ~ 2 (lit ~1.96-2.0; 4^3/35Ry ~2.17)
    assert 1.8 < z_mg < 2.4
    assert -2.4 < z_o < -1.8
    # isotropy: off-diagonals tiny relative to the diagonal
    for k in range(2):
        off = z[k] - np.diag(np.diag(z[k]))
        assert np.abs(off).max() < 0.02
    # diagonal is uniform
    assert np.ptp(np.diag(z[0])) < 0.02


def test_acoustic_sum_rule(mgo_born):
    # ASR residual is a k-mesh-convergence diagnostic; 4^3 sits at ~1e-2
    assert float(mgo_born["asr_max"]) < 0.05
