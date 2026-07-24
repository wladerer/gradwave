"""Supercell phonon task: diamond Si, PBE.

Slow tier: a small supercell finite-displacement run (13 warm-started SCFs).
Validates the full `run_phonons` path — the folded Γ frequencies must match the
primitive optical phonon (~521 cm⁻¹, experiment ~520), the acoustic modes must
sit near zero, and no branch may go strongly imaginary. The converged full-cell
dispersion vs QE ph.x / neutron data runs as a benchmark on asus, not here.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from gradwave.api import run_phonons
from gradwave.inputs import Input, KPointsParams, PhononParams, SmearingParams
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def si_phonons():
    torch.set_num_threads(8)
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    atoms = Atoms("Si2", positions=pos, cell=cell, pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=Path(PSEUDOS),
        pseudo_map={"Si": "Si_ONCV_PBE-1.2.upf"}, ecut=20 * RY, xc="pbe",
        kpoints=KPointsParams(mesh=(4, 4, 4)),
        smearing=SmearingParams(type="none"),
        phonons=PhononParams(supercell=(2, 2, 2), displacement=0.01,
                             npoints=60, dos_mesh=(6, 6, 6)))
    return run_phonons(inp, verbose=False)


def test_phonon_gamma_and_stability(si_phonons):
    ph = si_phonons
    freqs = np.array(ph["frequencies_cm1"])          # (nq, 6)
    x = np.array(ph["x"])
    labels = ph["labels"]
    # locate Γ (label "G") on the path
    gx = [xt for xt, lab in labels if lab in ("G", "Γ")][0]
    ig = int(np.argmin(np.abs(x - gx)))
    at_gamma = np.sort(freqs[ig])
    # 3 acoustic near zero, 3 optical near the Si Γ optical phonon (~520 cm⁻¹)
    assert np.abs(at_gamma[:3]).max() < 10.0            # acoustic ≈ 0
    assert 490.0 < at_gamma[3:].mean() < 545.0          # optical ~521
    # optical branch is threefold degenerate at Γ
    assert at_gamma[3:].std() < 8.0
    # no strongly imaginary branch anywhere on the path
    assert ph["min_frequency_cm1"] > -15.0


def test_phonon_dos_present(si_phonons):
    dos = si_phonons["dos"]
    g = np.array(dos["frequency_cm1"])
    d = np.array(dos["dos"])
    assert g.shape == d.shape and (d >= 0).all()
    assert d.sum() > 0.0
    # spectral weight extends up to the optical band, none far above it
    assert g[d > d.max() * 0.01].max() < 600.0
