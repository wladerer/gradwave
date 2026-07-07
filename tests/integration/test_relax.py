"""M2 acceptance: FIRE relaxation of rattled Si recovers the ideal geometry."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.optimize import FIRE

from gradwave.calculator import GradWave

RY = 13.605693122994
PSEUDO = "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf"


@pytest.mark.slow
def test_fire_relax_rattled_si(tmp_path):
    torch.set_num_threads(8)
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    ideal = np.array([[0.0, 0, 0], [a / 4] * 3])
    rng = np.random.default_rng(42)
    atoms = Atoms("Si2", positions=ideal + rng.normal(0, 0.08, (2, 3)), cell=cell, pbc=True)
    atoms.calc = GradWave(
        ecut=15 * RY, pseudopotentials={"Si": PSEUDO}, xc="lda", kpts=(2, 2, 2)
    )
    opt = FIRE(atoms, logfile=None)
    assert opt.run(fmax=0.01, steps=60)
    bond = np.linalg.norm(atoms.get_positions()[1] - atoms.get_positions()[0])
    assert abs(bond - a * np.sqrt(3) / 4) < 1e-3
    assert np.abs(atoms.get_forces()).max() < 0.01
