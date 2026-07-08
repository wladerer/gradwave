"""NVE energy-conservation test: rattled Si8, velocity-Verlet, GradWave/CUDA.

Total-energy drift measures force/energy consistency — for autograd
Hellmann-Feynman forces the drift should be at the SCF-noise floor.
"""
import sys
import time

import numpy as np
import torch

torch.set_num_threads(8)
sys.stdout.reconfigure(line_buffering=True)

# route every system built by the calculator to the GPU (bind before the
# calculator module imports setup_system)
import gradwave.scf.loop as _loop  # noqa: E402

_orig = _loop.setup_system
_loop.setup_system = lambda *a, **k: _orig(*a, **k).to("cuda")

from ase import units  # noqa: E402
from ase.build import bulk  # noqa: E402
from ase.md.verlet import VelocityVerlet  # noqa: E402

from gradwave.calculator import GradWave  # noqa: E402

RY = 13.605693122994
atoms = bulk("Si", "diamond", a=5.43, cubic=True)  # 8 atoms
rng = np.random.RandomState(42)
atoms.positions += 0.05 * rng.standard_normal(atoms.positions.shape)

calc = GradWave(
    ecut=30 * RY,
    pseudopotentials={"Si": "/home/wladerer/github/QSuite/tests/fixtures/qe/pseudos/"
                            "Si_ONCV_PBE-1.2.upf"},
    xc="pbe", kpts=(2, 2, 2), smearing="none",
    etol=1e-8, rhotol=1e-7, use_symmetry=False,
)
atoms.calc = calc
dyn = VelocityVerlet(atoms, timestep=2.0 * units.fs)
t_wall = time.time()


def log():
    ep = atoms.get_potential_energy()
    ek = atoms.get_kinetic_energy()
    t = dyn.get_time() / units.fs
    print(f"{t:8.1f} {ep:+.8f} {ek:+.8f} {ep + ek:+.10f} {time.time() - t_wall:7.1f}")


dyn.attach(log, interval=1)
log()
dyn.run(250)
print("NVE_DONE")
