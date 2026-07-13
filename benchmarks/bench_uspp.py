"""USPP/PAW SCF benchmark: Si kjpaw PBE, 30 Ry / 120 Ry, 4x4x4.

nosym keeps the TR-reduced 36 k (the batching-relevant regime); sym drops
to the 8-k IBZ. perk is the reference per-k generalized Davidson path.

Usage: uv run python benchmarks/bench_uspp.py [cpu|cuda] [batched|perk]
                                              [sym|nosym] [threads]
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
batched = (sys.argv[2] if len(sys.argv) > 2 else "batched") == "batched"
use_sym = (sys.argv[3] if len(sys.argv) > 3 else "nosym") == "sym"
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
torch.set_num_threads(threads)

RY = 13.605693122994
a = 5.43
cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
pos = np.array([[0.0, 0, 0], [a / 4] * 3])
root = Path(__file__).parents[1]
paw = parse_upf_paw(root / "tests/fixtures/qe/pseudos/Si.pbe-n-kjpaw_psl.1.0.0.UPF")

t0 = time.time()
system = setup_uspp(cell, pos, [0, 0], [paw], ecut=30 * RY, kmesh=(4, 4, 4),
                    ecutrho=120 * RY, use_symmetry=use_sym)
t_setup = time.time() - t0
if device != "cpu":
    system = system.to(device)

t0 = time.time()
res = scf_uspp(system, PBE(), etol=1e-9, rhotol=1e-8, batched=batched,
               verbose=False)
if device != "cpu":
    torch.cuda.synchronize()
t_scf = time.time() - t0

print(f"device={device} batched={batched} sym={use_sym} threads={threads} "
      f"nk={len(system.spheres)}")
print(f"setup: {t_setup:.1f} s   scf: {t_scf:.1f} s   ({res['n_iter']} "
      f"iterations, conv={res['converged']})")
print(f"F = {float(res['energies'].free_energy):.8f} eV")
