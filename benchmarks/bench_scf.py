"""SCF benchmark: Si LDA, 30 Ry (36 k TR-reduced; 8 k with symmetry).

Usage: uv run python benchmarks/bench_scf.py \
       [cpu|cuda] [threads] [sym|nosym] [mp|fp64] [kmesh]

  mp     enable the fp32 mixed-precision Davidson draft (default fp64)
  kmesh  k-mesh side (default 4 → 4x4x4); larger makes the run SCF-dominated
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
use_sym = (sys.argv[3] if len(sys.argv) > 3 else "nosym") == "sym"
mixed_precision = (sys.argv[4] if len(sys.argv) > 4 else "fp64") == "mp"
km = int(sys.argv[5]) if len(sys.argv) > 5 else 4
torch.set_num_threads(threads)

RY = 13.605693122994
a = 5.43
cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
pos = np.array([[0.0, 0, 0], [a / 4] * 3])
root = Path(__file__).parents[1]
si = parse_upf(root / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")

t0 = time.time()
system = setup_system(cell, pos, [0, 0], [si], ecut=30 * RY, kmesh=(km, km, km),
                      use_symmetry=use_sym)
t_setup = time.time() - t0
if device != "cpu":
    system = system.to(device)

t0 = time.time()
res = scf(system, LDA_PW92(), smearing="none", etol=1e-9, rhotol=1e-8,
          mixed_precision=mixed_precision, verbose=False)
if device != "cpu":
    torch.cuda.synchronize()
t_scf = time.time() - t0

print(f"device={device} threads={threads} sym={use_sym} mp={mixed_precision} "
      f"kmesh={km} nk={len(system.spheres)}")
print(f"setup: {t_setup:.1f} s   scf: {t_scf:.1f} s   ({res.n_iter} iterations, "
      f"conv={res.converged})")
print(f"E = {float(res.energies.total):.8f} eV  (QE ref: -213.94494866)")
