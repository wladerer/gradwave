"""A/B one SCF with the fp64-certified fp32-expansion Davidson on/off.

Same Si LDA system as bench_scf.py (30 Ry, 4x4x4), plus the apply-count split
the mode's speedup is bounded by: the fp32 fraction of band-vector H-applies
is the share of apply work that runs in complex64; the fp64 ("full") count
relative to the baseline's is the apply work that cannot be accelerated.

Usage: uv run python benchmarks/bench_fp32_expansion.py [cpu|cuda] [threads] [mode]
  mode: off | on | auto | ab   (default ab = run off then on and compare)

GPU-targeted (crippled-fp64 cards); on CPU it measures correctness overhead
only. Heavy runs go through scripts/gwq to asus, not a laptop.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import gradwave.solvers.registry as reg
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
mode = sys.argv[3] if len(sys.argv) > 3 else "ab"
torch.set_num_threads(threads)

RY = 13.605693122994
a = 5.43
cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
pos = np.array([[0.0, 0, 0], [a / 4] * 3])
root = Path(__file__).parents[1]
si = parse_upf(root / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")

# Tally the per-solve apply split out of the davidson adapter's diagnostics
# (the SCF loop itself drops the diagnostics dict).
tally = {"low": 0, "full": 0}
_orig = reg.get("davidson")


def _counting(*args, **kw):
    r = _orig(*args, **kw)
    tally["low"] += r.diagnostics.get("n_apply_low", 0)
    tally["full"] += r.diagnostics.get("n_apply_full", 0)
    return r


reg.register("davidson", _counting, overwrite=True)


def run(m):
    os.environ["GRADWAVE_FP32_EXPANSION"] = m
    tally["low"] = tally["full"] = 0
    system = setup_system(cell, pos, [0, 0], [si], ecut=30 * RY, kmesh=(4, 4, 4),
                          use_symmetry=False)
    if device != "cpu":
        system = system.to(device)
    t0 = time.time()
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-9, rhotol=1e-8,
              verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    dt = time.time() - t0
    tot = tally["low"] + tally["full"]
    frac = tally["low"] / tot if tot else 0.0
    print(f"mode={m:<4} scf={dt:7.1f} s  iters={res.n_iter:3d} conv={res.converged}  "
          f"E={float(res.energies.total):.8f} eV  "
          f"applies: low={tally['low']} full={tally['full']} fp32_frac={frac:.3f}")
    return dt, res, dict(tally)


print(f"device={device} threads={threads}")
if mode == "ab":
    t_off, r_off, c_off = run("off")
    t_on, r_on, c_on = run("on")
    de = abs(float(r_off.energies.total) - float(r_on.energies.total))
    print(f"A/B: wall {t_off:.1f} -> {t_on:.1f} s ({t_off / t_on:.2f}x)  "
          f"|dE|={de:.2e} eV  d_iters={r_on.n_iter - r_off.n_iter}  "
          f"fp64 applies {c_on['full']}/{c_off['full']} of baseline")
else:
    run(mode)
