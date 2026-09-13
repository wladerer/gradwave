"""Probe A2 — shipped Gamma real-wavefunction path e2e A/B (with conversion).

The realification lever's BROAD case (Gamma-only large cells: molecules,
defects, supercells) is already SHIPPED as the Gamma real path
(src/gradwave/core/gamma.py, GRADWAVE_GAMMA_REAL, default off per [D-009]). This
measures whether that shipped path actually banks a whole-SCF win INCLUDING the
half<->full conversion cost, on a subspace-dominated single-Gamma insulator.

If the shipped Gamma real path is <1.15x e2e over the complex path (conversion
eats the half-bytes win), the "realification is an unexploited lever" claim is
dead for the broad case too. Combined with A1 (the non-Gamma TRIM extension only
adds the handful of TRIM k in a mesh -> narrow), that settles the moonshot.

Runs locked alternating off/on pairs to cancel drift; spy-verifies res.gamma_real.

Usage:
  uv run python benchmarks/moonshot_doorclosers/probe_a2_gamma_real.py [npairs]
Run on an IDLE box (timing-sensitive). Thread cap via OMP_NUM_THREADS.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

RY = 13.605693122994


def build_si_gamma(ecut_ry: float, nrep: int):
    a = 5.43
    conv = np.array([
        [0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0],
        [1, 1, 1], [1, 3, 3], [3, 1, 3], [3, 3, 1],
    ], dtype=float) * (a / 4)
    reps = [conv + np.array([i, j, k]) * a
            for i in range(nrep) for j in range(nrep) for k in range(nrep)]
    pos = np.concatenate(reps, axis=0)
    cell = np.eye(3) * (nrep * a)
    root = Path(__file__).resolve().parents[2]
    si = parse_upf(root / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")
    return setup_system(cell, pos, [0] * len(pos), [si], ecut=ecut_ry * RY,
                        kmesh=(1, 1, 1), use_symmetry=False)


def run(mode: str, system):
    os.environ["GRADWAVE_GAMMA_REAL"] = mode  # "0" complex, "1" force real
    t0 = time.time()
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-9, rhotol=1e-8,
              verbose=False)
    dt = time.time() - t0
    return dt, res


def main():
    npairs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    torch.set_num_threads(threads)
    # Si-64 Gamma-only, 30 Ry: single Gamma k, big subspace -> RR/local-FFT heavy
    system = build_si_gamma(30.0, 2)
    print(f"SI64-Gamma npw={int(system.batch.npw_max)} grid={tuple(system.grid.shape)} "
          f"threads={threads} npairs={npairs}", flush=True)
    offs, ons = [], []
    for i in range(npairs):
        t_off, r_off = run("0", system)
        t_on, r_on = run("1", system)
        de = abs(float(r_off.energies.total) - float(r_on.energies.total))
        print(f"pair {i}: complex={t_off:.2f}s (gamma_real={r_off.gamma_real}, "
              f"it={r_off.n_iter}) real={t_on:.2f}s (gamma_real={r_on.gamma_real}, "
              f"it={r_on.n_iter})  dE={de:.2e}eV", flush=True)
        offs.append(t_off)
        ons.append(t_on)
    mo, mn = float(np.median(offs)), float(np.median(ons))
    print(f"MEDIAN complex={mo:.2f}s real={mn:.2f}s speedup(complex/real)={mo/mn:.3f}x",
          flush=True)


if __name__ == "__main__":
    main()
