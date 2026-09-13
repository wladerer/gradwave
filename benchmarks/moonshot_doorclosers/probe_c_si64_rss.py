"""Probe C — Si-64 / 30 Ry peak-RSS + paging confirmation (GO-2 premise).

GO-2 (CheFSI-for-memory) rests on "the ~10 GB [V,HV] working set swaps at
Si-64 30Ry." PR #479 (auto dense-box band-chunking) reportedly moved the peak
to ~6.4 GB native / ~8.3 GB eager with no paging on the 14 GB box.

Confirm on SHIPPED main: Si-64 (2x2x2 diamond, 30 Ry, 2x2x2 MP -> 4 IBZ k,
nb=128, LDA, smearing none), 8 threads. Record peak RSS (VmHWM) and whether the
run pages (vmstat si/so watched by the caller). Peak RSS is reached in the first
SCF solve, so max_iter is capped low to keep the run (and any contention with a
concurrent campaign) short.

Verdict:
  peak RSS ~6-8 GB, zero paging  -> GO-2 swap premise STALE (KILL: reverts to
                                    the already-measured-negative CheFSI-as-solver)
  peak RSS >11 GB or pages       -> #479 win regressed (flag loudly, real bug)

Usage:
  uv run python benchmarks/moonshot_doorclosers/probe_c_si64_rss.py [solver]
    solver in {davidson (default), davidson-native}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

RY = 13.605693122994


def peak_rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def main():
    solver = sys.argv[1] if len(sys.argv) > 1 else "davidson"
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[2]
    a = 5.43
    conv = np.array([
        [0, 0, 0], [0, 2, 2], [2, 0, 2], [2, 2, 0],
        [1, 1, 1], [1, 3, 3], [3, 1, 3], [3, 3, 1],
    ], dtype=float) * (a / 4)
    reps = [conv + np.array([i, j, k]) * a
            for i in range(2) for j in range(2) for k in range(2)]
    pos = np.concatenate(reps, axis=0)  # 64 atoms
    cell = np.eye(3) * (2 * a)
    si = parse_upf(root / "tests/fixtures/qe/pseudos/Si_ONCV_PBE-1.2.upf")
    system = setup_system(cell, pos, [0] * 64, [si], ecut=30 * RY,
                          kmesh=(2, 2, 2), use_symmetry=True, nbands=128)
    print(f"SI64/30Ry nk={len(system.spheres)} npw_max={int(system.batch.npw_max)} "
          f"grid={tuple(system.grid.shape)} solver={solver} rss_after_setup="
          f"{peak_rss_mb():.0f}MB", flush=True)

    t0 = time.time()
    try:
        res = scf(system, LDA_PW92(), smearing="none", max_iter=3,
                  etol=1e-9, rhotol=1e-8, verbose=False, eigensolver=solver)
        conv_s = f"iters={res.n_iter} conv={res.converged} E={float(res.energies.total):.4f}"
    except Exception as e:  # native solver may segfault/raise at this shape
        conv_s = f"SCF-RAISED: {type(e).__name__}: {e}"
    dt = time.time() - t0
    print(f"RESULT solver={solver} PEAK_RSS={peak_rss_mb():.0f}MB "
          f"({peak_rss_mb() / 1024:.2f}GB) wall={dt:.1f}s {conv_s}", flush=True)


if __name__ == "__main__":
    main()
