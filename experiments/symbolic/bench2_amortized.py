"""Benchmark v2: cached density-independent basis + compiled sparse assembly.

The adapted basis is built ONCE (density-independent) and reused; the reduced
blocks are assembled with C-backed sparse·dense products (symbasis.SymBasis).
We report, at fixed k=Γ across N:

  A: baseline full eigvalsh(H)
  P: per-solve symmetry (assembly + block eigh, basis cached)   ← the amortized cost
  S: one-time basis build (paid once, amortized over all solves at this k)

so per-solve speedup A/P is what you get in SCF iterations / DFPT / parametric
sweeps / property runs at fixed geometry, and the single-shot cost is S + P.

Run:  uv run python experiments/symbolic/bench2_amortized.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.api import build_system, run_scf
from gradwave.core.hamiltonian import HamiltonianK, projectors
from gradwave.grids import build_gsphere
from gradwave.inputs import load_input
from gradwave.postscf._kb import projector_data_at_k, species_projector_tables
from gradwave.postscf.irreps import little_group
from gradwave.symmetry import find_spacegroup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_walltime import median_time, spacegroup, symmetrize  # noqa: E402
from symbasis import SymBasis  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def bench_system(name, yaml, sweep):
    inp = load_input(str(HERE / yaml))
    system = build_system(inp)
    t0 = time.perf_counter()
    res = run_scf(inp, system=system, verbose=False)
    v_eff = res.v_eff if res.v_eff.ndim == 3 else res.v_eff[0]
    sg = spacegroup(system)
    grid = system.grid
    print(f"\n=== {name} (SCF {time.perf_counter()-t0:.0f}s, {sg.international}, "
          f"ecut {system.ecut:.0f}) ===")
    print(f"{'N':>5} {'blks':>4} {'maxblk':>6} {'1/R':>6} "
          f"{'A:eigh':>9} {'P:per-solve':>11} {'A/P':>6} {'S:build':>8} "
          f"{'breakeven':>9} {'err':>8}")
    for se in sweep:
        if se > system.ecut:
            continue
        sph = build_gsphere(grid, se, np.zeros(3))
        npw = sph.npw
        miller = sph.miller.cpu().numpy()
        ops = little_group(np.zeros(3), sg, grid.cell)
        Ws = [np.asarray(op["W"], int) for op in ops]
        beta_ls, dij_sp = species_projector_tables(system.upfs, None)
        pd = projector_data_at_k(sph, system.species_of_atom, system.upfs,
                                 beta_ls, dij_sp, grid.volume, None)
        p = projectors(pd, system.positions)
        h = HamiltonianK(sph, grid.shape, v_eff, pd, p)
        H = h.apply(torch.eye(npw, dtype=torch.complex128)).transpose(0, 1).numpy()
        H = 0.5 * (H + H.conj().T)
        H = symmetrize(H, miller, ops, np.zeros(3))

        tS, sb = median_time(lambda: SymBasis(miller, ops, Ws, np.zeros(3)), repeats=1)
        R = npw ** 3 / sum(s ** 3 for s in sb.sizes)
        tA, evA = median_time(lambda: np.sort(np.linalg.eigvalsh(H)))
        tP, evP = median_time(lambda: sb.block_diagonalize(H))
        err = float(np.abs(evA - evP).max())
        breakeven = tS / (tA - tP) if tA > tP else float("inf")
        print(f"{npw:5d} {len(sb.sizes):4d} {sb.max_star:6d} {1/R:6.3f} "
              f"{tA*1e3:8.1f} {tP*1e3:10.1f} {tA/tP:5.1f}× {tS*1e3:7.1f} "
              f"{breakeven:8.1f} {err:8.1e}")


def main() -> None:
    print("Benchmark v2 — cached basis + compiled sparse assembly (times ms)")
    bench_system("diamond primitive", "diamond_bench.yaml", [500, 700, 900])
    bench_system("diamond conventional (8-atom)", "diamond_conv.yaml", [300, 450, 600])


if __name__ == "__main__":
    main()
