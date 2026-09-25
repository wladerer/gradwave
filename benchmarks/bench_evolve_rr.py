"""Evolve-pilot bench: score the Davidson Rayleigh-Ritz candidate population.

    uv run python benchmarks/bench_evolve_rr.py [reps] [regime]

regime in {supercell (default), small}. Prints a leaderboard and, on the last
non-empty stdout line, the best admissible speedup vs baseline -- the metric the
gwq/_capture.py wrapper records. Run on asus via:

    scripts/gwq --host asus bench bench_evolve_rr 200 supercell

Timing is min-of-N under 8 threads (matching the repo's benches). This is a
PLUMBING + first-population run: the point is to prove the loop measures fitness,
enforces the correctness oracle, and ranks honestly -- not to claim a win on a
kernel the native large-dim path already mined.
"""

from __future__ import annotations

import os
import sys

# benchmarks/ (this file's dir) on sys.path so `evolve` imports as a top-level pkg.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from evolve.driver import run_population
from evolve.rr_problem import ORACLE_TOL, RRProblem

REGIMES = {
    # wide subspace + large basis: RR costs measurable time (supercell-like)
    "supercell": dict(nk=8, npw=4000, m=120, nw=32),
    # tiny: fast plumbing check, RR is microseconds (speedups unmeasurable)
    "small": dict(nk=4, npw=200, m=32, nw=8),
}


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    regime = sys.argv[2] if len(sys.argv) > 2 else "supercell"
    if regime not in REGIMES:
        raise SystemExit(f"regime must be one of {sorted(REGIMES)}, got {regime!r}")

    torch.set_num_threads(8)
    cand_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evolve", "candidates_rr")

    problem = RRProblem(**REGIMES[regime])
    results, base_ms = run_population(problem, cand_dir, reps=reps)

    print(f"# evolve pilot: {problem.name}  regime={regime}  "
          f"reps={reps}  threads={torch.get_num_threads()}")
    print(f"# workload: nk={problem.nk} npw={problem.npw} m={problem.m} nw={problem.nw} "
          f"dtype={problem.dtype}")
    base_line = f"# oracle tol={ORACLE_TOL:.0e}  baseline={base_ms:.4f} ms" if base_ms \
        else f"# oracle tol={ORACLE_TOL:.0e}  baseline INADMISSIBLE"
    print(base_line)
    print(f"{'candidate':<20} {'admiss':>6} {'max_err':>10} {'ms':>10} {'speedup':>8}  note")
    for r in results:
        ms = f"{r.fitness_ms:.4f}" if r.fitness_ms is not None else "-"
        sp = f"{r.speedup:.3f}x" if r.speedup is not None else "-"
        note = r.error or ("" if r.admissible else "FAILED ORACLE")
        print(f"{r.name:<20} {r.admissible!s:>6} {r.max_err:>10.2e} {ms:>10} {sp:>8}  {note}")

    # Winner among admissible non-baseline candidates (best measured speedup).
    winners = [r for r in results if r.admissible and r.name != "baseline"
               and r.speedup is not None]
    if winners:
        best = max(winners, key=lambda r: r.speedup)
        verdict = f"BEST={best.name} speedup={best.speedup:.3f}x max_err={best.max_err:.2e}"
    else:
        verdict = "BEST=none (no admissible non-baseline candidate)"
    print(verdict)  # last non-empty line -> captured as `reported`


if __name__ == "__main__":
    main()
