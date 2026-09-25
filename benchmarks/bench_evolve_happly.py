"""Evolve-pilot bench: score the H-apply candidate population (tolerance oracle).

    uv run python benchmarks/bench_evolve_happly.py [reps] [tol]

Prints a leaderboard and, on the last non-empty stdout line, the best admissible
speedup vs the fp64 FFT-path baseline. Run on asus via:

    scripts/gwq --host asus bench bench_evolve_happly 100 1e-5

``tol`` is the oracle's accuracy budget: 1e-5 (default, expansion grade) admits
the fp32 apply and the Toeplitz local path; ~1e-11 admits only exact
arithmetic-reordering variants. Timing is min-of-N under 8 threads.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from evolve.driver import run_population
from evolve.happly_problem import ORACLE_TOL, HApplyProblem
from evolve.report import print_leaderboard


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else ORACLE_TOL

    torch.set_num_threads(8)
    cand_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "evolve", "candidates_happly")

    problem = HApplyProblem(tol=tol)
    results, base_ms = run_population(problem, cand_dir, reps=reps)

    tol_line = (f"oracle tol={problem.tol:.0e}  baseline={base_ms:.4f} ms" if base_ms
                else f"oracle tol={problem.tol:.0e}  baseline INADMISSIBLE")
    meta = [
        f"reps={reps}  threads={torch.get_num_threads()}",
        f"workload: real Si  nk={problem.nk} npw_max={problem.npw_max} "
        f"nb={problem.nb} n_grid={problem.n_grid}",
        tol_line,
    ]
    print_leaderboard(f"evolve pilot: {problem.name}", meta, results, base_ms)


if __name__ == "__main__":
    main()
