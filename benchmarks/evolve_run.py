"""Run the evolutionary mutation loop on a kernel problem (autonomous backend).

    uv run python benchmarks/evolve_run.py [problem] [gens] [reps]

problem in {happly (default), rr}. Runs entirely in-process (so launch it on
asus, where the timing is meaningful). Prints a per-generation trace, a final
leaderboard over every genome evaluated, and the best genome on the last line
(captured by benchmarks/_capture.py as ``reported``).

    scripts/gwq --host asus bench evolve_run happly 6 100
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from evolve import mutate
from evolve.happly_problem import HApplyProblem
from evolve.rr_problem import RRProblem


def _setup(problem_name: str):
    if problem_name == "rr":
        return (RRProblem(nk=8, npw=4000, m=120, nw=32),
                mutate.RR_GENES, mutate.build_rr, mutate.RR_BASELINE)
    if problem_name == "happly":
        return (HApplyProblem(tol=1e-5),
                mutate.HAPPLY_GENES, mutate.build_apply, mutate.HAPPLY_BASELINE)
    if problem_name == "happly_large":
        return (HApplyProblem(tol=1e-5, cell="cubic8", ecut_ry=30.0, nb=32),
                mutate.HAPPLY_GENES, mutate.build_apply, mutate.HAPPLY_BASELINE)
    raise SystemExit(f"problem must be happly|happly_large|rr, got {problem_name!r}")


def main() -> None:
    problem_name = sys.argv[1] if len(sys.argv) > 1 else "happly"
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    torch.set_num_threads(8)
    problem, genes, builder, baseline = _setup(problem_name)
    space = 1
    for v in genes.values():
        space *= len(v)

    log, best, base_ms, results = mutate.evolve(
        problem, genes, builder, baseline, pop=8, gens=gens, reps=reps)

    print(f"# mutation loop: {problem.name}  genome-space={space}  "
          f"gens={len(log)}/{gens}  reps={reps}  threads={torch.get_num_threads()}")
    print(f"# baseline={base_ms:.4f} ms  evaluated={len(results)}/{space} genomes")
    print("# --- per-generation best ---")
    for rec in log:
        bn = rec.best_name or "-"
        bs = f"{rec.best_speedup:.3f}x" if rec.best_speedup else "-"
        print(f"#   gen {rec.gen}: best={bn}  {bs}")
    print(f"{'genome':<40} {'admiss':>6} {'max_err':>10} {'ms':>10} {'speedup':>8}")
    for r in results:
        ms = f"{r['ms']:.4f}" if r["ms"] is not None else "-"
        sp = f"{r['speedup']:.3f}x" if r["speedup"] is not None else "-"
        note = "" if r["admissible"] else (r["error"] or "FAILED ORACLE")
        print(f"{r['name']:<40} {r['admissible']!s:>6} {r['max_err']:>10.2e} "
              f"{ms:>10} {sp:>8}  {note}")

    print(f"BEST={best['name']} speedup={best['speedup']:.3f}x max_err={best['max_err']:.2e}")


if __name__ == "__main__":
    main()
