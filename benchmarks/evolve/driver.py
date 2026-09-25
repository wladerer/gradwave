"""Discover the candidate population, evaluate each against a Problem, rank.

Candidates are one ``*.py`` per file under a directory, each exposing ``rr`` with
the target signature. The driver loads them by path (so a new variant is just a
new file), evaluates every one through the shared evaluator, fills in speedup
relative to the named baseline, and returns the leaderboard sorted best-first with
inadmissible candidates last. No selection/mutation policy lives here -- this is
the fitness-evaluation half of the loop; proposing new candidate files is the
other half (an LLM/agent step), kept out so the measurement stays deterministic.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

from evolve.evaluator import EvalResult, Problem, evaluate


def load_candidates(cand_dir: str | Path) -> dict[str, Callable]:
    """Load ``rr`` from every non-underscore ``*.py`` in ``cand_dir``, by path."""
    out: dict[str, Callable] = {}
    for p in sorted(Path(cand_dir).glob("*.py")):
        if p.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"_evolve_cand_{p.stem}", p)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "rr", None)
        if callable(fn):
            out[p.stem] = fn
    return out


def run_population(
    problem: Problem,
    cand_dir: str | Path,
    *,
    baseline: str = "baseline",
    warmup: int = 10,
    reps: int = 200,
) -> tuple[list[EvalResult], float | None]:
    """Evaluate the whole population; return (ranked results, baseline_ms)."""
    cands = load_candidates(cand_dir)
    results = [evaluate(name, fn, problem, warmup=warmup, reps=reps)
               for name, fn in cands.items()]

    base = next((r for r in results if r.name == baseline), None)
    base_ms = base.fitness_ms if (base and base.admissible) else None
    if base_ms:
        for r in results:
            if r.admissible and r.fitness_ms:
                r.speedup = base_ms / r.fitness_ms

    results.sort(key=lambda r: (not r.admissible,
                                r.fitness_ms if r.fitness_ms is not None else float("inf")))
    return results, base_ms
