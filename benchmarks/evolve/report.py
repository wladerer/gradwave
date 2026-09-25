"""Shared leaderboard rendering for the evolve-pilot entrypoints.

Prints a header, one row per candidate (admissibility, worst error vs the
reference, min-of-N ms, speedup vs baseline), and a final one-line verdict that
``benchmarks/_capture.py`` records as ``reported``.
"""

from __future__ import annotations

from collections.abc import Iterable

from evolve.evaluator import EvalResult


def print_leaderboard(
    title: str,
    meta_lines: Iterable[str],
    results: list[EvalResult],
    base_ms: float | None,
) -> str:
    """Render the population table; return the verdict (also printed last)."""
    print(f"# {title}")
    for line in meta_lines:
        print(f"# {line}")
    print(f"{'candidate':<20} {'admiss':>6} {'max_err':>10} {'ms':>10} {'speedup':>8}  note")
    for r in results:
        ms = f"{r.fitness_ms:.4f}" if r.fitness_ms is not None else "-"
        sp = f"{r.speedup:.3f}x" if r.speedup is not None else "-"
        note = r.error or ("" if r.admissible else "FAILED ORACLE")
        print(f"{r.name:<20} {r.admissible!s:>6} {r.max_err:>10.2e} {ms:>10} {sp:>8}  {note}")

    winners = [r for r in results
               if r.admissible and r.name != "baseline" and r.speedup is not None]
    if base_ms is None:
        verdict = "BEST=none (baseline inadmissible)"
    elif winners:
        best = max(winners, key=lambda r: r.speedup)
        verdict = f"BEST={best.name} speedup={best.speedup:.3f}x max_err={best.max_err:.2e}"
    else:
        verdict = "BEST=none (no admissible non-baseline candidate)"
    print(verdict)  # last non-empty line -> captured as `reported`
    return verdict
