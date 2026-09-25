"""Kernel-agnostic candidate evaluator: correctness gate + min-of-N wall time.

A ``Problem`` freezes one workload and knows how to check a candidate's output
against a reference. ``evaluate`` runs a candidate once, gates it through the
oracle, and -- only if it is semantics-preserving -- times it. An inadmissible or
throwing candidate gets no fitness: it cannot win, by construction. This is the
whole anti-corpse mechanism, so keep the gate strictly before the timer.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class EvalResult:
    name: str
    admissible: bool  # passed the oracle (semantics preserved within tolerance)
    max_err: float  # worst deviation from the reference the oracle measured
    fitness_ms: float | None  # min-of-N wall time; None if inadmissible / errored
    speedup: float | None = None  # baseline_ms / fitness_ms; filled in by the driver
    error: str | None = None  # exception text if the candidate raised


class Problem(Protocol):
    """One frozen kernel-optimisation problem: a workload and a correctness oracle."""

    name: str

    def workload(self) -> tuple[Any, ...]:
        """The frozen positional args a candidate is called with (built once)."""
        ...

    def oracle(self, out: Any) -> tuple[bool, float]:
        """Return (semantics_preserved, max_error) for a candidate's output.

        Must be gauge/degeneracy robust and reference-anchored, so that only a
        genuinely equivalent implementation passes -- the constraint the fitness
        search is not allowed to cheat."""
        ...


def _time_min(fn: Callable[..., Any], args: tuple[Any, ...], *, warmup: int, reps: int) -> float:
    """Min wall time (ms) of ``fn(*args)`` over ``reps`` reps after ``warmup``.

    Min, not mean: it is the least-contaminated estimate of the kernel's own cost
    (scheduler noise, other tenants, and turbo dips only ever add time), matching
    the repo's documented "min of N reps on asus" practice. The caller pins
    ``torch.set_num_threads(8)`` once before evaluating the population.
    """
    for _ in range(warmup):
        fn(*args)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def evaluate(
    name: str,
    candidate: Callable[..., Any],
    problem: Problem,
    *,
    warmup: int = 10,
    reps: int = 200,
) -> EvalResult:
    """Gate ``candidate`` through ``problem``'s oracle, then time it if it passes."""
    args = problem.workload()
    try:
        out = candidate(*args)
    except Exception as exc:  # a candidate that throws is simply inadmissible
        return EvalResult(name, False, float("inf"), None, error=f"{type(exc).__name__}: {exc}")

    try:
        ok, max_err = problem.oracle(out)
    except Exception as exc:
        msg = f"oracle: {type(exc).__name__}: {exc}"
        return EvalResult(name, False, float("inf"), None, error=msg)

    if not ok:
        return EvalResult(name, False, max_err, None)

    try:
        ms = _time_min(candidate, args, warmup=warmup, reps=reps)
    except Exception as exc:
        return EvalResult(name, False, max_err, None, error=f"timing: {type(exc).__name__}: {exc}")

    return EvalResult(name, True, max_err, ms)
