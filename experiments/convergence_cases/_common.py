"""Shared helpers for the convergence case-study campaign.

Structure builders and pseudo paths for the hard SCF cases documented in
docs/manual/wisdom.md and docs/manual/performance.md. Kept separate so both the
Part 1 tag-validation driver and the Part 2 remediation experiment reuse one
definition of every system.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RY = 13.605693122994

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "qe" / "pseudos"
DG = ROOT / "benchmarks" / "delta_gauge" / "pseudos"
OUT = Path(__file__).resolve().parent
TRACES = OUT / "traces"


def fcc_cell(a: float) -> np.ndarray:
    return a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def bcc_cell(a: float) -> np.ndarray:
    return a / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])


def al_slab(nlayers: int, a: float = 4.05, vacuum: float = 8.0):
    """fcc(100) Al slab via ASE (same construction as benchmarks/bench_precond)."""
    from ase.build import fcc100

    slab = fcc100("Al", size=(1, 1, nlayers), a=a, vacuum=vacuum)
    return np.array(slab.cell), slab.get_positions(), [0] * len(slab)


def dump_trace(name: str, recorder) -> Path:
    TRACES.mkdir(parents=True, exist_ok=True)
    p = TRACES / f"{name}.json"
    p.write_text(json.dumps(recorder.to_trace_dict(), indent=1))
    return p


def low2_series(recorder) -> list[float]:
    """The 2-lowest-|G|-shell residual fraction per iteration (the sloshing signal)."""
    return [round(sum(i["shell_frac"][:2]), 4) for i in recorder.iters]


def tags_of(recorder) -> list[dict]:
    return [{"tag": t, "reason": r} for t, r in recorder.diagnose()]


def case_summary(name, *, converged, n_iter, energy, mag_abs, recorder, extra=None):
    d = {
        "case": name,
        "converged": bool(converged),
        "n_iter": int(n_iter),
        "energy_eV": None if energy is None else round(float(energy), 5),
        "mag_abs_muB": None if mag_abs is None else round(float(mag_abs), 4),
        "tags": tags_of(recorder),
        "low2_series": low2_series(recorder),
        "reorder_series": [i["reorder"] for i in recorder.iters],
        "drho_series": [round(i["drho"], 6) for i in recorder.iters],
        "summarize": recorder.summarize(),
    }
    if extra:
        d.update(extra)
    return d
