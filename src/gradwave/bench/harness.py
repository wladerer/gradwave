"""Run benchmark cases, stamp provenance, persist analyzable per-run records.

``run_case`` executes one (case × method) SCF with the flight recorder on, wraps
the per-iteration trace with a system descriptor, the method config, hardware/git
provenance, and the reduced outcome, and returns a :class:`RunRecord`.
``write_run`` drops it as one JSON sidecar per run — parallel-safe (one file per
run), stdlib-only. The tidy flattening for analysis lives in
:mod:`gradwave.bench.analyze` (needs the ``[bench]`` extra).
"""

from __future__ import annotations

import itertools
import json
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gradwave.bench.suite import BenchCase
from gradwave.scf.loop import scf

# method axes swept by default: quasi-Newton scheme × damping × Kerker precond.
DEFAULT_METHODS = dict(
    mixing_scheme=["pulay", "broyden"],
    mixing_alpha=[0.3, 0.5, 0.7],
    kerker=[True],
)


@dataclass
class RunRecord:
    run_id: str
    case: str
    hardness: str
    method: dict[str, Any]
    descriptor: dict[str, Any]
    provenance: dict[str, Any]
    outcome: dict[str, Any]
    trace: dict[str, Any] = field(default_factory=dict)


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=3,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _provenance() -> dict[str, Any]:
    prov: dict[str, Any] = {
        "git_sha": _git_sha(),
        "host": socket.gethostname(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        from gradwave.io import runinfo

        prov["cpu"] = runinfo.cpu_info()
        prov["memory"] = runinfo.memory_info()
        gpu = runinfo.gpu_info()
        if gpu:
            prov["gpu"] = gpu
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        pass
    return prov


def sweep_methods(axes: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Cartesian product of the method axes → a list of scf() keyword configs."""
    axes = axes or DEFAULT_METHODS
    keys = list(axes)
    return [dict(zip(keys, combo, strict=True))
            for combo in itertools.product(*(axes[k] for k in keys))]


def run_case(case: BenchCase, method: dict[str, Any], *,
             run_dir: str | Path | None = None) -> RunRecord:
    """Run one (case × method) SCF with the recorder on; return its RunRecord.

    A non-converging or failing solve is captured as an outcome, not raised, so a
    sweep completes even when a method blows up on a hard case (that failure IS
    signal for a model). ``run_dir`` (if given) persists the record immediately.
    """
    tag = "_".join(f"{k}{method[k]}" for k in sorted(method))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    run_id = f"{case.name}__{tag}__{stamp}"

    outcome: dict[str, Any] = {"converged": False, "error": None}
    trace: dict[str, Any] = {}
    t0 = time.perf_counter()
    try:
        system = case.build()
        res = scf(system, case.xc(), **case.scf_kwargs, **method)
        wall = time.perf_counter() - t0
        trace = res.recorder.to_trace_dict() if res.recorder is not None else {}
        summary = res.recorder.summarize() if res.recorder is not None else {}
        outcome = {
            "converged": bool(res.converged),
            "n_iter": int(res.n_iter),
            "wall_s": round(wall, 4),
            "final_residual": float(res.history[-1]["res"]) if res.history else None,
            "final_free_energy_eV": float(res.energies.free_energy),
            "diagnosis": summary.get("diagnosis", []),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — a blown-up solve is a recorded outcome
        outcome = {"converged": False, "wall_s": round(time.perf_counter() - t0, 4),
                   "error": f"{type(exc).__name__}: {exc}"}

    rec = RunRecord(run_id=run_id, case=case.name, hardness=case.hardness,
                    method=method, descriptor=case.descriptor,
                    provenance=_provenance(), outcome=outcome, trace=trace)
    if run_dir is not None:
        write_run(rec, run_dir)
    return rec


def write_run(rec: RunRecord, run_dir: str | Path) -> Path:
    """Persist one run as ``<run_dir>/<run_id>.json`` (one file per run)."""
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{rec.run_id}.json"
    path.write_text(json.dumps(asdict(rec), indent=None))
    return path
