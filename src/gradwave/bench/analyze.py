"""Flatten persisted benchmark runs into one tidy, model-ready table.

Each run JSON holds run-level metadata plus a per-iteration trace; ``load_runs``
explodes them into one row per iteration with the run-level features broadcast
across, so the result drops straight into pandas / DuckDB / a statistical model /
symbolic regression. Needs the ``[bench]`` extra (pandas, pyarrow).

    from gradwave.bench.analyze import load_runs, load_runs_summary
    df = load_runs("bench_runs/")          # one row per SCF iteration
    runs = load_runs_summary("bench_runs/")  # one row per run (the outcome table)
"""

from __future__ import annotations

import json
from pathlib import Path


def _iter_records(run_dir):
    for path in sorted(Path(run_dir).glob("*.json")):
        try:
            yield json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue


def _run_features(rec: dict) -> dict:
    """Run-level columns broadcast onto every iteration row."""
    d, m, o = rec.get("descriptor", {}), rec.get("method", {}), rec.get("outcome", {})
    prov = rec.get("provenance", {})
    cpu = prov.get("cpu", {}) if isinstance(prov, dict) else {}
    return {
        "run_id": rec.get("run_id"),
        "case": rec.get("case"),
        "hardness": rec.get("hardness"),
        # method (what we vary to improve performance)
        "scheme": m.get("mixing_scheme"),
        "alpha": m.get("mixing_alpha"),
        "kerker": m.get("kerker"),
        "history": m.get("mixing_history"),
        # problem descriptor (what the map "looks like")
        "n_atoms": d.get("n_atoms"),
        "n_electrons": d.get("n_electrons"),
        "z_valence": d.get("z_valence"),
        "ecut_ry": d.get("ecut_ry"),
        "kmesh": d.get("kmesh"),
        "nbands": d.get("nbands"),
        "smearing": d.get("smearing"),
        "gap_hint_eV": d.get("gap_hint_eV"),
        # outcome (broadcast; the label a model predicts)
        "converged": o.get("converged"),
        "n_iter": o.get("n_iter"),
        "wall_s": o.get("wall_s"),
        "final_residual": o.get("final_residual"),
        "error": o.get("error"),
        "git_sha": prov.get("git_sha") if isinstance(prov, dict) else None,
        "host": prov.get("host") if isinstance(prov, dict) else None,
        "torch_threads": cpu.get("torch_threads") if isinstance(cpu, dict) else None,
    }


def load_runs(run_dir):
    """One row per SCF iteration, run-level features broadcast across. → DataFrame."""
    import pandas as pd

    rows = []
    for rec in _iter_records(run_dir):
        feat = _run_features(rec)
        iters = (rec.get("trace") or {}).get("iterations", [])
        if not iters:  # a failed/empty run still contributes one row (error signal)
            rows.append({**feat, "iter": None})
            continue
        for it in iters:
            row = {**feat, **it}
            sf = it.get("shell_fraction")
            if isinstance(sf, list) and sf:  # collapse the |G|-shell vector to a scalar
                row["shell_frac_low2"] = float(sum(sf[:2]))
            row.pop("shell_fraction", None)
            rows.append(row)
    return pd.DataFrame(rows)


def load_runs_summary(run_dir):
    """One row per run — the outcome/method/descriptor table. → DataFrame."""
    import pandas as pd

    return pd.DataFrame([_run_features(rec) for rec in _iter_records(run_dir)])


def to_parquet(run_dir, out_path) -> str:
    """Load the per-iteration table and write it to Parquet. → path."""
    df = load_runs(run_dir)
    df.to_parquet(out_path, index=False)
    return str(out_path)
