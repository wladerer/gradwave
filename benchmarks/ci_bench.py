"""Run a curated set of benchmarks and emit a Markdown timing table.

Designed to run on CI (GitHub Actions) so performance checks never load a
laptop — see .github/workflows/bench.yml. Each benchmark is an existing
standalone script in this directory; we run it in a subprocess, time the whole
run, and capture the timing line it prints itself. Absolute numbers on shared
CI runners are noisy (2-4 variable vCPUs, no GPU), so read the table for
regression *trends* across runs, not as calibrated wall times — the asus box
(or a self-hosted runner) is the place for calibrated numbers.

Usage:
    uv run python benchmarks/ci_bench.py                # default quick set
    uv run python benchmarks/ci_bench.py bench_scf bench_precond
    THREADS=2 uv run python benchmarks/ci_bench.py      # cap threads on small runners
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> argv passed to the script. Keep the default set bounded so a CI run
# finishes in a few minutes; heavier sweeps stay opt-in on the command line.
BENCHES: dict[str, list[str]] = {
    "bench_scf": ["cpu", os.environ.get("THREADS", "2"), "nosym"],
    "bench_precond": ["nc"],
}

# Per-benchmark wall-clock ceiling; a benchmark that blows past it is reported
# as TIMEOUT rather than hanging the whole CI job.
TIMEOUT_S = int(os.environ.get("BENCH_TIMEOUT", "900"))


def run_one(name: str, argv: list[str]) -> dict:
    script = HERE / f"{name}.py"
    if not script.exists():
        return {"name": name, "status": "MISSING", "wall": 0.0, "detail": ""}
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(script), *argv],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "TIMEOUT", "wall": TIMEOUT_S,
                "detail": f"exceeded {TIMEOUT_S}s"}
    wall = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    # the scripts print their own timing summary on the last non-empty stdout line
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    detail = lines[-1] if lines else (proc.stderr.strip().splitlines()[-1:] or [""])[0]
    return {"name": name, "status": status, "wall": wall,
            "detail": detail, "argv": " ".join(argv)}


def main() -> int:
    requested = sys.argv[1:] or list(BENCHES)
    results = []
    for name in requested:
        argv = BENCHES.get(name, [os.environ.get("THREADS", "2")])
        print(f"# running {name} {argv} ...", file=sys.stderr, flush=True)
        results.append(run_one(name, argv))

    # Markdown table (renders in the GitHub Actions job summary)
    rows = ["| benchmark | args | wall (s) | status | reported |",
            "|---|---|---:|---|---|"]
    for r in results:
        rows.append(
            f"| `{r['name']}` | {r.get('argv', '')} | {r['wall']:.1f} | "
            f"{r['status']} | {r['detail']} |"
        )
    table = "\n".join(rows)
    print(table)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write("## Benchmark results\n\n" + table + "\n")

    # non-zero exit if any benchmark failed outright (not just slow)
    return 1 if any(r["status"].startswith(("FAIL", "TIMEOUT", "MISSING"))
                    for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
