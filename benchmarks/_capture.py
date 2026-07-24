"""Run a benchmark script and persist a structured result record.

`gwq bench <name> <args...>` submits `uv run python benchmarks/_capture.py
<name> <args...>` to the queue. This wrapper runs the named standalone bench in
a subprocess (same contract as ci_bench.py: the script prints its own timing
summary on the last non-empty stdout line), times the whole run, and writes a
JSON record to benchmarks/results/<host>/<name>-<sha>-<ts>.json.

That directory is gitignored — the records are the history a dashboard charts
across runs (wall time, the reported metric line, exit status) without churning
the repo. Absolute wall times are only comparable *within a host*; the record
stamps the host, git sha, and thread hints so cross-run trends stay honest.

Standalone: run a bench directly and still capture it with
    uv run python benchmarks/_capture.py bench_scf cpu 8 nosym
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TIMEOUT_S = int(os.environ.get("BENCH_TIMEOUT", "3600"))


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: _capture.py <bench_name> [args...]", file=sys.stderr)
        return 2
    name = sys.argv[1]
    argv = sys.argv[2:]
    script = HERE / f"{name}.py"
    if not script.exists():
        print(f"error: no benchmark named {name!r} ({script} missing)", file=sys.stderr)
        return 2

    host = socket.gethostname()
    sha = git_sha()
    started = datetime.now(timezone.utc)

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(script), *argv],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        status = "ok" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        status, stdout, stderr, rc = "TIMEOUT", e.stdout or "", e.stderr or "", 124
    wall = time.time() - t0

    lines = [ln for ln in (stdout or "").splitlines() if ln.strip()]
    reported = lines[-1] if lines else (
        (stderr or "").strip().splitlines()[-1:] or [""])[0]

    record = {
        "name": name,
        "argv": argv,
        "host": host,
        "git_sha": sha,
        "started_utc": started.isoformat(),
        "wall_s": round(wall, 2),
        "status": status,
        "returncode": rc,
        "reported": reported,
        "stdout_tail": lines[-10:],
        "threads_env": os.environ.get("THREADS") or os.environ.get("OMP_NUM_THREADS"),
    }

    ts = started.strftime("%Y%m%dT%H%M%SZ")
    outdir = REPO / "benchmarks" / "results" / host
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{name}-{sha}-{ts}.json"
    outpath.write_text(json.dumps(record, indent=1))

    # echo a one-line summary so `pueue log` / stdout stays useful
    print(f"[capture] {name} {' '.join(argv)}  {status}  {wall:.1f}s  -> {outpath}")
    print(f"[capture] reported: {reported}")
    # forward the child's stdout so nothing is lost in the log
    if stdout:
        sys.stdout.write(stdout)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
