"""Deep-profile a gradwave workload into a provenance-stamped artifact set.

This is the granular tier of the two-tier profiling harness. It runs the SAME
workload three ways, each answering a different question the others cannot:

* **torch.profiler** (primary, no new dependency) — the op-level view. Produces
  ``key_averages()`` (a structured op table, persisted as parquet + json for
  cross-commit diffing) and a Perfetto ``export_chrome_trace()`` timeline.
* **py-spy** (``[profiling]`` extra) — a sampling flamegraph of the Python
  dispatch / eager-glue that torch.profiler cannot see. Emits SVG, inlined into
  the report. Skipped with a note if the tool is absent or ptrace is denied.
* **memray** (``[profiling]`` extra) — an allocation flamegraph + high-water
  timeline for the memory question. Emits its own self-contained HTML, linked.
  Skipped with a note if unavailable.

Headline wall time and peak RSS come from a SEPARATE, UNPROFILED, thread-pinned
timed loop (median of ``n_timed`` runs, warmup discarded) — the profiled run's
wall time is distorted by profiler overhead and must never be reported as the
speed number. :func:`deep_profile` orchestrates all of this and hands a
:class:`ProfileResult` to :mod:`gradwave.profiling.report`.

Provenance (full + short git SHA, dirty flag, host, versions, CPU, threads)
comes from :func:`gradwave.io.runinfo.machine_snapshot`, and the output directory
``benchmarks/results/<host>/<short-sha>/<workload>/`` is keyed by that SHA so it
is the join key for :func:`gradwave.profiling.report.compare`, not merely a label.
"""

from __future__ import annotations

import gzip
import json
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from gradwave.io import runinfo

# Phase grouping: torch.profiler op-name substrings → the four SCF phases we care
# about. First match wins; anything unmatched lands in "other". Substrings are
# lowercased before matching. This is the "derive phase grouping from the op
# names" path from the design; it is a heuristic bucketing, not an exact split.
_PHASE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("diag", ("eigh", "syevd", "syev", "linalg_eig", "geev", "cusolver")),
    ("fft", ("fft", "_fft_", "rfft", "irfft")),
    ("h-apply", ("matmul", "bmm", "addmm", "mm", "linear", "einsum", "baddbmm")),
    ("mixing", ("add", "mul", "sub", "div", "axpy", "copy", "cat", "stack")),
)


@dataclass
class Workload:
    """One profileable unit of work.

    ``run`` executes the workload in-process (torch.profiler and the timed loop
    call it) and returns whatever the work produced — for a bench case that is a
    :class:`gradwave.bench.RunRecord`, whose ``trace`` supplies the per-iteration
    RSS / phase telemetry the report plots. ``command`` reproduces the SAME work
    in a fresh process for the process-level samplers (py-spy, memray); if None,
    those tools are skipped with a note (they cannot profile an in-process
    closure). ``spec`` is the human-facing workload descriptor shown in the
    report header (system name, n_atoms, ecut, k-mesh, what was profiled).
    """

    name: str
    spec: dict[str, Any]
    run: Callable[[], Any]
    command: list[str] | None = None


@dataclass
class ProfileResult:
    """Everything :mod:`gradwave.profiling.report` needs to render the summary.

    Serialized (minus the large table string and inlined SVG) to
    ``profile.json`` for machine-readable diffing; the op table also round-trips
    to ``op_table.parquet`` / ``op_table.json``.
    """

    workload: dict[str, Any]
    provenance: dict[str, Any]
    headline: dict[str, Any]
    op_rows: list[dict[str, Any]]
    op_table_str: str
    iterations: list[dict[str, Any]]
    phase_times_s: dict[str, float]
    out_dir: str
    trace_path: str | None = None
    memray_html: str | None = None
    flame_svg: str | None = None
    notes: list[str] = field(default_factory=list)


def workload_from_bench_case(case_name: str, *, alpha: float = 0.5) -> Workload:
    """Build a :class:`Workload` from a named :func:`gradwave.bench.build_suite`
    case. Laptop-safe cases (Si, C-diamond, Al, Cu) make the light validation
    example. The subprocess ``command`` targets :mod:`gradwave.profiling._runner`
    so py-spy/memray reproduce the identical SCF."""
    from gradwave.bench import run_case
    from gradwave.bench.suite import build_suite

    cases = {c.name: c for c in build_suite()}
    case = cases.get(case_name)
    if case is None:
        raise ValueError(
            f"unknown bench case {case_name!r}; available: {sorted(cases)}"
        )
    method = {"mixing_scheme": "pulay", "mixing_alpha": alpha, "kerker": True}
    spec = {
        "system": case.name,
        "hardness": case.hardness,
        "profiled": "one SCF solve (setup + iterations to convergence)",
        **case.descriptor,
    }
    command = [
        sys.executable,
        "-m",
        "gradwave.profiling._runner",
        "--case",
        case_name,
        "--alpha",
        str(alpha),
    ]
    return Workload(
        name=case_name,
        spec=spec,
        run=lambda: run_case(case, method),
        command=command,
    )


def _op_row(event: Any) -> dict[str, Any]:
    """One torch.profiler ``FunctionEventAvg`` → a flat, version-robust row.

    torch has renamed CUDA-time attributes across releases (``cuda_time_total``
    → ``self_device_time_total``); we read whatever is present via getattr so the
    same code works on 2.4 through current."""

    def _get(*names: str, default: float = 0.0) -> float:
        for n in names:
            v = getattr(event, n, None)
            if v is not None:
                return float(v)
        return default

    return {
        "name": str(getattr(event, "key", "?")),
        "count": int(getattr(event, "count", 0) or 0),
        "cpu_time_us": _get("cpu_time_total"),
        "self_cpu_time_us": _get("self_cpu_time_total"),
        "cuda_time_us": _get("cuda_time_total", "device_time_total"),
        "self_cuda_time_us": _get("self_cuda_time_total", "self_device_time_total"),
        "cpu_mem_bytes": _get("cpu_memory_usage"),
        "self_cpu_mem_bytes": _get("self_cpu_memory_usage"),
    }


def phase_breakdown(op_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Group op self-CPU time (seconds) into the four SCF phases + ``other``."""
    totals: dict[str, float] = {"h-apply": 0.0, "fft": 0.0, "diag": 0.0,
                                "mixing": 0.0, "other": 0.0}
    for row in op_rows:
        name = row["name"].lower()
        secs = row["self_cpu_time_us"] / 1e6
        phase = "other"
        for cand, needles in _PHASE_PATTERNS:
            if any(n in name for n in needles):
                phase = cand
                break
        totals[phase] += secs
    return totals


def _iterations_of(run_out: Any) -> list[dict[str, Any]]:
    """Pull the per-iteration trace (rss_mb, phase counts, t_s) out of whatever
    ``Workload.run`` returned. Handles a bench ``RunRecord`` (``.trace``) and a
    plain dict; returns [] if there is no trace to plot."""
    trace: Any = None
    if hasattr(run_out, "trace"):
        trace = run_out.trace
    elif isinstance(run_out, dict):
        trace = run_out.get("trace")
    if isinstance(trace, dict):
        iters = trace.get("iterations")
        if isinstance(iters, list):
            return [dict(it) for it in iters]
    return []


def _timed_runs(workload: Workload, *, n_timed: int, warmup: int) -> dict[str, Any]:
    """Unprofiled, thread-pinned timing loop → median wall + peak RSS.

    The profiler distorts wall time, so the headline speed/memory numbers come
    from HERE, not the profiled run — this separation is methodologically
    required and stated in the report. ``warmup`` runs are discarded (cold caches,
    first-touch allocation) before the median."""
    import torch

    walls: list[float] = []
    peak_rss_gb = 0.0
    last_out: Any = None
    for _ in range(n_timed):
        meter = runinfo.ProcessMeter()
        last_out = workload.run()
        m = meter.stop()
        walls.append(float(m["wall_s"]))
        rss = m.get("peak_rss_gb")
        if isinstance(rss, int | float):
            peak_rss_gb = max(peak_rss_gb, float(rss))
    kept = walls[warmup:] if len(walls) > warmup else walls
    return {
        "wall_median_s": round(statistics.median(kept), 4),
        "wall_min_s": round(min(kept), 4),
        "wall_all_s": [round(w, 4) for w in walls],
        "n_timed": n_timed,
        "warmup_discarded": min(warmup, len(walls)),
        "peak_rss_gb": round(peak_rss_gb, 3),
        "torch_threads": torch.get_num_threads(),
        "last_out": last_out,
    }


def _profiled_run(workload: Workload, *, row_limit: int, trace_path: Path | None,
                  notes: list[str]) -> tuple[list[dict[str, Any]], str, str | None]:
    """torch.profiler pass: op table (structured + string) + Perfetto trace.

    ``trace_path`` None (e.g. ``write=False``) skips the chrome-trace export."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, profile_memory=True,
                 record_shapes=True, with_stack=True) as prof:
        workload.run()

    ka = prof.key_averages()
    op_rows = [_op_row(e) for e in ka]
    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    try:
        table_str = ka.table(sort_by=sort_key, row_limit=row_limit)
    except Exception:  # sort key not present on this build; fall back to CPU
        table_str = ka.table(sort_by="self_cpu_time_total", row_limit=row_limit)

    written: str | None = None
    if trace_path is not None:
        try:
            raw = trace_path.with_name(trace_path.name.replace(".gz", ""))
            prof.export_chrome_trace(str(raw))
            with open(raw, "rb") as fh, gzip.open(trace_path, "wb") as gz:
                shutil.copyfileobj(fh, gz)
            raw.unlink(missing_ok=True)
            written = str(trace_path)
        except Exception as exc:  # a trace-export failure must not sink the profile
            notes.append(
                f"Perfetto chrome-trace export failed: {type(exc).__name__}: {exc}")
    return op_rows, table_str, written


def _run_pyspy(command: list[str], out_dir: Path, notes: list[str]) -> str | None:
    """py-spy sampling flamegraph of ``command`` → inlined SVG string, or None
    (degrade gracefully) with a report note explaining why."""
    if shutil.which("py-spy") is None:
        notes.append("py-spy flamegraph skipped: `py-spy` not on PATH "
                     "(install the [profiling] extra: uv sync --extra profiling).")
        return None
    svg = out_dir / "flame.svg"
    cmd = ["py-spy", "record", "--format", "flamegraph", "--subprocesses",
           "-o", str(svg), "--", *command]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except (OSError, subprocess.SubprocessError) as exc:
        notes.append(f"py-spy flamegraph skipped: {type(exc).__name__}: {exc}")
        return None
    if proc.returncode != 0 or not svg.exists():
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        notes.append("py-spy flamegraph skipped: py-spy exited "
                     f"{proc.returncode} ({tail[0]}). On Linux this is usually a "
                     "ptrace restriction (needs same-user + permissive "
                     "kernel.yama.ptrace_scope).")
        return None
    try:
        return svg.read_text()
    except OSError as exc:
        notes.append(f"py-spy flamegraph produced but unreadable: {exc}")
        return None


def _memray_target(command: list[str]) -> list[str]:
    """Translate a ``[python, -m, module, *args]`` (or ``[python, script, *args]``)
    workload command into the tail ``memray run`` expects, which drives the
    interpreter itself: ``-m module *args`` or ``script *args`` — NOT a nested
    ``python …`` (memray rejects that with "Only valid Python files or commands
    can be executed under memray")."""
    if len(command) >= 3 and command[1] == "-m":
        return ["-m", *command[2:]]
    if len(command) >= 2:  # [python, script.py, *args] → script.py *args
        return command[1:]
    return command


def _run_memray(command: list[str], out_dir: Path, notes: list[str]) -> str | None:
    """memray allocation flamegraph of ``command`` → relative path to its
    self-contained HTML (linked, not inlined), or None with a note."""
    if find_spec("memray") is None:
        notes.append("memray report skipped: `memray` not importable "
                     "(install the [profiling] extra: uv sync --extra profiling).")
        return None
    binfile = out_dir / "memray.bin"
    html = out_dir / "memray.html"
    try:
        rec = subprocess.run(
            [sys.executable, "-m", "memray", "run", "--force", "-o", str(binfile),
             *_memray_target(command)],
            capture_output=True, text=True, timeout=1800)
        if rec.returncode != 0:
            tail = (rec.stderr or "").strip().splitlines()[-1:] or [""]
            notes.append(f"memray report skipped: `memray run` exited "
                         f"{rec.returncode} ({tail[0]}).")
            return None
        fg = subprocess.run(
            [sys.executable, "-m", "memray", "flamegraph", "--force", "-o", str(html),
             str(binfile)],
            capture_output=True, text=True, timeout=600)
        if fg.returncode != 0 or not html.exists():
            tail = (fg.stderr or "").strip().splitlines()[-1:] or [""]
            notes.append(f"memray report skipped: `memray flamegraph` exited "
                         f"{fg.returncode} ({tail[0]}).")
            return None
    except (OSError, subprocess.SubprocessError) as exc:
        notes.append(f"memray report skipped: {type(exc).__name__}: {exc}")
        return None
    finally:
        binfile.unlink(missing_ok=True)
    return html.name


def _out_dir(root: Path, snap: dict[str, Any], workload_name: str) -> Path:
    """``<root>/<host>/<short-sha>/<workload>``. A dirty tree gets a ``-dirty``
    suffix on the sha segment so an uncommitted profile is never filed under, or
    mistaken for, a clean commit."""
    host = str(snap.get("host", {}).get("hostname", "unknown-host"))
    code = snap.get("code", {})
    sha = str(code.get("git") or "nogit")
    if code.get("git_dirty"):
        sha = f"{sha}-dirty"
    safe_wl = workload_name.replace("/", "_")
    return root / host / sha / safe_wl


def deep_profile(
    workload: Workload,
    *,
    out_root: str | Path = "benchmarks/results",
    n_timed: int = 5,
    warmup: int = 1,
    threads: int | None = None,
    row_limit: int = 25,
    run_pyspy: bool = True,
    run_memray: bool = True,
    write: bool = True,
) -> ProfileResult:
    """Deep-profile ``workload`` and (optionally) write the full artifact set.

    Returns a :class:`ProfileResult`. When ``write`` is True the report and all
    deep artifacts land under
    ``<out_root>/<host>/<short-sha>/<workload>/`` (see :func:`_out_dir`). Pass
    ``threads`` to pin ``torch.set_num_threads`` for comparability across
    commits; py-spy/memray are opt-out via their flags (and always skip
    gracefully if the tool or ptrace permission is missing).
    """
    import torch

    if threads is not None:
        torch.set_num_threads(int(threads))

    snap = runinfo.machine_snapshot()
    out_dir = Path(out_root) / "_pending"
    if write:
        out_dir = _out_dir(Path(out_root), snap, workload.name)
        out_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []

    # 1. Headline metrics from an UNPROFILED, thread-pinned timed loop.
    timed = _timed_runs(workload, n_timed=n_timed, warmup=warmup)
    iterations = _iterations_of(timed.pop("last_out"))
    headline = {
        "wall_median_s": timed["wall_median_s"],
        "wall_min_s": timed["wall_min_s"],
        "wall_all_s": timed["wall_all_s"],
        "peak_rss_gb": timed["peak_rss_gb"],
        "n_timed": timed["n_timed"],
        "warmup_discarded": timed["warmup_discarded"],
        "torch_threads": timed["torch_threads"],
        "note": ("wall/RSS are from a separate unprofiled timed loop (median of "
                 "post-warmup runs); the op table below is from a distinct "
                 "profiled run whose wall time is inflated by profiler overhead."),
    }

    # 2. torch.profiler op-level view + Perfetto trace (trace only when writing).
    trace_path = (out_dir / "trace.json.gz") if write else None
    op_rows, table_str, written_trace = _profiled_run(
        workload, row_limit=row_limit, trace_path=trace_path, notes=notes)
    phase_times = phase_breakdown(op_rows)

    # 3+4. Process-level samplers (graceful degrade).
    flame_svg: str | None = None
    memray_html: str | None = None
    if workload.command is None:
        notes.append("py-spy and memray skipped: workload has no reproducible "
                     "subprocess command (in-process closures cannot be profiled "
                     "by a process-level sampler).")
    else:
        if run_pyspy:
            flame_svg = _run_pyspy(workload.command, out_dir, notes)
        if run_memray:
            memray_html = _run_memray(workload.command, out_dir, notes)

    result = ProfileResult(
        workload={"name": workload.name, "spec": workload.spec},
        provenance=snap,
        headline=headline,
        op_rows=op_rows,
        op_table_str=table_str,
        iterations=iterations,
        phase_times_s=phase_times,
        out_dir=str(out_dir),
        trace_path=written_trace,
        memray_html=memray_html,
        flame_svg=flame_svg,
        notes=notes,
    )

    if write:
        write_artifacts(result, out_dir)
    return result


def write_artifacts(result: ProfileResult, out_dir: str | Path) -> Path:
    """Persist the machine-readable sidecars: ``profile.json`` (everything but the
    big table string / inlined SVG) and the op table as ``op_table.parquet`` +
    ``op_table.json``. Returns the output directory."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    payload = asdict(result)
    # Keep the big blobs out of the json sidecar (they live as their own files).
    payload.pop("op_table_str", None)
    payload.pop("flame_svg", None)
    (d / "profile.json").write_text(json.dumps(payload, indent=2, default=str))

    (d / "op_table.json").write_text(json.dumps(result.op_rows, indent=2))
    try:
        import pandas as pd

        pd.DataFrame(result.op_rows).to_parquet(d / "op_table.parquet", index=False)
    except Exception as exc:  # pandas/pyarrow are the [profiling] extra; degrade
        result.notes.append(
            f"op_table.parquet not written ({type(exc).__name__}: {exc}); "
            "install the [profiling] extra for parquet output.")
    return d
