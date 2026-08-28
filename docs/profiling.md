# Profiling

`gradwave.profiling` is a standardized, provenance-stamped deep-profiling
harness. It runs one workload three ways, each answering a question the others
cannot, and renders the result to a two-tier artifact: a glanceable
`summary.html` and the granular deep artifacts it links to. Every profile is
keyed by the git commit that produced it, so two profiles diff cleanly across
commits.

Install the tools (not core dependencies) with the `profiling` extra:

```bash
uv sync --extra profiling
```

`py-spy` and `memray` come from that extra; `torch.profiler` needs nothing extra.
If `py-spy` or `memray` is missing, or ptrace is denied, the harness skips that
tool and records the reason in the report rather than failing.

## Method — three views of one workload

| tool | question | output |
|---|---|---|
| **torch.profiler** (primary) | which ops cost what? | op table (parquet + json) + a Perfetto timeline |
| **py-spy** | where does Python dispatch / eager glue go? | sampling flamegraph (SVG, inlined) |
| **memray** | where does memory get allocated? | allocation flamegraph + high-water timeline (HTML, linked) |

The op table drives the per-phase breakdown (ops are grouped into
H-apply / FFT / diag / mixing by name). The Perfetto trace opens at
`ui.perfetto.dev` or `chrome://tracing`.

### Headline numbers come from a separate, unprofiled run

The profiled run's wall time is inflated by profiler overhead, so it must never
be reported as the speed number. The headline **wall time** and **peak RSS**
come from a distinct, thread-pinned timed loop — the median of `n_timed` runs
with the warmup discarded. This separation is methodologically required and is
stated on the report itself.

## Two-tier report

`summary.html` is one self-contained file that opens offline (matplotlib plots
are rendered to inline SVG; no JavaScript charting library, no CDN, no external
files). Top to bottom:

1. **Provenance** — full + short commit SHA, a dirty flag (so an uncommitted
   profile is never mistaken for a clean commit), hostname, torch/python/gradwave
   versions, CPU model, thread count, timestamp, and the workload spec.
2. **Headline metrics** — median wall time and peak RSS from the unprofiled loop.
3. **Plots** — per-phase time bar, RSS-vs-iteration line, top-N ops bar.
4. **Top-N op table** — collapsible, sortable by clicking a column header
   (inline vanilla JS, no library).
5. **py-spy flamegraph** — the eager-glue view, inlined under a `<details>`.

The **linked** deep artifacts sit in the same directory: the Perfetto trace
(`trace.json.gz`), the memray HTML (`memray.html`), and the raw op table as both
`op_table.parquet` and `op_table.json` (machine-readable, for diffing).

## Storage layout

Artifacts are written under

```
benchmarks/results/<host>/<short-sha>/<workload>/
```

so the commit SHA is the join key for cross-commit diffing, not just a label. A
dirty tree gets a `-dirty` suffix on the SHA segment. (`benchmarks/results/` is
gitignored.)

## How to run

The light, laptop-safe example profiles a small suite SCF case:

```python
from gradwave.profiling import deep_profile, workload_from_bench_case, write_summary

wl = workload_from_bench_case("Si")        # or C-diamond, Al, Cu
result = deep_profile(wl, threads=4)       # pin threads for comparability
write_summary(result)                      # summary.html in result.out_dir
```

`deep_profile` writes the full artifact set and returns a `ProfileResult`.
`workload_from_bench_case` builds both the in-process callable (for the timed
loop and torch.profiler) and a reproducible subprocess command (for py-spy and
memray).

### A custom workload

```python
from gradwave.profiling import Workload, deep_profile

wl = Workload(
    name="my-run",
    spec={"system": "…", "n_atoms": 8, "ecut_ry": 40, "profiled": "one SCF"},
    run=lambda: my_scf(),                  # in-process; return value may carry a
                                           # bench-style .trace for the RSS plot
    command=[sys.executable, "-m", "my_module", "--arg"],  # or None to skip
)                                          # py-spy/memray when there is no
result = deep_profile(wl)                  # reproducible subprocess
```

If `command` is `None`, py-spy and memray are skipped with a note (a
process-level sampler cannot profile an in-process closure); torch.profiler and
the timed loop still run.

### `light` mode for op-heavy workloads

`deep_profile(wl, light=True)` runs the torch.profiler pass with
`with_stack=False` and `record_shapes=False` (the two per-op trace-size
multipliers — a full Python call stack and the input shapes recorded for every
op) while KEEPING `profile_memory=True`, so the per-op memory column survives. A
note records the trade-off (no call-stack / shape attribution). Use it for
moderately op-heavy workloads whose full trace would otherwise grow too large.

### Limitation: a full GIPAW shielding eval cannot be profiled on a 14 GB box

torch.profiler holds an in-RAM event buffer for every op executed inside the
`profile()` context. A full GIPAW magnetic-shielding evaluation issues *millions*
of aten ops (per-k × per-q × per-band × per-site CG/Sternheimer resolvent
solves), and that buffer grows past **12–13 GB** — the box OOM-kills the process.
This is true in **every** configuration measured, `light` mode included:
`with_stack`/`record_shapes` off gets further, but the raw op-event **count** is
the wall, and `profile_memory` alone (recording every tensor allocation) still
overflows. The un-instrumented eval itself is under 1 GB — the memory is
entirely the profiler's trace, not the workload.

Two ways to profile shielding-scale work anyway:

1. **Profile one kernel invocation, not the whole eval.** A single resolvent
   solve (e.g. `_SMetricResolventCG.apply` — one matrix-free CG Sternheimer
   solve) is a few thousand ops, fits trivially, and is the *right* granularity
   for "which ops dominate a response solve, and where its bytes live". Build the
   context once (unprofiled) and wrap the single call as the `Workload.run`. See
   `experiments/autoapw/shielding_profile_workload.py`.
2. **Use the bench RSS telemetry** (`gradwave.bench`, per-iteration RSS + phase
   counts) for the whole-eval memory picture, which does not retain a per-op
   buffer.

Run any heavy real workload on `asus` (not the laptop), one at a time, after
checking the box has room (`ssh asus 'free -g; uptime'`), and use the `chunk_k`
streaming path.

## How to diff two commits

Profile the same workload on two commits, then compare their output directories
(each identified by its SHA):

```python
from gradwave.profiling import compare, write_compare

rows = compare(dir_a, dir_b)               # signed per-op delta (b − a), sorted
write_compare(dir_a, dir_b, "compare.html")  # color-coded side-by-side table
```

A positive delta means the op is slower or heavier on B — a regression, shown in
red; improvements are green. `compare` reads the persisted `op_table.json` /
`profile.json`, so it works entirely offline from two stored profiles.
