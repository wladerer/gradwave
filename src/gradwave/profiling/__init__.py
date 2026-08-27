"""Standardized deep-profiling harness + self-contained HTML report.

A two-tier, provenance-stamped profiling artifact for gradwave workloads:

* the **granular tier** (:mod:`gradwave.profiling.harness`) runs a workload under
  torch.profiler (op table + Perfetto trace), py-spy (eager-glue flamegraph), and
  memray (allocation flamegraph), with headline wall/RSS from a separate
  unprofiled timed loop; and
* the **summary tier** (:mod:`gradwave.profiling.report`) renders it all to one
  offline ``summary.html`` (matplotlib inline-SVG plots, a sortable op table, the
  inlined flamegraph) plus a cross-commit ``compare.html``.

py-spy and memray install via the ``[profiling]`` optional-dependency group
(``uv sync --extra profiling``) and degrade gracefully if absent. torch.profiler
needs no extra dependency.

    from gradwave.profiling import deep_profile, workload_from_bench_case
    from gradwave.profiling import write_summary

    wl = workload_from_bench_case("Si")
    result = deep_profile(wl, threads=4)
    write_summary(result)          # summary.html in result.out_dir
"""

from gradwave.profiling.harness import (
    ProfileResult,
    Workload,
    deep_profile,
    phase_breakdown,
    workload_from_bench_case,
    write_artifacts,
)
from gradwave.profiling.report import (
    build_compare_html,
    build_summary_html,
    compare,
    write_compare,
    write_summary,
)

__all__ = [
    "ProfileResult",
    "Workload",
    "build_compare_html",
    "build_summary_html",
    "compare",
    "deep_profile",
    "phase_breakdown",
    "workload_from_bench_case",
    "write_artifacts",
    "write_compare",
    "write_summary",
]
