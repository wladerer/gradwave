"""Subprocess entry point: run one profiling workload in a fresh process.

py-spy (sampling flamegraph) and memray (allocation flamegraph) both profile a
*process*, so they need a self-contained command that reproduces the workload
from the command line rather than an in-process Python callable. This module is
that command: ``python -m gradwave.profiling._runner --case Si`` builds a named
:mod:`gradwave.bench` suite case and runs one SCF, doing exactly the work the
in-process torch.profiler run does, so the three tools all measure the same
workload.

It intentionally imports only the standard library plus gradwave at call time so
that the process start-up cost the samplers see matches a real run.
"""

from __future__ import annotations

import argparse
import sys


def _run_case(case_name: str, method_alpha: float) -> None:
    from gradwave.bench import run_case
    from gradwave.bench.suite import build_suite

    cases = {c.name: c for c in build_suite()}
    case = cases.get(case_name)
    if case is None:
        raise SystemExit(
            f"unknown bench case {case_name!r}; available: {sorted(cases)}"
        )
    method = {"mixing_scheme": "pulay", "mixing_alpha": method_alpha, "kerker": True}
    run_case(case, method)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gradwave.profiling._runner",
        description="Run one profiling workload in a fresh process.",
    )
    parser.add_argument(
        "--case",
        required=True,
        help="name of a gradwave.bench.build_suite() case (e.g. Si, C-diamond, Al, Cu)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="mixing_alpha for the SCF (default 0.5)"
    )
    args = parser.parse_args(argv)
    _run_case(args.case, args.alpha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
