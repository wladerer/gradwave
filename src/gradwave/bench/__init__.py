"""Benchmarking substrate for solver/convergence research (``[bench]`` extra).

Turns the SCF flight recorder (:mod:`gradwave.scf.recorder`) into an analyzable
multi-run dataset: a hardness-graded suite of systems, a method sweep, and a
persistence + flattening layer so per-iteration solver telemetry lands in a tidy
table ready for a notebook, DuckDB, a statistical model, or symbolic regression.

    from gradwave.bench import SUITE, sweep_methods, run_case, write_run
    from gradwave.bench.analyze import load_runs   # needs the [bench] extra

The recorder and harness use only the standard library (JSON sidecars); the
``analyze`` module lazily imports pandas/pyarrow from the ``[bench]`` extra.
"""

from gradwave.bench.harness import RunRecord, run_case, sweep_methods, write_run
from gradwave.bench.suite import SUITE, BenchCase

__all__ = [
    "SUITE",
    "BenchCase",
    "RunRecord",
    "run_case",
    "sweep_methods",
    "write_run",
]
