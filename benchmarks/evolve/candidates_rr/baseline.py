"""Baseline candidate: the shipped ``_rr`` verbatim. The population is scored as
speedup relative to this, and it doubles as the plumbing no-op (it must come back
admissible with speedup ~1.0)."""

from gradwave.solvers.davidson import _rr as rr

__all__ = ["rr"]
