"""Per-iteration flight recorder for the FLAPW SCF — the all-electron sibling of
``scf.recorder.SCFRecorder``.

The PW recorder is structurally G-grid/torch-bound, so this is a duck-typed sibling, not a
subclass: it honors the same consumer contracts (``.iters`` truthy, ``summarize()`` for the
summary block, ``to_trace_dict()`` for the runinfo sidecar) with its own trace schema. It records
only floats the SCF loop has already computed — never tensors, never anything on the hot path.

Motivation (learned the hard way): every silent FLAPW failure of the TiO2 campaign — the
cold-coupled fullpot divergence, the aspherical loop that never converged, the basin swings —
was invisible because runs kept only endpoint numbers. The per-iteration story IS the diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TRACE_VERSION = 1


@dataclass
class FLAPWRecorder:
    """Append-only per-iteration telemetry for ``crystal_scf_multi``."""

    iters: list[dict] = field(default_factory=list)

    def record(self, **kw) -> None:
        """One iteration's scalars: it, span, d_span, r_v, r_nsph, beta_nsph, symmetry_dev,
        e_fermi, mt_phase, exact_solve, t_s (all floats/bools/ints; None where not applicable)."""
        self.iters.append(dict(kw))

    def summarize(self) -> dict:
        """The convergence story in one dict (consumed by the summary block)."""
        if not self.iters:
            return {}
        last = self.iters[-1]
        return {
            "n_iter": len(self.iters),
            "span_eV": last.get("span"),
            "d_span_eV": last.get("d_span"),
            "r_v": last.get("r_v"),
            "r_nsph": last.get("r_nsph"),
            "symmetry_dev": last.get("symmetry_dev"),
            "mt_phase_iters": sum(1 for it in self.iters if it.get("mt_phase")),
            "exact_solves": sum(1 for it in self.iters if it.get("exact_solve")),
            "wall_s": sum(it.get("t_s") or 0.0 for it in self.iters),
        }

    def to_trace_dict(self) -> dict:
        """Column-oriented trace for the runinfo sidecar / analysis frames."""
        keys = sorted({k for it in self.iters for k in it})
        return {"trace_version": TRACE_VERSION,
                "columns": {k: [it.get(k) for it in self.iters] for k in keys}}
