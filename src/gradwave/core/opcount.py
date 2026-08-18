"""Gated primitive-operation counters for SCF telemetry.

A process-global tally of the expensive primitives — FFT launches (``fft``),
subspace diagonalizations (``eigh``), and Hamiltonian applies (``hpsi``, counted
as band·k) — incremented at their chokepoints in ``core``/``solvers``. Off by
default, so non-SCF code (gradchecks, postscf) pays nothing; the SCF loop calls
:func:`enable` for the run when the flight recorder is active and snapshots the
per-outer-iteration delta for :class:`gradwave.scf.recorder.SCFRecorder`.

Pure side effect: never touches numerics. Mirrors the older
``core.batch._HAPPLY_TALLY`` pattern, generalized to several keys. Counts are a
best-effort launch/work tally (a FFT over a batched (nk, nbc) block is one
``fft`` launch); they are diagnostics, not a contract.
"""

from __future__ import annotations

_STATE = {"on": False}
_COUNTS = {"fft": 0, "eigh": 0, "hpsi": 0}


def enable() -> None:
    """Turn counting on (idempotent)."""
    _STATE["on"] = True


def disable() -> None:
    """Turn counting off (the default)."""
    _STATE["on"] = False


def bump(key: str, n: int = 1) -> None:
    """Increment ``key`` by ``n`` when enabled; a no-op otherwise."""
    if _STATE["on"]:
        _COUNTS[key] += n


def snapshot() -> dict[str, int]:
    """A copy of the current cumulative counts."""
    return dict(_COUNTS)


def since(prev: dict[str, int]) -> dict[str, int]:
    """Delta of the current counts vs a prior :func:`snapshot`."""
    return {k: _COUNTS[k] - prev.get(k, 0) for k in _COUNTS}
