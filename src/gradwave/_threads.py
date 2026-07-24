"""CPU thread default for gradwave (intra-op parallelism).

The SCF is dominated by small, latency-bound linear-algebra and FFT kernels.
On many-core and hybrid CPUs (Intel P+E cores) letting torch/BLAS fan out
across every logical core is *slower*, not faster: fork-join BLAS stalls the
fast P-cores behind the slow E-cores and the per-op synchronization cost dwarfs
the tiny kernels. A profile on a Core Ultra 7 155H (22 logical cores) found
running all threads 2-3x slower than ~8 threads, with 22 threads slower than a
single thread. Peak throughput was around 8 threads.

gradwave therefore picks a sane default on import -- ``min(cpu_count, 8)`` --
instead of inheriting whatever the caller's environment left torch at. This is
a *default*, not a policy: it is applied only when the user has expressed no
preference, and is trivially overridable.

Override, in precedence order (first that applies wins):

1. ``GRADWAVE_NUM_THREADS=<n>`` -- gradwave's own knob, honoured even when the
   BLAS/OpenMP env vars below are also set.
2. ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` present
   in the environment -- the user has already told the parallel stack what to
   do (torch reads ``OMP_NUM_THREADS`` at import), so gradwave leaves torch
   untouched and never clobbers that explicit choice.
3. otherwise the auto-default ``min(cpu_count, 8)`` is applied.

At runtime call :func:`gradwave.set_num_threads` to retune explicitly.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("gradwave.threads")

# Above this the fork-join / hybrid-core penalty outweighs the extra cores for
# gradwave's latency-bound kernels (see module docstring).
_MAX_DEFAULT_THREADS = 8

# Presence of any of these means the user has already configured the parallel
# stack; it suppresses gradwave's auto-default so an explicit choice is never
# clobbered. torch itself picks up OMP_NUM_THREADS at import.
_USER_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _cpu_count() -> int:
    # sched_getaffinity reflects cpuset / taskset pinning (containers, HPC job
    # allocations) more faithfully than os.cpu_count(); fall back where absent.
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or _MAX_DEFAULT_THREADS


def default_num_threads() -> int:
    """Thread count gradwave picks absent any override: ``min(cpu_count, 8)``,
    at least 1. Pure -- no side effects, safe to call for reporting."""
    return max(1, min(_cpu_count(), _MAX_DEFAULT_THREADS))


def set_num_threads(n: int) -> int:
    """Set the intra-op CPU thread count torch uses for gradwave work.

    A thin, explicit wrapper over ``torch.set_num_threads`` (clamped to >= 1)
    so callers need not import torch just to retune. Overrides the import-time
    default. Returns the value applied.
    """
    import torch

    n = max(1, int(n))
    torch.set_num_threads(n)
    logger.debug("set_num_threads(%d)", n)
    return n


def _gradwave_env_override() -> int | None:
    """Parse ``GRADWAVE_NUM_THREADS``; None if unset/invalid (with a warning)."""
    val = os.environ.get("GRADWAVE_NUM_THREADS")
    if val is None:
        return None
    try:
        n = int(val)
    except ValueError:
        logger.warning("ignoring non-integer GRADWAVE_NUM_THREADS=%r", val)
        return None
    if n < 1:
        logger.warning("ignoring non-positive GRADWAVE_NUM_THREADS=%r", val)
        return None
    return n


def apply_default_threads() -> int | None:
    """Apply gradwave's CPU thread policy once, at import.

    Precedence (see module docstring): GRADWAVE_NUM_THREADS wins; else an
    existing OMP/MKL/OpenBLAS env var means "hands off"; else apply
    ``min(cpu_count, 8)``.

    Returns the thread count applied, or None when gradwave deliberately left
    torch as-is because the user had already configured the stack.
    """
    import torch

    forced = _gradwave_env_override()
    if forced is not None:
        torch.set_num_threads(forced)
        logger.debug("threads: GRADWAVE_NUM_THREADS -> %d", forced)
        return forced

    if any(os.environ.get(v) for v in _USER_THREAD_ENV):
        logger.debug("threads: user thread env present, leaving torch at %d",
                     torch.get_num_threads())
        return None

    n = default_num_threads()
    torch.set_num_threads(n)
    logger.debug("threads: auto-default -> %d", n)
    return n
