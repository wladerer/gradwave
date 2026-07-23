"""Shared two-stage (low-precision draft → full-precision polish) scaffolding
for the mixed-precision batched eigensolvers.

davidson_batched_ms and chebyshev_filtered_batched_ms are the same skeleton:
skip to a single full-precision solve when mixed precision does not apply,
otherwise draft in complex64 down to a loose `crossover`, cast the drafted
eigenvectors back up, and polish to `tol` warm-started from them. Only the
per-solver stages differ (which extra args to thread, whether the Chebyshev
warm start is renormalized, whether Lanczos bounds are precomputed), so those
are supplied as closures built lazily in the non-skip path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # only for annotations — importing at runtime would cycle
    from gradwave.solvers.davidson import BatchedDavidsonResult

LOW = torch.complex64  # the mixed-precision draft dtype


def mixed_precision_solve(
    x0: torch.Tensor,
    tol: float,
    crossover: float,
    mixed_precision: bool,
    *,
    full: Callable[[], BatchedDavidsonResult],
    make_stages: Callable[
        [], tuple[Callable[[torch.Tensor], BatchedDavidsonResult],
                  Callable[[torch.Tensor], BatchedDavidsonResult]]
    ],
) -> BatchedDavidsonResult:
    """Run the draft→polish schedule, or `full()` when MS does not apply.

    Skipped (a single full-precision solve via `full()`) when mixed_precision is
    off, x0 is already low precision, or crossover ≥ tol. Otherwise `make_stages`
    is invoked once (so any per-solve setup, e.g. Lanczos bounds, is paid only on
    this path) to build (draft, polish): `draft` receives x0 cast to LOW and must
    solve to `crossover`; its eigenvectors are cast back to x0's dtype and handed
    to `polish`, which solves to `tol` and whose result is returned.
    """
    if (not mixed_precision) or x0.dtype == LOW or crossover >= tol:
        return full()
    hi_dtype = x0.dtype
    draft, polish = make_stages()
    d = draft(x0.to(LOW))
    return polish(d.eigenvectors.to(hi_dtype))
