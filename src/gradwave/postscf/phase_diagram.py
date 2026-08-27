"""Temperature-composition phase diagram by common tangent (phase-diagram phase 4).

At a fixed temperature each phase has a free energy per atom G(x) against
composition. Two compositions coexist where a single line is tangent to the
free energy at both, which equalizes the chemical potentials of both species.
That common tangent is exactly an edge of the lower convex hull of G(x), so the
construction reuses the hull from phase 2. A hull edge that spans more than one
grid interval is a tie-line, and the compositions it connects are the phase
boundaries of a two-phase region.

Sweeping temperature traces the boundaries, and for a free energy with a
concave region (a positive-interaction solution) the two boundaries are the
binodal of a miscibility gap that closes at the critical temperature.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np

from gradwave.postscf.convex_hull import lower_convex_hull


def two_phase_regions(
    x: np.ndarray, g: np.ndarray, gap_factor: float = 1.5
) -> list[tuple[float, float]]:
    """Two-phase composition intervals at one temperature. A lower-hull edge that
    spans more than gap_factor times the local grid spacing is a tie-line, so its
    endpoints bound a two-phase region. Returns a list of (x_left, x_right)."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(g, dtype=float)
    order = np.argsort(x)
    x, g = x[order], g[order]
    verts = lower_convex_hull(x, g)
    verts = sorted(verts, key=lambda i: x[i])
    dx = np.min(np.diff(x))
    regions = []
    for a, b in itertools.pairwise(verts):
        if x[b] - x[a] > gap_factor * dx:
            regions.append((float(x[a]), float(x[b])))
    return regions


def binodal(
    temperatures: np.ndarray,
    x_grid: np.ndarray,
    g_of_x_t: Callable[[np.ndarray, float], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace the widest two-phase region across temperature. g_of_x_t(x, T)
    returns the free energy per atom on the composition grid at temperature T.
    Returns temperatures, and the left and right boundary compositions, with NaN
    where no two-phase region exists (above the critical temperature)."""
    temperatures = np.asarray(temperatures, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    left = np.full(temperatures.shape, np.nan)
    right = np.full(temperatures.shape, np.nan)
    for i, temp in enumerate(temperatures):
        regions = two_phase_regions(x_grid, np.asarray(g_of_x_t(x_grid, temp)))
        if regions:
            widest = max(regions, key=lambda r: r[1] - r[0])
            left[i], right[i] = widest
    return temperatures, left, right


def critical_temperature(temperatures: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Highest temperature that still shows a two-phase region, the estimated
    critical temperature where the miscibility gap closes."""
    has_gap = ~np.isnan(left)
    if not np.any(has_gap):
        return float("nan")
    return float(np.asarray(temperatures)[has_gap].max())
