"""Monkhorst–Pack k-meshes with time-reversal reduction.

Full spglib space-group reduction lives in symmetry.reduce_mesh; this module
does the lighter time-reversal reduction (k ≡ −k mod G, valid without
spin-orbit) which already halves most meshes. Weights sum to 1. QE 'shift'
convention: shift_i ∈ {0, 1} displaces the mesh by half a grid step along
axis i.
"""

from __future__ import annotations

import warnings

import numpy as np


def _axis_vacuum_gaps(
    cell: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Largest vacuum gap [Å] along each of the three cell axes.

    For each axis, project the atoms onto the fractional coordinate of that axis,
    sort, and measure the widest gap between consecutive occupied planes (with the
    periodic wrap-around gap included). A slab has one axis whose widest gap is the
    vacuum layer; a bulk cell has small gaps on every axis. Working in fractional
    coordinates and taking the *largest* gap makes it robust to a protruding
    adsorbate — the adsorbate just shrinks the vacuum gap on the open axis, it does
    not move the detection to a periodic axis. The gap is returned in Å (fraction ×
    the axis length ``|a_i|``), the physically meaningful vacuum thickness for a
    slab whose open axis is orthogonal to the periodic pair.
    """
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    lengths = np.linalg.norm(cell, axis=1)
    if len(pos) == 0:
        return np.zeros(3)
    frac = pos @ np.linalg.inv(cell)
    gaps = np.zeros(3)
    for i in range(3):
        f = np.sort(np.mod(frac[:, i], 1.0))
        if len(f) == 1:
            widest = 1.0  # a single plane leaves the whole axis empty otherwise
        else:
            internal = np.diff(f)
            wrap = (f[0] + 1.0) - f[-1]  # gap across the periodic boundary
            widest = float(max(internal.max(), wrap))
        gaps[i] = widest * lengths[i]
    return gaps


def slab_kmesh(
    cell: np.ndarray,
    positions: np.ndarray,
    kspacing: float,
    *,
    vacuum_axis: int | None = None,
    min_vacuum_gap: float = 5.0,
) -> tuple[int, int, int]:
    """Slab-aware anisotropic Monkhorst–Pack mesh from a target ``kspacing`` [Å⁻¹].

    Each axis gets ``max(1, ceil(|b_i| / kspacing))`` k-points from the reciprocal
    lattice length ``|b_i| = |2π (cell⁻ᵀ)_i|`` (the VASP KSPACING convention), then
    the detected vacuum axis is pinned to a single Γ point: bands do not disperse
    across a vacuum-separated image, so sampling that direction is pure waste (a
    (6,6,1) slab refined uniformly to (9,9,2) doubles the k-count for no accuracy).

    The vacuum axis is auto-detected as the axis with the widest inter-atomic gap
    (:func:`_axis_vacuum_gaps`); it is only pinned when that gap is at least
    ``min_vacuum_gap`` Å, so a bulk cell (no real vacuum) is returned untouched —
    every axis solved from ``kspacing``. Pass ``vacuum_axis`` to force a specific
    axis; a forced axis whose gap is below ``min_vacuum_gap`` still pins but warns.

    Returns the ``(n1, n2, n3)`` mesh, intended to be paired with a Γ-centered
    shift ``(0, 0, 0)`` (``symmetry.reduce_mesh`` rejects a shifted mesh).
    """
    if kspacing <= 0.0:
        raise ValueError(f"kspacing must be positive, got {kspacing}")
    cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    recip = 2.0 * np.pi * np.linalg.inv(cell).T  # rows b_i
    blen = np.linalg.norm(recip, axis=1)
    mesh = [max(1, int(np.ceil(b / kspacing))) for b in blen]

    gaps = _axis_vacuum_gaps(cell, positions)
    if vacuum_axis is None:
        axis = int(np.argmax(gaps))
        if gaps[axis] >= min_vacuum_gap:
            vacuum_axis = axis
    else:
        if not 0 <= vacuum_axis <= 2:
            raise ValueError(f"vacuum_axis must be 0, 1 or 2, got {vacuum_axis}")
        if gaps[vacuum_axis] < min_vacuum_gap:
            warnings.warn(
                f"slab_kmesh: forced vacuum_axis={vacuum_axis} has only "
                f"{gaps[vacuum_axis]:.2f} Å of vacuum (< min_vacuum_gap="
                f"{min_vacuum_gap} Å); pinning it to 1 k-point anyway, but the "
                "structure may not be a slab along that axis.",
                stacklevel=2)

    if vacuum_axis is not None:
        mesh[vacuum_axis] = 1
    return (mesh[0], mesh[1], mesh[2])


def monkhorst_pack(
    mesh: tuple[int, int, int] | list[int],
    shift: tuple[int, int, int] = (0, 0, 0),
    time_reversal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (k_frac (nk,3), weights (nk,)) in fractional coordinates, folded to (-1/2, 1/2]."""
    mesh_arr = np.asarray(mesh, dtype=np.int64)
    shift_arr = np.asarray(shift, dtype=np.float64)
    if mesh_arr.shape != (3,) or np.any(mesh_arr < 1):
        raise ValueError(f"bad k-mesh {mesh_arr}")
    if not np.all(np.isin(shift_arr, (0.0, 1.0))):
        raise ValueError("shift components must be 0 or 1 (QE convention)")

    def fold(x: np.ndarray) -> np.ndarray:
        """Map fractional coordinates to (-1/2, 1/2]."""
        return -((-x + 0.5) % 1.0 - 0.5)

    # QE convention: k_i = (m + s/2)/n, m = 0..n-1 — Γ-centered at zero shift
    grids = [(np.arange(n) + 0.5 * s) / n for n, s in zip(mesh_arr, shift_arr, strict=True)]
    k = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)
    k = fold(k)
    w = np.full(len(k), 1.0 / len(k))

    if time_reversal:
        seen: dict[tuple[float, ...], int] = {}
        keep_k, keep_w = [], []
        for ki, wi in zip(k, w, strict=True):
            key = tuple(np.round(ki, 9))
            neg = tuple(np.round(fold(-ki), 9))
            if neg in seen:
                keep_w[seen[neg]] += wi
            else:
                seen[key] = len(keep_k)
                keep_k.append(ki)
                keep_w.append(wi)
        k = np.array(keep_k)
        w = np.array(keep_w)

    assert abs(w.sum() - 1.0) < 1e-12
    return k, w
