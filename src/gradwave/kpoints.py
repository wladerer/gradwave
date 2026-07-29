"""Monkhorst–Pack k-meshes with time-reversal reduction.

Full spglib space-group reduction lives in symmetry.reduce_mesh; this module
does the lighter time-reversal reduction (k ≡ −k mod G, valid without
spin-orbit) which already halves most meshes. Weights sum to 1. QE 'shift'
convention: shift_i ∈ {0, 1} displaces the mesh by half a grid step along
axis i.
"""

from __future__ import annotations

import numpy as np


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
