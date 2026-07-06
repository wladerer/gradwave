"""Monkhorst–Pack k-meshes with time-reversal reduction.

Full spglib symmetry reduction is a later milestone; time reversal (k ≡ −k
mod G, valid without spin-orbit) already halves most meshes. Weights sum
to 1. QE 'shift' convention: shift_i ∈ {0, 1} displaces the mesh by half a
grid step along axis i.
"""

from __future__ import annotations

import numpy as np


def monkhorst_pack(mesh, shift=(0, 0, 0), time_reversal: bool = True):
    """Return (k_frac (nk,3), weights (nk,)) in fractional coordinates, folded to (-1/2, 1/2]."""
    mesh = np.asarray(mesh, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.float64)
    if mesh.shape != (3,) or np.any(mesh < 1):
        raise ValueError(f"bad k-mesh {mesh}")
    if not np.all(np.isin(shift, (0.0, 1.0))):
        raise ValueError("shift components must be 0 or 1 (QE convention)")

    def fold(x):
        """Map fractional coordinates to (-1/2, 1/2]."""
        return -((-x + 0.5) % 1.0 - 0.5)

    # QE convention: k_i = (m + s/2)/n, m = 0..n-1 — Γ-centered at zero shift
    grids = [(np.arange(n) + 0.5 * s) / n for n, s in zip(mesh, shift, strict=True)]
    k = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, 3)
    k = fold(k)
    w = np.full(len(k), 1.0 / len(k))

    if time_reversal:
        seen: dict[tuple, int] = {}
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
