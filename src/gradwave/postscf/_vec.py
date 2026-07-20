"""Small shared vector helpers for the magnetism postscf modules.

``mae``, ``spin_exchange``, and ``moment_config`` each carried a private
``_unit`` (with differing zero-norm semantics) and two of them duplicated
the perpendicular-seed construction. The shared versions here use the
safest semantics: the norm is floored by ``clamp_min``, so a zero vector
maps to zero instead of NaN. On the live call paths the inputs are never
zero (callers gate on the norm or pass unit-ish directions), so the guard
is a no-op there.
"""

from __future__ import annotations

import torch


def unit(v: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """v / |v| along the last axis, guarded against |v| = 0 (works on a
    single 3-vector or an (na, 3) batch)."""
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def perpendicular_unit(ref: torch.Tensor) -> torch.Tensor:
    """A deterministic unit vector perpendicular to the unit vector ``ref``.

    Seeds with x̂ (ŷ when ref is nearly parallel to x̂) and projects out the
    ``ref`` component — the construction both the MAE antiparallel-rotation
    axis and the exchange transverse basis used to duplicate.
    """
    seed = torch.tensor([1.0, 0.0, 0.0], dtype=ref.dtype, device=ref.device)
    if abs(float(torch.dot(ref, seed))) > 0.9:
        seed = torch.tensor([0.0, 1.0, 0.0], dtype=ref.dtype, device=ref.device)
    return unit(seed - torch.dot(seed, ref) * ref)
