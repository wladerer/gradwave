"""Type-II Anderson mixing for the FLAPW self-consistency loops."""

from __future__ import annotations

import torch
from torch import Tensor


def anderson_next(v_hist: list[Tensor], r_hist: list[Tensor], beta: float = 0.5,
                  m: int = 5) -> Tensor:
    """Next input from the history of inputs ``v_hist`` and residuals ``r_hist = F(v) - v``."""
    n = len(r_hist)
    if n == 1:
        return v_hist[-1] + beta * r_hist[-1]
    mm = min(m, n - 1)
    d_r = torch.stack([r_hist[-1] - r_hist[-1 - i] for i in range(1, mm + 1)], dim=1)
    d_v = torch.stack([v_hist[-1] - v_hist[-1 - i] for i in range(1, mm + 1)], dim=1)
    # Regularized normal equations (Tikhonov) rather than lstsq: robust to a rank-deficient
    # residual history, which makes LAPACK's gelsy driver assert. Falls back to a linear step
    # when the residual differences collapse.
    a = d_r.T @ d_r
    tr = torch.trace(a)
    if tr < 1e-30:
        return v_hist[-1] + beta * r_hist[-1]
    reg = 1e-8 * tr / a.shape[0]
    eye = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    gamma = torch.linalg.solve(a + reg * eye, d_r.T @ r_hist[-1])
    return (v_hist[-1] - d_v @ gamma) + beta * (r_hist[-1] - d_r @ gamma)
