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
    gamma = torch.linalg.lstsq(d_r, r_hist[-1]).solution
    return (v_hist[-1] - d_v @ gamma) + beta * (r_hist[-1] - d_r @ gamma)
