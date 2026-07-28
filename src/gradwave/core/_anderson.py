"""Anderson-accelerated fixed-point mixer for the linear-response solvers.

The Sternheimer/adjoint fixed points u = G(u) in dielectric.py, hubbard_u.py,
uspp_implicit.py, and uspp_position.py all have gain > 1 modes where plain
damping diverges (the NiO magnetization channel, the stiff becsum↔ddd feedback),
so they all ran the same Anderson recursion inline. It lives here once.

Given the current iterate u and its residual r = G(u) − u (raw or
preconditioned, the caller's choice), step() returns the next iterate

    u⁺ = u + β r − (ΔU + β ΔR) γ,   γ = argmin_γ ‖r − ΔR γ‖,

where ΔU, ΔR are the last `history` secant pairs (Δuₖ, Δrₖ).
"""

from __future__ import annotations

import torch


class AndersonMixer:
    """Type-II Anderson mixing over a rolling window of `history` secant pairs."""

    def __init__(self, history: int, beta: float):
        self.history = history
        self.beta = beta
        self.prev_u = None
        self.prev_r = None
        self.hist_du: list[torch.Tensor] = []
        self.hist_dr: list[torch.Tensor] = []

    def step(self, u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Advance the fixed point given the iterate u and its residual r."""
        if self.prev_r is not None:
            self.hist_du.append(u - self.prev_u)
            self.hist_dr.append(r - self.prev_r)
            if len(self.hist_dr) > self.history:
                self.hist_du.pop(0)
                self.hist_dr.pop(0)
        self.prev_u, self.prev_r = u, r
        if self.hist_dr:
            dr_m = torch.stack(self.hist_dr, dim=1)
            du_m = torch.stack(self.hist_du, dim=1)
            # Rank-safe least squares for gamma = argmin ‖r − ΔR γ‖. A plain
            # torch.linalg.lstsq uses CUDA's gels driver, which assumes ΔR is
            # full rank and returns garbage on the (common) rank-deficient
            # secant matrix. Solve the Tikhonov-damped normal equations
            # (ΔRᴴΔR + λI) γ = ΔRᴴr instead; λ is a tiny fraction of the ΔRᴴΔR
            # diagonal, so well-conditioned windows match the old lstsq result.
            drh = dr_m.conj().transpose(-2, -1)
            ata = drh @ dr_m
            lam = 1e-12 * ata.diagonal().abs().max().clamp_min(1e-300)
            ata = ata + lam * torch.eye(ata.shape[0], dtype=ata.dtype,
                                        device=ata.device)
            gamma = torch.linalg.solve(ata, drh @ r[:, None])[:, 0]
            return u + self.beta * r - (du_m + self.beta * dr_m) @ gamma
        return u + self.beta * r
