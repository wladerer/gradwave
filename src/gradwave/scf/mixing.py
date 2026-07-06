"""Density mixing: linear, Kerker-preconditioned, Pulay/Anderson (Layer B).

Mixing operates on ρ(G) over the density sphere (complex vectors). The G=0
component is pinned (both ρ_in and ρ_out integrate to N_e, so the residual
there is zero by construction — asserted).

Kerker: R̃(G) = R(G)·G²/(G² + q0²) suppresses long-wavelength charge
sloshing in metals; q0 default 1.1 Å⁻¹ (~0.58 bohr⁻¹). Off for insulators
(it slows their convergence).

Pulay: minimize ‖Σ c_i R_i‖ with Σ c_i = 1 in the Kerker-weighted inner
product ⟨R, R'⟩ = Σ_G Re[R*R']/(G² + q0²) (QE-style metric emphasizing
long-range components), via a bordered linear system; restart on
ill-conditioning.
"""

from __future__ import annotations

import torch


class PulayMixer:
    def __init__(
        self,
        g2: torch.Tensor,  # (nG,) |G|² over the density sphere, G=0 first entry allowed
        alpha: float = 0.7,
        history: int = 8,
        kerker: bool = False,
        q0: float = 1.1,
    ):
        self.g2 = g2
        self.alpha = alpha
        self.history = history
        self.kerker = kerker
        self.q0 = q0
        self._rho_in: list[torch.Tensor] = []
        self._res: list[torch.Tensor] = []

    def _precondition(self, r: torch.Tensor) -> torch.Tensor:
        if not self.kerker:
            return self.alpha * r
        fac = self.g2 / (self.g2 + self.q0**2)
        return self.alpha * fac * r

    def _metric(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        w = 1.0 / (self.g2 + self.q0**2)
        return (a.conj() * b * w).sum().real

    def reset(self):
        self._rho_in.clear()
        self._res.clear()

    def step(self, rho_in: torch.Tensor, rho_out: torch.Tensor) -> torch.Tensor:
        """Next ρ_in(G) from the current (ρ_in, ρ_out) pair."""
        res = rho_out - rho_in
        assert res[0].abs() < 1e-8, "G=0 residual nonzero — density not normalized"

        self._rho_in.append(rho_in)
        self._res.append(res)
        if len(self._res) > self.history:
            self._rho_in.pop(0)
            self._res.pop(0)

        m = len(self._res)
        if m == 1:
            return rho_in + self._precondition(res)

        # bordered system: [B 1; 1ᵀ 0][c; λ] = [0; 1], B_ij = <R_i, R_j>
        b = torch.zeros((m + 1, m + 1), dtype=torch.float64, device=rho_in.device)
        for i in range(m):
            for j in range(i, m):
                bij = self._metric(self._res[i], self._res[j])
                b[i, j] = b[j, i] = bij
        b[:m, m] = 1.0
        b[m, :m] = 1.0
        rhs = torch.zeros(m + 1, dtype=torch.float64, device=rho_in.device)
        rhs[m] = 1.0

        # ill-conditioning → drop oldest history and retry, else linear step
        try:
            cond = torch.linalg.cond(b[:m, :m])
            if not torch.isfinite(cond) or cond > 1e12:
                raise RuntimeError
            coeff = torch.linalg.solve(b, rhs)[:m]
        except RuntimeError:
            self._rho_in = self._rho_in[-1:]
            self._res = self._res[-1:]
            return rho_in + self._precondition(res)

        rho_opt = sum(c * r for c, r in zip(coeff, self._rho_in, strict=True))
        res_opt = sum(c * r for c, r in zip(coeff, self._res, strict=True))
        return rho_opt + self._precondition(res_opt)
