"""PBE exchange-correlation (Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996)),
spin-unpolarized.

Exchange:    e_x = e_x^LDA · F_x(s),  F_x = 1 + κ − κ/(1 + μs²/κ)
Correlation: e_c = ρ(ε_c^PW92 + H(rs, t)),
             H = γ ln[1 + (β/γ) t² (1 + A t²)/(1 + A t² + A² t⁴)]
             A = (β/γ)/(exp(−ε_c^PW92/γ) − 1)

s = |∇ρ|/(2 k_F ρ),  k_F = (3π²ρ)^{1/3};  t = |∇ρ|/(2 k_s ρ),  k_s = √(4 k_F/π).
All internal math in Hartree atomic units; σ = |∇ρ|² is supplied by the
caller in Å units and converted here. In the uniform limit (σ → 0) PBE
reduces exactly to LDA_PW92 — that is a unit test.
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.xc._pbe_kernels import KAPPA, MU, pbe_enhancement, pbe_h
from gradwave.core.xc.base import XCFunctional, to_au
from gradwave.core.xc.lda_pw92 import eps_c_pw92, eps_x_lda


class PBE(XCFunctional):
    needs_gradient = True

    # Exchange enhancement parameters (κ, μ). Fixed at the PBE values here;
    # LearnableX subclasses this and overrides them with trainable tensors, so
    # the energy_density body below is shared verbatim between the two.
    kappa = KAPPA
    mu = MU

    def energy_density(
        self, rho: torch.Tensor, sigma: torch.Tensor | None = None, tau=None
    ) -> torch.Tensor:
        if sigma is None:
            raise ValueError("PBE requires sigma = |grad rho|^2")
        rho_au = to_au(rho)
        # σ [e²/Å⁸] → a.u.: |∇ρ|² scales by (Bohr³/Å³ · Å/Bohr)² per length⁻⁴ → BOHR⁸... :
        # ρ: ×BOHR³, ∇: ×BOHR per derivative ⇒ σ_au = σ_ang · BOHR_ANG⁸
        sigma_au = torch.clamp(sigma * BOHR_ANG**8, min=0.0)
        grad_au = torch.sqrt(sigma_au + 1e-30)

        kf = (3.0 * math.pi**2 * rho_au) ** (1.0 / 3.0)
        s = grad_au / (2.0 * kf * rho_au)
        eps_x = eps_x_lda(rho_au) * pbe_enhancement(s * s, self.kappa, self.mu)

        eps_c_lda = eps_c_pw92(rho_au)
        ks = torch.sqrt(4.0 * kf / math.pi)
        t = grad_au / (2.0 * ks * rho_au)
        eps_c = eps_c_lda + pbe_h(t * t, eps_c_lda)

        return rho * (eps_x + eps_c) * HARTREE_EV
