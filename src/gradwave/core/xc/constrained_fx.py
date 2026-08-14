"""Constrained flexible exchange enhancement factor — the Stage-C / SR ansatz.

The searchable object for functional learning: a richer-than-(κ,μ) exchange
enhancement `F_x(s²)` whose exact constraints are baked into the SKELETON, so the
extra flexibility can only ever be "weird PBE", never unphysical garbage. The
free interior term (here a single s⁴ coefficient c₂; later an SR-searched form) is
BOUNDED so it cannot break the constraints:

    F_x(s²) = 1 + κ − κ / (1 + (μ/κ) s² + (c₂/κ) s⁴)

- **UEG limit** F_x(0) = 1                          — exact (denominator → 1).
- **Lieb-Oxford bound** F_x < 1 + κ (= 1.804 at PBE κ) — exact for c₂ ≥ 0
  (denominator ≥ 1 ⇒ 0 ≤ κ/den ≤ κ).
- **GE2 gradient expansion** F_x → 1 + μ s² + …      — the small-s coefficient is
  μ (pin to 10/81 for the solid-exact value; PBE uses 0.2195).

c₂ = 0 reproduces PBE / LearnableX exactly; c₂ > 0 adds curvature between the UEG
and large-s regimes without touching either limit. This is the minimal richer
constrained form and the seed the symbolic-regression layer generalizes — SR
searches the FORM of the bounded interior term, evaluated (and its constants
fit) through gradwave's differentiable SCF. See docs/design/learned-functional.md.
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.xc._pbe_kernels import KAPPA, MU, pbe_h
from gradwave.core.xc.base import to_au
from gradwave.core.xc.lda_pw92 import eps_c_pw92, eps_x_lda
from gradwave.core.xc.learnable import _LearnableKappaMu
from gradwave.core.xc.pbe import PBE


def _inv_softplus(y: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(float(y))))


def constrained_enhancement(
    s2: torch.Tensor,
    kappa: float | torch.Tensor = KAPPA,
    mu: float | torch.Tensor = MU,
    c2: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """F_x(s²) = 1 + κ − κ/(1 + (μ/κ)s² + (c₂/κ)s⁴). c₂=0 → PBE. c₂≥0 keeps the
    UEG limit F(0)=1 and the Lieb-Oxford bound F<1+κ exact."""
    den = 1.0 + mu * s2 / kappa + c2 * s2 * s2 / kappa
    return 1.0 + kappa - kappa / den


class ConstrainedFx(_LearnableKappaMu, PBE):
    """PBE-form exchange with a learnable, constraint-preserving s⁴ correction
    (κ, μ, c₂). Reduces to PBE at c₂=0; UEG limit and Lieb-Oxford bound hold by
    construction for any trained (κ>0, μ>0, c₂≥0)."""

    needs_gradient = True

    def __init__(self, kappa: float = KAPPA, mu: float = MU, c2: float = 0.0) -> None:
        super().__init__(kappa=kappa, mu=mu)
        # softplus-parameterized c₂ ≥ 0 (init ~0 reproduces PBE to numerical noise)
        self.raw_c2 = torch.nn.Parameter(_inv_softplus(max(c2, 1e-8)))

    @property
    def c2(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_c2)

    def energy_density(self, rho, sigma=None, tau=None):  # noqa: D102 — mirrors PBE
        if sigma is None:
            raise ValueError("ConstrainedFx requires sigma = |grad rho|^2")
        rho_au = to_au(rho)
        sigma_au = torch.clamp(sigma * BOHR_ANG**8, min=0.0)
        grad_au = torch.sqrt(sigma_au + 1e-30)
        kf = (3.0 * math.pi**2 * rho_au) ** (1.0 / 3.0)
        s = grad_au / (2.0 * kf * rho_au)
        eps_x = eps_x_lda(rho_au) * constrained_enhancement(s * s, self.kappa, self.mu, self.c2)
        eps_c_lda = eps_c_pw92(rho_au)
        ks = torch.sqrt(4.0 * kf / math.pi)
        t = grad_au / (2.0 * ks * rho_au)
        eps_c = eps_c_lda + pbe_h(t * t, eps_c_lda)
        return rho * (eps_x + eps_c) * HARTREE_EV
