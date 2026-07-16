"""Shared PBE exchange/correlation kernels (Perdew–Burke–Ernzerhof, PRL 77,
3865 (1996)).

The spin-unpolarized (pbe.py), spin-polarized (spin.py), and learnable
(learnable.py) functionals all use the same enhancement factor and the same
gradient-correction H term. They live here so the physics-sensitive constants
and the two formulas exist once.
"""

from __future__ import annotations

import math

import torch

KAPPA = 0.804
MU = 0.2195149727645171
BETA = 0.06672455060314922
GAMMA = (1.0 - math.log(2.0)) / math.pi**2


def pbe_enhancement(s2, kappa=KAPPA, mu=MU):
    """PBE exchange enhancement F_x(s²) = 1 + κ − κ/(1 + μ s²/κ)."""
    return 1.0 + kappa - kappa / (1.0 + mu * s2 / kappa)


def pbe_h(t2, eps_c_lda, phi3=1.0):
    """PBE correlation gradient term H(rs, ζ, t) [Ha/electron].

    phi3 = φ³ is the spin-scaling factor; φ = 1 (phi3 = 1) in the unpolarized
    case, which recovers the plain PBE H.
    """
    expo = torch.exp(-eps_c_lda / (GAMMA * phi3))
    a = (BETA / GAMMA) / torch.clamp(expo - 1.0, min=1e-30)
    num = 1.0 + a * t2
    den = 1.0 + a * t2 + (a * t2) ** 2
    return GAMMA * phi3 * torch.log1p((BETA / GAMMA) * t2 * num / den)
