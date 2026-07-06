"""LDA: Slater exchange + Perdew–Wang 1992 correlation (spin-unpolarized).

References: Slater exchange ε_x = −(3/4)(3/π)^{1/3} ρ^{1/3};
PW92: Perdew & Wang, PRB 45, 13244 (1992), Eq. (10) with the standard
unpolarized parameters. This matches QE's `input_dft = 'sla+pw'`.

All internal math in Hartree atomic units (base.to_au / eps_to_ev_density).
"""

from __future__ import annotations

import math

import torch

from gradwave.core.xc.base import XCFunctional, eps_to_ev_density, to_au

# PW92 unpolarized (A doubled in the log arg per the paper's 2A convention)
_A = 0.031091
_ALPHA1 = 0.21370
_BETA1 = 7.5957
_BETA2 = 3.5876
_BETA3 = 1.6382
_BETA4 = 0.49294

_CX = -0.75 * (3.0 / math.pi) ** (1.0 / 3.0)  # Slater exchange coefficient


def eps_x_lda(rho_au: torch.Tensor) -> torch.Tensor:
    """Slater exchange energy per electron [Ha]."""
    return _CX * rho_au ** (1.0 / 3.0)


def eps_c_pw92(rho_au: torch.Tensor) -> torch.Tensor:
    """PW92 correlation energy per electron [Ha], unpolarized."""
    rs = (3.0 / (4.0 * math.pi * rho_au)) ** (1.0 / 3.0)
    srs = torch.sqrt(rs)
    q0 = -2.0 * _A * (1.0 + _ALPHA1 * rs)
    q1 = 2.0 * _A * (_BETA1 * srs + _BETA2 * rs + _BETA3 * rs * srs + _BETA4 * rs * rs)
    return q0 * torch.log1p(1.0 / q1)


class LDA_PW92(XCFunctional):
    needs_gradient = False

    def energy_density(self, rho: torch.Tensor, sigma=None) -> torch.Tensor:
        rho_au = to_au(rho)
        eps = eps_x_lda(rho_au) + eps_c_pw92(rho_au)
        return eps_to_ev_density(rho, eps)
