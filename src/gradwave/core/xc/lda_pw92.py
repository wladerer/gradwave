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

# PW92 correlation G-function parameter set (A, α1, β1, β2, β3, β4). _EC0 is the
# spin-unpolarized set; the polarized and stiffness sets live in xc/spin.py and
# reuse _g_pw92 from here. A enters the log arg doubled (the paper's 2A form).
_EC0 = (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

_CX = -0.75 * (3.0 / math.pi) ** (1.0 / 3.0)  # Slater exchange coefficient


def eps_x_lda(rho_au: torch.Tensor) -> torch.Tensor:
    """Slater exchange energy per electron [Ha]."""
    return _CX * rho_au ** (1.0 / 3.0)


def _g_pw92(rs: torch.Tensor, p) -> torch.Tensor:
    """PW92 G(rs) for one parameter set p = (A, α1, β1, β2, β3, β4) [Ha]."""
    a, a1, b1, b2, b3, b4 = p
    srs = torch.sqrt(rs)
    q0 = -2.0 * a * (1.0 + a1 * rs)
    q1 = 2.0 * a * (b1 * srs + b2 * rs + b3 * rs * srs + b4 * rs * rs)
    return q0 * torch.log1p(1.0 / q1)


def eps_c_pw92(rho_au: torch.Tensor) -> torch.Tensor:
    """PW92 correlation energy per electron [Ha], unpolarized."""
    rs = (3.0 / (4.0 * math.pi * rho_au)) ** (1.0 / 3.0)
    return _g_pw92(rs, _EC0)


class LDA_PW92(XCFunctional):
    needs_gradient = False

    def energy_density(self, rho: torch.Tensor, sigma=None, tau=None) -> torch.Tensor:
        rho_au = to_au(rho)
        eps = eps_x_lda(rho_au) + eps_c_pw92(rho_au)
        return eps_to_ev_density(rho, eps)
