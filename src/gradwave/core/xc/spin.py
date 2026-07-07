"""Spin-polarized XC: LSDA (PW92 spin interpolation) and spin-PBE (Layer A).

Exchange uses the exact spin-scaling relation
    E_x[ρ↑, ρ↓] = ½ (E_x[2ρ↑] + E_x[2ρ↓])          (σ_σσ → 4σ_σσ for GGAs)

PW92 correlation interpolates between unpolarized/polarized fits:
    ε_c(rs, ζ) = ε_c(rs,0) + α_c(rs)·f(ζ)/f″(0)·(1−ζ⁴) + [ε_c(rs,1)−ε_c(rs,0)]·f(ζ)·ζ⁴
    f(ζ) = [(1+ζ)^{4/3} + (1−ζ)^{4/3} − 2] / (2^{4/3} − 2)

Spin-PBE correlation: H(rs, ζ, t) with φ(ζ) = [(1+ζ)^{2/3}+(1−ζ)^{2/3}]/2,
t = |∇ρ|/(2 φ k_s ρ), and the γφ³ prefactors of the original paper.

v_xc↑/v_xc↓ (including all GGA divergence terms) come from autograd on
these expressions — no hand-coded potentials anywhere. ζ = 0 reduces
EXACTLY to the unpolarized LDA_PW92/PBE (unit-tested).
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.xc.base import RHO_FLOOR_AU
from gradwave.core.xc.lda_pw92 import eps_x_lda

_F_DD0 = 1.709920934161365  # f″(0)
_FZ_DEN = 2.0 ** (4.0 / 3.0) - 2.0

# PW92 G-function parameter sets: (A, α1, β1, β2, β3, β4)
_EC0 = (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
_EC1 = (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
_MAC = (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)  # −α_c

_KAPPA = 0.804
_MU = 0.2195149727645171
_BETA = 0.06672455060314922
_GAMMA = (1.0 - math.log(2.0)) / math.pi**2


def _g_pw92(rs, p):
    a, a1, b1, b2, b3, b4 = p
    srs = torch.sqrt(rs)
    q0 = -2.0 * a * (1.0 + a1 * rs)
    q1 = 2.0 * a * (b1 * srs + b2 * rs + b3 * rs * srs + b4 * rs * rs)
    return q0 * torch.log1p(1.0 / q1)


def eps_c_pw92_spin(rho_au: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
    """PW92 ε_c(rs, ζ) [Ha/electron]."""
    rs = (3.0 / (4.0 * math.pi * rho_au)) ** (1.0 / 3.0)
    ec0 = _g_pw92(rs, _EC0)
    ec1 = _g_pw92(rs, _EC1)
    alpha_c = -_g_pw92(rs, _MAC)
    zc = torch.clamp(zeta, -1.0 + 1e-15, 1.0 - 1e-15)
    fz = ((1.0 + zc) ** (4.0 / 3.0) + (1.0 - zc) ** (4.0 / 3.0) - 2.0) / _FZ_DEN
    z4 = zc**4
    return ec0 + alpha_c * fz / _F_DD0 * (1.0 - z4) + (ec1 - ec0) * fz * z4


class SpinXC(torch.nn.Module):
    """Base: maps per-spin grid densities (and gradients) to e_xc [eV/Å³]."""

    needs_gradient: bool = False

    def energy_density(self, rho_up, rho_dn, sigma_uu=None, sigma_dd=None, sigma_tot=None):
        raise NotImplementedError

    def energy(self, rho_up, rho_dn, volume, sigma_uu=None, sigma_dd=None, sigma_tot=None):
        e = self.energy_density(rho_up, rho_dn, sigma_uu, sigma_dd, sigma_tot)
        return e.sum() * (volume / e.numel())


def _to_au(rho):
    return torch.clamp(rho * BOHR_ANG**3, min=RHO_FLOOR_AU)


class LSDA_PW92(SpinXC):
    needs_gradient = False

    def energy_density(self, rho_up, rho_dn, sigma_uu=None, sigma_dd=None, sigma_tot=None):
        ru, rd = _to_au(rho_up), _to_au(rho_dn)
        rho = ru + rd
        zeta = (ru - rd) / rho
        # exchange by spin scaling of the unpolarized form
        ex = 0.5 * (2.0 * ru * eps_x_lda(2.0 * ru) + 2.0 * rd * eps_x_lda(2.0 * rd)) / rho
        ec = eps_c_pw92_spin(rho, zeta)
        return (rho_up + rho_dn) * (ex + ec) * HARTREE_EV


class SpinPBE(SpinXC):
    needs_gradient = True

    def energy_density(self, rho_up, rho_dn, sigma_uu=None, sigma_dd=None, sigma_tot=None):
        ru, rd = _to_au(rho_up), _to_au(rho_dn)
        rho = ru + rd
        zeta = (ru - rd) / rho

        # exchange: spin scaling, per channel with its own gradient;
        # accumulate ρ_au·ε_x [Ha·bohr⁻³], divide by ρ_au at the end
        ex_dens = torch.zeros_like(rho)
        for r_s, sig in ((ru, sigma_uu), (rd, sigma_dd)):
            r2 = 2.0 * r_s
            s2au = torch.clamp(4.0 * sig * BOHR_ANG**8, min=0.0)
            grad = torch.sqrt(s2au + 1e-30)
            kf = (3.0 * math.pi**2 * r2) ** (1.0 / 3.0)
            s_red = grad / (2.0 * kf * r2)
            fx = 1.0 + _KAPPA - _KAPPA / (1.0 + _MU * s_red * s_red / _KAPPA)
            ex_dens = ex_dens + 0.5 * r2 * eps_x_lda(r2) * fx
        eps_x = ex_dens / rho  # per-electron [Ha]

        # correlation: PW92(rs, ζ) + H(rs, ζ, t) with the total gradient
        ec_lda = eps_c_pw92_spin(rho, zeta)
        zc = torch.clamp(zeta, -1.0 + 1e-12, 1.0 - 1e-12)
        phi = 0.5 * ((1.0 + zc) ** (2.0 / 3.0) + (1.0 - zc) ** (2.0 / 3.0))
        kf = (3.0 * math.pi**2 * rho) ** (1.0 / 3.0)
        ks = torch.sqrt(4.0 * kf / math.pi)
        sig_t = torch.clamp(sigma_tot * BOHR_ANG**8, min=0.0)
        t2 = sig_t / (2.0 * phi * ks * rho) ** 2
        phi3 = phi**3
        expo = torch.exp(-ec_lda / (_GAMMA * phi3))
        a = (_BETA / _GAMMA) / torch.clamp(expo - 1.0, min=1e-30)
        num = 1.0 + a * t2
        den = 1.0 + a * t2 + (a * t2) ** 2
        h = _GAMMA * phi3 * torch.log1p((_BETA / _GAMMA) * t2 * num / den)

        return (rho_up + rho_dn) * (eps_x + ec_lda + h) * HARTREE_EV
