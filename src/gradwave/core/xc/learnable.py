"""Learnable exchange enhancement factor — the functional-learning slot (M4).

Exchange: e_x = e_x^LDA(ρ) · F_θ(s²), with the PBE functional form but
LEARNABLE κ, μ (initialization = PBE values reproduces PBE exactly).
Correlation: fixed PW92 + PBE-H gradient correction. The parameterization
inherits the uniform-gas limit (F(0) = 1) and the Lieb–Oxford-motivated
bound (F < 1 + κ) by construction — a badly trained functional is "weird
PBE", not unphysical garbage.

Training gradients dE/dθ are FREE at SCF convergence: the energy is
variational in the density, so dE/dθ = ∂E_xc/∂θ at fixed (detached) ρ —
no response solve needed (energy_param_grads below). Losses that depend on
the DENSITY itself need the implicit-diff SCF backward (scf/implicit.py).
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.xc.base import XCFunctional, to_au
from gradwave.core.xc.lda_pw92 import eps_c_pw92, eps_x_lda
from gradwave.core.xc.spin import SpinXC, eps_c_pw92_spin

_BETA = 0.06672455060314922
_GAMMA = (1.0 - math.log(2.0)) / math.pi**2

PBE_KAPPA = 0.804
PBE_MU = 0.2195149727645171


class LearnableX(XCFunctional):
    """PBE-form exchange with learnable (κ, μ); PW92+PBE-H correlation fixed."""

    needs_gradient = True

    def __init__(self, kappa: float = PBE_KAPPA, mu: float = PBE_MU):
        super().__init__()
        # softplus-parameterized to keep κ, μ > 0 under unconstrained training
        self.raw_kappa = torch.nn.Parameter(_inv_softplus(kappa))
        self.raw_mu = torch.nn.Parameter(_inv_softplus(mu))

    @property
    def kappa(self):
        return torch.nn.functional.softplus(self.raw_kappa)

    @property
    def mu(self):
        return torch.nn.functional.softplus(self.raw_mu)

    def energy_density(self, rho: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        if sigma is None:
            raise ValueError("LearnableX requires sigma")
        rho_au = to_au(rho)
        sigma_au = torch.clamp(sigma * BOHR_ANG**8, min=0.0)
        grad_au = torch.sqrt(sigma_au + 1e-30)
        kf = (3.0 * math.pi**2 * rho_au) ** (1.0 / 3.0)
        s2 = (grad_au / (2.0 * kf * rho_au)) ** 2

        kappa, mu = self.kappa, self.mu
        fx = 1.0 + kappa - kappa / (1.0 + mu * s2 / kappa)
        eps_x = eps_x_lda(rho_au) * fx

        eps_c_lda = eps_c_pw92(rho_au)
        ks = torch.sqrt(4.0 * kf / math.pi)
        t2 = sigma_au / (2.0 * ks * rho_au) ** 2
        expo = torch.exp(-eps_c_lda / _GAMMA)
        a = (_BETA / _GAMMA) / torch.clamp(expo - 1.0, min=1e-30)
        num = 1.0 + a * t2
        den = 1.0 + a * t2 + (a * t2) ** 2
        h = _GAMMA * torch.log1p((_BETA / _GAMMA) * t2 * num / den)
        return rho * (eps_x + eps_c_lda + h) * HARTREE_EV


class LearnableSpinX(SpinXC):
    """Spin-PBE with the same learnable (κ, μ) exchange as LearnableX —
    exact spin scaling per channel, PW92(rs, ζ) + spin-PBE-H correlation
    fixed. At the PBE initialization this reproduces SpinPBE exactly, and
    for ζ = 0 it reduces to LearnableX with the same parameters."""

    needs_gradient = True

    def __init__(self, kappa: float = PBE_KAPPA, mu: float = PBE_MU):
        super().__init__()
        self.raw_kappa = torch.nn.Parameter(_inv_softplus(kappa))
        self.raw_mu = torch.nn.Parameter(_inv_softplus(mu))

    @property
    def kappa(self):
        return torch.nn.functional.softplus(self.raw_kappa)

    @property
    def mu(self):
        return torch.nn.functional.softplus(self.raw_mu)

    def energy_density(self, rho_up, rho_dn, sigma_uu=None, sigma_dd=None,
                       sigma_tot=None):
        ru, rd = to_au(rho_up), to_au(rho_dn)
        rho = ru + rd
        zeta = (ru - rd) / rho
        kappa, mu = self.kappa, self.mu

        # exchange: spin scaling, per channel with its own gradient
        ex_dens = torch.zeros_like(rho)
        for r_s, sig in ((ru, sigma_uu), (rd, sigma_dd)):
            r2 = 2.0 * r_s
            s2au = torch.clamp(4.0 * sig * BOHR_ANG**8, min=0.0)
            grad = torch.sqrt(s2au + 1e-30)
            kf = (3.0 * math.pi**2 * r2) ** (1.0 / 3.0)
            s_red = grad / (2.0 * kf * r2)
            fx = 1.0 + kappa - kappa / (1.0 + mu * s_red * s_red / kappa)
            ex_dens = ex_dens + 0.5 * r2 * eps_x_lda(r2) * fx
        eps_x = ex_dens / rho

        # correlation: PW92(rs, ζ) + H(rs, ζ, t), fixed (matches SpinPBE)
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


def _inv_softplus(y: float) -> torch.Tensor:
    y = torch.tensor(float(y), dtype=torch.float64)
    return y + torch.log(-torch.expm1(-y))


def energy_param_grads(res, xc: XCFunctional) -> dict[str, torch.Tensor]:
    """dE_total/dθ for all parameters of `xc`, at the converged SCF point.

    Valid by variational stationarity: total-energy derivative w.r.t.
    functional parameters equals ∂E_xc/∂θ at fixed converged density.
    """
    from gradwave.core.density import sigma_from_rho

    grid = res.system.grid
    rho = res.rho.detach()
    sigma = sigma_from_rho(rho, grid.g_cart) if xc.needs_gradient else None
    e_xc = xc.energy(rho, grid.volume, sigma)
    grads = torch.autograd.grad(e_xc, list(xc.parameters()), allow_unused=True)
    return {
        name: g
        for (name, _), g in zip(xc.named_parameters(), grads, strict=True)
    }
