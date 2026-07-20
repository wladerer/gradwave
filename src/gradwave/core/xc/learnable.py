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

import torch

from gradwave.core.xc._pbe_kernels import KAPPA, MU
from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE

# PBE reference values, re-exported for callers that initialize at PBE.
PBE_KAPPA, PBE_MU = KAPPA, MU


class _LearnableKappaMu:
    """Mixin: softplus-parameterized trainable (κ, μ) exposed as read-only
    properties, so the inherited PBE/SpinPBE energy_density (which reads
    self.kappa/self.mu) trains the exchange enhancement with no other change.
    At the default (PBE) initialization the properties return the PBE values
    and the functional reproduces its fixed-parameter base class exactly."""

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


class LearnableX(_LearnableKappaMu, PBE):
    """PBE-form exchange with learnable (κ, μ); PW92+PBE-H correlation fixed.
    Shares PBE.energy_density verbatim — only (κ, μ) become trainable."""

    needs_gradient = True


class LearnableSpinX(_LearnableKappaMu, SpinPBE):
    """Spin-PBE with the same learnable (κ, μ) exchange as LearnableX —
    exact spin scaling per channel, PW92(rs, ζ) + spin-PBE-H correlation
    fixed. At the PBE initialization this reproduces SpinPBE exactly, and
    for ζ = 0 it reduces to LearnableX with the same parameters. Shares
    SpinPBE.energy_density verbatim — only (κ, μ) become trainable."""

    needs_gradient = True


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
