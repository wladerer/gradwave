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

from typing import TYPE_CHECKING

import torch
from typing_extensions import override

from gradwave.core.xc._pbe_kernels import KAPPA, MU
from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE, SpinXC

# Positivity floor for the ζ-modulated κ(ζ), μ(ζ): keeps the F_x denominator
# (1 + μ s²/κ) and the κ prefactor strictly positive even where a trained κ₁/μ₁
# would otherwise drive them non-positive. Well below any physical value.
_KMU_FLOOR = 1e-8

if TYPE_CHECKING:
    # gradwave.scf.loop imports LearnableSpinX from this module, so importing
    # SCFResult here for real would be circular (same pattern as core.hubbard's
    # System import; see PR #182's precedent).
    from gradwave.scf.loop import SCFResult

# PBE reference values, re-exported for callers that initialize at PBE.
PBE_KAPPA, PBE_MU = KAPPA, MU


class _LearnableKappaMu:
    """Mixin: softplus-parameterized trainable (κ, μ) exposed as read-only
    properties, so the inherited PBE/SpinPBE energy_density (which reads
    self.kappa/self.mu) trains the exchange enhancement with no other change.
    At the default (PBE) initialization the properties return the PBE values
    and the functional reproduces its fixed-parameter base class exactly."""

    def __init__(self, kappa: float = PBE_KAPPA, mu: float = PBE_MU) -> None:
        super().__init__()
        # softplus-parameterized to keep κ, μ > 0 under unconstrained training
        self.raw_kappa = torch.nn.Parameter(_inv_softplus(kappa))
        self.raw_mu = torch.nn.Parameter(_inv_softplus(mu))

    @property
    def kappa(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_kappa)

    @property
    def mu(self) -> torch.Tensor:
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


class LearnableSpinXZeta(_LearnableKappaMu, SpinPBE):
    """Spin-adapted exchange: PBE enhancement whose (κ, μ) depend on the LOCAL
    spin polarization ζ(r) = (ρ↑−ρ↓)/(ρ↑+ρ↓),

        κ(ζ) = κ₀ + κ₁·ζ²,   μ(ζ) = μ₀ + μ₁·ζ²,

    with κ₀, μ₀ the base (learnable spin-PBE) parameters and κ₁, μ₁ the NEW
    trainable spin-adaptation parameters. The dependence is on ζ² (never ζ): a
    ↑↔↓ swap sends ζ→−ζ and must leave exchange invariant, which ζ² satisfies and
    an odd term would break. Both spin channels see the same ζ(r), so the two
    channels' enhancements share one κ(r), μ(r) at each grid point.

    Initialization κ₁ = μ₁ = 0 makes κ(ζ) ≡ κ₀, μ(ζ) ≡ μ₀, so the functional
    reduces to LearnableSpinX — and at the PBE parameters to SpinPBE — EXACTLY
    (machine precision), the non-negotiable recovery gate.

    Physics trade-off (documented honestly): modulating the per-channel exchange
    enhancement by the shared local ζ(r) DELIBERATELY breaks the exact exchange
    spin-scaling identity E_x[ρ↑,ρ↓] = ½(E_x[2ρ↑] + E_x[2ρ↓]). This is the
    intended spin adaptation — the ζ-dependent term is no longer pure spin-scaled
    exchange; it absorbs a spin-dependent, correlation-like contribution into the
    exchange enhancement. The uniform-gas limit F_x(0) = 1 is untouched (holds for
    any positive κ, μ). Lieb–Oxford κ(ζ) ≤ 0.804 is enforced pointwise (see
    _exchange_kappa_mu), which — since κ(ζ) peaks at |ζ| = 1 — is exactly the
    κ₀ + κ₁ ≤ 0.804 cap; κ(ζ), μ(ζ) are floored strictly positive. ζ is clamped to
    [−1, 1] before squaring and the ζ denominator is ρ-floored upstream (to_au), so
    fully polarized (ζ→±1) and ρ→0 regions are NaN-safe."""

    needs_gradient = True

    def __init__(
        self,
        kappa: float = PBE_KAPPA,
        mu: float = PBE_MU,
        kappa1: float = 0.0,
        mu1: float = 0.0,
    ) -> None:
        super().__init__(kappa=kappa, mu=mu)
        # Unconstrained (plain) parameters so the PBE-recovery init is EXACTLY 0
        # — a softplus reparam could never represent 0 exactly. The physical
        # constraints (Lieb–Oxford cap, positivity) are enforced downstream in
        # _exchange_kappa_mu, not by the parameterization, so κ₁, μ₁ stay free to
        # take either sign during training.
        self.kappa1 = torch.nn.Parameter(torch.tensor(float(kappa1), dtype=torch.float64))
        self.mu1 = torch.nn.Parameter(torch.tensor(float(mu1), dtype=torch.float64))

    @override
    def _exchange_kappa_mu(
        self, zeta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(κ(ζ), μ(ζ)) per grid point. ζ clamped to [−1, 1] before squaring
        (fp guard at full polarization); κ(ζ) capped at the Lieb–Oxford bound and
        both floored strictly positive."""
        z2 = torch.clamp(zeta, -1.0, 1.0) ** 2
        kappa = torch.clamp(self.kappa + self.kappa1 * z2, min=_KMU_FLOOR, max=KAPPA)
        mu = torch.clamp(self.mu + self.mu1 * z2, min=_KMU_FLOOR)
        return kappa, mu


def _inv_softplus(y: float) -> torch.Tensor:
    yt = torch.tensor(float(y), dtype=torch.float64)
    return yt + torch.log(-torch.expm1(-yt))


def energy_param_grads(
    res: SCFResult, xc: XCFunctional | SpinXC
) -> dict[str, torch.Tensor]:
    """dE_total/dθ for all parameters of `xc`, at the converged SCF point.

    Valid by variational stationarity: total-energy derivative w.r.t.
    functional parameters equals ∂E_xc/∂θ at fixed converged density. The
    density is detached (held fixed) and θ carries the graph, so σ — a function
    of ρ, not θ — is detached with it. Handles both the charge-only functionals
    (res.rho, the total density) and the collinear spin functionals (res.rho_spin
    = [ρ↑, ρ↓]); the spin branch is what surfaces dE/dκ₁, dE/dμ₁ for the
    ζ-adaptive LearnableSpinXZeta on a magnetic result.
    """
    from gradwave.core.density import sigma_from_rho

    grid = res.system.grid
    params = list(xc.parameters())
    if isinstance(xc, SpinXC):
        if res.rho_spin is None:
            raise ValueError("spin energy_param_grads needs res.rho_spin (nspin=2)")
        ru, rd = res.rho_spin[0].detach(), res.rho_spin[1].detach()
        if xc.needs_gradient:
            s_uu = sigma_from_rho(ru, grid.g_cart)
            s_dd = sigma_from_rho(rd, grid.g_cart)
            s_tot = sigma_from_rho(ru + rd, grid.g_cart)
        else:
            s_uu = s_dd = s_tot = None
        e_xc = xc.energy(ru, rd, grid.volume, s_uu, s_dd, s_tot)
    else:
        rho = res.rho.detach()
        sigma = sigma_from_rho(rho, grid.g_cart) if xc.needs_gradient else None
        e_xc = xc.energy(rho, grid.volume, sigma)
    grads = torch.autograd.grad(e_xc, params, allow_unused=True)
    return {
        name: g
        for (name, _), g in zip(xc.named_parameters(), grads, strict=True)
    }
