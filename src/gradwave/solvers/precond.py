"""Teter–Payne–Allan kinetic preconditioner for plane-wave eigensolvers."""

from __future__ import annotations

import torch


def teter(residual: torch.Tensor, t_g: torch.Tensor, t_band: torch.Tensor) -> torch.Tensor:
    """K·r with the TPA rational filter.

    residual: (nb, npw); t_g: (npw,) kinetic energies (ħ²/2m)|k+G|²;
    t_band: (nb,) band kinetic expectation ⟨ψ|T|ψ⟩ (must be > 0).
    """
    x = t_g[None, :] / torch.clamp(t_band[:, None], min=1e-12)
    x2 = x * x
    num = 27.0 + 18.0 * x + 12.0 * x2 + 8.0 * x2 * x
    return residual * (num / (num + 16.0 * x2 * x2))


def teter_b(residual: torch.Tensor, t_g: torch.Tensor, t_band: torch.Tensor) -> torch.Tensor:
    """Batched TPA filter: residual (nk, nb, npw), t_g (nk, npw), t_band (nk, nb).

    Padded slots have t_g = 0 → K = 1, residual there is 0 anyway.

    The rational filter coefficient is formed in float64 whatever the residual's
    precision. In the complex64 mixed-precision draft a band with a near-zero
    kinetic expectation (an initial-guess transient, present identically in the
    fp64 path) sends x = t_g/t_band to ~1e14, and 16·x⁴ then OVERFLOWS float32 to
    inf so the coefficient goes inf/inf = nan and poisons the whole SCF (#136).
    fp64 holds x⁴ comfortably and the filter tends to 0 as it should. The
    coefficient is in [0, 1], so casting it back to the residual precision is
    lossless, and the long residual vector itself stays complex64. A no-op for
    the fp64 path — mp-off is bit-identical.
    """
    x = t_g.double()[:, None, :] / torch.clamp(t_band.double()[..., None], min=1e-12)
    x2 = x * x
    num = 27.0 + 18.0 * x + 12.0 * x2 + 8.0 * x2 * x
    coeff = num / (num + 16.0 * x2 * x2)
    return residual * coeff.to(residual.real.dtype)
