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
    """
    x = t_g[:, None, :] / torch.clamp(t_band[..., None], min=1e-12)
    x2 = x * x
    num = 27.0 + 18.0 * x + 12.0 * x2 + 8.0 * x2 * x
    return residual * (num / (num + 16.0 * x2 * x2))
