"""Analytic Fourier transform of a solid-ball indicator — the Θ(G) form factor that lets a
moving region boundary (a muffin-tin sphere) carry an autograd-recoverable surface term.

The indicator of a ball of radius ``R`` centred at the origin has the closed-form transform

    W(G) = ∫_{|r|<R} e^{-iG·r} d³r = 4π (sin x - x cos x) / |G|³,   x = |G|·R,

with ``W(0) = 4πR³/3`` (the ball volume). A ball centred at ``τ`` is ``W(G)·e^{-iG·τ}``: the
entire ``τ``-dependence lives in a smooth phase, so a two-region energy built from this form
factor differentiates through ``τ`` cleanly and recovers the moving-boundary *surface* term —
the same structure gradwave already uses for the USPP/PAW compensation charge
(``postscf.uspp_frozen``). A naive real-space grid indicator ``1[|r_n-τ| < R]`` instead hides
``τ`` inside a Heaviside the FFT grid never resolves, so its autograd gradient is identically
zero; see ``experiments/autoapw/surface_term_toy.py`` for the demonstration and control.

Pure geometry: ``|G|`` in Å⁻¹, ``R`` in Å, result in Å³. fp64; autograd- and
``torch.func``-traceable in both ``|G|`` and ``R``.
"""

from __future__ import annotations

import torch
from torch import Tensor

# |G|·R below this uses the Taylor series, avoiding the 0/0 in the closed form near G=0.
_SMALL = 1.0e-2


def ball_ff(g_norm: Tensor, radius: Tensor | float) -> Tensor:
    """Fourier transform ``W(|G|)`` of a solid ball of the given radius.

    Args:
        g_norm: ``|G|`` magnitudes (Å⁻¹), any shape.
        radius: ball radius ``R`` (Å); scalar or broadcastable to ``g_norm``.

    Returns:
        ``W`` (Å³), same shape as ``g_norm``. ``W(0) = 4πR³/3``.

    The small-``x`` branch uses ``(sin x - x cos x)/x³ = 1/3 - x²/30 + x⁴/840 - x⁶/45360 + …``.
    """
    r = torch.as_tensor(radius, dtype=g_norm.dtype, device=g_norm.device)
    x = g_norm * r
    x2 = x * x

    # Closed form on a |G| clamped away from 0 so the unused branch cannot emit a NaN that would
    # poison the torch.where backward.
    g_safe = g_norm.clamp_min(_SMALL / r)
    x_safe = g_safe * r
    full = 4.0 * torch.pi * (torch.sin(x_safe) - x_safe * torch.cos(x_safe)) / (g_safe**3)

    series = (4.0 * torch.pi * r**3) * (
        1.0 / 3.0 - x2 / 30.0 + x2 * x2 / 840.0 - x2 * x2 * x2 / 45360.0
    )

    return torch.where(x < _SMALL, series, full)
