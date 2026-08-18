"""Electrostatics for the FLAPW stack (eV/Å units): the radial Hartree Poisson and Weinert's method.

- ``hartree`` — the l=0 radial Hartree of an isolated spherical density (used by the atomic SCF).
- ``fft_poisson`` — the periodic interstitial Poisson ``V_H(G) = 4π E2 ρ(G)/|G|²`` on a cubic grid.
- ``sphere_pseudocharge`` — a smooth l=0 pseudocharge with a prescribed monopole (Weinert's trick:
  the exterior potential depends only on the multipole moment, not the interior shape).
- ``radial_poisson_to_R`` — the l=0 sphere radial Poisson with a Dirichlet boundary at R_MT.

Together these solve the periodic FLAPW Coulomb: replace each sharp muffin-tin charge by a smooth
pseudocharge of the same moments, FFT-solve the resulting smooth density for the interstitial
potential, then solve the true sphere charge's radial Poisson matched to that potential at R_MT.

Poisson convention: ``∇²V = -4π E2 ρ``, so ``V(G) = 4π E2 ρ(G)/|G|²``; the G=0 term is 0 (a
neutralizing background — the density passed for a crystal should be net-neutral).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from gradwave.constants import E2


def ball_ff_np(g, R):
    """Numpy Fourier transform of a solid-ball indicator, ``W(|G|) = 4π(sin x - x cos x)/g³``,
    ``x=gR`` (Å³). The numpy counterpart of ``gradwave.core.sphere_ff.ball_ff``, used in the LAPW
    interstitial overlap and the Weinert pseudocharge."""
    x = g * R
    small = x < 1e-2
    gs = np.where(small, 1.0, g)
    full = 4 * math.pi * (np.sin(x) - x * np.cos(x)) / gs**3
    series = 4 * math.pi * R**3 * (1 / 3 - x**2 / 30 + x**4 / 840)
    return np.where(small, series, full)


def hartree(rho: Tensor, r: Tensor, dx: float) -> Tensor:
    """Radial Poisson: ``V_H(r) = e²[ (1/r)∫_0^r 4πρr'²dr' + ∫_r^∞ 4πρr'dr' ]`` on a log mesh."""
    dr = r * dx
    q_in = torch.cumsum(4 * math.pi * rho * r * r * dr, dim=0)
    tail = torch.flip(torch.cumsum(torch.flip(4 * math.pi * rho * r * dr, [0]), 0), [0])
    return E2 * (q_in / r + tail)


def g2_grid(n: int, L: float):
    """``|G|²`` (Å⁻²) and the G-vector components on an ``n³`` FFT grid for a cubic cell side L."""
    f = np.fft.fftfreq(n, d=1.0 / n) * (2 * math.pi / L)
    Gx, Gy, Gz = np.meshgrid(f, f, f, indexing="ij")
    return Gx**2 + Gy**2 + Gz**2, (Gx, Gy, Gz)


def fft_poisson(rho_r: np.ndarray, L: float) -> np.ndarray:
    """Periodic Hartree potential (eV) of a smooth density ``rho_r`` (e/Å³) on a cubic grid."""
    n = rho_r.shape[0]
    g2, _ = g2_grid(n, L)
    rho_g = np.fft.fftn(rho_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        vg = np.where(g2 > 1e-12, 4 * math.pi * E2 * rho_g / g2, 0.0)
    return np.fft.ifftn(vg).real


def sphere_pseudocharge(q00: float, R: float, center, n: int, L: float,
                        npow: int = 4) -> np.ndarray:
    """Smooth l=0 pseudocharge on the grid with monopole ``q00``: ``ρ̃(d) ∝ (1-(d/R)²)^npow``."""
    ax = np.arange(n) * (L / n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")

    def mi(a, c):
        return (a - c) - L * np.round((a - c) / L)

    d = np.sqrt(mi(X, center[0]) ** 2 + mi(Y, center[1]) ** 2 + mi(Z, center[2]) ** 2)
    shape = np.where(d < R, (1 - (d / R) ** 2) ** npow, 0.0)
    norm = shape.sum() * (L / n) ** 3
    return shape * (q00 / norm)


def radial_poisson_to_R(rho: np.ndarray, r: np.ndarray, R: float, drw=None) -> np.ndarray:
    """l=0 radial Poisson inside R with the density contained in R (ρ=0 beyond):
    ``V_part(r) = E2[(1/r)∫_0^r 4πρr'²dr' + ∫_r^R 4πρr'dr']``.

    ``drw`` = per-point ∫dr weight. ``None`` -> uniform mesh; pass ``r·dx`` for a log mesh.
    """
    w = np.full_like(r, float(r[1] - r[0])) if drw is None else np.asarray(drw)
    q_in = np.cumsum(4 * math.pi * rho * r**2 * w)
    tail = np.cumsum((4 * math.pi * rho * r * w)[::-1])[::-1] - (4 * math.pi * rho * r * w)
    return E2 * (q_in / np.maximum(r, 1e-12) + tail)
