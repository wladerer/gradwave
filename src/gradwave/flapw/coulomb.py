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

import itertools
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


def cell_matrix(L) -> np.ndarray:
    """The 3×3 cell (Å, rows = lattice vectors) from a scalar (cubic side), a length-3 array
    (orthorhombic edges), or a 3×3 matrix (triclinic). The one place cell shape is decoded."""
    a = np.asarray(L, dtype=float)
    if a.ndim == 0:
        return np.eye(3) * float(a)
    if a.shape == (3,):
        return np.diag(a)
    if a.shape == (3, 3):
        return a
    raise ValueError(f"cell must be scalar, (3,), or (3,3); got shape {a.shape}")


def reciprocal(A: np.ndarray) -> np.ndarray:
    """Reciprocal lattice ``B = 2π (A⁻¹)ᵀ`` (rows = reciprocal vectors); ``a_i·b_j = 2π δ_ij``."""
    return 2 * math.pi * np.linalg.inv(A).T


def _min_image_dist(cfrac, n, A):
    """Cartesian minimum-image distance from fractional centre ``cfrac`` to every ``n³`` grid point.
    Per-component wrap suffices for orthogonal axes; general cells search neighbour images."""
    fi = np.arange(n) / n
    FX, FY, FZ = np.meshgrid(fi, fi, fi, indexing="ij")
    df = np.stack([FX - cfrac[0], FY - cfrac[1], FZ - cfrac[2]], axis=-1)
    df -= np.round(df)                                    # wrap fractional to (-1/2, 1/2]
    metric = A @ A.T
    if np.allclose(metric - np.diag(np.diag(metric)), 0.0):
        return np.sqrt(((df @ A) ** 2).sum(-1))          # orthogonal: the wrapped image is nearest
    d2 = np.full((n, n, n), np.inf)                       # triclinic: search the 27 nearest images
    for s in itertools.product((-1, 0, 1), repeat=3):
        d2 = np.minimum(d2, (((df + np.asarray(s)) @ A) ** 2).sum(-1))
    return np.sqrt(d2)


def g2_grid(n: int, L):
    """``|G|²`` (Å⁻²) and the G-vector components on an ``n³`` FFT grid over the fractional cell.
    ``L`` is a cubic side, length-3 orthorhombic edges, or a 3×3 triclinic cell; ``G = m·B``."""
    B = reciprocal(cell_matrix(L))
    fi = np.fft.fftfreq(n, d=1.0 / n)
    MX, MY, MZ = np.meshgrid(fi, fi, fi, indexing="ij")
    Gx = MX * B[0, 0] + MY * B[1, 0] + MZ * B[2, 0]
    Gy = MX * B[0, 1] + MY * B[1, 1] + MZ * B[2, 1]
    Gz = MX * B[0, 2] + MY * B[1, 2] + MZ * B[2, 2]
    return Gx**2 + Gy**2 + Gz**2, (Gx, Gy, Gz)


def fft_poisson(rho_r: np.ndarray, L) -> np.ndarray:
    """Periodic Hartree potential (eV) of a smooth density ``rho_r`` (e/Å³) on the fractional grid.
    ``L`` cubic side, length-3 orthorhombic edges, or a 3×3 triclinic cell."""
    n = rho_r.shape[0]
    g2, _ = g2_grid(n, L)
    rho_g = np.fft.fftn(rho_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        vg = np.where(g2 > 1e-12, 4 * math.pi * E2 * rho_g / g2, 0.0)
    return np.fft.ifftn(vg).real


def sphere_pseudocharge(q00: float, R: float, center, n: int, L, npow: int = 4) -> np.ndarray:
    """Smooth l=0 pseudocharge on the grid with monopole ``q00``: ``ρ̃(d) ∝ (1-(d/R)²)^npow``.
    ``center`` is Cartesian (Å); ``L`` any cell (Cartesian distance, minimum image)."""
    A = cell_matrix(L)
    cfrac = np.asarray(center) @ np.linalg.inv(A)
    d = _min_image_dist(cfrac, n, A)
    shape = np.where(d < R, (1 - (d / R) ** 2) ** npow, 0.0)
    norm = shape.sum() * (abs(np.linalg.det(A)) / n**3)
    return shape * (q00 / norm)


def _min_image_vec(cfrac, n, A):
    """The Cartesian minimum-image displacement vector (n,n,n,3) from fractional centre ``cfrac`` to
    each ``n³`` grid point — the vector form of ``_min_image_dist`` (needed for angular shapes)."""
    fi = np.arange(n) / n
    fx, fy, fz = np.meshgrid(fi, fi, fi, indexing="ij")
    df = np.stack([fx - cfrac[0], fy - cfrac[1], fz - cfrac[2]], axis=-1)
    df -= np.round(df)
    metric = A @ A.T
    if np.allclose(metric - np.diag(np.diag(metric)), 0.0):
        return df @ A
    disp = (df @ A)
    best = (disp ** 2).sum(-1)
    for s in itertools.product((-1, 0, 1), repeat=3):
        if s == (0, 0, 0):
            continue
        rc = (df + np.asarray(s)) @ A
        d2 = (rc ** 2).sum(-1)
        m = d2 < best
        disp = np.where(m[..., None], rc, disp)
        best = np.minimum(best, d2)
    return disp


def sphere_pseudocharge_lm(qlm, R: float, center, n: int, L, big_l: int,
                           npow: int = 4) -> np.ndarray:
    """Smooth angular-momentum-``big_l`` pseudocharge on the grid matching the sphere's charge
    moments ``qlm = {M: Q_LM}`` (M=-L..L): ``ρ̃(r,Ω) = Σ_M c_M d^L (1-(d/R)²)^npow Y_LM(Ω)`` with
    ``c_M = Q_LM / (R^{2L+3} ∫_0^1 x^{2L+2}(1-x²)^npow dx)`` so its L-moment ``∫ρ̃ r^{L+2} Y*`` is
    ``Q_LM``. This is the general-L part of Weinert's pseudocharge: matching every multipole in play
    (not just the monopole) makes the interstitial FFT potential carry each sphere's full exterior
    multipole field — the inter-atomic (lattice) terms of the aspherical potential and the EFG."""
    from scipy.special import sph_harm_y
    a = cell_matrix(L)
    disp = _min_image_vec(np.asarray(center) @ np.linalg.inv(a), n, a)
    d = np.linalg.norm(disp, axis=-1)
    inside = d < R
    ds = np.where(d < 1e-12, 1.0, d)
    theta = np.arccos(np.clip(disp[..., 2] / ds, -1.0, 1.0))
    phi = np.arctan2(disp[..., 1], disp[..., 0])
    x = np.linspace(0.0, 1.0, 4000)
    i_n = float(np.trapezoid(x ** (2 * big_l + 2) * (1 - x**2) ** npow, x))
    radial = np.where(inside, d**big_l * (1 - (d / R) ** 2) ** npow, 0.0)
    out = np.zeros(d.shape)
    for m in range(-big_l, big_l + 1):
        c_m = qlm[m] / (R ** (2 * big_l + 3) * i_n)
        out = out + (c_m * radial * sph_harm_y(big_l, m, theta, phi)).real
    return out


def sphere_pseudocharge_l2(q2m, R: float, center, n: int, L, npow: int = 4) -> np.ndarray:
    """The l=2 case of ``sphere_pseudocharge_lm`` (kept as the historical entry point)."""
    return sphere_pseudocharge_lm(q2m, R, center, n, L, 2, npow=npow)


def radial_poisson_to_R(rho: np.ndarray, r: np.ndarray, R: float, drw=None) -> np.ndarray:
    """l=0 radial Poisson inside R with the density contained in R (ρ=0 beyond):
    ``V_part(r) = E2[(1/r)∫_0^r 4πρr'²dr' + ∫_r^R 4πρr'dr']``.

    ``drw`` = per-point ∫dr weight. ``None`` -> uniform mesh; pass ``r·dx`` for a log mesh.
    """
    w = np.full_like(r, float(r[1] - r[0])) if drw is None else np.asarray(drw)
    q_in = np.cumsum(4 * math.pi * rho * r**2 * w)
    tail = np.cumsum((4 * math.pi * rho * r * w)[::-1])[::-1] - (4 * math.pi * rho * r * w)
    return E2 * (q_in / np.maximum(r, 1e-12) + tail)
