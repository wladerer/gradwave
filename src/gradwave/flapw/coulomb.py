"""Electrostatics for the FLAPW stack (eV/Å units): the radial Hartree Poisson and Weinert's method.

- ``hartree`` — the l=0 radial Hartree of an isolated spherical density (used by the atomic SCF).
- ``fft_poisson`` — the periodic interstitial Poisson ``V_H(G) = 4π E2 ρ(G)/|G|²`` on a cubic grid.
- ``sphere_pseudocharge`` — a smooth l=0 pseudocharge with a prescribed monopole (Weinert's trick:
  the exterior potential depends only on the multipole moment, not the interior shape).
- ``gvec_ylm_tables`` / ``sphere_interstitial_moments`` / ``sphere_pseudocharge_ft`` — the general-L
  Weinert machinery, entirely in G-space: analytic in-sphere multipole moments of a band-limited
  density (Bessel projection, exact for the Fourier series) and the analytic Fourier synthesis of
  the smooth moment-matched pseudocharge ``r^L (1-(r/R)²)^N Y_LM``. No real-space sampling anywhere,
  so there is no grid aliasing: the pseudocharge's moments are exact up to Fourier truncation, and
  the own-sphere multipole field can be subtracted ANALYTICALLY from the sphere-boundary values
  (Elk ``zpotcoul``'s ``z1 = (zlm − zvclmt(R))/R^l`` homogeneous-solution construction). The
  earlier real-space-sampled per-M pseudocharge (Gram-corrected grid moments) left a ~20%
  own-field residue from sampling aliasing that fed a runaway aspherical fixed point.
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


def _dfact(n: int) -> float:
    """Odd double factorial ``n!!`` (``n ≥ -1``; ``(-1)!! = 1``)."""
    return float(math.prod(range(n, 0, -2))) if n > 0 else 1.0


_GTAB_CACHE: dict = {}


def gvec_ylm_tables(cell, n: int, lmax: int):
    """Flattened G-vector table of the ``n³`` FFT grid with harmonics: ``(gvec (n³,3), gnorm (n³,),
    ylm {(L,M): Y_LM(Ĝ) (n³,)})`` for ``L ≤ lmax`` — the shared geometry of the G-space Weinert
    machinery (``sphere_interstitial_moments`` / ``sphere_pseudocharge_ft``). Cached per
    (cell, n, lmax), 2-entry LRU (a few tens of MB at production sizes)."""
    a = cell_matrix(cell)
    key = (a.tobytes(), n, lmax)
    hit = _GTAB_CACHE.get(key)
    if hit is not None:
        return hit
    from scipy.special import sph_harm_y
    b = reciprocal(a)
    fi = np.fft.fftfreq(n, d=1.0 / n)
    mx, my, mz = np.meshgrid(fi, fi, fi, indexing="ij")
    gvec = np.stack([mx * b[0, 0] + my * b[1, 0] + mz * b[2, 0],
                     mx * b[0, 1] + my * b[1, 1] + mz * b[2, 1],
                     mx * b[0, 2] + my * b[1, 2] + mz * b[2, 2]], axis=-1).reshape(-1, 3)
    gnorm = np.linalg.norm(gvec, axis=-1)
    gs = np.where(gnorm < 1e-12, 1.0, gnorm)
    th = np.arccos(np.clip(gvec[:, 2] / gs, -1.0, 1.0))
    ph = np.arctan2(gvec[:, 1], gvec[:, 0])
    ylm = {(lang, m): sph_harm_y(lang, m, th, ph)
           for lang in range(lmax + 1) for m in range(-lang, lang + 1)}
    while len(_GTAB_CACHE) >= 2:
        _GTAB_CACHE.pop(next(iter(_GTAB_CACHE)))
    _GTAB_CACHE[key] = (gvec, gnorm, ylm)
    return gvec, gnorm, ylm


def sphere_interstitial_moments(rho_g, R: float, center, gvec, gnorm, ylm, big_ls):
    """Analytic multipole moments ``q^I_LM = ∫_{r<R} ρ(τ+r) r^L Y*_LM(r̂) d³r`` of a band-limited
    density about the sphere centre ``τ`` (Weinert / Elk ``zpotcoul``):

        q^I_LM = 4π i^L R^{L+3} Σ_G [j_{L+1}(GR)/(GR)] ρ_G e^{iG·τ} Y*_LM(Ĝ),

    with the G=0 term ``(4π/3)R³ρ₀Y₀₀δ_L0``. ``rho_g`` = flattened Fourier coefficients
    (``fftn(ρ)/n³``). Exact for the Fourier series — replaces jagged in-sphere grid sums (whose
    sampling error broke the own-field cancellation). Returns ``{(L,M): complex}``."""
    from scipy.special import spherical_jn
    x = gnorm * R
    g0 = gnorm < 1e-12
    xs = np.where(g0, 1.0, x)
    z = rho_g * np.exp(1j * (gvec @ np.asarray(center, dtype=float)))
    out = {}
    for bl in big_ls:
        rad = spherical_jn(bl + 1, x) / xs
        rad[g0] = (1.0 / 3.0) if bl == 0 else 0.0          # lim j_{L+1}(x)/x = δ_L0/3
        base = (4 * math.pi) * (1j ** bl) * R ** (bl + 3) * (z * rad)
        for m in range(-bl, bl + 1):
            out[(bl, m)] = complex(np.sum(base * np.conj(ylm[(bl, m)])))
    return out


def sphere_pseudocharge_ft(qlm, R: float, center, vol: float, npow: int, gvec, gnorm, ylm):
    """Fourier coefficients (flattened, ``fftn/n³`` convention) of Weinert's smooth in-sphere
    pseudocharge with prescribed multipole moments ``qlm = {(L,M): Q_LM}`` — the shape
    ``Σ_LM c_LM (r/R)^L (1-(r/R)²)^npow Y_LM`` synthesized ANALYTICALLY in G-space:

        ρ̃(G) = (4π/Ω) e^{-iG·τ} Σ_LM (-i)^L [(2L+2npow+3)!!/(2L+1)!!]
                · j_{L+npow+1}(GR)/((GR)^{npow+1} R^L) · Q_LM Y_LM(Ĝ),

    G=0 → ``√4π Q_00/Ω``. No real-space sampling → no grid aliasing; the pseudocharge's moments
    equal ``Q_LM`` exactly up to the Fourier truncation of the grid, which decays as
    ``(GR)^{-(npow+2)}`` — choose ``npow ≈ R·Gmax/4`` (Weinert) so the series has converged at the
    grid's Nyquist edge."""
    from scipy.special import spherical_jn
    x = gnorm * R
    g0 = gnorm < 1e-12
    xs = np.where(g0, 1.0, x)
    acc = np.zeros(gnorm.shape, dtype=complex)
    for bl in sorted({lm[0] for lm in qlm}):
        pref = ((-1j) ** bl) * _dfact(2 * bl + 2 * npow + 3) / (_dfact(2 * bl + 1) * R ** bl)
        rad = spherical_jn(bl + npow + 1, x) / xs ** (npow + 1)
        ang = sum(qlm.get((bl, m), 0.0) * ylm[(bl, m)] for m in range(-bl, bl + 1))
        acc += pref * rad * ang
    out = (4 * math.pi / vol) * np.exp(-1j * (gvec @ np.asarray(center, dtype=float))) * acc
    out[g0] = math.sqrt(4 * math.pi) * complex(qlm.get((0, 0), 0.0)) / vol
    return out


def radial_poisson_to_R(rho: np.ndarray, r: np.ndarray, R: float, drw=None) -> np.ndarray:
    """l=0 radial Poisson inside R with the density contained in R (ρ=0 beyond):
    ``V_part(r) = E2[(1/r)∫_0^r 4πρr'²dr' + ∫_r^R 4πρr'dr']``.

    ``drw`` = per-point ∫dr weight. ``None`` -> uniform mesh; pass ``r·dx`` for a log mesh.
    """
    w = np.full_like(r, float(r[1] - r[0])) if drw is None else np.asarray(drw)
    q_in = np.cumsum(4 * math.pi * rho * r**2 * w)
    tail = np.cumsum((4 * math.pi * rho * r * w)[::-1])[::-1] - (4 * math.pi * rho * r * w)
    return E2 * (q_in / np.maximum(r, 1e-12) + tail)
