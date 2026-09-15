"""Real spherical harmonics beyond l=3 and real Gaunt coefficients (setup layer).

The ultrasoft/PAW augmentation Q_ij(r⃗) carries the product Y_{l₁m₁}Y_{l₂m₂},
expanded as Σ_LM c·Y_LM with c the REAL Gaunt coefficient ∫Y_LM Y₁ Y₂ dΩ and
L up to l₁+l₂ (= 4 for d-channel projectors, beyond core/ylm.py's l ≤ 3).

ylm_np evaluates the same convention as core/ylm.py (verified for l ≤ 3 by a
unit test): Y_l0 = √((2l+1)/4π)·P_l, and for m > 0 the cos/sin pair

    Y_{l,+m-slot} = √2·N_lm·P_l^m(cosθ)·cos(mφ)      (no Condon–Shortley)
    Y_{l,−m-slot} = √2·N_lm·P_l^m(cosθ)·sin(mφ)

ordered (l,0),(l,1c),(l,1s),(l,2c),(l,2s),… densely in l² … (l+1)²−1.

Gaunt coefficients are computed by Gauss–Legendre × uniform-φ quadrature,
exact for the band-limited integrand (degree ≤ 3·lmax); numpy only, runs once
per setup.
"""

from __future__ import annotations

import numpy as np
from scipy.special import lpmv, roots_legendre


def gauss_legendre_sphere(n_cos: int, n_phi: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre(cosθ) × uniform-φ sphere quadrature.

    Returns ``(dirs, w)`` with ``dirs`` the (n_cos·n_phi, 3) Cartesian unit
    directions and ``w`` the matching (n_cos·n_phi,) weights (Gauss–Legendre
    weight in cosθ times the uniform 2π/n_phi step), so ``Σ w·f ≈ ∫ f dΩ`` and
    is exact for band-limited integrands (degree ≤ 2·n_cos−1 in cosθ). numpy
    only; the shared node/weight layout for the Gaunt table, becsum
    symmetrization, and the one-center angular grid.
    """
    z, wz = roots_legendre(n_cos)
    phi = np.arange(n_phi) * (2.0 * np.pi / n_phi)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    st = np.sqrt(1.0 - zz**2)
    dirs = np.stack([st * np.cos(pp), st * np.sin(pp), zz], axis=-1).reshape(-1, 3)
    w = (wz[:, None] * np.full(n_phi, 2.0 * np.pi / n_phi)).reshape(-1)
    return dirs, w


def _norm_lm(l: int, m: int) -> float:
    from math import factorial, pi, sqrt

    return sqrt((2 * l + 1) / (4.0 * pi) * factorial(l - m) / factorial(l + m))


def ylm_np(lmax: int, vecs: np.ndarray) -> np.ndarray:
    """Real Y_lm for l ≤ lmax at directions of vecs (..., 3) → (..., (lmax+1)²).

    Zero vectors get Y_00 only (matches core/ylm.py's guard).
    """
    v = np.asarray(vecs, dtype=np.float64)
    norm = np.linalg.norm(v, axis=-1)
    safe = np.where(norm < 1e-14, 1.0, norm)
    unit = v / safe[..., None]
    zero = norm < 1e-14
    ct = np.where(zero, 1.0, unit[..., 2])  # cosθ
    phi = np.arctan2(unit[..., 1], unit[..., 0])

    out = np.empty((*v.shape[:-1], (lmax + 1) ** 2))
    for l in range(lmax + 1):
        out[..., l * l] = _norm_lm(l, 0) * lpmv(0, l, ct)
        for m in range(1, l + 1):
            # lpmv includes Condon–Shortley (−1)^m — remove it
            plm = ((-1.0) ** m) * lpmv(m, l, ct)
            fac = np.sqrt(2.0) * _norm_lm(l, m) * plm
            out[..., l * l + 2 * m - 1] = fac * np.cos(m * phi)
            out[..., l * l + 2 * m] = fac * np.sin(m * phi)
    if np.any(zero):
        y00 = out[..., 0].copy()
        out[zero] = 0.0
        out[..., 0] = y00
    return out


def real_gaunt_table(lmax_beta: int) -> np.ndarray:
    """c[LM, i, j] = ∫ Y_LM Y_i Y_j dΩ for i, j ≤ (lmax_beta+1)², L ≤ 2·lmax_beta.

    Quadrature: Gauss–Legendre in cosθ (exact through degree 2n−1) ×
    trapezoid in φ (exact for band-limited 2π-periodic integrands).
    """
    lmax_aug = 2 * lmax_beta
    deg = 3 * lmax_beta + 2
    nct = deg + 2
    nphi = 2 * (3 * lmax_beta) + 4
    dirs, w = gauss_legendre_sphere(nct, nphi)

    y_all = ylm_np(lmax_aug, dirs)  # (npt, (2lb+1)²)
    nb = (lmax_beta + 1) ** 2
    yb = y_all[:, :nb]
    return np.einsum("pL,pi,pj,p->Lij", y_all, yb, yb, w)
