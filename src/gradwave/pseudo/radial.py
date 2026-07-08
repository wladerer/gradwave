"""Radial-grid quadrature and spherical Bessel transforms (setup layer, numpy/scipy).

Works for any UPF mesh (linear SG15-style or logarithmic PseudoDojo-style):
the mesh derivative PP_RAB = dr/di is the quadrature weight, so composite
Simpson in the index variable i handles both.

These run once per pseudopotential/geometry setup; results are frozen into
torch buffers by the callers. Nothing here is differentiable and nothing
here may be called from Layer A.
"""

from __future__ import annotations

import numpy as np

# (2l+1)!! for the small-argument series (l=4 appears in USPP/PAW
# augmentation channels, L ≤ 2·l_max_beta)
_DFACT = {0: 1.0, 1: 3.0, 2: 15.0, 3: 105.0, 4: 945.0}


def simpson(fvals: np.ndarray, rab: np.ndarray) -> np.ndarray:
    """∫ f dr on the UPF mesh via composite Simpson in the mesh index.

    fvals: (..., n) integrand values; rab: (n,) dr/di weights.
    For even point counts, Simpson covers the first n-1 points and the last
    interval is closed with a trapezoid (matches QE's accuracy class).
    """
    g = fvals * rab
    n = g.shape[-1]
    if n < 4:
        raise ValueError("need at least 4 mesh points")
    w = np.zeros(n)
    if n % 2 == 1:
        w[:] = 2.0 / 3.0
        w[1::2] = 4.0 / 3.0
        w[0] = w[-1] = 1.0 / 3.0
    else:
        # 1/3 rule over the first n-3 points (odd count), 3/8 rule over the
        # last 4 points — uniformly O(h⁴), unlike a trapezoid closure.
        m = n - 3
        w[:m] = 2.0 / 3.0
        w[1:m:2] = 4.0 / 3.0
        w[0] = 1.0 / 3.0
        w[m - 1] = 1.0 / 3.0 + 3.0 / 8.0
        w[m], w[m + 1] = 9.0 / 8.0, 9.0 / 8.0
        w[m + 2] = 3.0 / 8.0
    return (g * w).sum(axis=-1)


def sph_jl(l: int, x: np.ndarray) -> np.ndarray:
    """Spherical Bessel j_l (l ≤ 3) to full float64 precision.

    The closed trigonometric forms (j₁ = sin x/x² − cos x/x, ...) suffer
    catastrophic cancellation for small x — worse with rising l (j₃ cancels
    five digits at x = 0.5). Below x = 2 use the ascending power series,
    converged to machine precision; above, the trig forms are stable.
    (scipy.special.spherical_jn is only ~1e-9 accurate in this regime, so we
    do not use it.)
    """
    if l not in _DFACT:
        raise ValueError(f"l={l} out of supported range 0..3")
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)

    small = np.abs(x) < 2.0
    xs = x[small]
    x2 = xs * xs
    term = xs**l / _DFACT[l]
    acc = term.copy()
    for k in range(1, 30):
        term = term * (-0.5 * x2) / (k * (2 * l + 2 * k + 1))
        acc += term
        if np.all(np.abs(term) <= 1e-17 * (np.abs(acc) + 1e-300)):
            break
    out[small] = acc

    xb = x[~small]
    s, c = np.sin(xb), np.cos(xb)
    if l == 0:
        big = s / xb
    elif l == 1:
        big = s / xb**2 - c / xb
    elif l == 2:
        big = (3.0 / xb**3 - 1.0 / xb) * s - 3.0 / xb**2 * c
    elif l == 3:
        big = (15.0 / xb**4 - 6.0 / xb**2) * s - (15.0 / xb**3 - 1.0 / xb) * c
    else:
        big = (105.0 / xb**5 - 45.0 / xb**3 + 1.0 / xb) * s + (
            -105.0 / xb**4 + 10.0 / xb**2
        ) * c
    out[~small] = big
    return out


def sbt(l: int, g: np.ndarray, r: np.ndarray, rab: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Spherical Bessel transform table:  F(q) = ∫ g(r) · j_l(q r) dr.

    The caller supplies g with all r-powers already included (e.g. g = (r·β)·r
    for projector form factors, g = 4πr²ρ for densities, g = V_sr·r² for the
    local potential), so nothing here ever divides by r.

    q: (nq,) in Å⁻¹; g, r, rab: (n,) on the UPF mesh. Returns (nq,).
    """
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    jl = sph_jl(l, np.outer(q, r))  # (nq, n)
    return simpson(jl * g, rab)
