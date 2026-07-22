"""Radial-grid quadrature and spherical Bessel transforms (setup layer, numpy/scipy).

Works for any UPF mesh (linear SG15-style or logarithmic PseudoDojo-style):
the mesh derivative PP_RAB = dr/di is the quadrature weight, so composite
Simpson in the index variable i handles both.

These run once per pseudopotential/geometry setup; results are frozen into
torch buffers by the callers. Nothing here is differentiable and nothing
here may be called from Layer A.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from gradwave.pseudo._bessel_data import DOUBLE_FACTORIAL, SERIES_TERMS, SERIES_X

# sph_jl / simpson are element-wise numpy over the (nq, n) transform and release
# the GIL, so a large sbt splits across threads for a near-linear speedup. Small
# transforms (per-atom PDOS/Hubbard qmag, few-shell cells) keep the serial path:
# the pool spin-up would cost more than it saves.
_ncpu = os.cpu_count() or 1
# default: use the cores, capped at 16 (setup is a one-time burst while torch's
# own pool is idle, so oversubscription isn't a concern); override via env.
_SBT_THREADS = max(1, min(int(os.environ.get("GRADWAVE_SBT_THREADS", "0"))
                          or min(_ncpu, 16), _ncpu))
# |q| per chunk: small enough that _SBT_THREADS chunks live at once stay cheap
# (each chunk is a (chunk, n_radial) transform with ~10 sph_jl temporaries, so
# peak ≈ threads × chunk × n_radial); large enough that numpy vectorization and
# GIL-release dominate the pool dispatch overhead. 512 keeps the threaded slab
# setup near the serial memory footprint.
_SBT_CHUNK_MIN = 512


def _simpson_index_weights(n: int) -> np.ndarray:
    """Composite-Simpson weights in the mesh index, so ∫f dr = Σ f·rab·w.

    Odd n: the plain 1/3 rule. Even n: 1/3 rule over the first n-3 points plus a
    3/8 closure over the last 4 — uniformly O(h⁴), unlike a trapezoid closure.
    """
    if n < 4:
        raise ValueError("need at least 4 mesh points")
    w = np.zeros(n)
    if n % 2 == 1:
        w[:] = 2.0 / 3.0
        w[1::2] = 4.0 / 3.0
        w[0] = w[-1] = 1.0 / 3.0
    else:
        m = n - 3
        w[:m] = 2.0 / 3.0
        w[1:m:2] = 4.0 / 3.0
        w[0] = 1.0 / 3.0
        w[m - 1] = 1.0 / 3.0 + 3.0 / 8.0
        w[m], w[m + 1] = 9.0 / 8.0, 9.0 / 8.0
        w[m + 2] = 3.0 / 8.0
    return w


def simpson(fvals: np.ndarray, rab: np.ndarray) -> np.ndarray:
    """∫ f dr on the UPF mesh via composite Simpson in the mesh index.

    fvals: (..., n) integrand values; rab: (n,) dr/di weights.
    """
    g = fvals * rab
    return (g * _simpson_index_weights(g.shape[-1])).sum(axis=-1)


def sph_jl(l: int, x: np.ndarray) -> np.ndarray:
    """Spherical Bessel j_l (l ≤ 4) to full float64 precision.

    The closed trigonometric forms (j₁ = sin x/x² − cos x/x, ...) suffer
    catastrophic cancellation for small x — worse with rising l (j₃ cancels
    five digits at x = 0.5). Below SERIES_X use the ascending power series,
    converged to machine precision; above, the trig forms are stable.
    (scipy.special.spherical_jn is only ~1e-9 accurate in this regime, so we
    do not use it.)
    """
    if not 0 <= l <= 4:
        raise ValueError(f"l={l} out of supported range 0..4")
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)

    small = np.abs(x) < SERIES_X
    xs = x[small]
    x2 = xs * xs
    term = xs**l / DOUBLE_FACTORIAL[l]
    acc = term.copy()
    for k in range(1, SERIES_TERMS):
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

    def _kernel(qc: np.ndarray) -> np.ndarray:
        jl = sph_jl(l, np.outer(qc, r))  # (nq, n)
        return simpson(jl * g, rab)

    if _SBT_THREADS <= 1 or q.size < 2 * _SBT_CHUNK_MIN:
        return _kernel(q)
    # FIXED-size chunks (not a fixed count): each holds ~_SBT_CHUNK_MIN transforms,
    # and the pool runs at most _SBT_THREADS at once, so peak memory is bounded by
    # _SBT_THREADS × chunk regardless of how large q gets. A fixed *count* would
    # make each chunk q/N — huge for a dense high-ecutrho grid — and N of them live
    # at once blows up memory. np.array_split preserves order, so concatenation is
    # bit-identical to the serial transform (pure parallelization, not an approx).
    nchunks = -(-q.size // _SBT_CHUNK_MIN)  # ceil(q.size / chunk)
    chunks = np.array_split(q, nchunks)
    with ThreadPoolExecutor(max_workers=_SBT_THREADS) as ex:
        return np.concatenate(list(ex.map(_kernel, chunks)))
