"""Cross-step density-extrapolation coefficient solve (``relax.extrapolation``).

The nested relaxation seeds each SCF from an extrapolation of the previous
geometries' converged states. The coefficients are chosen QE-style: least-squares
match the new positions from the last two (linear) or three (quadratic)
geometries, then apply the same combination to the stored remainder densities.
These tests pin the coefficient solve — the exact recovery on a consistent system
and the graceful fallback on a degenerate one — since the density combination is
only ever as good as the coefficients driving it.
"""

import torch

from gradwave.calculator import _extrapolation_coeffs
from gradwave.dtypes import RDTYPE


def _vecs(n=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, generator=g, dtype=RDTYPE) for _ in range(3)]


def test_quadratic_recovers_analytic_coeffs():
    # positions t-2, t-1, t and the two difference vectors that span the fit
    p_tm2, p_tm1, p_t = _vecs(seed=1)
    a1 = p_t - p_tm1
    a2 = p_tm1 - p_tm2
    alpha, beta = 0.73, -0.41
    d = alpha * a1 + beta * a2  # new displacement lies exactly in the span
    c = _extrapolation_coeffs(d, [a1, a2])
    assert abs(c[0] - alpha) < 1e-10
    assert abs(c[1] - beta) < 1e-10


def test_linear_recovers_analytic_coeff():
    p_tm2, p_tm1, p_t = _vecs(seed=2)
    a1 = p_t - p_tm1
    alpha = 1.37
    d = alpha * a1
    c = _extrapolation_coeffs(d, [a1])
    assert len(c) == 1
    assert abs(c[0] - alpha) < 1e-10


def test_collinear_history_drops_to_linear():
    # a1 and a2 collinear (a repeated *direction*): the 2x2 Gram is singular, so
    # the quadratic term drops out and the scalar (linear) fit stands.
    _p2, _p1, p_t = _vecs(seed=3)
    a1 = p_t
    a2 = 2.0 * p_t
    d = 3.0 * p_t
    c = _extrapolation_coeffs(d, [a1, a2])
    assert c[1] == 0.0
    assert abs(c[0] - 3.0) < 1e-10


def test_repeated_geometry_falls_back_to_reuse():
    # a vanishing most-recent difference (the geometry did not move) leaves the
    # solve fully degenerate: every coefficient is zero, i.e. plain reuse.
    _p2, _p1, p_t = _vecs(seed=4)
    zero = torch.zeros_like(p_t)
    c = _extrapolation_coeffs(zero.clone(), [zero.clone()])
    assert c == [0.0]
    c2 = _extrapolation_coeffs(zero.clone(), [zero.clone(), zero.clone()])
    assert c2 == [0.0, 0.0]
