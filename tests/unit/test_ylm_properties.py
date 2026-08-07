"""Property-based contracts for the real solid/spherical harmonics ``ylm_all``.

``tests/unit/test_ylm.py`` checks orthonormality on a quadrature grid, parity at
one fixed lmax, and the zero-vector limit. The tests here generalize the pointwise
identities over Hypothesis-generated directions **and** random lmax, so a bug that
only bites at, say, l = 5 or for a single component cannot hide behind the one
hardcoded lmax:

- parity:            Y_lm(−n) = (−1)^l Y_lm(n)
- scale invariance:  Y_lm(c·n) = Y_lm(n) for c > 0 (harmonics see direction only)
- zero-vector limit: only the l=0 channel survives, at the constant C00

All are pure and fast (a closed-form evaluation on a handful of points).
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from gradwave.core.ylm import C00, ylm_all

_LMAX = st.integers(0, 4)  # ylm_all supports lmax <= 4 (raises above that)


def _parity_signs(lmax: int) -> torch.Tensor:
    """(-1)^l laid out over the (lmax+1)^2 harmonic channels."""
    return torch.tensor(
        [(-1.0) ** l for l in range(lmax + 1) for _ in range(2 * l + 1)],
        dtype=torch.float64,
    )


@st.composite
def directions(draw):
    """A small batch of (k, 3) directions with norm bounded away from zero.

    The near-zero ball is excluded (``assume``) because the direction there is
    ill-defined — that degenerate limit is the separate zero-vector test.
    """
    k = draw(st.integers(1, 8))
    v = np.array(
        [[draw(st.floats(-5.0, 5.0)) for _ in range(3)] for _ in range(k)],
        dtype=np.float64,
    )
    assume(np.all(np.linalg.norm(v, axis=1) > 0.5))
    return torch.tensor(v, dtype=torch.float64)


@settings(max_examples=60, deadline=None)
@given(lmax=_LMAX, n=directions())
def test_ylm_parity(lmax, n):
    """Y_lm(−n) = (−1)^l Y_lm(n) across every channel, for any lmax."""
    yp = ylm_all(lmax, n)
    ym = ylm_all(lmax, -n)
    assert torch.allclose(ym, yp * _parity_signs(lmax), atol=1e-12)


@settings(max_examples=60, deadline=None)
@given(lmax=_LMAX, n=directions(), c=st.floats(0.1, 10.0))
def test_ylm_scale_invariance(lmax, n, c):
    """Real spherical harmonics depend only on n/|n|: Y_lm(c·n) = Y_lm(n), c>0."""
    y1 = ylm_all(lmax, n)
    y2 = ylm_all(lmax, c * n)
    assert torch.allclose(y1, y2, atol=1e-12, rtol=1e-11)


@settings(max_examples=30, deadline=None)
@given(lmax=_LMAX, k=st.integers(1, 4))
def test_ylm_zero_vector(lmax, k):
    """At n = 0 only the l=0 channel is nonzero (the constant C00), any lmax."""
    y = ylm_all(lmax, torch.zeros(k, 3, dtype=torch.float64))
    assert torch.allclose(y[:, 0], torch.full((k,), C00, dtype=torch.float64))
    if y.shape[1] > 1:
        assert torch.all(y[:, 1:] == 0)
