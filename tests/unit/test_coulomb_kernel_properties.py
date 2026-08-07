"""Property-based contracts for the range-separated Coulomb kernels.

``tests/unit/test_coulomb_kernel.py`` checks the range-separation sum and the
screening limits on fixed |q+G|² grids. The tests here hold the same identities
over Hypothesis-generated grids **and** random screening ω, which is the point:
the exact-exchange / HSE build sweeps a whole box of |q+G|² and a (learnable) ω,
so the contract must hold everywhere, not on one linspace.

- full = short_range + long_range, pointwise away from q+G = 0 (any ω)
- 0 ≤ short_range ≤ full and 0 ≤ long_range ≤ full (screening only removes weight)
- the q+G = 0 cell: full and long_range vanish; short_range is finite at π e²/ω²

All are pure and fast (elementwise kernel arithmetic).
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from gradwave.constants import E2
from gradwave.postscf.coulomb_kernel import coulomb_kernel

# ω [Å⁻¹]: physical range-separation values (HSE uses ~0.1–0.2), kept off zero.
_OMEGA = st.floats(0.05, 2.0)


@st.composite
def qg2_grid(draw):
    """A grid of strictly-positive |q+G|² values (above the q+G=0 special case)."""
    k = draw(st.integers(1, 32))
    vals = [draw(st.floats(0.05, 50.0)) for _ in range(k)]
    return torch.tensor(vals, dtype=torch.float64)


@settings(max_examples=60, deadline=None)
@given(qg2=qg2_grid(), omega=_OMEGA)
def test_range_separation_sums_to_full(qg2, omega):
    """short_range + long_range = full, pointwise, for any grid and any ω."""
    k_full = coulomb_kernel(qg2, "full")
    k_sr = coulomb_kernel(qg2, "short_range", omega)
    k_lr = coulomb_kernel(qg2, "long_range", omega)
    assert torch.allclose(k_sr + k_lr, k_full, rtol=1e-11, atol=1e-12)


@settings(max_examples=60, deadline=None)
@given(qg2=qg2_grid(), omega=_OMEGA)
def test_screened_kernels_bracketed_by_full(qg2, omega):
    """Each screened kernel is non-negative and no larger than the bare kernel."""
    k_full = coulomb_kernel(qg2, "full")
    k_sr = coulomb_kernel(qg2, "short_range", omega)
    k_lr = coulomb_kernel(qg2, "long_range", omega)
    tol = 1e-11 * k_full  # scale-relative slack for the upper bracket
    assert torch.all(k_sr >= -1e-12) and torch.all(k_lr >= -1e-12)
    assert torch.all(k_sr <= k_full + tol) and torch.all(k_lr <= k_full + tol)


@settings(max_examples=40, deadline=None)
@given(omega=_OMEGA)
def test_zero_cell_values(omega):
    """q+G = 0: full/long_range → 0; short_range → π e²/ω² (finite), any ω."""
    qg2 = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)
    assert float(coulomb_kernel(qg2, "full")[0]) == 0.0
    assert float(coulomb_kernel(qg2, "long_range", omega)[0]) == 0.0
    sr0 = float(coulomb_kernel(qg2, "short_range", omega)[0])
    assert sr0 == pytest.approx(math.pi * E2 / omega**2, rel=1e-12)
