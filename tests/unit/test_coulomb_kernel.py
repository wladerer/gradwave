"""Range-separated Coulomb kernels for exchange (Layer C).

The erfc/erf range separation must satisfy K_short + K_long = K_full pointwise
away from q+G = 0, and the screened (short-range) kernel must stay finite at
q+G = 0 where the bare kernel diverges — the property that lets HSE-style
screened hybrids skip the singularity correction.
"""

import math

import pytest
import torch

from gradwave.constants import E2
from gradwave.postscf.coulomb_kernel import coulomb_kernel


def test_range_separation_sums_to_full_away_from_zero():
    qg2 = torch.linspace(0.05, 20.0, 64, dtype=torch.float64)
    omega = torch.tensor(0.3, dtype=torch.float64)
    k_full = coulomb_kernel(qg2, "full")
    k_sr = coulomb_kernel(qg2, "short_range", omega)
    k_lr = coulomb_kernel(qg2, "long_range", omega)
    assert torch.allclose(k_sr + k_lr, k_full, rtol=1e-12, atol=1e-12)


def test_zero_cell_handling():
    qg2 = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)
    omega = torch.tensor(0.3, dtype=torch.float64)
    assert float(coulomb_kernel(qg2, "full")[0]) == 0.0
    assert float(coulomb_kernel(qg2, "long_range", omega)[0]) == 0.0
    # short-range limit at q+G→0 is π e²/ω², finite
    sr0 = float(coulomb_kernel(qg2, "short_range", omega)[0])
    assert sr0 == pytest.approx(math.pi * E2 / 0.3 ** 2, rel=1e-12)


def test_screening_limits():
    qg2 = torch.linspace(0.1, 10.0, 32, dtype=torch.float64)
    k_full = coulomb_kernel(qg2, "full")
    # ω → 0: the erf crossover recedes to infinity, so short-range → full 1/|q+G|²
    k_sr_unscreened = coulomb_kernel(qg2, "short_range", torch.tensor(1e-4, dtype=torch.float64))
    assert torch.allclose(k_sr_unscreened, k_full, rtol=1e-6)
    # ω → ∞: erfc(ωr) collapses to the origin, so the short-range part vanishes
    k_sr_screened = coulomb_kernel(qg2, "short_range", torch.tensor(1e3, dtype=torch.float64))
    assert float(k_sr_screened.abs().max()) < 1e-6 * float(k_full.abs().max())


def test_kernel_differentiable_in_omega():
    qg2 = torch.linspace(0.1, 8.0, 16, dtype=torch.float64)
    omega = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    coulomb_kernel(qg2, "short_range", omega).sum().backward()
    assert omega.grad is not None and torch.isfinite(omega.grad).all()


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        coulomb_kernel(torch.ones(3), "screened")
    with pytest.raises(ValueError):
        coulomb_kernel(torch.ones(3), "short_range")  # missing omega
