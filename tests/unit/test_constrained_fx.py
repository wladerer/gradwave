"""The constrained exchange enhancement factor (the functional-learning / SR
ansatz) satisfies its exact constraints BY CONSTRUCTION, and reduces to PBE.

These are the guardrails that make the searchable form "weird PBE, never garbage":
a trained (or SR-discovered) interior term cannot break the UEG limit or the
Lieb-Oxford bound. See docs/design/learned-functional.md.
"""

import torch

from gradwave.core.xc._pbe_kernels import KAPPA, MU, pbe_enhancement
from gradwave.core.xc.constrained_fx import ConstrainedFx, constrained_enhancement


def test_reduces_to_pbe_at_c2_zero():
    s2 = torch.tensor([0.0, 0.1, 0.5, 2.0, 50.0, 1e4])
    assert torch.allclose(
        constrained_enhancement(s2, KAPPA, MU, 0.0),
        pbe_enhancement(s2, KAPPA, MU),
        atol=1e-12,
    )


def test_ueg_limit_exact():
    # F_x(s=0) = 1 for any (κ, μ, c₂) — the uniform-electron-gas limit
    for c2 in (0.0, 0.05, 1.0):
        f0 = constrained_enhancement(torch.zeros(1), KAPPA, MU, c2)
        assert abs(float(f0) - 1.0) < 1e-12


def test_lieb_oxford_bound_holds_for_c2_nonneg():
    # F_x < 1 + κ (Lieb-Oxford) for all s and any c₂ ≥ 0
    s2 = torch.logspace(-3, 6, 500, dtype=torch.float64)
    for c2 in (0.0, 0.01, 0.1, 5.0):
        f = constrained_enhancement(s2, KAPPA, MU, c2)
        assert float(f.max()) < 1.0 + KAPPA  # strict Lieb-Oxford bound (fp64)
        assert float(f.min()) >= 1.0 - 1e-9  # monotone increasing from the UEG limit


def test_c2_has_effect_and_stays_bounded():
    s2 = torch.tensor([1.0, 4.0])
    base = constrained_enhancement(s2, KAPPA, MU, 0.0)
    bumped = constrained_enhancement(s2, KAPPA, MU, 0.2)
    assert not torch.allclose(base, bumped)  # the extra knob does something
    assert float(bumped.max()) < 1.0 + KAPPA  # ...without breaking the bound


def test_class_energy_density_matches_pbe_at_init():
    from gradwave.core.xc.pbe import PBE

    rho = torch.tensor([0.05, 0.2, 1.0], dtype=torch.float64)
    sigma = torch.tensor([1e-3, 1e-2, 5e-2], dtype=torch.float64)
    e_pbe = PBE().energy_density(rho, sigma)
    e_cfx = ConstrainedFx().energy_density(rho, sigma)  # c₂ ≈ 0 at init
    assert torch.allclose(e_pbe, e_cfx, atol=1e-6)
