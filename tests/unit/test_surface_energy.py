"""Fiorentini-Methfessel surface energy fit (postscf.surface_energy), no SCF.

The physics (a real metal's γ vs experiment) needs a slab-thickness SCF sweep and
is a DEFERRED integration test; here we only check the slope method recovers a
known γ from a synthetic E_slab(N) = 2γA + N·E_bulk line and validates input.
"""

import numpy as np
import pytest

from gradwave.constants import EV_A2_TO_JM2
from gradwave.postscf.surface_energy import (
    SurfaceEnergyFit,
    surface_energy_fm,
    surface_energy_subtraction,
)


def test_fm_recovers_known_gamma_and_bulk():
    # synthetic sweep with known surface energy and per-layer bulk energy
    gamma, e_bulk, area = 0.075, -6.4, 40.0  # eV/Å², eV/layer, Å²
    n = np.array([3, 4, 5, 6, 7, 8], dtype=float)
    e_slab = 2 * gamma * area + n * e_bulk  # exact FM line
    fit = surface_energy_fm(n, e_slab, area)
    assert isinstance(fit, SurfaceEnergyFit)
    assert fit.gamma == pytest.approx(gamma, abs=1e-10)
    assert fit.e_bulk == pytest.approx(e_bulk, abs=1e-10)
    assert fit.intercept == pytest.approx(2 * gamma * area, abs=1e-9)
    assert fit.rms_residual_eV < 1e-9  # exact line → machine-precision residual
    assert fit.gamma_Jm2 == pytest.approx(gamma * EV_A2_TO_JM2, rel=1e-12)
    assert fit.gamma_mJm2 == pytest.approx(gamma * EV_A2_TO_JM2 * 1e3, rel=1e-12)


def test_fm_one_sided_slab():
    # a one-sided construction divides by a single surface
    gamma, e_bulk, area = 0.05, -3.1, 25.0
    n = np.arange(4, 11, dtype=float)
    e_slab = 1 * gamma * area + n * e_bulk
    fit = surface_energy_fm(n, e_slab, area, n_surfaces=1)
    assert fit.gamma == pytest.approx(gamma, abs=1e-10)


def test_fm_slope_gives_bulk_independent_of_thickness_offset():
    # the slope method is invariant to where N starts (only the span matters)
    gamma, e_bulk, area = 0.09, -5.0, 33.3
    for start in (1, 5, 20):
        n = np.arange(start, start + 5, dtype=float)
        e = 2 * gamma * area + n * e_bulk
        fit = surface_energy_fm(n, e, area)
        assert fit.gamma == pytest.approx(gamma, abs=1e-9)
        assert fit.e_bulk == pytest.approx(e_bulk, abs=1e-9)


def test_fm_residual_flags_nonlinearity():
    # quantum-size oscillation on top of the line → non-zero residual
    gamma, e_bulk, area = 0.07, -4.2, 30.0
    n = np.arange(3, 12, dtype=float)
    e = 2 * gamma * area + n * e_bulk + 0.01 * np.cos(n)
    fit = surface_energy_fm(n, e, area)
    assert fit.rms_residual_eV > 1e-3


def test_subtraction_matches_fm_on_exact_line():
    # given the true per-layer bulk energy, the direct formula agrees with FM
    gamma, e_bulk, area = 0.11, -7.7, 50.0
    n = np.arange(4, 9, dtype=float)
    e = 2 * gamma * area + n * e_bulk
    fit = surface_energy_fm(n, e, area)
    for ni, ei in zip(n, e, strict=True):
        g = surface_energy_subtraction(ei, ni, fit.e_bulk, area)
        assert g == pytest.approx(gamma, abs=1e-9)


def test_input_validation():
    with pytest.raises(ValueError, match="at least two"):
        surface_energy_fm([3.0], [1.0], 10.0)
    with pytest.raises(ValueError, match="distinct thicknesses"):
        surface_energy_fm([4.0, 4.0], [1.0, 2.0], 10.0)
    with pytest.raises(ValueError, match="equal length"):
        surface_energy_fm([3.0, 4.0, 5.0], [1.0, 2.0], 10.0)
    with pytest.raises(ValueError, match="area must be positive"):
        surface_energy_fm([3.0, 4.0], [1.0, 2.0], 0.0)
    with pytest.raises(ValueError, match="n_surfaces"):
        surface_energy_fm([3.0, 4.0], [1.0, 2.0], 10.0, n_surfaces=0)
