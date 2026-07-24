"""Unit tests for the Birch-Murnaghan fitter (postscf.eos), no SCF involved.

The physics (B0 of a real crystal vs WIEN2k) is exercised in
tests/integration/test_eos.py; here we only check the fitter recovers known
parameters, agrees with ASE's independent implementation, and validates input.
"""

import numpy as np
import pytest

from gradwave.postscf.eos import (
    EV_A3_TO_GPA,
    birch_murnaghan,
    delta_value,
    fit_bm3,
)


def test_bm3_fit_recovers_known_parameters():
    # synthesize a curve with known (e0, v0, b0, b0') and fit it back
    e0, v0, b0, b0p = -10.5, 20.4, 0.55, 4.3  # b0 in eV/Å³ (~88 GPa)
    v = np.linspace(0.92 * v0, 1.08 * v0, 9)
    e = birch_murnaghan(v, e0, v0, b0, b0p)
    fit = fit_bm3(v, e)
    assert fit.v0 == pytest.approx(v0, abs=1e-6)
    assert fit.b0 == pytest.approx(b0, abs=1e-6)
    assert fit.b0_prime == pytest.approx(b0p, abs=1e-5)
    assert fit.e0 == pytest.approx(e0, abs=1e-6)
    assert fit.b0_GPa == pytest.approx(b0 * EV_A3_TO_GPA, rel=1e-9)
    assert fit.rms_residual_eV < 1e-9  # exact curve → ~machine-precision residual


def test_bm3_matches_ase_birchmurnaghan():
    # independent cross-check: the same (V, E) points fit by ASE's 3rd-order
    # Birch-Murnaghan must give the same V0 and B0 as our fitter
    from ase.eos import EquationOfState

    e0, v0, b0, b0p = -8.0, 16.5, 0.48, 4.6
    v = np.linspace(0.94 * v0, 1.06 * v0, 7)
    e = birch_murnaghan(v, e0, v0, b0, b0p)
    ours = fit_bm3(v, e)
    eos = EquationOfState(v.tolist(), e.tolist(), eos="birchmurnaghan")
    v0_ase, e0_ase, b_ase = eos.fit()  # b_ase in eV/Å³
    assert ours.v0 == pytest.approx(v0_ase, rel=1e-4)
    assert ours.b0 == pytest.approx(b_ase, rel=1e-3)


def test_delta_value_self_is_zero():
    fit = fit_bm3(*_curve(-5.0, 23.9, 0.37, 5.0))
    assert delta_value(fit, fit) == pytest.approx(0.0, abs=1e-9)


def test_delta_value_accepts_raw_tuples():
    # an all-electron reference is a bare (e0, v0, b0, b0p) tuple
    a = fit_bm3(*_curve(-5.0, 20.0, 0.55, 4.0))
    ref = (0.0, 20.6, 0.56, 4.1)  # slightly different equilibrium
    d = delta_value(a, ref)
    assert d > 0.0 and np.isfinite(d)


def test_fit_bm3_rejects_too_few_points():
    with pytest.raises(ValueError, match=">=4"):
        fit_bm3([1.0, 2.0, 3.0], [0.0, -1.0, 0.0])


def test_fit_bm3_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="matching"):
        fit_bm3([1.0, 2.0, 3.0, 4.0], [0.0, -1.0, 0.0])


def _curve(e0, v0, b0, b0p):
    v = np.linspace(0.94 * v0, 1.06 * v0, 7)
    return v, birch_murnaghan(v, e0, v0, b0, b0p)
