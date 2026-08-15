"""Charged-cell Makov-Payne corrections (pure electrostatics)."""

import numpy as np

from gradwave.constants import E2
from gradwave.postscf.charged import madelung_energy, makov_payne_correction

ALPHA_M = 2.8372975  # simple-cubic Madelung constant


def test_madelung_cubic_reproduces_alpha():
    """A unit charge + jellium in a cubic cell gives E_M = −α_M e²/(2L), and the
    Ewald sum reproduces the textbook simple-cubic Madelung constant."""
    for L in (8.0, 12.0, 20.0):
        e_m = madelung_energy(L * np.eye(3))
        expect = -ALPHA_M * E2 / (2 * L)
        assert abs(e_m - expect) < 1e-4, f"L={L}: {e_m} vs {expect}"


def test_makov_payne_sign_and_scaling():
    """The correction is positive (adds back the spurious lowering) and scales as
    q²/L and 1/ε."""
    L = 10.0
    c1 = makov_payne_correction(1.0, L * np.eye(3))
    assert c1 > 0
    assert abs(c1 - ALPHA_M * E2 / (2 * L)) < 1e-4
    # q² scaling
    assert abs(makov_payne_correction(2.0, L * np.eye(3)) - 4 * c1) < 1e-9
    # 1/L scaling (double L → half correction)
    assert abs(makov_payne_correction(1.0, 20.0 * np.eye(3)) - c1 / 2) < 1e-4
    # 1/ε screening
    assert abs(makov_payne_correction(1.0, L * np.eye(3), epsilon=4.0) - c1 / 4) < 1e-9


def test_madelung_nonorthogonal():
    """madelung_energy works for a non-cubic cell (fcc primitive), where there is
    no closed-form α_M — it just has to be finite and negative and scale as 1/L."""
    fcc = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    e1 = madelung_energy(6.0 * fcc)
    e2 = madelung_energy(12.0 * fcc)
    assert e1 < 0 and e2 < 0
    assert abs(e1 / e2 - 2.0) < 1e-4  # 1/L scaling
