"""The NMR observable layer: EFG V_zz (eV/Å²) → quadrupolar coupling C_Q (MHz)."""

from __future__ import annotations

import math

import pytest

from gradwave.flapw.nmr import NUCLEAR_Q, quadrupolar_coupling


def test_cq_conversion_constant():
    """C_Q[MHz] = 2.4180·Q[barn]·V_zz[eV/Å²] (the 234.965 a.u. formula in eV/Å² units)."""
    out = quadrupolar_coupling(1.0, 0.0, "17O")
    q = NUCLEAR_Q["17O"][0]
    assert math.isclose(out["C_Q_MHz"], 2.4180 * q * 1.0, rel_tol=1e-6)
    assert math.isclose(out["abs_C_Q_MHz"], abs(2.4180 * q), rel_tol=1e-6)


def test_cq_scales_linearly_with_vzz():
    """C_Q is linear in V_zz."""
    a = quadrupolar_coupling(2.0, 0.0, "27Al")["C_Q_MHz"]
    b = quadrupolar_coupling(6.0, 0.0, "27Al")["C_Q_MHz"]
    assert math.isclose(b, 3.0 * a, rel_tol=1e-9)


def test_nu_q_spin_formula():
    """ν_Q = 3 C_Q / [2 I (2I−1)] for spin I (here ¹⁷O, I=5/2)."""
    out = quadrupolar_coupling(5.0, 0.0, "17O")
    spin = NUCLEAR_Q["17O"][1]
    expected = 3.0 * out["abs_C_Q_MHz"] / (2.0 * spin * (2.0 * spin - 1.0))
    assert math.isclose(out["nu_Q_MHz"], expected, rel_tol=1e-9)


def test_unknown_isotope_raises():
    with pytest.raises(KeyError):
        quadrupolar_coupling(1.0, 0.0, "99Zz")
