"""Unit tests for the elastic-constants numerics (postscf.elastic), no SCF.

The physics (Si C11/C12/C44 vs literature, K vs the EOS bulk modulus) is in
tests/integration/test_elastic.py; here we check the Voigt bookkeeping, the FD
driver against a linear stress model, the Voigt–Reuss–Hill averages against
known Si polycrystalline values, and the Born stability test.
"""

import numpy as np
import pytest

from gradwave.postscf.elastic import (
    EV_A3_TO_GPA,
    elastic_tensor,
    is_mechanically_stable,
    moduli_from_cij,
    stress_to_voigt,
    voigt_strain_tensor,
)


def _cubic_c(c11, c12, c44):
    c = np.zeros((6, 6))
    for i in range(3):
        c[i, i] = c11
        c[3 + i, 3 + i] = c44
    for i, j in ((0, 1), (0, 2), (1, 2)):
        c[i, j] = c[j, i] = c12
    return c


def test_moduli_match_known_silicon():
    # experimental Si single-crystal constants → tabulated polycrystalline moduli
    m = moduli_from_cij(_cubic_c(165.7, 63.9, 79.6))
    # cubic: Voigt and Reuss bulk moduli coincide, both = (C11+2C12)/3
    assert m.bulk_voigt == pytest.approx((165.7 + 2 * 63.9) / 3, rel=1e-9)
    assert m.bulk_reuss == pytest.approx(m.bulk_voigt, rel=1e-9)
    assert m.bulk_hill == pytest.approx(97.83, abs=0.1)
    assert m.shear_hill == pytest.approx(66.5, abs=0.5)
    assert m.young == pytest.approx(162.7, abs=1.0)   # Si polycrystalline E ~ 160 GPa
    assert m.poisson == pytest.approx(0.223, abs=0.005)


def test_fd_driver_recovers_linear_model_with_shear_convention():
    # a linear σ = C·ε model (engineering shear) must be recovered exactly,
    # which pins the factor-of-2 shear convention in voigt_strain_tensor
    c_true = _cubic_c(165.7, 63.9, 79.6)

    def stress_model(eps):
        ev = np.array([eps[0, 0], eps[1, 1], eps[2, 2],
                       2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]])
        sig_v = (c_true / EV_A3_TO_GPA) @ ev  # eV/Å³
        s = np.zeros((3, 3))
        for k, (a, b) in enumerate([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]):
            s[a, b] = s[b, a] = sig_v[k]
        return s

    c_fd = elastic_tensor(stress_model, h=0.004)
    assert np.allclose(c_fd, c_true, atol=1e-6)


def test_voigt_strain_tensor_shear_is_half():
    # ε_voigt_4 = 2 ε_yz ⇒ a unit Voigt shear h gives tensor entry h/2
    eps = voigt_strain_tensor(3, 0.01)  # yz
    assert eps[1, 2] == pytest.approx(0.005) and eps[2, 1] == pytest.approx(0.005)
    assert eps[0, 0] == 0.0
    eps_xx = voigt_strain_tensor(0, 0.01)
    assert eps_xx[0, 0] == pytest.approx(0.01)


def test_stress_to_voigt_order():
    s = np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])
    assert np.allclose(stress_to_voigt(s), [1, 2, 3, 4, 5, 6])


def test_stability_flags_negative_definite():
    assert is_mechanically_stable(_cubic_c(165.7, 63.9, 79.6))
    # C12 > C11 violates the cubic Born criterion C11 > |C12|
    assert not is_mechanically_stable(_cubic_c(50.0, 120.0, 79.6))
