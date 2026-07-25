"""Directional Poisson ratio (postscf.elastic), no SCF.

The isotropic limit pins the compliance-unfold factor convention and the einsum
contraction against a closed form. The cubic-metal cases check that the sphere
scan finds the known auxetic minimum of FCC Cu along ⟨110⟩ and that covalent Si
stays positive everywhere.
"""

import numpy as np
import pytest

from gradwave.postscf.elastic import (
    compliance_tensor,
    directional_poisson,
    min_directional_poisson,
    moduli_from_cij,
)


def _cubic_c(c11, c12, c44):
    c = np.zeros((6, 6))
    for i in range(3):
        c[i, i] = c11
        c[3 + i, 3 + i] = c44
    for i, j in ((0, 1), (0, 2), (1, 2)):
        c[i, j] = c[j, i] = c12
    return c


def _isotropic_c(k, g):
    # isotropic stiffness in cubic Voigt form: C11 = K + 4G/3, C12 = K - 2G/3, C44 = G
    return _cubic_c(k + 4.0 * g / 3.0, k - 2.0 * g / 3.0, g)


def _rand_perp_pair(rng):
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    m = rng.normal(size=3)
    m -= m.dot(n) * n
    m /= np.linalg.norm(m)
    return n, m


def _is_110_type(v, atol=0.03):
    # ⟨110⟩: sorted |components| ≈ [0, 1/√2, 1/√2]
    s = np.sort(np.abs(_unit(v)))
    return np.allclose(s, [0.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], atol=atol)


def _is_001_or_1m10(v, atol=0.03):
    s = np.sort(np.abs(_unit(v)))
    is_001 = np.allclose(s, [0.0, 0.0, 1.0], atol=atol)
    return is_001 or _is_110_type(v, atol)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_compliance_tensor_matches_isotropic_closed_form():
    # isotropic S_ijkl = -(ν/E) δ_ij δ_kl + ((1+ν)/2E)(δ_ik δ_jl + δ_il δ_jk)
    k, g = 130.0, 47.0
    st = compliance_tensor(_isotropic_c(k, g))
    e = 9.0 * k * g / (3.0 * k + g)
    nu = (3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g))
    d = np.eye(3)
    ref = -(nu / e) * np.einsum("ij,kl->ijkl", d, d) + ((1.0 + nu) / (2.0 * e)) * (
        np.einsum("ik,jl->ijkl", d, d) + np.einsum("il,jk->ijkl", d, d)
    )
    assert np.allclose(st, ref, atol=1e-12)


def test_directional_poisson_isotropic_is_constant_and_matches_hill():
    k, g = 100.0, 60.0
    c = _isotropic_c(k, g)
    nu_ana = (3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g))
    assert moduli_from_cij(c).poisson == pytest.approx(nu_ana, abs=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(8):
        n, m = _rand_perp_pair(rng)
        assert directional_poisson(c, n, m) == pytest.approx(nu_ana, abs=1e-10)


def test_cu_is_auxetic_along_110():
    # experimental FCC Cu: C11=169, C12=122, C44=75 GPa
    nu_min, n_hat, m_hat = min_directional_poisson(_cubic_c(169.0, 122.0, 75.0))
    assert nu_min < 0.0
    # the Hill isotropic average misses it
    assert moduli_from_cij(_cubic_c(169.0, 122.0, 75.0)).poisson > 0.0
    assert _is_110_type(n_hat), n_hat
    assert _is_001_or_1m10(m_hat), m_hat


def test_si_is_not_auxetic():
    # covalent cubic Si: C11=166, C12=64, C44=80 GPa
    nu_min, _, _ = min_directional_poisson(_cubic_c(166.0, 64.0, 80.0))
    assert nu_min > 0.0
