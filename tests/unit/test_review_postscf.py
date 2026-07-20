"""Unit coverage for the postscf code-review fixes (pure logic, no SCF — runs
in the fast gate):

- hubbard_u._assemble_u now guards against inequivalent two-site manifolds
- magnetism uses the shared KB_EV and a single MOMENT_TOL_MUB constant
- phonons.gamma_frequencies derives its cm⁻¹ constant from gradwave.constants
"""

import math

import numpy as np
import pytest
import torch

from gradwave.constants import KB_EV
from gradwave.postscf import magnetism
from gradwave.postscf.hessian import SQRT_EV_AMU_ANG2_TO_CM1
from gradwave.postscf.hubbard_u import _assemble_u
from gradwave.postscf.phonons import (
    _SQRT_EV_AMU_ANG2_TO_CM1,
    gamma_frequencies,
)


def _sites(l0, l1):
    return [{"atom": 0, "l": l0, "start": 0, "dim": 2 * l0 + 1},
            {"atom": 1, "l": l1, "start": 2 * l0 + 1, "dim": 2 * l1 + 1}]


def test_assemble_u_two_equivalent_sites():
    chi = torch.tensor([-0.30, -0.05], dtype=torch.float64)
    chi0 = torch.tensor([-0.50, -0.02], dtype=torch.float64)
    out = _assemble_u(chi, chi0, site=0, sites=_sites(2, 2),
                      species_of_atom=[3, 3])
    # (χ0⁻¹ − χ⁻¹)_00 of the symmetric [[a,b],[b,a]] reconstruction
    m_chi = np.array([[-0.30, -0.05], [-0.05, -0.30]])
    m_chi0 = np.array([[-0.50, -0.02], [-0.02, -0.50]])
    expect = (np.linalg.inv(m_chi0) - np.linalg.inv(m_chi))[0, 0]
    assert out["U_eV"] == pytest.approx(expect, rel=1e-12)


def test_assemble_u_inequivalent_l_raises():
    chi = torch.tensor([-0.30, -0.05], dtype=torch.float64)
    chi0 = torch.tensor([-0.50, -0.02], dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="different species"):
        _assemble_u(chi, chi0, site=0, sites=_sites(2, 1),
                    species_of_atom=[3, 3])


def test_assemble_u_inequivalent_species_raises():
    chi = torch.tensor([-0.30, -0.05], dtype=torch.float64)
    chi0 = torch.tensor([-0.50, -0.02], dtype=torch.float64)
    with pytest.raises(NotImplementedError):
        _assemble_u(chi, chi0, site=0, sites=_sites(2, 2),
                    species_of_atom=[3, 4])


def test_assemble_u_single_site_scalar():
    chi = torch.tensor([-0.30], dtype=torch.float64)
    chi0 = torch.tensor([-0.50], dtype=torch.float64)
    out = _assemble_u(chi, chi0, site=0, sites=_sites(2, 2)[:1],
                      species_of_atom=[3])
    assert out["U_eV"] == pytest.approx(1.0 / -0.50 - 1.0 / -0.30, rel=1e-12)


def test_magnetism_uses_shared_kb():
    # local truncated KB copy is gone; the module references the CODATA KB_EV
    assert not hasattr(magnetism, "KB")
    assert magnetism.KB_EV == KB_EV


def test_magnetism_moment_tol_single_constant():
    assert magnetism.MOMENT_TOL_MUB == 0.15
    # _classify's default arg is the module constant, not a duplicated literal
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    below = (0.9 * magnetism.MOMENT_TOL_MUB) * torch.stack([z, z])
    assert magnetism._classify(below, torch.linalg.norm(below, dim=-1),
                               None) == "nonmagnetic"


def test_gamma_frequencies_constant_derivation():
    # derived-from-constants value matches the older explicit-SI form to ~13
    # significant digits and the sibling hessian.py copy to ~1e-7 relative
    assert _SQRT_EV_AMU_ANG2_TO_CM1 == pytest.approx(521.4708983725066, rel=1e-12)
    assert _SQRT_EV_AMU_ANG2_TO_CM1 == pytest.approx(
        SQRT_EV_AMU_ANG2_TO_CM1, rel=1e-7)


def test_gamma_frequencies_diagonal_hessian():
    # one atom, isotropic spring k on each Cartesian axis, unit mass
    k = 4.0
    hess = np.zeros((1, 3, 1, 3))
    for i in range(3):
        hess[0, i, 0, i] = k
    freqs = gamma_frequencies(hess, [1.0])
    assert np.allclose(freqs, _SQRT_EV_AMU_ANG2_TO_CM1 * math.sqrt(k))


def test_gamma_frequencies_imaginary_sign():
    # a negative eigenvalue comes back as a negative ("imaginary") frequency
    hess = np.zeros((1, 3, 1, 3))
    hess[0, 0, 0, 0] = -1.0
    freqs = gamma_frequencies(hess, [1.0])
    assert freqs.min() == pytest.approx(-_SQRT_EV_AMU_ANG2_TO_CM1, rel=1e-12)
