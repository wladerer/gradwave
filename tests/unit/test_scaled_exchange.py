"""Scaled-exchange PBE functional for global hybrids."""

import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.hybrid import ScaledExchangePBE


def _rho_sigma(n=64):
    torch.manual_seed(0)
    rho = torch.rand(n, dtype=torch.float64) * 0.5 + 0.05
    sigma = torch.rand(n, dtype=torch.float64) * 0.2
    return rho, sigma


def test_zero_fraction_is_plain_pbe():
    rho, sigma = _rho_sigma()
    e_pbe = PBE().energy_density(rho, sigma)
    e_scaled = ScaledExchangePBE(0.0).energy_density(rho, sigma)
    assert torch.allclose(e_pbe, e_scaled, rtol=1e-14, atol=1e-14)


def test_scaling_removes_a_fraction_of_exchange():
    rho, sigma = _rho_sigma()
    e_pbe = PBE().energy_density(rho, sigma)
    e_full_exx = ScaledExchangePBE(1.0).energy_density(rho, sigma)  # exchange fully removed
    # PBE exchange is negative, so removing it raises the energy density
    assert torch.all(e_full_exx > e_pbe)
    # linear in the fraction: E(α) = E_pbe − α·(E_pbe − E(1))
    e_quarter = ScaledExchangePBE(0.25).energy_density(rho, sigma)
    expected = e_pbe - 0.25 * (e_pbe - e_full_exx)
    assert torch.allclose(e_quarter, expected, rtol=1e-12, atol=1e-12)


def test_bad_fraction_raises():
    with pytest.raises(ValueError):
        ScaledExchangePBE(1.5)
    with pytest.raises(ValueError):
        ScaledExchangePBE(-0.1)
