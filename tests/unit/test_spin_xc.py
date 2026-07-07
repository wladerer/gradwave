"""Spin XC: exact limits. ζ=0 must reduce to the unpolarized functionals to
machine precision; full polarization follows the exchange spin-scaling and
the PW92 ε_c(rs, 1) branch."""

import math

import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92, eps_x_lda
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import _EC1, LSDA_PW92, SpinPBE, _g_pw92, eps_c_pw92_spin


def grids():
    gen = torch.Generator().manual_seed(9)
    rho = 0.02 + 0.3 * torch.rand(8, generator=gen, dtype=torch.float64)
    sigma = 0.05 * torch.rand(8, generator=gen, dtype=torch.float64)
    return rho, sigma


def test_lsda_unpolarized_limit():
    rho, _ = grids()
    e_spin = LSDA_PW92().energy_density(rho / 2, rho / 2)
    e_ref = LDA_PW92().energy_density(rho)
    assert torch.allclose(e_spin, e_ref, rtol=1e-12)


def test_spin_pbe_unpolarized_limit():
    rho, sigma = grids()
    # ρσ = ρ/2 ⇒ σ_σσ = σ/4
    e_spin = SpinPBE().energy_density(rho / 2, rho / 2, sigma / 4, sigma / 4, sigma)
    e_ref = PBE().energy_density(rho, sigma)
    assert torch.allclose(e_spin, e_ref, rtol=1e-12)


def test_full_polarization_exchange_scaling():
    rho, _ = grids()
    zero = torch.full_like(rho, 1e-15)
    e = LSDA_PW92().energy_density(rho, zero)
    # E_x[ρ,0] = ½E_x[2ρ] ⇒ ε_x = 2^{1/3}·ε_x_unpol(ρ)
    from gradwave.constants import HARTREE_EV
    from gradwave.core.xc.base import to_au

    rho_au = to_au(rho)
    ex_expect = 2.0 ** (1.0 / 3.0) * eps_x_lda(rho_au)
    ec_expect = eps_c_pw92_spin(rho_au, torch.ones_like(rho))
    e_expect = rho * (ex_expect + ec_expect) * HARTREE_EV
    assert torch.allclose(e, e_expect, rtol=1e-9)


def test_pw92_polarized_branch():
    # ζ=1 ⇒ ε_c = G(_EC1) exactly
    rho_au = torch.tensor([3.0 / (4.0 * math.pi * 2.0**3)], dtype=torch.float64)
    rs = torch.tensor([2.0], dtype=torch.float64)
    ec = eps_c_pw92_spin(rho_au, torch.ones_like(rho_au))
    assert torch.allclose(ec, _g_pw92(rs, _EC1), rtol=1e-10)


def test_gradcheck_spin_functionals():
    gen = torch.Generator().manual_seed(3)
    ru = (0.02 + 0.2 * torch.rand(5, generator=gen, dtype=torch.float64)).requires_grad_(True)
    rd = (0.03 + 0.15 * torch.rand(5, generator=gen, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda a, b: LSDA_PW92().energy_density(a, b).sum(), (ru, rd), atol=1e-8)
    suu = (0.01 * torch.rand(5, generator=gen, dtype=torch.float64)).requires_grad_(True)
    sdd = (0.01 * torch.rand(5, generator=gen, dtype=torch.float64)).requires_grad_(True)
    stt = (0.04 * torch.rand(5, generator=gen, dtype=torch.float64) + 0.02).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda a, b, x, y, z: SpinPBE().energy_density(a, b, x, y, z).sum(),
        (ru, rd, suu, sdd, stt), atol=1e-7)
