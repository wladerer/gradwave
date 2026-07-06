import math

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.xc.lda_pw92 import LDA_PW92, eps_c_pw92, eps_x_lda
from gradwave.core.xc.pbe import PBE


def rho_from_rs(rs: float) -> float:
    """ρ [e/Å³] for a given Wigner–Seitz rs [bohr]."""
    rho_au = 3.0 / (4.0 * math.pi * rs**3)
    return rho_au / BOHR_ANG**3


def test_slater_exchange_value():
    # ε_x(rs) = −(3/4)(3/π)^{1/3} ρ^{1/3} = −0.458165.../rs Ha
    for rs in (1.0, 2.0, 5.0):
        rho_au = torch.tensor(3.0 / (4.0 * math.pi * rs**3), dtype=torch.float64)
        assert np.isclose(eps_x_lda(rho_au).item(), -0.4581652932831429 / rs, rtol=1e-12)


def test_pw92_values():
    # Regression pins (this implementation, Eq. 10 parameters, 1e-9) plus a
    # loose literature check (~1e-4) against commonly quoted PW92 numbers.
    pinned = {1.0: -0.059773864, 2.0: -0.044759590, 5.0: -0.028216261, 10.0: -0.018572298}
    literature = {1.0: -0.0598, 2.0: -0.0448, 5.0: -0.0282, 10.0: -0.0186}
    for rs, ref in pinned.items():
        rho_au = torch.tensor(3.0 / (4.0 * math.pi * rs**3), dtype=torch.float64)
        val = eps_c_pw92(rho_au).item()
        assert np.isclose(val, ref, atol=1e-9), rs
        assert np.isclose(val, literature[rs], atol=1e-4), rs


def test_pbe_uniform_limit_is_lda():
    rho = torch.tensor([rho_from_rs(r) for r in (0.8, 1.5, 3.0, 6.0)], dtype=torch.float64)
    sigma = torch.zeros_like(rho)
    e_pbe = PBE().energy_density(rho, sigma)
    e_lda = LDA_PW92().energy_density(rho)
    assert torch.allclose(e_pbe, e_lda, rtol=1e-10)


def test_pbe_enhancement_bounds():
    # exchange enhancement 1 ≤ F_x < 1.804 ⇒ e_x(PBE) more negative than LDA-x,
    # bounded by the Lieb-Oxford-motivated kappa limit
    rho = torch.full((5,), rho_from_rs(2.0), dtype=torch.float64)
    sigma = torch.tensor([0.0, 1e-4, 1e-2, 1.0, 100.0], dtype=torch.float64)
    rho_au = rho * BOHR_ANG**3
    ex_lda = (eps_x_lda(rho_au) * rho * HARTREE_EV).numpy()
    # isolate exchange by comparing full PBE minus (LDA-c + PBE-H correction)… simpler:
    # check monotone decrease with sigma and the 1.804 cap on the total exchange part
    e = PBE().energy_density(rho, sigma).numpy()
    assert all(e[i + 1] <= e[i] + 1e-15 for i in range(4))  # more binding with gradient
    # crude cap: |e_pbe| can't exceed |LDA exchange|·1.804 + |LDA corr|·(reasonable)
    assert abs(e[-1]) < 1.804 * abs(ex_lda[0]) + 0.1 * abs(ex_lda[0])


def test_vxc_via_autograd_matches_finite_difference():
    # v_xc = ∂e_xc/∂ρ for LDA (per grid point); autograd vs central differences
    rho = torch.tensor([0.01, 0.05, 0.2], dtype=torch.float64, requires_grad=True)
    xc = LDA_PW92()
    e = xc.energy_density(rho).sum()
    (v,) = torch.autograd.grad(e, rho)
    h = 1e-6
    for i in range(3):
        rp = rho.detach().clone()
        rm = rho.detach().clone()
        rp[i] += h
        rm[i] -= h
        fd = (xc.energy_density(rp).sum() - xc.energy_density(rm).sum()).item() / (2 * h)
        assert abs(fd - v[i].item()) < 1e-6 * max(1.0, abs(fd))


def test_gradcheck_lda_and_pbe():
    gen = torch.Generator().manual_seed(4)
    rho = (0.02 + 0.2 * torch.rand(6, generator=gen, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda r: LDA_PW92().energy_density(r).sum(), (rho,))
    sigma = (0.01 * torch.rand(6, generator=gen, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda r, s: PBE().energy_density(r, s).sum(), (rho, sigma), atol=1e-8
    )
    # second derivatives exist (needed for M4 Hessian-vector products)
    assert torch.autograd.gradgradcheck(lambda r: LDA_PW92().energy_density(r).sum(), (rho,))
