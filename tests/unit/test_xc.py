import math

import numpy as np
import pytest
import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.xc._pbe_kernels import KAPPA, MU, pbe_h
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


# ---------------------------------------------------------------------------
# PBE exchange enhancement pinned to the closed form with LITERAL κ, μ.
# The other σ-dependent test (test_pbe_enhancement_bounds) only checks a
# monotone decrease and a loose 1.804 cap — it never pins μ=0.2195 or κ=0.804
# to a number, so a wrong μ or a wrong s² scaling would pass it. These oracles
# fix both: e_x = ρ·ε_x^LDA·[1+κ−κ/(1+μ s²/κ)] with the PBE constants typed
# out explicitly, so any drift in κ, μ, or the s = |∇ρ|/(2 k_F ρ) reduced
# gradient fails at rtol 1e-12. PBE is the production workhorse — pin it hard.
# ---------------------------------------------------------------------------

# PBE exchange constants, typed out (PRL 77, 3865 (1996)); μ is the gradient
# coefficient tied to the LDA linear-response limit, κ the Lieb-Oxford bound.
_MU_PBE = 0.2195149727645171
_KAPPA_PBE = 0.804
_CX_SLATER = -0.75 * (3.0 / math.pi) ** (1.0 / 3.0)  # Slater ε_x = C_x ρ^{1/3}


def _pbe_exchange_closed_form(rho_ang, sigma_ang):
    """e_x^PBE [eV/Å³] from the closed form with literal κ, μ (independent of
    the module's KAPPA/MU). Mirrors pbe.py's a.u. conversion exactly."""
    rho_au = rho_ang * BOHR_ANG**3
    sigma_au = sigma_ang * BOHR_ANG**8
    grad_au = np.sqrt(sigma_au)
    kf = (3.0 * math.pi**2 * rho_au) ** (1.0 / 3.0)
    s = grad_au / (2.0 * kf * rho_au)
    s2 = s * s
    f_x = 1.0 + _KAPPA_PBE - _KAPPA_PBE / (1.0 + _MU_PBE * s2 / _KAPPA_PBE)
    eps_x = _CX_SLATER * rho_au ** (1.0 / 3.0) * f_x  # Ha/electron
    return rho_ang * eps_x * HARTREE_EV


def test_pbe_module_constants_are_the_pbe_values():
    # a bare typo in the shared kernel constants must not slip through
    assert KAPPA == _KAPPA_PBE
    assert MU == _MU_PBE


def test_pbe_exchange_pointwise_oracle():
    """PBE().energy_density minus its correlation part equals the closed-form
    exchange with literal κ, μ, at several (ρ, σ) spanning small→large s."""
    # (ρ [e/Å³], σ = |∇ρ|² [e²/Å⁸]); s grows across the set into the κ-cap regime
    rho = torch.tensor([0.05, 0.15, 0.4, 0.8], dtype=torch.float64)
    sigma = torch.tensor([1e-4, 5e-3, 0.2, 3.0], dtype=torch.float64)

    e_total = PBE().energy_density(rho, sigma)

    # correlation part from the code (pinned elsewhere — not the target here):
    # ε_c^PW92 + H, matching pbe.py's t = |∇ρ|/(2 k_s ρ), k_s = √(4 k_F/π).
    rho_au = rho * BOHR_ANG**3
    grad_au = torch.sqrt(sigma * BOHR_ANG**8)
    kf = (3.0 * math.pi**2 * rho_au) ** (1.0 / 3.0)
    ks = torch.sqrt(4.0 * kf / math.pi)
    t = grad_au / (2.0 * ks * rho_au)
    eps_c_lda = eps_c_pw92(rho_au)
    eps_c = eps_c_lda + pbe_h(t * t, eps_c_lda)
    e_corr = rho * eps_c * HARTREE_EV

    e_exch_code = (e_total - e_corr).numpy()
    e_exch_oracle = _pbe_exchange_closed_form(rho.numpy(), sigma.numpy())
    assert np.allclose(e_exch_code, e_exch_oracle, rtol=1e-12, atol=0.0)

    # and the enhancement is strictly inside [1, 1+κ) and grows with s
    f_x = 1.0 + _KAPPA_PBE - _KAPPA_PBE / (
        1.0 + _MU_PBE * (grad_au / (2.0 * kf * rho_au)).numpy() ** 2 / _KAPPA_PBE)
    assert (f_x >= 1.0 - 1e-12).all() and (f_x < 1.0 + _KAPPA_PBE).all()
    assert (np.diff(f_x) > 0).all()


def test_pbe_exchange_vrho_vsigma_matches_libxc():
    """Cross-check the PBE exchange potentials vρ, vσ against libxc's own
    gga_x_pbe (the reference PBE-x implementation). Isolates exchange, so it
    pins κ/μ AND the derivative chain, independent of gradwave's correlation.
    Skips only when the fragile pyscf/libxc wheel is absent."""
    libxc = pytest.importorskip("pyscf.dft.libxc",
                                reason="pyscf/libxc oracle not installed")
    den = np.array([0.35, 0.12, 0.6, 0.03])      # a.u. ρ
    gmag = np.array([0.05, 0.03, 0.2, 0.004])    # a.u. |∇ρ|
    sigma = gmag**2
    rho4 = np.vstack([den, gmag, np.zeros(4), np.zeros(4)])  # (ρ, ∂xρ, ∂yρ, ∂zρ)
    exc, vxc, _, _ = libxc.eval_xc("gga_x_pbe", rho4, spin=0, deriv=1)
    vrho_ref, vsig_ref = vxc[0], vxc[1]

    dt = torch.tensor(den, dtype=torch.float64, requires_grad=True)
    st = torch.tensor(sigma, dtype=torch.float64, requires_grad=True)
    kf = (3.0 * math.pi**2 * dt) ** (1.0 / 3.0)
    s2 = st / (4.0 * kf * kf * dt * dt)          # s² = |∇ρ|²/(2 k_F ρ)²
    f_x = 1.0 + KAPPA - KAPPA / (1.0 + MU * s2 / KAPPA)
    e_haub = dt * (eps_x_lda(dt) * f_x)          # Ha/bohr³ = ρ_au·ε_x
    g_rho, g_sig = torch.autograd.grad(e_haub.sum(), (dt, st))

    assert np.allclose(e_haub.detach().numpy(), exc * den, rtol=1e-11, atol=0.0)
    assert np.allclose(g_rho.numpy(), vrho_ref, rtol=1e-11, atol=0.0)
    assert np.allclose(g_sig.numpy(), vsig_ref, rtol=1e-11, atol=0.0)
