"""LearnableSpinXZeta: ζ-adaptive spin-PBE exchange with κ(ζ)=κ₀+κ₁ζ²,
μ(ζ)=μ₀+μ₁ζ². The non-negotiable gate is exact (machine-precision) recovery of
the base learnable spin-PBE — and hence PBE — when κ₁=μ₁=0. All grid-level, so
this runs in the fast tier."""

import math

import torch

from gradwave.core.xc.learnable import (
    PBE_KAPPA,
    PBE_MU,
    LearnableSpinX,
    LearnableSpinXZeta,
)
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE


def _polarized_grid():
    gen = torch.Generator().manual_seed(11)
    ru = 0.01 + 0.3 * torch.rand(64, generator=gen, dtype=torch.float64)
    rd = 0.01 + 0.3 * torch.rand(64, generator=gen, dtype=torch.float64)
    suu = 0.1 * torch.rand(64, generator=gen, dtype=torch.float64)
    sdd = 0.1 * torch.rand(64, generator=gen, dtype=torch.float64)
    stt = suu + sdd + 0.05 * torch.rand(64, generator=gen, dtype=torch.float64)
    return ru, rd, suu, sdd, stt


def test_zeta_zero_params_recover_spin_pbe_exactly():
    """κ₁=μ₁=0 at the PBE base ⇒ identical to SpinPBE to machine precision on a
    spin-polarized density (this is the ζ≠0 path with a flat κ(ζ)=κ₀)."""
    ru, rd, suu, sdd, stt = _polarized_grid()
    a = SpinPBE().energy_density(ru, rd, suu, sdd, stt)
    b = LearnableSpinXZeta().energy_density(ru, rd, suu, sdd, stt)
    assert float((a.detach() - b.detach()).abs().max()) < 1e-14


def test_zeta_zero_params_recover_learnable_spin_x():
    """κ₁=μ₁=0 at an off-PBE base (κ₀≤0.804, so the LO cap is inactive) reduces
    to LearnableSpinX with the same (κ₀, μ₀), to machine precision."""
    ru, rd, suu, sdd, stt = _polarized_grid()
    a = LearnableSpinX(kappa=0.70, mu=0.18).energy_density(ru, rd, suu, sdd, stt)
    b = LearnableSpinXZeta(kappa=0.70, mu=0.18).energy_density(ru, rd, suu, sdd, stt)
    assert float((a.detach() - b.detach()).abs().max()) < 1e-14


def test_closed_shell_is_pbe_for_any_zeta_params():
    """A spin-unpolarized system (ζ≡0) gives exactly PBE for ANY κ₁, μ₁ — the ζ²
    term vanishes identically. Equal channels: ρσ=ρ/2, σσσ=σ/4, σ_tot=σ."""
    gen = torch.Generator().manual_seed(3)
    rho = 0.02 + 0.3 * torch.rand(32, generator=gen, dtype=torch.float64)
    sig = 0.05 * torch.rand(32, generator=gen, dtype=torch.float64)
    ref = PBE().energy_density(rho, sig)
    for k1, m1 in ((0.5, 0.3), (-0.9, -0.1), (2.0, 1.5)):
        xc = LearnableSpinXZeta(kappa1=k1, mu1=m1)
        e = xc.energy_density(rho / 2, rho / 2, sig / 4, sig / 4, sig)
        assert float((e - ref).detach().abs().max()) < 1e-13, (k1, m1)


def test_spin_swap_invariance():
    """E_x invariant under ρ↑↔ρ↓ (ζ→−ζ, ζ² unchanged) — the reason the
    dependence is on ζ² and not ζ. Holds for nonzero κ₁, μ₁."""
    ru, rd, suu, sdd, stt = _polarized_grid()
    xc = LearnableSpinXZeta(kappa1=-0.3, mu1=0.4)
    e = xc.energy_density(ru, rd, suu, sdd, stt)
    e_swap = xc.energy_density(rd, ru, sdd, suu, stt)
    assert float((e - e_swap).detach().abs().max()) < 1e-14


def test_lieb_oxford_cap_and_positivity():
    """κ(ζ) is capped at the Lieb–Oxford bound 0.804 and κ(ζ), μ(ζ) stay
    strictly positive even for aggressive κ₁, μ₁ and full polarization ζ=±1."""
    zeta = torch.linspace(-1.0, 1.0, 41, dtype=torch.float64)
    xc = LearnableSpinXZeta(kappa=0.804, mu=0.2195, kappa1=5.0, mu1=-10.0)
    kappa, mu = (t.detach() for t in xc._exchange_kappa_mu(zeta))
    assert float(kappa.max()) <= PBE_KAPPA + 1e-15
    assert float(kappa.min()) > 0.0
    assert float(mu.min()) > 0.0
    # ζ=±1 with a huge negative μ₁ would drive μ<0 without the floor
    assert torch.isfinite(kappa).all() and torch.isfinite(mu).all()


def test_no_nan_at_full_polarization_and_low_density():
    """Fully polarized (ρ↓→0 ⇒ ζ→1) and near-floor densities give finite e_x and
    finite v via autograd — the ζ denominator and the F_x chain are NaN-safe."""
    ru = torch.tensor([0.3, 0.1, 1e-6, 5e-4], dtype=torch.float64, requires_grad=True)
    rd = torch.tensor([1e-12, 1e-8, 1e-6, 2e-4], dtype=torch.float64, requires_grad=True)
    suu = torch.tensor([0.05, 0.02, 1e-6, 1e-3], dtype=torch.float64)
    sdd = torch.tensor([1e-12, 1e-8, 1e-6, 1e-3], dtype=torch.float64)
    stt = suu + sdd + 0.01
    xc = LearnableSpinXZeta(kappa1=-0.4, mu1=0.6)
    e = xc.energy_density(ru, rd, suu, sdd, stt)
    assert torch.isfinite(e).all()
    (vu, vd) = torch.autograd.grad(e.sum(), (ru, rd))
    assert torch.isfinite(vu).all() and torch.isfinite(vd).all()


def test_potential_matches_finite_difference():
    """v_σ = ∂e_x/∂ρ_σ from autograd matches central finite differences on a
    small spin-polarized grid — the ζ path differentiates correctly. Exchange-
    only comparison (κ₁≠0) isolates the new ζ term."""
    gen = torch.Generator().manual_seed(5)
    ru = (0.05 + 0.2 * torch.rand(6, generator=gen, dtype=torch.float64))
    rd = (0.03 + 0.2 * torch.rand(6, generator=gen, dtype=torch.float64))
    suu = 0.02 * torch.rand(6, generator=gen, dtype=torch.float64)
    sdd = 0.02 * torch.rand(6, generator=gen, dtype=torch.float64)
    stt = suu + sdd + 0.01
    xc = LearnableSpinXZeta(kappa1=-0.3, mu1=0.5)

    def e_of(a, b):
        return xc.energy_density(a, b, suu, sdd, stt).sum()

    a = ru.clone().requires_grad_(True)
    b = rd.clone().requires_grad_(True)
    vu, vd = torch.autograd.grad(e_of(a, b), (a, b))

    h = 1e-6
    for i in range(ru.numel()):
        for rho, v in ((ru, vu), (rd, vd)):
            rp = rho.clone()
            rp[i] += h
            rm = rho.clone()
            rm[i] -= h
            if rho is ru:
                fd = (e_of(rp, rd) - e_of(rm, rd)) / (2 * h)
            else:
                fd = (e_of(ru, rp) - e_of(ru, rm)) / (2 * h)
            assert abs(float(fd) - float(v[i])) < 1e-5, (i, float(fd), float(v[i]))


def test_gradcheck_zeta_functional():
    """Full autograd gradcheck of the ζ-adaptive energy_density w.r.t. the
    per-channel densities and sigmas at a nonzero (κ₁, μ₁)."""
    gen = torch.Generator().manual_seed(13)
    ru = (0.02 + 0.2 * torch.rand(5, generator=gen, dtype=torch.float64)).requires_grad_(True)
    rd = (0.03 + 0.15 * torch.rand(5, generator=gen, dtype=torch.float64)).requires_grad_(True)
    suu = (0.01 * torch.rand(5, generator=gen, dtype=torch.float64)).requires_grad_(True)
    sdd = (0.01 * torch.rand(5, generator=gen, dtype=torch.float64)).requires_grad_(True)
    stt = (0.04 * torch.rand(5, generator=gen, dtype=torch.float64) + 0.02).requires_grad_(True)
    xc = LearnableSpinXZeta(kappa1=-0.2, mu1=0.3)
    assert torch.autograd.gradcheck(
        lambda a, b, x, y, z: xc.energy_density(a, b, x, y, z).sum(),
        (ru, rd, suu, sdd, stt), atol=1e-7)


def test_params_include_zeta_adaptation():
    """κ₁, μ₁ are ordinary nn.Parameters (so energy_param_grads / any optimizer
    iterating xc.parameters() picks them up) and are exactly 0 at init."""
    xc = LearnableSpinXZeta()
    names = dict(xc.named_parameters())
    assert {"raw_kappa", "raw_mu", "kappa1", "mu1"} <= set(names)
    assert float(names["kappa1"]) == 0.0 and float(names["mu1"]) == 0.0
    # base κ₀, μ₀ recover PBE
    assert abs(float(xc.kappa) - PBE_KAPPA) < 1e-15
    assert abs(float(xc.mu) - PBE_MU) < 1e-15


def test_zeta_term_actually_changes_energy():
    """Sanity: a nonzero κ₁ on a genuinely polarized density DOES move the
    exchange energy (the term is not silently dead)."""
    ru, rd, suu, sdd, stt = _polarized_grid()
    base = LearnableSpinXZeta().energy_density(ru, rd, suu, sdd, stt)
    pert = LearnableSpinXZeta(kappa1=-0.3).energy_density(ru, rd, suu, sdd, stt)
    assert float((base - pert).detach().abs().max()) > 1e-6
    assert not math.isnan(float(pert.sum()))
