"""SpinAdaptedPBE — the shipped preset whose ζ² spin-adaptation parameters
(κ₁, μ₁) are fit to the moments of Fe/Ni/Co. Grid-level (fast tier): the
non-negotiable guards are (1) it loads from the registry, (2) it is EXACTLY PBE
on a closed-shell density (ζ≡0, so the fit cannot damage non-magnetic systems),
and (3) on a spin-polarized density it actually differs from plain SpinPBE — the
fitted adaptation is live, which is what shifts the moment through an SCF."""

import torch

from gradwave.api._common import SPIN_XC_REGISTRY, XC_REGISTRY
from gradwave.core.xc.learnable import (
    SPIN_ADAPTED_PBE_KAPPA1,
    SPIN_ADAPTED_PBE_MU1,
    SpinAdaptedPBE,
)
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE


def _polarized_grid():
    gen = torch.Generator().manual_seed(7)
    ru = 0.01 + 0.3 * torch.rand(64, generator=gen, dtype=torch.float64)
    rd = 0.01 + 0.3 * torch.rand(64, generator=gen, dtype=torch.float64)
    suu = 0.1 * torch.rand(64, generator=gen, dtype=torch.float64)
    sdd = 0.1 * torch.rand(64, generator=gen, dtype=torch.float64)
    stt = suu + sdd + 0.05 * torch.rand(64, generator=gen, dtype=torch.float64)
    return ru, rd, suu, sdd, stt


def test_registry_loads_preset():
    """`xc: spin_adapted_pbe` resolves to the fitted preset (spin path) and to
    plain PBE on the closed-shell charge-only path (ζ≡0 ⇒ PBE)."""
    assert SPIN_XC_REGISTRY["spin_adapted_pbe"] is SpinAdaptedPBE
    assert XC_REGISTRY["spin_adapted_pbe"] is PBE
    xc = SPIN_XC_REGISTRY["spin_adapted_pbe"]()
    assert isinstance(xc, SpinAdaptedPBE)


def test_preset_params_are_the_fitted_values():
    """The default-constructed preset carries the fitted κ₁, μ₁ (module-level
    constants), and at least one is nonzero — the adaptation is actually on."""
    xc = SpinAdaptedPBE()
    assert float(xc.kappa1) == SPIN_ADAPTED_PBE_KAPPA1
    assert float(xc.mu1) == SPIN_ADAPTED_PBE_MU1
    assert abs(SPIN_ADAPTED_PBE_KAPPA1) + abs(SPIN_ADAPTED_PBE_MU1) > 0.0


def test_closed_shell_is_pbe_exactly():
    """Spin-unpolarized (ζ≡0) ⇒ exactly PBE, for the preset's fitted params —
    the structural guarantee that the fit does not touch non-magnetic systems."""
    gen = torch.Generator().manual_seed(3)
    rho = 0.02 + 0.3 * torch.rand(32, generator=gen, dtype=torch.float64)
    sig = 0.05 * torch.rand(32, generator=gen, dtype=torch.float64)
    ref = PBE().energy_density(rho, sig)
    e = SpinAdaptedPBE().energy_density(rho / 2, rho / 2, sig / 4, sig / 4, sig)
    assert float((e - ref).detach().abs().max()) < 1e-13


def test_polarized_density_differs_from_spin_pbe():
    """On a spin-polarized density the preset departs from plain SpinPBE by a
    finite, ζ²-weighted amount — the grid-level signature of the moment shift the
    fit installs (a full-SCF moment shift is the standard-tier validation)."""
    ru, rd, suu, sdd, stt = _polarized_grid()
    e_pbe = SpinPBE().energy_density(ru, rd, suu, sdd, stt)
    e_sa = SpinAdaptedPBE().energy_density(ru, rd, suu, sdd, stt)
    assert float((e_sa - e_pbe).detach().abs().max()) > 1e-6
