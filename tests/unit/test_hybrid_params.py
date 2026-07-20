"""Learnable hybrid-exchange parameters (α mixing, ω screening)."""

import torch

from gradwave.postscf.exchange_multik import HybridExchangeParams


def test_defaults_are_physical_hse_like():
    p = HybridExchangeParams()
    assert p.mode == "short_range"
    assert abs(float(p.alpha.detach()) - 0.25) < 1e-6
    assert abs(float(p.omega.detach()) - 0.2) < 1e-6


def test_reparameterization_keeps_alpha_in_unit_interval_and_omega_positive():
    p = HybridExchangeParams(alpha=0.05, omega=0.9)
    assert abs(float(p.alpha.detach()) - 0.05) < 1e-6
    assert abs(float(p.omega.detach()) - 0.9) < 1e-6
    # push raw parameters hard; the reparameterization keeps them physical
    with torch.no_grad():
        p.raw_alpha += 20.0
        p.raw_omega -= 20.0
    assert 0.0 < float(p.alpha.detach()) < 1.0
    assert float(p.omega.detach()) > 0.0


def test_parameters_are_registered_and_grad_flows():
    p = HybridExchangeParams()
    names = {n for n, _ in p.named_parameters()}
    assert names == {"raw_alpha", "raw_omega"}
    (p.alpha * p.omega).backward()
    assert p.raw_alpha.grad is not None
    assert p.raw_omega.grad is not None
