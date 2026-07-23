"""Learned hybrid: differentiable exchange mixing (α) and screening (ω).

A converged hybrid SCF is turned into a trainable objective by the stationary-
energy (Hellmann–Feynman) derivative: at self-consistency the density is
variational, so dE_total/dθ = ∂E_total/∂θ — the explicit θ-dependence of the
exchange terms on the *frozen* converged orbitals. ``differentiable_hybrid_energy``
returns a scalar equal to ``res.energies.total`` whose gradient in (α, ω) is that
exact derivative, so an optimizer over ``HybridExchangeParams`` trains the hybrid.

The gates: the differentiable energy equals the SCF total in value; its α- and
ω-gradients match a finite difference of *re-converged* SCF energies (the exact
stationary derivative, not a frozen-orbital approximation); and a backward pass
populates the raw-parameter gradients so a training step moves (α, ω).
"""

import pytest
import torch

from gradwave.postscf.exchange_multik import HybridExchangeParams
from gradwave.postscf.hybrid import (
    differentiable_hybrid_energy,
    hybrid_energy_gradient,
    hybrid_scf,
)
from gradwave.scf.loop import setup_system
from tests.helpers import RY, si_fcc, si_upf


def _system():
    cell, pos = si_fcc()
    upf = si_upf()
    return setup_system(cell, pos, [0, 0], [upf], ecut=14 * RY, kmesh=(2, 1, 1),
                        use_symmetry=False, time_reversal=False, nbands=8)


def _converge(alpha, mode="full", omega=None):
    res = hybrid_scf(_system(), alpha=alpha, mode=mode, omega=omega, smearing="none",
                     etol=1e-10, rhotol=1e-9, verbose=False, max_iter=150)
    assert res.converged
    return res


@pytest.fixture(scope="module")
def pbe0():
    return _converge(0.25, "full")


@pytest.fixture(scope="module")
def screened():
    return _converge(0.25, "short_range", 0.30)


def test_differentiable_energy_matches_total(pbe0):
    params = HybridExchangeParams(alpha=0.25, mode="full")
    e = differentiable_hybrid_energy(pbe0, params)
    assert abs(float(e.detach()) - float(pbe0.energies.total)) < 1e-9


def test_alpha_gradient_matches_finite_difference(pbe0):
    a0, h = 0.25, 0.02
    params = HybridExchangeParams(alpha=a0, mode="full")
    d_alpha, d_omega = hybrid_energy_gradient(pbe0, params)
    assert d_omega is None  # no screening in full mode
    fd = (float(_converge(a0 + h).energies.total)
          - float(_converge(a0 - h).energies.total)) / (2 * h)
    assert abs(d_alpha - fd) / abs(fd) < 1e-3


def test_omega_gradient_matches_finite_difference(screened):
    a0, w0, h = 0.25, 0.30, 0.01
    params = HybridExchangeParams(alpha=a0, omega=w0, mode="short_range")
    _, d_omega = hybrid_energy_gradient(screened, params)
    fd = (float(_converge(a0, "short_range", w0 + h).energies.total)
          - float(_converge(a0, "short_range", w0 - h).energies.total)) / (2 * h)
    assert abs(d_omega - fd) / abs(fd) < 5e-3


def test_training_step_updates_params(screened):
    params = HybridExchangeParams(alpha=0.25, omega=0.30, mode="short_range")
    opt = torch.optim.SGD(params.parameters(), lr=1e-3)
    e = differentiable_hybrid_energy(screened, params)
    loss = (e - (float(screened.energies.total) - 1.0)) ** 2  # pull energy down by 1 eV
    params.zero_grad(set_to_none=True)
    loss.backward()
    assert params.raw_alpha.grad is not None and float(params.raw_alpha.grad) != 0.0
    assert params.raw_omega.grad is not None and float(params.raw_omega.grad) != 0.0
    a_before, w_before = float(params.alpha.detach()), float(params.omega.detach())
    opt.step()
    assert float(params.alpha.detach()) != a_before
    assert float(params.omega.detach()) != w_before
