"""Phase 5 of the differentiable-composition method: the surrogate fit and the
inverse-design optimizer. Validated on synthetic quadratic landscapes, so no SCF
runs here. The DFT bridge (sample_alchemical) rides on the phase 1-4 engine."""

import numpy as np
import torch

from gradwave.postscf.composition_design import (
    CompositionSurrogate,
    fit_surrogate,
    optimize_composition,
)


def _true_model(na, seed):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=na)
    M = rng.normal(size=(na, na))
    A = 0.5 * (M + M.T)  # symmetric
    c0 = float(rng.normal())
    return c0, b, A


def test_fit_recovers_quadratic_with_gradients():
    na = 3
    c0, b, A = _true_model(na, seed=1)
    true = CompositionSurrogate(c0, b, A)
    rng = np.random.default_rng(7)
    # 5 samples with gradient supervision pin the 10-parameter 3-site model; an
    # energy-only fit would need 10 samples
    lams = rng.uniform(0, 1, size=(5, na))
    energies = [float(true.energy(lam)) for lam in lams]
    grads = [true.gradient(lam).numpy() for lam in lams]

    fit = fit_surrogate(lams, energies, grads)
    # parameters recovered from exactly-quadratic data
    assert np.allclose(fit.b.numpy(), b, atol=1e-8)
    assert np.allclose(fit.A.numpy(), A, atol=1e-8)
    assert abs(float(fit.c0) - c0) < 1e-8
    # prediction matches at a held-out point
    test_lam = rng.uniform(0, 1, size=na)
    assert abs(float(fit.energy(test_lam)) - float(true.energy(test_lam))) < 1e-8


def test_optimize_finds_minimizer():
    # convex surrogate (A positive definite): unconstrained min at -0.5 A^-1 b
    na = 2
    A = np.array([[2.0, 0.3], [0.3, 1.5]])
    b = np.array([-1.2, -0.8])
    surr = CompositionSurrogate(0.0, b, A)
    lam_star = -0.5 * np.linalg.solve(A, b)  # inside [0,1]^2 by construction
    assert np.all((lam_star > 0) & (lam_star < 1))
    out = optimize_composition(surr, x0=np.full(na, 0.5), steps=800, lr=0.05)
    assert np.allclose(out.numpy(), lam_star, atol=1e-3), (out.numpy(), lam_star)


def test_optimize_hits_target_energy():
    na = 2
    c0, b, A = _true_model(na, seed=3)
    surr = CompositionSurrogate(c0, b, A)
    target = float(surr.energy(torch.tensor([0.7, 0.2], dtype=torch.float64)))
    out = optimize_composition(surr, x0=np.full(na, 0.5), target=target,
                               steps=1500, lr=0.03)
    assert abs(float(surr.energy(out)) - target) < 1e-3
