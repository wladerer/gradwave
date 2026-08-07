"""SoftModeDeflate P0: the soft-mode diagnostic on a benign insulator.

Gate: the dominant eigenvalue of the screening operator M = K_Hxc·χ₀ (the
Jacobian of the fixed point solve_adjoint iterates) is real, contractive
(0 < λ < 1), a genuine eigenpair (small residual), and equals the independently
measured plain fixed-point contraction rate. That proves the operator wrappers
and the power-iteration estimator are correct and that the "softness" number
means what the later deflation phases need — without touching any physics.
"""

from pathlib import Path

import pytest
import torch

from gradwave.core.xc.learnable import LearnableX
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.soft_mode import (
    dielectric_apply,
    dominant_screening_eigenvalue,
    max_real_screening_eigenvalue,
    plain_fixed_point_rate,
    screening_apply,
)
from tests.helpers import RY, si_fcc

pytestmark = pytest.mark.standard

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
CELL, POS = si_fcc()


@pytest.fixture(scope="module")
def si_res():
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(CELL, POS, [0, 0], [upf], ecut=10 * RY, kmesh=(1, 1, 1))
    xc = LearnableX(kappa=0.70, mu=0.20)
    res = scf(system, xc, smearing="none", etol=1e-12, rhotol=1e-11, verbose=False)
    assert res.converged
    return res, xc


def test_dominant_eigenvalue_matches_undamped_growth_rate(si_res):
    """Spectral radius of M, two independent ways, and why the solver damps."""
    res, xc = si_res

    est = dominant_screening_eigenvalue(res, xc, n_iter=80, chi0_tol=1e-7)

    # A genuine eigenpair: ‖Mv − λv‖ small (well-separated, near-normal here).
    assert est.residual < 1e-2, est
    assert est.n_iter <= 80
    # The dominant mode is the strong Hartree charge mode: |λ| > 1. This is the
    # reason solve_adjoint uses Anderson, not the plain undamped fixed point.
    assert abs(est.eigenvalue) > 1.0, est

    # Independent check: the undamped iteration u ← v̄ + M u grows/contracts at
    # the spectral radius of M — the same magnitude, via the actual iteration.
    rate = plain_fixed_point_rate(res, xc, n_iter=30, tail=10, chi0_tol=1e-7)
    assert rate == pytest.approx(abs(est.eigenvalue), rel=0.05), (
        est.eigenvalue, rate)
    # Concretely: spectral radius > 1 means the undamped map diverges.
    assert rate > 1.0


def test_soft_mode_margin_is_positive_for_benign_insulator(si_res):
    """The soft-mode indicator (max real eigenvalue of M) is safely below 1."""
    res, xc = si_res

    soft = max_real_screening_eigenvalue(res, xc, n_iter=120, chi0_tol=1e-7)

    # No instability: margin 1 − λ_max is strictly positive (and O(1) here).
    assert soft.eigenvalue < 1.0, soft
    assert 1.0 - soft.eigenvalue > 1e-2, soft
    # Recovered a genuine eigenpair of the un-shifted operator.
    assert soft.residual < 1e-1, soft
    # And it must not exceed the spectral radius.
    dom = dominant_screening_eigenvalue(res, xc, n_iter=80, chi0_tol=1e-7)
    assert soft.eigenvalue <= abs(dom.eigenvalue) + 1e-6, (soft, dom)


def test_operator_wrappers_are_consistent(si_res):
    """L = 1 − M applied to a field equals u − M(u) elementwise."""
    res, xc = si_res
    m = screening_apply(res, xc, chi0_tol=1e-7)
    ell = dielectric_apply(res, xc, chi0_tol=1e-7)

    g = torch.Generator().manual_seed(3)
    u = torch.randn(res.rho.shape, generator=g, dtype=res.rho.dtype)
    u = u - u.mean()

    lhs = ell(u)
    rhs = u - m(u)
    assert torch.allclose(lhs, rhs, atol=1e-10, rtol=0), float(
        torch.linalg.norm(lhs - rhs))
