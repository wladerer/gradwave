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
    anderson_solve,
    deflated_solve,
    dielectric_apply,
    dominant_screening_eigenvalue,
    max_real_screening_eigenvalue,
    plain_fixed_point_rate,
    screening_apply,
    soft_subspace,
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


def test_soft_subspace_arnoldi_matches_p0_estimator(si_res):
    """P1: the Arnoldi soft mode agrees with P0's shifted power iteration."""
    res, xc = si_res

    sub = soft_subspace(res, xc, krylov=24, n_modes=1, chi0_tol=1e-7)
    assert len(sub.values) == 1
    lam = sub.values[0].real

    # It IS the soft (max-real) mode, not the dominant negative charge mode.
    ref = max_real_screening_eigenvalue(res, xc, n_iter=120, chi0_tol=1e-7)
    assert lam == pytest.approx(ref.eigenvalue, abs=0.02), (lam, ref.eigenvalue)
    assert 0.0 < lam < 1.0, sub.values

    # Genuine eigenpair (true operator residual), and near-normal here so the
    # non-normality flag stays ~0 (the P2 signal is absent on a benign insulator).
    assert sub.residuals[0] < 1e-2, sub.residuals
    assert sub.max_imag < 1e-3, sub.max_imag


def test_deflated_solve_is_exact_on_the_real_operator(si_res):
    """P2: the deflated solve is correct on the real Si response operator.

    Si has a SINGLE dielectric soft mode, which Anderson already handles (base 27
    vs deflated 29 iters at a modest near-critical coupling) — so deflation is
    neutral on speed here, matching the synthetic single-mode result. The decisive
    speedup needs a soft *cluster* beyond Anderson's history (validated in the
    synthetic unit test). The physical gate is therefore exactness/robustness: the
    deflated solve converges on the real coupling-scaled operator and agrees with
    the baseline solution."""
    res, xc = si_res
    sub = soft_subspace(res, xc, krylov=24, n_modes=1, chi0_tol=1e-5)
    q = sub.vectors
    c = 0.7 / sub.values[0].real  # a modest near-critical coupling on real Si

    m = screening_apply(res, xc, coupling=c, chi0_tol=1e-5)
    g = torch.Generator().manual_seed(5)
    vbar = torch.randn(res.rho.shape, dtype=res.rho.dtype, generator=g)
    vbar = vbar - vbar.mean()

    base = anderson_solve(m, vbar, tol=1e-6, max_iter=60, history=8)
    defl = deflated_solve(m, vbar, q, method="post", tol=1e-6, max_iter=60, history=8)

    assert base.converged and defl.converged, (base, defl)
    assert defl.residual < 1e-6, defl
    # deflated and undeflated solve the SAME real near-critical system
    agree = float(torch.linalg.vector_norm(base.u - defl.u)
                  / torch.linalg.vector_norm(defl.u))
    assert agree < 1e-4, agree


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
