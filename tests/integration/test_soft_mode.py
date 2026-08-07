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
from gradwave.scf.implicit import solve_adjoint
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
    soft_subspace_from_operator,
    solve_adjoint_deflated,
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


def test_fxc_scaling_drives_a_physical_soft_cluster(si_res):
    """P2b: fxc-selective scaling makes a real, bounded soft cluster on Si, and the
    deflation must capture the whole (degenerate) cluster to help.

    Scaling only the xc kernel drives cubic Si's 3-fold degenerate dielectric mode
    toward +1 while leaving the Hartree charge spectrum fixed (unlike uniform
    coupling-scaling, which amplifies the −2.2 charge mode and swamps the smoother).
    That is a genuine physical near-critical soft *cluster* — deflation's regime —
    and it confirms on the real operator what the synthetic cluster test showed:
    deflating the whole cluster beats the baseline, deflating only part of it does
    not."""
    res, xc = si_res
    m = screening_apply(res, xc, fxc_scale=2.4, chi0_tol=1e-5)  # near-critical

    sub = soft_subspace_from_operator(m, res.rho, krylov=30, n_modes=3, seed=0)
    top = [v.real for v in sub.values]
    assert len(top) == 3
    assert 0.95 < top[0] < 1.0, top             # near-critical, still stable
    assert abs(top[0] - top[2]) < 0.02, top     # a 3-fold degenerate triplet
    assert sub.max_imag < 1e-3, sub.max_imag    # near-normal (not defective yet)

    q3 = sub.vectors
    q1 = q3[:1]
    g = torch.Generator().manual_seed(5)
    vbar = torch.randn(res.rho.shape, dtype=res.rho.dtype, generator=g)
    vbar = vbar - vbar.mean()

    base = anderson_solve(m, vbar, tol=1e-6, max_iter=120, history=8)
    d1 = deflated_solve(m, vbar, q1, method="post", tol=1e-6, max_iter=120, history=8)
    d3 = deflated_solve(m, vbar, q3, method="post", tol=1e-6, max_iter=120, history=8)
    assert base.converged and d1.converged and d3.converged, (base, d1, d3)

    # capturing the WHOLE degenerate cluster beats the baseline...
    assert d3.n_iter < base.n_iter, (d3.n_iter, base.n_iter)
    # ...while deflating only 1 of the 3 does not help (needs the full cluster)
    assert d3.n_iter < d1.n_iter, (d3.n_iter, d1.n_iter)


def test_solve_adjoint_deflated_is_a_faithful_drop_in(si_res):
    """Production entry: on benign Si, auto falls back to plain Anderson and lands
    on the same solution as the shipped solve_adjoint."""
    res, xc = si_res
    g = torch.Generator().manual_seed(7)
    vbar = torch.randn(res.rho.shape, dtype=res.rho.dtype, generator=g)
    vbar = vbar - vbar.mean()

    u_ref = solve_adjoint(res, xc, vbar, tol=1e-8)
    # Si's soft mode (0.15) is below the 0.9 near-critical threshold → plain solve
    got = solve_adjoint_deflated(res, xc, vbar, auto_threshold=0.9, tol=1e-8,
                                 chi0_tol=1e-8)
    assert got.converged
    rel = float(torch.linalg.vector_norm(got.u - u_ref)
                / torch.linalg.vector_norm(u_ref))
    assert rel < 1e-5, rel


def test_solve_adjoint_deflated_auto_and_recycle_near_critical(si_res):
    """Production entry: near a soft cluster, auto detects and deflates, and a
    recycled (precomputed) subspace gives the identical solution."""
    res, xc = si_res
    g = torch.Generator().manual_seed(7)
    vbar = torch.randn(res.rho.shape, dtype=res.rho.dtype, generator=g)
    vbar = vbar - vbar.mean()

    auto = solve_adjoint_deflated(res, xc, vbar, fxc_scale=2.4, auto_threshold=0.9,
                                  tol=1e-6, max_iter=120, chi0_tol=1e-5)
    assert auto.converged

    # recycle a cluster extracted once (the sweep use-case)
    m = screening_apply(res, xc, fxc_scale=2.4, chi0_tol=1e-5)
    q = soft_subspace_from_operator(m, res.rho, krylov=30, n_modes=3, seed=0).vectors
    recyc = solve_adjoint_deflated(res, xc, vbar, subspace=q, fxc_scale=2.4,
                                   tol=1e-6, max_iter=120, chi0_tol=1e-5)
    assert recyc.converged
    rel = float(torch.linalg.vector_norm(auto.u - recyc.u)
                / torch.linalg.vector_norm(recyc.u))
    assert rel < 1e-4, rel


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
