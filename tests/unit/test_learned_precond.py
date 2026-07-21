"""Learned multi-pole Kerker preconditioner: filter algebra, mixer wiring, and
the differentiable fit against a diagonal response model."""

import torch

from gradwave.dtypes import RDTYPE
from gradwave.scf.learned_precond import (
    MultipoleKerkerPrecond,
    _diis_unroll_logres,
    fit_multipole,
    response_from_residuals,
    spectral_radius,
)
from gradwave.scf.mixing import PulayMixer

torch.manual_seed(0)


def _g2_grid(n=200, gmax=40.0):
    """A density-sphere-like |G|² vector including the pinned G=0 component."""
    g2 = torch.linspace(0.0, gmax, n, dtype=RDTYPE)
    return g2


def test_kerker_special_case_matches_bare_filter():
    """K=1, w=1, q1=q0 reproduces G²/(G²+q0²) to round-off."""
    g2 = _g2_grid()
    q0 = 1.1
    P = MultipoleKerkerPrecond.kerker(g2, q0)
    ref = g2 / (g2 + q0**2)
    assert torch.allclose(P.filter_vals(), ref, atol=1e-12, rtol=0)


def test_g0_component_preserved():
    """f(0) = 0: the filter never touches the pinned charge, for any poles."""
    g2 = _g2_grid()
    P = MultipoleKerkerPrecond.init_poles(g2, n_poles=4, requires_grad=False)
    fac = P.filter_vals()
    assert float(fac[0].abs()) < 1e-14  # g2[0] == 0
    # applied to a residual, the G=0 entry stays exactly zero
    r = torch.randn(g2.shape[0], dtype=torch.complex128)
    r[0] = 0.0
    assert float(P(r)[0].abs()) < 1e-14


def test_precond_op_wiring_matches_builtin_kerker():
    """A Kerker-equivalent learned filter used as mixer.precond_op yields the
    exact same damped step as the mixer's built-in Kerker path."""
    g2 = torch.linspace(0.0, 30.0, 128, dtype=RDTYPE)
    q0, alpha = 1.1, 0.7
    r = torch.randn(g2.shape[0], dtype=torch.complex128)
    r[0] = 0.0

    builtin = PulayMixer(g2, alpha=alpha, kerker=True, q0=q0, check_g0=False)
    learned = PulayMixer(g2, alpha=alpha, kerker=False, check_g0=False)
    learned.precond_op = MultipoleKerkerPrecond.kerker(g2, q0)

    assert torch.allclose(builtin._damped(r), learned._damped(r),
                          atol=1e-12, rtol=0)


def test_fit_beats_best_single_pole_on_two_scale_response():
    """When the response carries two length scales, a fitted multi-pole filter
    reaches a smaller spectral radius (faster convergence) than the best
    single-pole Kerker, whose optimum is inside the multi-pole hypothesis class."""
    alpha = 0.7
    # smallest |G|² set by a finite cell (~16 Å box); the multi-scale advantage
    # lives in the resolved mid-range, not the unreachable G→0 corner
    g2 = torch.linspace(0.15, 40.0, 300, dtype=RDTYPE)
    q1, q2 = 0.3, 2.5  # two well-separated response length scales
    d = 0.5 * g2 / (g2 + q1**2) + 0.5 * g2 / (g2 + q2**2)

    # best single-pole Kerker over a q0 sweep
    best = min(
        float(spectral_radius(g2 / (g2 + q0**2), d, alpha))
        for q0 in torch.linspace(0.2, 4.0, 60).tolist()
    )

    P, info = fit_multipole(g2, d, n_poles=3, alpha=alpha, n_unroll=40,
                            steps=500, lr=0.05)
    assert info["rho_final"] < info["rho_init"]          # fit made progress
    assert info["rho_final"] < 0.9 * best                # and beats one pole


def test_diis_unroll_is_differentiable_and_fit_beats_kerker():
    """The Pulay-DIIS unroll is differentiable in the pole parameters, and a
    DIIS-aware fit reaches a smaller post-DIIS residual than bare Kerker on a
    two-scale response — i.e. the filter finds room DIIS's finite history leaves."""
    alpha = 0.7
    g2 = torch.linspace(0.15, 40.0, 150, dtype=RDTYPE)
    d = 0.5 * g2 / (g2 + 0.3**2) + 0.5 * g2 / (g2 + 2.5**2)
    metric = torch.ones_like(g2) / (g2 + 1.1**2)

    # differentiable: gradient flows to both pole weights and positions
    Q = MultipoleKerkerPrecond.init_poles(g2, n_poles=3, requires_grad=True)
    loss = _diis_unroll_logres(Q.filter_vals(), d, metric, alpha, 20, 8)
    loss.backward()
    assert Q.w_raw.grad is not None and float(Q.w_raw.grad.norm()) > 0
    assert Q.logq2.grad is not None and float(Q.logq2.grad.norm()) > 0

    # DIIS-aware fit beats Kerker's post-DIIS residual
    kerker = MultipoleKerkerPrecond.kerker(g2, 1.1).filter_vals()
    res_kerker = float(_diis_unroll_logres(kerker, d, metric, alpha, 25, 8))
    P, _ = fit_multipole(g2, d, n_poles=3, alpha=alpha, mixer="diis",
                         history=8, n_unroll=25, steps=500)
    res_learned = float(_diis_unroll_logres(P.filter_vals(), d, metric, alpha,
                                            25, 8))
    assert res_learned < res_kerker - 1.0  # at least e¹× smaller residual


def test_response_estimate_recovers_known_denominator():
    """response_from_residuals inverts res_{n+1}/res_n = 1 − α·d back to d(G)."""
    alpha = 0.5
    g2 = torch.linspace(0.0, 25.0, 400, dtype=RDTYPE)
    d_true = g2 / (g2 + 0.8**2)                      # single-pole truth
    amp = (1.0 - alpha * d_true).to(torch.complex128)
    r0 = torch.ones(g2.shape[0], dtype=torch.complex128)
    r0[0] = 0.0                                     # pinned G=0
    res_hist = [r0 * amp**n for n in range(8)]

    centers, d_shell, count = response_from_residuals(
        res_hist, g2, alpha, n_bins=30, skip=0)
    d_expect = centers / (centers + 0.8**2)
    # bins away from the noisy G→0 turn-on recover d to a few percent
    mid = centers > 2.0
    assert torch.allclose(d_shell[mid], d_expect[mid], atol=0.03, rtol=0.05)
