"""Learned multi-pole Kerker preconditioner: filter algebra, mixer wiring, and
the differentiable fit against a diagonal response model."""

import math

import torch

from gradwave.dtypes import RDTYPE
from gradwave.scf.learned_precond import (
    BlockPrecond,
    MultipoleKerkerPrecond,
    _diis_unroll_logres,
    diis_objective,
    fit_multipole,
    fit_multipole_robust,
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


def test_const_term_keeps_g0_alive():
    """The magnetization-channel form f = w0 + Σ wᵢ g²/(g²+qᵢ²) has f(0) = w0 ∈
    (0,1), so the uniform moment mode can move, unlike the charge form's f(0)=0."""
    g2 = _g2_grid()
    P = MultipoleKerkerPrecond.init_poles(g2, n_poles=3, const=True,
                                          const_init=0.4, requires_grad=True)
    w0 = float(P.const_val())
    assert 0.0 < w0 < 1.0
    assert abs(float(P.filter_vals()[0]) - w0) < 1e-12   # g2[0]==0 → f(0)=w0
    # the const is a learnable parameter and the loss differentiates through it
    assert any(p is P.c_raw for p in P.params)
    loss = P.filter_vals().sum()
    loss.backward()
    assert P.c_raw.grad is not None and float(P.c_raw.grad.abs()) > 0
    # charge form (no const) still pins G=0
    Q = MultipoleKerkerPrecond.init_poles(g2, n_poles=3, requires_grad=False)
    assert float(Q.filter_vals()[0].abs()) < 1e-14


def test_block_precond_applies_per_block():
    """BlockPrecond routes each contiguous segment to its own operator: Kerker on
    the total block, a learned filter on the magnetization block, identity where
    the operator is None."""
    ng = 64
    g2 = torch.linspace(0.0, 30.0, ng, dtype=RDTYPE)
    kerker = MultipoleKerkerPrecond.kerker(g2, 1.1)
    mag = MultipoleKerkerPrecond.init_poles(g2, n_poles=2, const=True,
                                            requires_grad=False)
    block = BlockPrecond([(ng, kerker), (ng, mag)])
    assert block.acts_on == "grid"

    r = torch.randn(2 * ng, dtype=torch.complex128)
    r[0] = 0.0                                     # pinned total G=0
    out = block(r)
    assert torch.allclose(out[:ng], kerker(r[:ng]), atol=1e-12)
    assert torch.allclose(out[ng:], mag(r[ng:]), atol=1e-12)
    # total G=0 stays pinned; mag G=0 is alive (w0·r ≠ 0 for r≠0)
    assert float(out[0].abs()) < 1e-14
    assert float(out[ng].abs()) > 0.0

    # a None block is identity (plain damping passthrough)
    ident = BlockPrecond([(ng, kerker), (ng, None)])
    assert torch.allclose(ident(r)[ng:], r[ng:], atol=1e-14)


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


# -- robust fit: multi-seed, quality weighting, model selection, abstention ----

def _two_scale_res_hist(alpha, g2, noise=0.0, seed=0, n=10):
    """Synthetic probe residual history whose ratios encode a two-scale d(G)."""
    d_true = 0.5 * g2 / (g2 + 0.3**2) + 0.5 * g2 / (g2 + 2.5**2)
    amp = (1.0 - alpha * d_true).to(torch.complex128)
    r0 = torch.ones(g2.shape[0], dtype=torch.complex128)
    r0[0] = 0.0                                        # pinned G=0
    hist = [r0 * amp**k for k in range(n)]
    if noise:
        g = torch.Generator().manual_seed(seed)
        hist = [r * (1.0 + noise * torch.randn(r.shape, generator=g,
                                               dtype=torch.float64)) for r in hist]
    return hist


def test_robust_fit_selection_is_noise_stable():
    """The recorded failure: ~1e-14 last-bit noise in the residual history steered
    the deterministic optimizer into a worse local minimum (the #91 rfft round-trip
    flipped Cu 10→8 back to a tie). Multi-seed objective selection must make the
    deployed filter's objective and synthetic iteration count stable across noise
    realizations — the selection must not flip to a worse minimum."""
    alpha = 0.5
    g2 = torch.linspace(0.05, 25.0, 120, dtype=RDTYPE)
    # one CLEAN reference response to score every realization's deployed filter
    # on, so objective differences reflect the filter alone, not the probe noise.
    hist0 = _two_scale_res_hist(alpha, g2, noise=0.0)
    ref_centers, ref_d, _ = response_from_residuals(hist0, g2, alpha,
                                                    n_bins=24, skip=1)
    objs, ks, iters = [], [], []
    for seed in range(3):
        hist = _two_scale_res_hist(alpha, g2, noise=1e-14, seed=seed)
        centers, d_shell, count, quality = response_from_residuals(
            hist, g2, alpha, n_bins=24, skip=1, return_quality=True)
        P, info = fit_multipole_robust(
            centers, d_shell, alpha=alpha, q0=1.1, history=6, n_unroll=15,
            steps=80, k_max=2, n_seeds=3, q_min=0.1, q_max=4.0,
            weight=count, quality=quality)
        obj = diis_objective(P, ref_centers, ref_d, alpha=alpha, q0=1.1,
                             n_unroll=15, history=6)
        objs.append(obj)
        ks.append(info["selected_k"])
        # synthetic iterations-to-1e-8 at the unrolled per-step DIIS log rate
        iters.append(math.log(1e-8) / (obj / 15.0))
    # 1e-14 probe noise must never flip the deployed pole count, must leave the
    # synthetic iteration count essentially fixed, and must move the deployed
    # objective by far less than the gap between distinct minima (the Kerker vs
    # multi-pole objectives differ by tens of log units here; the DIIS-floor
    # conditioning limits agreement to O(1) log units, which is iteration-
    # equivalent to a rounding error).
    assert len(set(ks)) == 1
    assert max(iters) - min(iters) < 0.75
    assert max(objs) - min(objs) < 1.5


def test_robust_fit_keeps_two_scale_win():
    """On a genuinely two-scale response the gate does NOT abstain: it deploys a
    multi-pole filter whose unrolled-DIIS objective beats bare Kerker by more than
    the abstention margin (the win the whole mechanism exists to capture)."""
    alpha = 0.7
    g2 = torch.linspace(0.15, 40.0, 100, dtype=RDTYPE)
    d = 0.5 * g2 / (g2 + 0.3**2) + 0.5 * g2 / (g2 + 2.5**2)
    P, info = fit_multipole_robust(g2, d, alpha=alpha, q0=1.1, history=8,
                                   n_unroll=20, steps=150, k_max=3, n_seeds=3,
                                   q_min=0.2, q_max=4.0)
    assert not info["abstained"]
    assert info["selected_kind"] == "fit"
    assert info["selected_k"] >= 2                       # genuinely multi-pole
    # beats the best SINGLE-pole candidate by more than the gate margin
    assert info["obj_ref"] - info["obj_deployed"] > info["margin"]


def test_robust_fit_abstains_to_kerker_on_single_scale():
    """A single-scale response (the response denominator is itself a single Kerker
    pole) offers no multi-scale room, so the abstention gate returns bare Kerker
    exactly — losing to Kerker is impossible by construction."""
    alpha = 0.7
    g2 = torch.linspace(0.15, 40.0, 100, dtype=RDTYPE)
    q0 = 1.1
    d = g2 / (g2 + q0**2)                               # single scale at q0
    P, info = fit_multipole_robust(g2, d, alpha=alpha, q0=q0, history=8,
                                   n_unroll=20, steps=150, k_max=3, n_seeds=3,
                                   q_min=0.2, q_max=4.0)
    assert info["abstained"]
    assert info["selected_kind"] == "kerker"
    kerker = MultipoleKerkerPrecond.kerker(g2, q0)
    assert torch.allclose(P.filter_vals(), kerker.filter_vals(), atol=1e-12)


def test_robust_fit_model_selection_prefers_fewer_poles():
    """Model selection prefers the smallest pole count whose objective is within
    tolerance of the best: a single-scale response model-selects K=1 (not the
    larger K_max fit), and the K=1 selection deploys as bare Kerker."""
    alpha = 0.7
    g2 = torch.linspace(0.15, 40.0, 100, dtype=RDTYPE)
    d = g2 / (g2 + 0.9**2)                              # single scale
    P, info = fit_multipole_robust(g2, d, alpha=alpha, q0=1.1, history=8,
                                   n_unroll=20, steps=150, k_max=3, n_seeds=3,
                                   q_min=0.2, q_max=4.0)
    assert info["selected_k_model"] == 1
    assert info["selected_k"] == 1


def test_robust_fit_pole_floor_from_quality():
    """Probe-quality weighting forbids a pole below the lowest trustworthy shell:
    when the low-|G| shells carry no trustworthy data (quality 0 there), the fitted
    poles all sit at or above the reported pole floor, killing the q≈0.03 Å⁻¹
    overfit-to-lowest-shell-noise failure mode structurally."""
    alpha = 0.5
    g2 = torch.linspace(0.05, 25.0, 120, dtype=RDTYPE)
    hist = _two_scale_res_hist(alpha, g2, noise=1e-12, seed=7)
    centers, d_shell, count, quality = response_from_residuals(
        hist, g2, alpha, n_bins=24, skip=1, return_quality=True)
    # zero out the two lowest shells' confidence: they become untrusted, so the
    # floor must climb above them and no pole may land there.
    quality = quality.clone()
    quality[:2] = 0.0
    # gate disabled (model_tol=0 picks the absolute best fit, margins off) to
    # isolate the mechanism under test: the FLOOR on fitted pole positions.
    P, info = fit_multipole_robust(
        centers, d_shell, alpha=alpha, q0=1.1, history=6, n_unroll=15, steps=80,
        k_max=2, n_seeds=3, q_min=0.05, q_max=4.0, weight=count, quality=quality,
        model_tol=0.0, abstain_margin=-1e9, rel_margin=0.0)
    trust = quality > 0.0
    q_lo_trust = float(centers[trust].min().sqrt())
    assert info["q_floor"] >= q_lo_trust - 1e-9
    # every fitted pole sits at or above the floor (bare Kerker's q0 is exempt:
    # it is the safe default, not a probe-noise artifact).
    if info["selected_kind"] == "fit":
        assert float(P.q2().sqrt().min()) >= info["q_floor"] - 1e-9
    else:                                   # K=1 model-selected → bare Kerker out
        assert torch.allclose(
            P.filter_vals(),
            MultipoleKerkerPrecond.kerker(centers, 1.1).filter_vals(), atol=1e-12)
    # and directly on the fitter: a seed BELOW the floor is projected back onto
    # it and stays there through the whole optimization.
    Q, _ = fit_multipole(centers, d_shell, alpha=alpha, mixer="diis", history=6,
                         n_unroll=15, steps=40, seed_q=[0.03, 2.5],
                         q_floor=info["q_floor"], q_ceil=4.0, weight=quality)
    assert float(Q.q2().sqrt().min()) >= info["q_floor"] - 1e-9
