"""Fast unit tests for the parallel line-search math and adaptive trigger.

Pure functions only — no SCF, no worker processes. Covers:
  * the α schedule generator,
  * the cubic fit from (E, g) samples and its interpolated minimum, including
    the recovery of a known parabola/cubic minimum and the fallback guard,
  * the adaptive struggle trigger (fires on energy-increase / max|F| stall,
    dormant on monotone progress).
"""

import numpy as np
import pytest

from gradwave.opt.line_search import (
    choose_alpha,
    cubic_line_min,
    default_alpha_schedule,
    fit_cubic,
    struggle_trigger,
)

# --- α schedule ---------------------------------------------------------------


def test_default_schedule_includes_unit_step_and_is_geometric():
    assert default_alpha_schedule(3, 10.0) == (0.5, 1.0, 2.0)
    assert default_alpha_schedule(4, 10.0) == (0.25, 0.5, 1.0, 2.0)
    # 1.0 (the plain BFGS step) is always a sample
    for n in range(2, 8):
        assert 1.0 in default_alpha_schedule(n, 10.0)


def test_schedule_clamps_to_max_alpha_and_drops_collapsed_duplicates():
    sched = default_alpha_schedule(5, 2.5)  # raw {0.25,0.5,1,2,4} → 4 clamped to 2.5
    assert max(sched) <= 2.5
    assert len(sched) == len(set(sched))  # no duplicates after clamping


# --- cubic fit from (E, g) recovers a known minimum ---------------------------


def test_cubic_recovers_parabola_minimum():
    # E(α) = (α − 2)², g(α) = E'(α) = 2(α − 2); min at α* = 2
    alphas = np.array([0.0, 1.0, 3.0])
    energies = (alphas - 2.0) ** 2
    grads = 2.0 * (alphas - 2.0)
    coeffs = fit_cubic(alphas, energies, grads)
    a_star, ok = cubic_line_min(coeffs, 0.0, 3.0)
    assert ok
    assert a_star == pytest.approx(2.0, abs=1e-6)


def test_cubic_recovers_true_cubic_minimum():
    # p(α) = (α − 0.8)²(α + 3) has an interior min near 0.8; sample E and g
    def p(a):
        return (a - 0.8) ** 2 * (a + 3.0)

    def dp(a):
        return 2.0 * (a - 0.8) * (a + 3.0) + (a - 0.8) ** 2

    alphas = np.array([0.0, 0.5, 1.0, 2.0])
    coeffs = fit_cubic(alphas, p(alphas), dp(alphas))
    a_star, ok = cubic_line_min(coeffs, 0.0, 2.0)
    assert ok
    assert a_star == pytest.approx(0.8, abs=1e-6)


def test_choose_alpha_takes_cubic_minimum_within_bracket():
    alphas = [0.0, 0.5, 1.0, 2.0]
    energies = [(a - 1.3) ** 2 for a in alphas]
    grads = [2.0 * (a - 1.3) for a in alphas]
    samples = list(zip(alphas, energies, grads, strict=True))
    alpha, how = choose_alpha(samples, max_alpha=2.5)
    assert how == "cubic"
    assert alpha == pytest.approx(1.3, abs=1e-6)


# --- fallback guard -----------------------------------------------------------


def test_choose_alpha_falls_back_when_min_outside_bracket():
    # monotone-decreasing energy with negative slope everywhere: the true min is
    # beyond the largest sampled α, so the cubic has no interior minimum in the
    # bracket → fall back to the best ACTUALLY-SAMPLED α (the largest, lowest E).
    alphas = [0.0, 0.5, 1.0, 2.0]
    energies = [4.0, 3.0, 2.0, 1.0]
    grads = [-2.0, -2.0, -2.0, -2.0]
    samples = list(zip(alphas, energies, grads, strict=True))
    alpha, how = choose_alpha(samples, max_alpha=2.5)
    assert how == "fallback"
    assert alpha == 2.0  # best sampled positive α by energy


def test_choose_alpha_respects_max_alpha_cap():
    # cubic minimum sits past max_alpha → not accepted, fall back to a sample
    alphas = [0.0, 1.0, 2.0]
    energies = [(a - 5.0) ** 2 for a in alphas]
    grads = [2.0 * (a - 5.0) for a in alphas]
    samples = list(zip(alphas, energies, grads, strict=True))
    alpha, how = choose_alpha(samples, max_alpha=2.0)
    assert how == "fallback"
    assert 0.0 < alpha <= 2.0


def test_cubic_line_min_rejects_nonconvex_fit():
    # a concave parabola (maximum, not minimum) inside the bracket is rejected
    coeffs = fit_cubic([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], [2.0, 0.0, -2.0])
    _a, ok = cubic_line_min(coeffs, 0.0, 2.0)
    assert not ok


# --- adaptive struggle trigger ------------------------------------------------


def test_trigger_dormant_on_monotone_progress():
    # energy and max|F| both decreasing every step → not struggling
    progress = [(-10.0, 0.5), (-10.5, 0.3), (-10.8, 0.15)]
    assert struggle_trigger(progress, patience=1) is False


def test_trigger_fires_on_energy_increase():
    progress = [(-10.5, 0.3), (-10.2, 0.25)]  # energy went UP on the last step
    assert struggle_trigger(progress, patience=1) is True


def test_trigger_fires_on_force_stall():
    # energy still drops but max|F| did not decrease over the last step (stall)
    progress = [(-10.5, 0.30), (-10.6, 0.32)]
    assert struggle_trigger(progress, patience=1) is True


def test_trigger_needs_history():
    assert struggle_trigger([], patience=1) is False
    assert struggle_trigger([(-10.0, 0.5)], patience=1) is False


def test_trigger_patience_window_allows_transient_stall():
    # over a 2-step patience window, max|F| net-decreases (0.30 → 0.20) despite a
    # transient uptick, and energy is monotone → dormant
    progress = [(-10.0, 0.30), (-10.3, 0.35), (-10.6, 0.20)]
    assert struggle_trigger(progress, patience=2) is False
    # but with patience=1 the last single step is a stall (0.35 → 0.20 decreases,
    # so still dormant here); make the last step a real stall to confirm firing
    progress2 = [(-10.0, 0.30), (-10.3, 0.20), (-10.6, 0.25)]
    assert struggle_trigger(progress2, patience=1) is True
