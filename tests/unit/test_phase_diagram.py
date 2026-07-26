"""Phase 4 of the phase-diagram builder: the common-tangent construction. The
oracle is the symmetric regular solution, whose miscibility gap has a critical
temperature Omega / (2 k_B) and a binodal at the two free-energy minima."""

import numpy as np

from gradwave.postscf.phase_diagram import binodal, critical_temperature, two_phase_regions

KB = 8.617333262e-5  # eV/K
OMEGA = 0.10         # eV, positive so the solution unmixes
TC_EXACT = OMEGA / (2.0 * KB)  # ~ 580 K


def _regular_solution(x, temp):
    return OMEGA * x * (1 - x) + KB * temp * (x * np.log(x) + (1 - x) * np.log(1 - x))


def test_ideal_solution_has_no_gap():
    # zero interaction gives a convex free energy, so no two-phase region
    x = np.linspace(0.002, 0.998, 300)
    g = KB * 400.0 * (x * np.log(x) + (1 - x) * np.log(1 - x))
    assert two_phase_regions(x, g) == []


def test_regular_solution_binodal_and_critical_temperature():
    x = np.linspace(0.002, 0.998, 400)
    temps = np.linspace(200.0, 640.0, 45)
    _, left, right = binodal(temps, x, _regular_solution)

    # the gap closes at the critical temperature, within the temperature grid
    tc_est = critical_temperature(temps, left, right)
    assert abs(tc_est - TC_EXACT) < 1.5 * (temps[1] - temps[0]), (tc_est, TC_EXACT)
    # no gap above the critical temperature
    assert np.all(np.isnan(left[temps > TC_EXACT + 15]))

    # at a temperature well inside the gap the recovered boundary sits at the
    # free-energy minimum (the tangent point of the symmetric double well)
    ti = np.argmin(np.abs(temps - 300.0))
    g300 = _regular_solution(x, temps[ti])
    x_min_left = x[np.argmin(np.where(x < 0.5, g300, np.inf))]
    assert abs(left[ti] - x_min_left) < 2 * (x[1] - x[0])
    # symmetric gap
    assert abs((left[ti] + right[ti]) - 1.0) < 2 * (x[1] - x[0])
