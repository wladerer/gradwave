"""Phase 3 of the phase-diagram builder: lattice Monte Carlo for the
configurational thermodynamics. The order-disorder oracle is the square-lattice
Ising model, whose exact transition (Onsager) is at 2 / ln(1 + sqrt(2))."""

import numpy as np
import pytest

from gradwave.postscf.lattice_mc import (
    configurational_entropy,
    order_disorder_temperature,
    scan_temperature,
)

ONSAGER_TC = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ~ 2.269


@pytest.mark.standard
def test_ising_order_disorder_transition():
    temps = np.linspace(1.6, 3.2, 17)
    out = scan_temperature((16, 16), temps, n_equil=1200, n_sample=2000, seed=3)
    tc = order_disorder_temperature(temps, out["cv"])
    # the finite-lattice specific-heat peak sits just above the exact transition
    assert abs(tc - ONSAGER_TC) < 0.2, (tc, ONSAGER_TC)
    # ordered below the transition, disordered above
    assert out["abs_m"][0] > 0.85     # T = 1.6
    assert out["abs_m"][-1] < 0.3     # T = 3.2
    # the specific heat peaks near the transition, not at the grid ends
    assert 0 < int(np.argmax(out["cv"])) < len(temps) - 1


def test_configurational_entropy_matches_analytic():
    # a constant specific heat integrates to S(T) = S_inf - c ln(Tmax / T)
    temps = np.linspace(1.0, 20.0, 400)
    c = 0.3
    cv = np.full_like(temps, c)
    s_inf = np.log(2.0)
    s = configurational_entropy(temps, cv, s_infinity=s_inf)
    analytic = s_inf - c * np.log(temps[-1] / temps)
    assert np.allclose(s, analytic, atol=1e-3)
    assert abs(s[-1] - s_inf) < 1e-12          # reference holds at the top
    assert np.all(np.diff(s) > 0)              # entropy rises with temperature
