"""Phase 2 of the phase-diagram builder: the zero-temperature convex hull. A
synthetic binary with a hand-checked ground-state set, so no SCF runs here."""

import numpy as np

from gradwave.postscf.convex_hull import (
    formation_energy,
    ground_states,
    hull_distance,
    leave_one_out_rmse,
    lower_convex_hull,
)


def test_formation_energy_zero_at_endpoints():
    # E - (1-x) E_A - x E_B is zero for the pure endpoints
    ef = formation_energy([-5.0, -6.0], x=[0.0, 1.0], e_a=-5.0, e_b=-6.0)
    assert np.allclose(ef, [0.0, 0.0])


# A hand-worked binary. Tie-line A->0.5 at x=0.25 is -0.05, so the x=0.25 point
# at -0.02 sits above the hull and decomposes; the x=0.75 point at -0.08 lies
# below the 0.5->B tie-line (-0.05), so it joins the hull as a ground state.
X = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
EF = np.array([0.0, -0.02, -0.10, -0.08, 0.0])


def test_hull_identifies_ground_states():
    gs = set(ground_states(X, EF))
    assert gs == {0, 2, 3, 4}  # x=0.25 is metastable, excluded


def test_hull_vertices_sorted_by_composition():
    verts = lower_convex_hull(X, EF)
    assert verts == [0, 2, 3, 4]
    assert list(X[verts]) == [0.0, 0.5, 0.75, 1.0]


def test_hull_distance_signs_and_metastable_value():
    d = hull_distance(X, EF)
    assert np.all(d >= -1e-12)                 # nothing below the hull
    assert np.allclose(d[[0, 2, 3, 4]], 0.0)   # ground states sit on it
    assert np.isclose(d[1], 0.03)              # x=0.25 decomposes by 0.03 eV/atom


def test_leave_one_out_rmse_on_linear_data():
    # a linear energy vs composition is reproduced exactly by a linear fit, which
    # extrapolates to the held-out endpoints (unlike clamped interpolation)
    x = np.linspace(0, 1, 8)
    e = 2.0 - 0.5 * x

    def linfit(tx, te, qx):
        return float(np.polyval(np.polyfit(tx, te, 1), qx))

    assert leave_one_out_rmse(x, e, linfit) < 1e-10
