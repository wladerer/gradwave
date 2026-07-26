"""Zero-temperature convex hull for binary alloys (phase-diagram phase 2).

The ground states of a binary A-B alloy are the structures whose formation energy
lies on the lower convex hull of formation energy against composition. A
structure above the hull is unstable, since it lowers its energy by phase
separating into the two hull neighbours that bracket its composition, and the
energy it gains by doing so is its distance above the hull. This module builds
the hull, labels the ground states, and reports each structure's decomposition
energy. Energies are per atom, so the tie-lines are the common-tangent lines the
finite-temperature construction generalizes.
"""

from __future__ import annotations

import numpy as np


def formation_energy(energy_per_atom, x, e_a, e_b):
    """Formation energy per atom relative to the pure endpoints, E - (1-x) E_A -
    x E_B. energy_per_atom and x may be arrays. e_a and e_b are the per-atom
    energies of pure A (x=0) and pure B (x=1)."""
    x = np.asarray(x, dtype=float)
    return np.asarray(energy_per_atom, dtype=float) - (1.0 - x) * e_a - x * e_b


def lower_convex_hull(x, e):
    """Indices of the points on the lower convex hull of (x, e), sorted by x.
    Monotone chain keeping only left turns, so the result is the lower envelope
    that minimizes energy at every composition."""
    x = np.asarray(x, dtype=float)
    e = np.asarray(e, dtype=float)
    order = sorted(range(len(x)), key=lambda i: (x[i], e[i]))

    def cross(o, a, b):
        return ((x[a] - x[o]) * (e[b] - e[o])
                - (e[a] - e[o]) * (x[b] - x[o]))

    hull: list[int] = []
    for i in order:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], i) <= 0:
            hull.pop()
        hull.append(i)
    return hull


def hull_distance(x, e):
    """Energy of each point above the lower hull [eV/atom]. Zero on the hull,
    positive above it. The positive value is the energy released by decomposing
    into the two bracketing ground states."""
    x = np.asarray(x, dtype=float)
    e = np.asarray(e, dtype=float)
    verts = lower_convex_hull(x, e)
    xv = x[verts]
    ev = e[verts]
    e_on_hull = np.interp(x, xv, ev)
    return e - e_on_hull


def ground_states(x, e, tol=1e-9):
    """Indices of the stable structures, the ones on the hull to within tol."""
    d = hull_distance(x, e)
    return [i for i in range(len(x)) if d[i] <= tol]


def leave_one_out_rmse(x, energies, fit_predict):
    """Leave-one-out cross-validation RMSE [eV/atom] for an energy model.
    fit_predict(train_x, train_e, test_x) fits on the training subset and
    returns the predicted energy at the held-out composition, so any model (a
    cluster expansion from composition_design, a spline) plugs in."""
    x = np.asarray(x, dtype=float)
    energies = np.asarray(energies, dtype=float)
    n = len(x)
    err = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        pred = fit_predict(x[mask], energies[mask], x[i])
        err[i] = pred - energies[i]
    return float(np.sqrt(np.mean(err ** 2)))
