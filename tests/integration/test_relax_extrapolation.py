"""Cross-step density extrapolation cuts SCF iterations on a relax trajectory.

``relax.extrapolation`` seeds each ionic step's SCF from an extrapolation of the
previous converged steps instead of re-converging from the atomic superposition.
This drives a fixed sequence of geometries (a curved march, so second-order
extrapolation has real signal) through one persistent calculator per mode — the
nested-relax reuse pattern — and checks the two properties that matter: the seed
never moves the fixed point (identical converged energies across modes), and the
extrapolating modes reach it in no more SCF iterations than plain reuse, and
strictly fewer than a cold start.

Standard tier: a handful of tightly-converged Si2 SCFs is seconds of work, above
the fast gate's per-test budget.
"""

import numpy as np
import pytest
from ase import Atoms

from gradwave.calculator import GradWave
from tests.helpers import RY, SI_ONCV, pseudo

pytestmark = pytest.mark.standard

A0 = 5.43
_IDEAL = A0 / 4 * np.array([[0.0, 0, 0], [1, 1, 1]])
_CELL = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _positions_path(n=5, amp=0.10, seed=0):
    """A curved (quadratic-in-step) march of the second atom toward the ideal
    site: smooth enough for extrapolation to predict the next density well."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(0, 1, 3)
    direction /= np.linalg.norm(direction)
    path = []
    for t in np.linspace(1.0, 0.0, n):
        pos = _IDEAL.copy()
        pos[1] = pos[1] + amp * (t * t) * direction
        path.append(pos)
    return path


def _calc(mode):
    return GradWave(pseudopotentials={"Si": pseudo(SI_ONCV)}, ecut=12 * RY,
                    xc="lda", kpts=(2, 2, 2), use_symmetry=False, etol=1e-9,
                    rhotol=1e-8, diago_tol=1e-10, max_iter=200,
                    extrapolation=mode)


def _run(mode, path):
    """Total SCF iterations and per-step converged energies over the path, using
    one calculator so the extrapolation history accumulates across steps."""
    atoms = Atoms("Si2", positions=path[0], cell=_CELL, pbc=True)
    atoms.calc = _calc(mode)
    total, energies = 0, []
    for pos in path:
        atoms.set_positions(pos)
        energies.append(atoms.get_potential_energy())
        total += int(atoms.calc.last_result.n_iter)
    return total, energies


def test_extrapolation_cuts_iterations_same_fixed_point():
    path = _positions_path()
    tot_none, en_none = _run("none", path)
    tot_reuse, en_reuse = _run("reuse", path)
    tot_quad, en_quad = _run("quadratic", path)

    # (a) the seed is a starting point only: every mode converges to the same
    # energy at each geometry, to tight tolerance.
    for a, b in zip(en_reuse, en_none, strict=True):
        assert abs(a - b) < 1e-6
    for a, b in zip(en_quad, en_none, strict=True):
        assert abs(a - b) < 1e-6

    # (b) extrapolation never costs more iterations than plain reuse, and beats a
    # cold start outright.
    assert tot_quad <= tot_reuse
    assert tot_quad < tot_none
