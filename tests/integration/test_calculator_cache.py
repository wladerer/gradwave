"""Self-oracle for the calculator's single-pass properties + geometry guard.

The GradWave calculator computes energy, forces, AND stress in one
calculate() pass and skips the SCF when the incoming atoms match the state
the stored result was computed at. The motivating access pattern is ASE's
FrechetCellFilter, which requests forces and stress as two separate
get_property calls: without the single pass every variable-cell BFGS step
paid a second (warm) SCF plus a full forces evaluation just to append the
stress.

Two oracles:

1. SCF-call counting (monkeypatched ``gradwave.calculator.scf``): one
   calculate() populates energy/free_energy/forces/stress, and neither
   repeated property requests, a direct re-entrant calculate(), nor a
   reset()-then-request re-run the SCF at unchanged geometry — while a real
   move does.

2. Identity under the guard: one BFGS + FrechetCellFilter step gives the
   same energy/forces/stress/geometry with the guard active as with it
   disabled (every calculate() re-solving, i.e. the old two-call behavior).
   The reused state and the re-solved state must be the same fixed point;
   1e-10 is far below any physically meaningful difference and observed
   agreement is bit-exact.
"""

import numpy as np
import torch
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

import gradwave.calculator as calcmod
from gradwave.calculator import GradWave
from tests.helpers import RY, SI_ONCV, pseudo

A0 = 5.43
# slight rattle so forces/stress are nonzero and the geometry is generic
RATTLE = np.array([[0.0, 0.0, 0.0], [0.02, -0.01, 0.015]])


def _atoms(scale=1.0):
    cell = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]) * scale
    pos = (A0 / 4 * np.array([[0.0, 0, 0], [1, 1, 1]]) + RATTLE) * scale
    return Atoms("Si2", positions=pos, cell=cell, pbc=True)


def _calc(**extra):
    return GradWave(ecut=10 * RY, pseudopotentials={"Si": pseudo(SI_ONCV)},
                    xc="lda", kpts=(1, 1, 1), **extra)


def test_single_pass_populates_all_and_guard_skips_scf(monkeypatch):
    torch.set_num_threads(4)
    calls = {"n": 0}
    real_scf = calcmod.scf

    def counting_scf(*args, **kwargs):
        calls["n"] += 1
        return real_scf(*args, **kwargs)

    monkeypatch.setattr(calcmod, "scf", counting_scf)

    atoms = _atoms()
    atoms.calc = _calc()
    atoms.get_potential_energy()
    assert calls["n"] == 1
    # one pass computed everything, stress included (not gated on request)
    for name in ("energy", "free_energy", "forces", "stress"):
        assert name in atoms.calc.results, name

    # ASE's forces-then-stress sequence (what FrechetCellFilter issues) is
    # answered from the stored results — no SCF re-run
    atoms.get_forces()
    atoms.get_stress()
    assert calls["n"] == 1

    # direct re-entrant calculate() at unchanged geometry: guard skips the SCF
    atoms.calc.calculate(atoms, ["energy"], [])
    assert calls["n"] == 1

    # reset() clears self.results; a request then re-enters calculate() with
    # unchanged geometry and must restore, not re-solve
    stress = atoms.get_stress().copy()
    atoms.calc.reset()
    assert np.array_equal(atoms.get_stress(), stress)
    assert calls["n"] == 1

    # a genuine move invalidates the guard and re-runs the SCF
    pos = atoms.get_positions()
    pos[1, 0] += 0.05
    atoms.set_positions(pos)
    atoms.get_forces()
    assert calls["n"] == 2


def test_cellfilter_step_identical_with_guard_disabled(monkeypatch):
    torch.set_num_threads(4)

    def run(disable_guard):
        atoms = _atoms(scale=1.01)  # strained so the cell DOF has work to do
        atoms.calc = _calc(etol=1e-10, rhotol=1e-9, diago_tol=1e-11,
                           max_iter=200)
        if disable_guard:
            # a key that never matches: every calculate() re-runs the SCF,
            # reproducing the old re-entrant forces-then-stress behavior
            atoms.calc._state_key = lambda a: object()
        opt = BFGS(FrechetCellFilter(atoms), logfile=None)
        opt.run(fmax=1e-8, steps=1)
        return (atoms.get_potential_energy(), atoms.get_forces(),
                atoms.get_stress(), atoms.get_positions(), atoms.cell.array)

    e_new, f_new, s_new, p_new, c_new = run(False)
    e_old, f_old, s_old, p_old, c_old = run(True)

    assert abs(e_new - e_old) < 1e-10
    assert np.abs(f_new - f_old).max() < 1e-10
    assert np.abs(s_new - s_old).max() < 1e-10
    assert np.abs(p_new - p_old).max() < 1e-10
    assert np.abs(c_new - c_old).max() < 1e-10
