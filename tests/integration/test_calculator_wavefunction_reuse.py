"""Reusing the previous ionic step's eigenvectors as the Davidson seed.

A relaxation/MD trajectory moves the atoms a little each step. The density warm
start already carries the converged charge to the next geometry; the *orbitals*
are just as reusable — the previous step's eigenvectors are a far better Davidson
starting subspace than a fresh random/identity block, so the eigensolver reaches
tolerance in fewer H-applications. ``reuse_wavefunctions`` (default on) seeds the
solver from ``last_result.coeffs`` alongside the density warm start; turning it
off recovers the density-only warm start (cold orbital seed).

``BatchedHamiltonian.apply`` is the single chokepoint every batched Davidson
round funnels H|ψ⟩ through (NC and USPP/PAW alike), so ``reset_happly_tally`` /
``happly_tally`` count the eigensolver work directly.

Oracles:
 1. reuse must not move the fixed point — same energy as the density-only warm
    start (and as a cold start), to tight tolerance;
 2. reuse must cut H-applications versus the density-only warm start on a
    positions-only step;
 3. across a variable-cell step that changes the FFT-grid shape, the remapped
    eigenvectors (Miller-index match onto the new G-spheres) still land on the
    cold fixed point.

use_symmetry is off here: with it on, the NC calculator caches the system on
(cell, symbols) and a positions-only move reuses the first geometry's IBZ/
symmetrizer, which shifts the warm energy independently of wavefunction reuse
(a separate concern) and would confound the comparison.
"""

import numpy as np
import torch
from ase import Atoms

from gradwave.calculator import GradWave
from gradwave.core.batch import happly_tally, reset_happly_tally
from tests.helpers import RY, SI_ONCV, pseudo

A0 = 5.43


def _si(rattle=0.0, scale=1.0):
    cell = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]) * scale
    pos = (A0 / 4 * np.array([[0.0, 0, 0], [1, 1, 1]])) * scale
    pos[1] = pos[1] + rattle
    return Atoms("Si2", positions=pos, cell=cell, pbc=True)


def _calc(**extra):
    tight = dict(ecut=12 * RY, xc="lda", kpts=(2, 2, 2), use_symmetry=False,
                 etol=1e-10, rhotol=1e-9, diago_tol=1e-11, max_iter=200)
    tight.update(extra)
    return GradWave(pseudopotentials={"Si": pseudo(SI_ONCV)}, **tight)


def _step2_happly(reuse):
    """Converge step 1, take a positions-only step 2, return (energy, H-apply
    count for step 2). reuse toggles the orbital seed; the density warm start is
    on either way."""
    a = _si()
    a.calc = _calc(reuse_wavefunctions=reuse)
    a.get_potential_energy()
    a.set_positions(_si(rattle=0.03).get_positions())
    reset_happly_tally()
    e = a.get_potential_energy()
    return e, happly_tally()


def test_reuse_cuts_davidson_work_same_fixed_point():
    torch.set_num_threads(4)

    e_reuse, h_reuse = _step2_happly(reuse=True)
    e_density, h_density = _step2_happly(reuse=False)

    # cold reference at the step-2 geometry (fresh calculator, no warm start)
    cold = _si(rattle=0.03)
    cold.calc = _calc()
    e_cold = cold.get_potential_energy()

    # (1) reuse does not move the fixed point
    assert abs(e_reuse - e_density) < 1e-9, (
        f"reuse shifted the energy: {abs(e_reuse - e_density):.2e} eV")
    assert abs(e_reuse - e_cold) < 1e-8 * len(cold), (
        f"warm fixed point differs from cold: {abs(e_reuse - e_cold):.2e} eV")

    # (2) reusing the eigenvectors cuts eigensolver work (deterministic counts)
    assert h_reuse < h_density, (
        f"reuse did not cut H-applications: {h_reuse} !< {h_density}")


def test_reuse_survives_variable_cell_grid_change():
    torch.set_num_threads(4)
    big = 1.1  # 10% dilation moves the ecut-fixed FFT box (15,15,15)->(18,18,18)

    cold = _si(rattle=0.02, scale=big)
    cold.calc = _calc(ecut=10 * RY, kpts=(1, 1, 1))
    e_cold = cold.get_potential_energy()
    shape_big = tuple(cold.calc.last_result.system.grid.shape)

    warm = _si(rattle=0.02, scale=1.0)
    warm.calc = _calc(ecut=10 * RY, kpts=(1, 1, 1))
    warm.get_potential_energy()
    shape_small = tuple(warm.calc.last_result.system.grid.shape)
    assert shape_small != shape_big, "cells must force a grid-shape change"
    assert warm.calc._warm_start_remaps == 0

    target = _si(rattle=0.02, scale=big)
    warm.set_cell(target.cell, scale_atoms=False)
    warm.set_positions(target.get_positions())
    e_warm = warm.get_potential_energy()

    # the grid-remap warm-start path (density + eigenvectors) ran once, on-grid
    assert warm.calc._warm_start_remaps == 1
    assert tuple(warm.calc.last_result.system.grid.shape) == shape_big
    # remapped eigenvectors seed the same fixed point the cold start finds
    assert abs(e_warm - e_cold) < 1e-8 * len(warm), (
        f"remapped warm start moved the fixed point: {abs(e_warm - e_cold):.2e} eV")
