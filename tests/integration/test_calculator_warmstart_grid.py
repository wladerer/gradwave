"""Warm-start density survives an FFT-grid-shape change.

During variable-cell relaxation the cell strains; at fixed ecut the FFT box
dims change from step to step. The old warm start bailed to a cold SAD start
whenever the previous density's grid shape differed from the new one, so every
such step paid the full 4-8x cold-start SCF cost. The calculator now remaps the
previous density onto the new grid in G-space (match Miller indices, zero-fill
new high-G components, truncate ones that no longer fit) and renormalizes to the
electron count on the new cell volume, then warm-starts from that.

Oracle: a variable-cell step that forces a grid-shape change must
 1. take the remap path (``calc._warm_start_remaps`` increments), and
 2. converge to the SAME energy as a cold start on the strained cell (the
    fixed point is start-independent), in FEWER SCF iterations than the cold
    SAD start.
"""

import numpy as np
import torch
from ase import Atoms

from gradwave.calculator import GradWave
from tests.helpers import RY, SI_ONCV, pseudo

A0 = 5.43
# slight rattle so the geometry is generic (nonzero forces/stress)
RATTLE = np.array([[0.0, 0.0, 0.0], [0.02, -0.01, 0.015]])
# a 10% dilation moves the ecut-fixed FFT box from (15,15,15) to (18,18,18)
BIG = 1.1


def _atoms(scale=1.0):
    cell = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]) * scale
    pos = (A0 / 4 * np.array([[0.0, 0, 0], [1, 1, 1]]) + RATTLE) * scale
    return Atoms("Si2", positions=pos, cell=cell, pbc=True)


def _calc(**extra):
    tight = dict(etol=1e-10, rhotol=1e-9, diago_tol=1e-11, max_iter=200)
    tight.update(extra)
    return GradWave(ecut=10 * RY, pseudopotentials={"Si": pseudo(SI_ONCV)},
                    xc="lda", kpts=(1, 1, 1), **tight)


def test_warmstart_survives_grid_shape_change():
    torch.set_num_threads(4)

    # cold reference on the strained (big) cell
    cold = _atoms(scale=BIG)
    cold.calc = _calc()
    e_cold = cold.get_potential_energy()
    n_cold = cold.calc.last_result.n_iter
    shape_big = tuple(cold.calc.last_result.system.grid.shape)

    # warm path: converge on the small cell, then take one variable-cell step to
    # the big cell (both cell and positions change) reusing the same calculator
    warm = _atoms(scale=1.0)
    warm.calc = _calc()
    warm.get_potential_energy()
    shape_small = tuple(warm.calc.last_result.system.grid.shape)
    assert shape_small != shape_big, (
        f"cells must force a grid-shape change ({shape_small} vs {shape_big})")
    assert warm.calc._warm_start_remaps == 0  # nothing to remap yet

    big = _atoms(scale=BIG)
    warm.set_cell(big.cell, scale_atoms=False)
    warm.set_positions(big.get_positions())
    e_warm = warm.get_potential_energy()
    n_warm = warm.calc.last_result.n_iter

    # the grid-remap warm-start path was taken exactly once
    assert warm.calc._warm_start_remaps == 1
    # and it landed on the new grid
    assert tuple(warm.calc.last_result.system.grid.shape) == shape_big

    # same fixed point as the cold start, to ~1e-8 eV/atom
    assert abs(e_warm - e_cold) < 1e-8 * len(warm), (
        f"warm fixed point moved: {abs(e_warm - e_cold):.2e} eV")
    # the remapped seed pays off: fewer SCF iterations than the cold SAD start
    assert n_warm < n_cold, f"warm {n_warm} not < cold {n_cold} iterations"


def test_extrapolation_survives_cell_volume_change():
    """#259: density extrapolation across a cell-VOLUME change at fixed FFT grid
    must keep the SCF seed normalized.

    ``_extrapolated_remainder`` renormalized the seed to N_e at the CURRENT cell
    volume, but the seed is tagged with the PREVIOUS system, so
    ``warm_start_densities`` rescaled it by prev/current again — leaving the
    electron count (the density's G=0 component) off by the volume ratio, which
    the mixer rejected as "density not normalized". It bit every extrapolating
    order (linear and quadratic), only under a changing cell; fixed-cell relaxes
    were fine (ratio 1). Tiny isotropic strains keep the grid at (15,15,15) so the
    history accumulates and the order-1 then order-2 extrapolation both fire.
    """
    torch.set_num_threads(4)
    calc = _calc(extrapolation="quadratic")
    cell0 = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos0 = A0 / 4 * np.array([[0.0, 0, 0], [1, 1, 1]])
    at = Atoms("Si2", positions=pos0, cell=cell0, pbc=True)
    at.calc = calc
    shapes: set = set()
    energies = []
    for s in (1.0, 1.001, 1.002, 1.003):   # grid fixed, volume changes each step
        at.set_cell(cell0 * s, scale_atoms=False)
        at.set_positions(pos0 * s)
        energies.append(at.get_potential_energy())   # must NOT raise (#259)
        shapes.add(tuple(calc.last_result.system.grid.shape))

    assert shapes == {(15, 15, 15)}, f"grid must stay fixed to exercise it: {shapes}"
    assert len(calc._history) >= 3   # the order-2 (quadratic) extrapolation engaged
    assert all(np.isfinite(e) for e in energies)
