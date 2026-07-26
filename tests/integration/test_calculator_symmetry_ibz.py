"""Regression: the NC calculator must rebuild the IBZ/symmetrizer when a
positions-only move breaks the spacegroup ops it reduced against (#128).

With ``use_symmetry=True`` the calculator reduces the k-mesh to the irreducible
Brillouin zone and symmetrizes the density using the operations spglib finds for
the *current* geometry. During a relaxation/MD the atoms move a little each step,
and a move can break some of those operations — the moved configuration has a
lower-symmetry group and therefore a larger IBZ.

The bug: ``_get_system`` cached the built System on ``(cell, symbols)`` only and,
on a positions-only move, handed back that cached System with the new positions
swapped in — keeping the *first* geometry's high-symmetry IBZ and density
symmetrizer. The SCF then converged against too few k-points / an over-symmetric
density, shifting the warm-start energy by up to ~0.3 eV versus a fresh
calculator at the moved geometry. (The USPP path already rebuilt every call under
symmetry; only the NC path kept the stale reduction.)

Oracle: converge a high-symmetry Si diamond cell, take a positions-only step that
breaks the symmetry, and require the warm-started energy to match a cold-start
(fresh calculator) energy at the moved geometry to tight tolerance. The two are
the same physical fixed point; a stale IBZ is the only thing that moves them
apart, and it moved them ~0.3 eV apart before the fix.
"""

import numpy as np
import torch
from ase import Atoms

from gradwave.calculator import GradWave
from tests.helpers import RY, SI_ONCV, pseudo

A0 = 5.43


def _si(disp=None):
    """Si diamond primitive cell. disp (a Cartesian vector, Å) displaces the
    second atom off its ideal Wyckoff site, breaking the diamond spacegroup."""
    cell = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = A0 / 4 * np.array([[0.0, 0, 0], [1, 1, 1]])
    if disp is not None:
        pos = pos.copy()
        pos[1] = pos[1] + np.asarray(disp, dtype=float)
    return Atoms("Si2", positions=pos, cell=cell, pbc=True)


def _calc(**extra):
    tight = dict(ecut=12 * RY, xc="lda", kpts=(2, 2, 2), use_symmetry=True,
                 etol=1e-10, rhotol=1e-9, diago_tol=1e-11, max_iter=200)
    tight.update(extra)
    return GradWave(pseudopotentials={"Si": pseudo(SI_ONCV)}, **tight)


def test_positions_move_breaking_symmetry_rebuilds_ibz():
    torch.set_num_threads(4)

    # a generic displacement that lowers the diamond group (asymmetric on all
    # three axes so it is not a residual symmetry direction)
    disp = np.array([0.08, -0.05, 0.11])

    # warm: converge the high-symmetry cell first, then take the positions-only
    # step. Before the fix this reuses the ideal cell's high-symmetry IBZ.
    warm = _si()
    warm.calc = _calc()
    warm.get_potential_energy()
    warm.set_positions(_si(disp).get_positions())
    e_warm = warm.get_potential_energy()

    # cold: a fresh calculator sees only the moved (lower-symmetry) geometry, so
    # spglib gives it the correct, larger IBZ.
    cold = _si(disp)
    cold.calc = _calc()
    e_cold = cold.get_potential_energy()

    # same physical fixed point; the stale IBZ was the sole source of the shift.
    assert abs(e_warm - e_cold) < 1e-6, (
        f"warm-start energy differs from cold: {abs(e_warm - e_cold):.3e} eV "
        f"(stale IBZ/symmetrizer reused across a symmetry-breaking move)")
