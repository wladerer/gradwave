"""IBZ reduction on a perfect supercell (the Si-64-class case).

A perfect n×n×n supercell keeps the full point group of the primitive crystal
(spglib returns every rotation×centering coset; ``find_spacegroup`` dedups to one
representative per unique rotation — the point group). The decisive property this
pins: on an EVEN Monkhorst–Pack mesh, time-reversal alone (``kpoints.monkhorst_pack``)
does NO reduction — every mesh point has half-integer components and is its own
−k mod G — so the full space-group ``reduce_mesh`` is what recovers the IBZ. This
is the reduction a benchmark harness forfeits if it builds the mesh with TR-only
folding (``use_symmetry=False``) instead of the space group.
"""

import numpy as np

from gradwave.kpoints import monkhorst_pack
from gradwave.symmetry import find_spacegroup, reduce_mesh


def _si_supercell(nrep):
    """nrep³ repetitions of the 8-atom conventional diamond cell (cubic, all Si)."""
    a = 5.43
    base = np.array(
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
         [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]
    )
    frac = np.vstack([
        (base + np.array([i, j, k])) / nrep
        for i in range(nrep) for j in range(nrep) for k in range(nrep)
    ])
    cell = a * nrep * np.eye(3)
    return cell, frac, [0] * (8 * nrep**3)


def test_si64_supercell_keeps_full_point_group():
    """The 2×2×2 (64-atom) Si supercell is still Fd-3m with the full 48-rotation
    point group — spglib's pure-lattice-translation cosets are deduped to one rep
    per rotation, not dropped down to a smaller group."""
    cell, frac, soa = _si_supercell(2)
    sg = find_spacegroup(cell, frac, soa)
    assert sg.international == "Fd-3m"
    assert sg.n_ops == 48


def test_si64_even_mesh_reduces_only_under_spacegroup():
    """On the 2×2×2 mesh, TR-only folding leaves all 8 points (each is its own
    −k on an even mesh); the full space group folds them to QE's 4 irreducible
    k-points with weights summing to 1. This 8→4 is the reduction a
    ``use_symmetry=False`` benchmark run forfeits."""
    cell, frac, soa = _si_supercell(2)
    sg = find_spacegroup(cell, frac, soa)

    k_tr, w_tr = monkhorst_pack((2, 2, 2), (0, 0, 0), time_reversal=True)
    assert len(k_tr) == 8  # TR does nothing on an even mesh (all points self-inverse)

    k_sg, w_sg = reduce_mesh((2, 2, 2), (0, 0, 0), sg, time_reversal=True)
    assert len(k_sg) == 4  # matches QE "number of k points = 4" for this supercell/mesh
    assert abs(w_sg.sum() - 1.0) < 1e-12
    # the four stars are Γ(1) + ⟨½00⟩(3) + ⟨½½0⟩(3) + ½½½(1), summing to 8/8
    assert np.allclose(sorted(np.round(w_sg * 8)), [1, 1, 3, 3])


def test_si64_odd_mesh_spacegroup_beats_tr():
    """A no-regression contrast on the 4×4×4 mesh: TR-only → 36, space group → 10
    (both weight-normalized). Pins that the supercell fold is the space-group
    fold, not an accidental TR count."""
    cell, frac, soa = _si_supercell(2)
    sg = find_spacegroup(cell, frac, soa)
    k_tr, _ = monkhorst_pack((4, 4, 4), (0, 0, 0), time_reversal=True)
    k_sg, w_sg = reduce_mesh((4, 4, 4), (0, 0, 0), sg, time_reversal=True)
    assert len(k_tr) == 36
    assert len(k_sg) == 10
    assert abs(w_sg.sum() - 1.0) < 1e-12


def test_al_conventional_4mesh_no_regression():
    """Al-4-class path (the config where TR-only reads 36/64): the 4-atom fcc
    conventional cell, full space group on a 4×4×4 mesh. Pins that the plain
    space-group reduction still fires and beats TR-only there too."""
    a = 4.05
    cell = a * np.eye(3)
    frac = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    sg = find_spacegroup(cell, frac, [0, 0, 0, 0])
    assert sg.international == "Fm-3m"
    k_tr, _ = monkhorst_pack((4, 4, 4), (0, 0, 0), time_reversal=True)
    k_sg, w_sg = reduce_mesh((4, 4, 4), (0, 0, 0), sg, time_reversal=True)
    assert len(k_tr) == 36  # the TR-only count a use_symmetry=False run would report
    assert len(k_sg) < len(k_tr)
    assert abs(w_sg.sum() - 1.0) < 1e-12
