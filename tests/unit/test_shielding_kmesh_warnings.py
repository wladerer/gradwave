"""Unit tests for the NMR-shielding k-mesh-suitability + equivalent-site warnings."""
import pytest

pytest.importorskip("spglib")
from ase.build import bulk
from ase.spacegroup import crystal

from gradwave.api.flapw import _equivalent_site_groups, _kmesh_symmetry_broken


def test_cubic_equal_mesh_is_suitable():
    si = bulk("Si", "diamond", a=5.431)
    assert _kmesh_symmetry_broken(si, [2, 2, 2])[0] == 0
    assert _kmesh_symmetry_broken(si, [4, 4, 4])[0] == 0


def test_cubic_unequal_mesh_breaks_symmetry():
    si = bulk("Si", "diamond", a=5.431)
    broken, total = _kmesh_symmetry_broken(si, [3, 3, 2])
    assert broken > 0 and total == 48        # cubic point group, unequal axis breaks it


def test_hexagonal_needs_equal_inplane():
    q = crystal(["Si", "O"], basis=[(0.4697, 0, 1 / 3), (0.4133, 0.2672, 0.2144)],
                spacegroup=152, cellpar=[4.9137, 4.9137, 5.4047, 90, 90, 120])
    assert _kmesh_symmetry_broken(q, [3, 3, 2])[0] == 0      # n_a=n_b ok
    assert _kmesh_symmetry_broken(q, [2, 3, 2])[0] > 0       # n_a≠n_b breaks the 3-fold


def test_equivalent_site_groups_quartz():
    q = crystal(["Si", "O"], basis=[(0.4697, 0, 1 / 3), (0.4133, 0.2672, 0.2144)],
                spacegroup=152, cellpar=[4.9137, 4.9137, 5.4047, 90, 90, 120])
    groups = _equivalent_site_groups(q)
    # the 3 Si form one orbit, the 6 O another
    sizes = sorted(len(g) for g in groups)
    assert sizes == [3, 6]
