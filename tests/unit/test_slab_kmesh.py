"""Slab-aware anisotropic k-mesh autoconfig (kpoints.slab_kmesh) and the
converge-time vacuum-axis pin (api.converge._scaled_mesh).

A slab's vacuum axis carries no dispersion (bands are flat across a
vacuum-separated image), so sampling it with more than one k-point is pure
waste — a (6,6,1) mesh refined uniformly to (9,9,2) doubles the k-count for no
accuracy. These are pure-numeric guards (no SCF): the mesh arithmetic, the
vacuum-axis detection (robust to a protruding adsorbate), the bulk-untouched
case, and the forced-axis warning.
"""

import numpy as np
import pytest

from gradwave.api.converge import _scaled_mesh
from gradwave.kpoints import _axis_vacuum_gaps, slab_kmesh


def _slab(lz=20.0, top=5.0, n_layer=4, a=3.0):
    """Orthorhombic slab: atoms stacked 0..top Å along z, vacuum to lz."""
    cell = np.diag([a, a, lz]).astype(float)
    zs = np.linspace(0.0, top, n_layer)
    pos = np.array([[0.3 * a, 0.3 * a, z] for z in zs])
    return cell, pos


def test_slab_kmesh_returns_n_n_1():
    cell, pos = _slab()
    mesh = slab_kmesh(cell, pos, kspacing=0.3)
    # in-plane |b| = 2π/3 ≈ 2.094 Å⁻¹ → ceil(2.094/0.3) = 7
    assert mesh == (7, 7, 1)
    # Γ-centered pairing is the caller's contract; the vacuum axis is pinned.
    assert mesh[2] == 1


def test_vacuum_axis_detected_is_z():
    cell, pos = _slab()
    gaps = _axis_vacuum_gaps(cell, pos)
    assert int(np.argmax(gaps)) == 2
    assert gaps[2] > gaps[0] and gaps[2] > gaps[1]


def test_protruding_adsorbate_does_not_move_detection():
    # A CO-like adsorbate sticking up into the vacuum shrinks the z gap but must
    # not shift detection onto a periodic axis.
    cell, pos = _slab(lz=20.0, top=5.0)
    pos = np.vstack([pos, [1.5, 1.5, 7.0]])  # adsorbate at z=7, vacuum 7..20
    mesh = slab_kmesh(cell, pos, kspacing=0.3)
    assert mesh == (7, 7, 1)


def test_bulk_cell_untouched():
    # A dense-packed cubic cell has small gaps on every axis: no pin, all three
    # axes solved from kspacing.
    cell = np.diag([3.0, 3.0, 3.0]).astype(float)
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
    mesh = slab_kmesh(cell, pos, kspacing=0.3)
    assert mesh == (7, 7, 7)
    assert 1 not in mesh  # nothing pinned


def test_forced_axis_below_threshold_warns_but_pins():
    cell = np.diag([3.0, 3.0, 3.0]).astype(float)
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
    with pytest.warns(UserWarning, match="vacuum"):
        mesh = slab_kmesh(cell, pos, kspacing=0.3, vacuum_axis=0)
    assert mesh[0] == 1
    assert mesh[1] == 7 and mesh[2] == 7


def test_kspacing_must_be_positive():
    cell, pos = _slab()
    with pytest.raises(ValueError, match="positive"):
        slab_kmesh(cell, pos, kspacing=0.0)


def test_bad_forced_axis_rejected():
    cell, pos = _slab()
    with pytest.raises(ValueError, match="vacuum_axis"):
        slab_kmesh(cell, pos, kspacing=0.3, vacuum_axis=3)


# ---- converge-time uniform-scaling pin -------------------------------------


def test_scaled_mesh_pins_vacuum_axis():
    # the whole point: a slab mesh does not grow its pinned axis under refinement
    assert _scaled_mesh((6, 6, 1), 1.5) == (9, 9, 1)
    assert _scaled_mesh((6, 6, 1), 2.0) == (12, 12, 1)


def test_scaled_mesh_bulk_unchanged():
    assert _scaled_mesh((4, 4, 4), 2.0) == (8, 8, 8)
    assert _scaled_mesh((4, 4, 4), 1.5) == (6, 6, 6)


def test_scaled_mesh_wire_pins_two_axes():
    # a 1D wire (two vacuum axes) keeps both pinned — up to ~6x k-waste avoided
    assert _scaled_mesh((1, 1, 4), 2.0) == (1, 1, 8)
