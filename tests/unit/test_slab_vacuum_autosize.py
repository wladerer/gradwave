"""Density-tail vacuum auto-sizer for open-boundary (ESM) slabs (api._slab).

Covers the pure-geometry core (vacuum-axis detection, tail trim, centering,
observable-aware margin, no-op-when-tight) with synthetic profiles, and the
gated ``resolve_slab_box`` wiring on a real Al slab / bulk cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from gradwave.api._slab import (
    _axis_vacuum_gaps,
    _npw_estimate,
    resolve_slab_box,
    size_box_from_planar_density,
)
from gradwave.inputs import Input, InputError, KPointsParams, SCFParams, SlabParams
from tests.helpers import pseudo

ALUPF = pseudo("Al_ONCV_PBE-1.2.upf")


# ---------------------------------------------------------------------------
# vacuum-axis detection
# ---------------------------------------------------------------------------
def test_axis_gaps_detects_z_vacuum():
    cell = np.diag([4.0, 4.0, 30.0])
    pos = np.array([[2.0, 2.0, 10.0], [2.0, 2.0, 12.5], [2.0, 2.0, 15.0]])
    gaps = _axis_vacuum_gaps(cell, pos)
    assert int(np.argmax(gaps)) == 2
    assert gaps[2] > 20.0  # ~25 Å of vacuum along z
    assert gaps[2] > gaps[0] and gaps[2] > gaps[1]  # z is the clear vacuum axis


def test_axis_gaps_bulk_has_no_large_gap():
    """A dense bulk cell has small gaps on every axis — no single vacuum axis."""
    cell = np.diag([4.0, 4.0, 4.0])
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 0.0, 2.0],
                    [0.0, 2.0, 2.0]])
    gaps = _axis_vacuum_gaps(cell, pos)
    assert gaps.max() <= 2.0 + 1e-9  # half the cell at most, no real vacuum


def test_axis_gaps_adsorbate_robust():
    """A protruding adsorbate shrinks the open-axis gap but must not move
    detection onto a periodic axis."""
    cell = np.diag([4.0, 4.0, 30.0])
    # slab z in [10,15], adsorbate sticking out at z=18
    pos = np.array([[2.0, 2.0, 10.0], [2.0, 2.0, 12.5], [2.0, 2.0, 15.0],
                    [2.0, 2.0, 18.0]])
    gaps = _axis_vacuum_gaps(cell, pos)
    assert int(np.argmax(gaps)) == 2


# ---------------------------------------------------------------------------
# pure-geometry trim core
# ---------------------------------------------------------------------------
def _gaussian_slab(L=30.0, nz=300, center=15.0, sigma=1.4, peak=0.5):
    z = (np.arange(nz) + 0.5) * (L / nz)
    return z, peak * np.exp(-((z - center) ** 2) / (2.0 * sigma**2))


def test_size_trims_to_tail_and_centers():
    cell = np.diag([4.0, 4.0, 30.0])
    pos = np.array([[2.0, 2.0, 12.0], [2.0, 2.0, 15.0], [2.0, 2.0, 18.0]])
    _, rho = _gaussian_slab()
    new_cell, new_pos, info = size_box_from_planar_density(
        cell, pos, rho, 2, rho_tol=1e-4, margin=1.0, min_vacuum=3.0)
    new_L = new_cell[2, 2]
    assert new_L < 30.0  # trimmed
    # in-plane coordinates untouched (open axis ⊥ in-plane pair)
    assert np.allclose(new_pos[:, :2], pos[:, :2])
    # slab recentered: mean atom z at the box midplane
    assert new_pos[:, 2].mean() == pytest.approx(new_L / 2.0, abs=0.3)
    # the occupied density extent + 2 margin (floor 12 Å not binding here)
    assert new_L == pytest.approx(info["extent"] + 2.0, abs=1e-6)


def test_workfunction_margin_is_more_conservative_than_energy():
    """Same slab, same tail: the plateau-limited workfunction target keeps a
    larger box than the tail-limited energy target."""
    cell = np.diag([4.0, 4.0, 30.0])
    pos = np.array([[2.0, 2.0, 13.0], [2.0, 2.0, 15.0], [2.0, 2.0, 17.0]])
    _, rho = _gaussian_slab()
    e_cell, _, _ = size_box_from_planar_density(
        cell, pos, rho, 2, rho_tol=1e-4, margin=1.0, min_vacuum=3.0)
    w_cell, _, _ = size_box_from_planar_density(
        cell, pos, rho, 2, rho_tol=1e-5, margin=4.0, min_vacuum=3.0)
    assert w_cell[2, 2] > e_cell[2, 2]


def test_size_never_grows_when_box_already_tight():
    """A box that is already just the tail + margin is left untouched."""
    cell = np.diag([4.0, 4.0, 12.0])
    pos = np.array([[2.0, 2.0, 5.0], [2.0, 2.0, 6.0], [2.0, 2.0, 7.0]])
    _, rho = _gaussian_slab(L=12.0, nz=120, center=6.0, sigma=1.4)
    new_cell, new_pos, info = size_box_from_planar_density(
        cell, pos, rho, 2, rho_tol=1e-4, margin=4.0, min_vacuum=3.0)
    assert new_cell[2, 2] == pytest.approx(12.0)
    assert np.allclose(new_pos, pos)
    assert "already tight" in info["reason"]


def test_size_safety_floor_binds_and_is_flagged():
    """A tiny tolerance-margin combo that wants less than min_vacuum/face is
    floored at the safety minimum."""
    cell = np.diag([4.0, 4.0, 30.0])
    pos = np.array([[2.0, 2.0, 14.0], [2.0, 2.0, 15.0], [2.0, 2.0, 16.0]])
    # narrow density, tiny margin: extent+2*margin would be < atom_span+2*3
    _, rho = _gaussian_slab(center=15.0, sigma=0.5)
    new_cell, _, info = size_box_from_planar_density(
        cell, pos, rho, 2, rho_tol=1e-2, margin=0.1, min_vacuum=3.0)
    atom_span = 2.0
    assert new_cell[2, 2] == pytest.approx(atom_span + 2 * 3.0, abs=1e-6)
    assert info["hit_floor"] is True


def test_size_no_trim_when_density_fills_axis():
    cell = np.diag([4.0, 4.0, 10.0])
    rho = np.full(100, 0.5)  # everywhere above tol
    pos = np.array([[2.0, 2.0, 5.0]])
    new_cell, _, info = size_box_from_planar_density(
        cell, pos, rho, 2, rho_tol=1e-4, margin=1.0, min_vacuum=3.0)
    assert new_cell[2, 2] == pytest.approx(10.0)
    assert "no vacuum" in info["reason"]


# ---------------------------------------------------------------------------
# npw estimate
# ---------------------------------------------------------------------------
def test_npw_estimate_scales_with_open_axis():
    """npw ∝ Ω ∝ open-axis length — the core of the trim's arithmetic win."""
    ecut = 400.0
    n_thick = _npw_estimate(np.diag([4.0, 4.0, 30.0]), ecut)
    n_thin = _npw_estimate(np.diag([4.0, 4.0, 15.0]), ecut)
    assert n_thick == pytest.approx(2 * n_thin, rel=0.05)


# ---------------------------------------------------------------------------
# resolve_slab_box wiring + gates (real Al slab)
# ---------------------------------------------------------------------------
def _al_slab(c=32.0, ecut=300.0, boundary="open_z", slab=None) -> Input:
    a = 4.05  # Å (fcc Al), 3-layer (100)-like column with a big z vacuum
    atoms = Atoms(
        "Al3", cell=[a / np.sqrt(2), a / np.sqrt(2), c], pbc=True,
        positions=[[0, 0, 10.0], [0, 0, 12.0], [0, 0, 14.0]])
    return Input(
        atoms=atoms, pseudo_dir=Path(ALUPF).parent,
        pseudo_map={"Al": Path(ALUPF).name}, ecut=ecut,
        kpoints=KPointsParams(mesh=(1, 1, 1)),
        scf=SCFParams(boundary=boundary),
        slab=slab or SlabParams(), verbose=False)


def test_resolve_noop_when_disabled():
    inp = _al_slab(slab=SlabParams(vacuum_autosize=False))
    box = resolve_slab_box(inp, [_load()], [0, 0, 0])
    assert box.trimmed is False
    assert box.length_after == 0.0 or box.length_after == box.length_before
    assert np.allclose(box.cell, inp.atoms.cell.array)


def test_resolve_noop_when_boundary_periodic():
    inp = _al_slab(boundary="periodic",
                   slab=SlabParams(vacuum_autosize=True, npw_gate=0))
    box = resolve_slab_box(inp, [_load()], [0, 0, 0])
    assert box.trimmed is False
    assert "not an open" in box.reason


def test_resolve_noop_below_npw_gate():
    """A small box (low npw) is latency-bound — leave it, even with vacuum."""
    inp = _al_slab(ecut=40.0, slab=SlabParams(vacuum_autosize=True))
    box = resolve_slab_box(inp, [_load()], [0, 0, 0])
    assert box.trimmed is False
    assert "latency-bound" in box.reason


def test_resolve_noop_below_vacuum_fraction_gate():
    """A cell with little vacuum (bulk-ish) is left untouched — the gate fires
    before any density is computed."""
    a = 4.05
    atoms = Atoms(  # 4 Å-thick slab in a 9 Å box → vacuum fraction ~0.55
        "Al3", cell=[a / np.sqrt(2), a / np.sqrt(2), 9.0], pbc=True,
        positions=[[0, 0, 2.5], [0, 0, 4.5], [0, 0, 6.5]])
    inp = Input(
        atoms=atoms, pseudo_dir=Path(ALUPF).parent,
        pseudo_map={"Al": Path(ALUPF).name}, ecut=300.0,
        kpoints=KPointsParams(mesh=(1, 1, 1)),
        scf=SCFParams(boundary="open_z"),
        slab=SlabParams(vacuum_autosize=True, npw_gate=0,
                        vacuum_fraction_gate=0.6), verbose=False)
    box = resolve_slab_box(inp, [_load()], [0, 0, 0])
    assert box.trimmed is False
    assert "vacuum fraction" in box.reason


def test_resolve_trims_real_al_slab():
    """Gates satisfied → the Al slab box is trimmed to the SAD tail; the atoms
    stay in-plane and re-center along the open axis."""
    inp = _al_slab(c=32.0, ecut=300.0,
                   slab=SlabParams(vacuum_autosize=True, npw_gate=0,
                                   vacuum_target="energy"))
    box = resolve_slab_box(inp, [_load()], [0, 0, 0])
    assert box.trimmed is True
    assert box.axis == 2
    assert box.length_after < box.length_before
    # sanity: enough vacuum kept (slab ~4 Å thick + margins) but well under 32
    assert 8.0 < box.length_after < 28.0
    assert np.allclose(box.positions[:, :2], inp.atoms.get_positions()[:, :2])
    # workfunction target keeps a larger box than energy on the same slab
    inp_wf = _al_slab(c=32.0, ecut=300.0,
                      slab=SlabParams(vacuum_autosize=True, npw_gate=0,
                                      vacuum_target="workfunction"))
    box_wf = resolve_slab_box(inp_wf, [_load()], [0, 0, 0])
    assert box_wf.length_after > box.length_after


def test_resolve_bulk_cell_untouched():
    """A genuine bulk cell (no vacuum axis) never trims, gates aside."""
    a = 4.05
    atoms = Atoms("Al4", cell=[a, a, a], pbc=True,
                  positions=[[0, 0, 0], [a / 2, a / 2, 0],
                             [a / 2, 0, a / 2], [0, a / 2, a / 2]])
    inp = Input(
        atoms=atoms, pseudo_dir=Path(ALUPF).parent,
        pseudo_map={"Al": Path(ALUPF).name}, ecut=300.0,
        kpoints=KPointsParams(mesh=(1, 1, 1)),
        scf=SCFParams(boundary="open_z"),
        slab=SlabParams(vacuum_autosize=True, npw_gate=0), verbose=False)
    box = resolve_slab_box(inp, [_load()], [0, 0, 0, 0])
    # bulk fills the box on every axis: no contiguous sub-tolerance vacuum run,
    # so nothing is trimmed (whichever guard catches it, the box is untouched).
    assert box.trimmed is False
    assert np.allclose(box.cell, inp.atoms.cell.array)


# ---------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------
def test_slabparams_rejects_bad_target():
    with pytest.raises(InputError):
        SlabParams(vacuum_target="dipole")


def test_slabparams_rejects_nonpositive_tol():
    with pytest.raises(InputError):
        SlabParams(vacuum_tol=0.0)


# ---------------------------------------------------------------------------
def _load():
    from gradwave.pseudo.upf import parse_upf
    return parse_upf(ALUPF)
