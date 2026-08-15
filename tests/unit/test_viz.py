"""gradwave.io.viz: pymatviz structure and periodic-table helpers.

Skipped in full when the ``viz`` extra (pymatviz) is not installed, mirroring how
the analysis helpers gate on pandas/matplotlib.
"""

import json

import pytest

pytest.importorskip("pymatviz")

import plotly.graph_objects as go  # noqa: E402
from ase.build import bulk  # noqa: E402

from gradwave.io import viz  # noqa: E402


def test_structure_view_from_atoms():
    fig = viz.structure_view(bulk("Si", "diamond", a=5.43))
    assert isinstance(fig, go.Figure)


def test_structure_view_resolves_dot_atoms():
    class _InputLike:
        atoms = bulk("Al", "fcc", a=4.05)

    fig = viz.structure_view(_InputLike(), mode="2d")
    assert isinstance(fig, go.Figure)


def test_structure_view_bad_atoms_raises():
    class _Bad:
        atoms = 42

    with pytest.raises(TypeError):
        viz.structure_view(_Bad())


def test_ptable_delta_from_dict():
    fig = viz.ptable_delta(
        {"Si": {"delta_wien2k": 0.11}, "Cu": {"delta_wien2k": 5.9}}
    )
    assert isinstance(fig, go.Figure)


def test_ptable_delta_selects_key():
    data = {"Si": {"delta_wien2k": 0.11, "dfact_pdojo": 0.15}}
    # both keys resolve without error; picking a present key is enough here
    assert isinstance(viz.ptable_delta(data, key="dfact_pdojo"), go.Figure)


def test_ptable_delta_missing_key_raises():
    with pytest.raises(ValueError):
        viz.ptable_delta({"Si": {"other": 1.0}})


def test_ptable_delta_from_json_path(tmp_path):
    p = tmp_path / "delta_summary.json"
    p.write_text(json.dumps({"Al": {"delta_wien2k": 0.87}}))
    assert isinstance(viz.ptable_delta(p), go.Figure)


def _signed_blob(n=12):
    """A signed 3-D field with a + and a − lobe (stand-in for a dρ/dλ)."""
    import numpy as np

    ax = np.linspace(0, 1, n, endpoint=False)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    pos = np.exp(-40 * ((x - 0.35) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2))
    neg = np.exp(-40 * ((x - 0.65) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2))
    return pos - neg


def test_field_isosurface_signed():
    import numpy as np

    field = _signed_blob()
    cell = 5.0 * np.eye(3)
    positions = np.array([[1.75, 2.5, 2.5], [3.25, 2.5, 2.5]])
    fig = viz.field_isosurface(field, cell, positions, ["Na", "Cl"])
    assert isinstance(fig, go.Figure)
    # two isosurfaces (+/- lobes) plus the atom markers
    assert len(fig.data) == 3


def test_field_isosurface_magnitude():
    import numpy as np

    fig = viz.field_isosurface(np.abs(_signed_blob()), 5.0 * np.eye(3), None,
                               signed=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # single surface, no atoms


def test_field_isosurface_rejects_non_3d():
    import numpy as np
    import pytest

    with pytest.raises(ValueError, match="3-D grid"):
        viz.field_isosurface(np.zeros((4, 4)), np.eye(3), None)
