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
