"""Honest k-mesh reduction label (output._mesh_label).

The parameters block must name *what* reduced the mesh: spatial symmetry folds
to the IBZ, but a time-reversal pairing halves it even with symmetry off, so an
unsymmetrized run is "TR-reduced" — not "IBZ". Under a distributed k-shard the
label reports this rank's slice of the reduced mesh.
"""

from gradwave.io.output import _mesh_label


def test_symmetry_on_labels_ibz():
    par = {"kmesh": [6, 6, 6], "nk_total": 216, "nk": 112, "symmetry": True}
    assert _mesh_label(par) == "6×6×6 = 216 k (112 IBZ)"


def test_symmetry_off_labels_tr_reduced_not_ibz():
    par = {"kmesh": [6, 6, 6], "nk_total": 216, "nk": 112, "symmetry": False}
    label = _mesh_label(par)
    assert label == "6×6×6 = 216 k (112 TR-reduced)"
    assert "IBZ" not in label


def test_distributed_shard_reports_local_slice():
    par = {"kmesh": [6, 6, 6], "nk_total": 216, "nk": 112, "symmetry": False,
           "nk_local": 56}
    assert _mesh_label(par) == "6×6×6 = 216 k (56 of 112 TR-reduced on this rank)"


def test_local_equal_to_reduced_is_not_annotated():
    # a single-rank run (or a rank that owns the whole reduced mesh) must not
    # print a redundant "112 of 112 on this rank"
    par = {"kmesh": [6, 6, 6], "nk_total": 216, "nk": 112, "symmetry": True,
           "nk_local": 112}
    assert _mesh_label(par) == "6×6×6 = 216 k (112 IBZ)"


def test_total_only_and_bare_mesh_fallbacks():
    assert _mesh_label({"kmesh": [4, 4, 4], "nk_total": 64}) == "4×4×4 = 64 k"
    assert _mesh_label({"kmesh": [2, 2, 2], "nk": 3}) == "2×2×2 → 3 k"
    assert _mesh_label({"kmesh": [1, 1, 1]}) == "1×1×1"
