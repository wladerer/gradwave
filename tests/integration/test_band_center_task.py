"""Band center reachable from the YAML input, end to end (io catch-up).

`postscf.band_center.band_center` computed the d-band-center / width descriptor
from a ProjectedDOS but had no input key or summary surface. This gate covers the
wiring: `projections.band_center` rides the already-wired PDOS path and attaches
a center/width sub-block to summary["pdos"], and the report renders it.

Diamond C (PseudoDojo PP_PSWFC) gives a fast, well-conditioned SCF; the descriptor
is exercised on the carbon p band (l='p') — the machinery is l-agnostic, and the
numeric accuracy of the moment integral is covered in tests/unit/test_band_center.py.
"""

from pathlib import Path

import pytest

from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard  # a small SCF + projection analysis


def _diamond_input(tmp_path: Path, ref: str = "fermi"):
    from gradwave.inputs import load_input

    a = 3.567
    h = a / 2
    body = f"""
structure:
  cell: [[0.0, {h}, {h}], [{h}, 0.0, {h}], [{h}, {h}, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [C, C]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: PD_C_PBE_std.upf}}
ecut: {30 * RY}
xc: pbe
kpoints:
  mesh: [2, 2, 2]
nbands: 8
projections:
  enabled: true
  group_by: l
  band_center:
    enabled: true
    l: p
    ref: {ref}
output:
  dir: {tmp_path}
  checkpoint: false
error_estimate: false
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


def test_band_center_input_runs_and_reports(tmp_path):
    import torch

    from gradwave.api import run

    torch.set_num_threads(8)
    inp = _diamond_input(tmp_path)
    assert inp.projections.band_center.enabled
    assert inp.projections.band_center.l == "p"

    summary = run(inp, verbose=False)
    assert summary["scf"]["converged"]
    bc = summary["pdos"]["band_center"]
    assert bc["available"]
    assert bc["l"] == "p"
    # center and width are finite; width is a positive spread in eV
    import math
    assert math.isfinite(bc["center_eV"])
    assert bc["width_eV"] > 0.0
    # referenced to E_F: the occupied p states sit below the Fermi level, so the
    # p-band center (weighted over the whole projected band) is well within a few
    # tens of eV of the reference — a sanity bound, not a physics assertion
    assert abs(bc["center_eV"]) < 60.0
    # the human report carries the band-center line
    assert "band center" in (tmp_path / "scf.out").read_text()


def test_band_center_unavailable_without_l_grouping(tmp_path):
    import torch

    from gradwave.api import run
    from gradwave.inputs import load_input

    torch.set_num_threads(8)
    a = 3.567
    h = a / 2
    body = f"""
structure:
  cell: [[0.0, {h}, {h}], [{h}, 0.0, {h}], [{h}, {h}, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [C, C]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{C: PD_C_PBE_std.upf}}
ecut: {30 * RY}
xc: pbe
kpoints:
  mesh: [2, 2, 2]
nbands: 8
projections:
  enabled: true
  group_by: total
  band_center:
    enabled: true
    l: p
output:
  dir: {tmp_path}
  checkpoint: false
error_estimate: false
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    inp = load_input(p)
    summary = run(inp, verbose=False)
    # group_by='total' has no l-resolved groups → graceful unavailable
    bc = summary["pdos"]["band_center"]
    assert bc["available"] is False and "reason" in bc
