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
    pdos = summary["pdos"]
    bc = pdos["band_center"]
    assert bc["available"]
    assert bc["l"] == "p"
    # center and width are finite; width is a positive spread in eV
    import math

    import numpy as np

    assert math.isfinite(bc["center_eV"])
    assert bc["width_eV"] > 0.0

    # --- exact recompute of the descriptor from summary["pdos"] itself ---
    # The driver's band_center is the 1st/2nd moment of the l='p' projected DOS
    # referenced to E_F. Redo that integral directly from the summary's own energy
    # grid + p-projected group arrays and require equality. This catches a wrong
    # E_F reference (ref='fermi' not applied, or a Ry/Ha-scaled fermi), the wrong
    # angular channel being selected, or a bad grid↔weight pairing — none of which
    # the loose |center|<60 null bound (the grid runs to +32 eV) could see.
    energy = np.asarray(pdos["energy_eV"], dtype=float)
    fermi = float(pdos["fermi_eV"])
    p_dos = np.zeros_like(energy)
    for key, arr in pdos["groups"].items():
        # group keys look like 'atom1:2P'; take the trailing angular letter
        label = key.split(":", 1)[1].split("_", 1)[0] if ":" in key else ""
        letters = [c for c in label if c.isalpha()]
        if letters and letters[-1].upper() == "P":
            p_dos = p_dos + np.asarray(arr, dtype=float)
    assert p_dos.sum() > 0.0, "no p-projected weight found in summary['pdos']"
    abs_center = (p_dos * energy).sum() / p_dos.sum()
    center_recomp = abs_center - fermi                      # 1st moment about E_F
    width_recomp = math.sqrt((p_dos * (energy - abs_center) ** 2).sum() / p_dos.sum())
    assert center_recomp == pytest.approx(bc["center_eV"], rel=1e-6, abs=1e-6)
    assert width_recomp == pytest.approx(bc["width_eV"], rel=1e-6, abs=1e-6)

    # d-band-center literature anchor (Cu-like ε_d a few eV below E_F) is SKIPPED
    # here with reason: this task's fixture is diamond C — an insulator exercised
    # on the p band, not a transition metal — and no small d-band-metal PSWFC
    # system is already run by this test, so a sign/offset anchor would require a
    # brand-new metal SCF. The exact recompute above is the load-bearing plumbing
    # check; the physical accuracy of ε_d is covered in tests/unit/test_band_center.py.

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
