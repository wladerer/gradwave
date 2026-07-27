"""COHP reachable from the YAML input, end to end (io catch-up).

`postscf.cohp.cohp` worked but had no input key, no summary block, and no plot
kind. This gate covers the wiring added for the catch-up: `projections.cohp`
computes a COHP block alongside the SCF, the summary/report carry it, and
`gradwave plot --kind cohp` renders it. The bonded-pair bonding sign follows the
same physics check the reference test uses (a real bond gives ICOHP < 0); the
COHP sum-rule/accuracy is validated in tests/integration/test_cohp.py (owned by
the test-runtime worker, not touched here). Diamond C is used instead of the O2
dimer purely for a fast, well-conditioned SCF under the capped thread budget —
same PseudoDojo (PP_PSWFC) projection path.
"""

from pathlib import Path

import pytest

from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard  # a small SCF + projection analysis


def _diamond_input(tmp_path: Path):
    from gradwave.inputs import load_input

    a = 3.567  # diamond lattice constant [Å]; a/2 scales the fcc primitive cell
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
  enabled: false
  cohp:
    enabled: true
    pairs: [[0, 1]]
    width: 0.2
output:
  dir: {tmp_path}
  checkpoint: false
error_estimate: false
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


def test_cohp_input_runs_and_plots(tmp_path):
    import torch

    from gradwave.api import run
    from gradwave.cli import main

    torch.set_num_threads(8)
    inp = _diamond_input(tmp_path)
    assert inp.projections.cohp.enabled and inp.projections.cohp.pairs == ((0, 1),)

    summary = run(inp, verbose=False)
    assert summary["scf"]["converged"]
    cohp = summary["cohp"]
    assert cohp["available"]
    # the C–C bond is bonding: ICOHP integrated to E_F is negative
    assert cohp["pair_icohp"]["1-2"] < 0.0
    assert cohp["total_icohp"] < 0.0
    # the human report carries the COHP section
    assert "COHP" in (tmp_path / "scf.out").read_text()

    # `gradwave plot --kind cohp` renders a figure from the JSON
    png = tmp_path / "cohp.png"
    assert main(["plot", str(tmp_path / "scf.json"),
                 "--kind", "cohp", "-o", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 0
