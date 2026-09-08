"""Bader charges reachable from the YAML input, end to end (io catch-up).

`postscf.bader.bader` worked but had no input key, no summary block, and no
report section. This gate covers the wiring: `bader: enabled` computes a Bader
block alongside the SCF, and the summary/report carry the per-atom charge table.

The partition invariants (charge conservation, cell neutrality) are the value
assertions — they hold for any density regardless of the pseudopotential caveat.
The physics/accuracy of the on-grid partition is validated in
tests/unit/test_bader.py (not touched here). Si (norm-conserving) gives a fast,
well-conditioned SCF under the capped thread budget.
"""

from pathlib import Path

import pytest

from tests.helpers import PSEUDOS, RY, SI_ONCV

pytestmark = pytest.mark.standard  # a small SCF + Bader partition


def _si_input(tmp_path: Path):
    from gradwave.inputs import load_input

    a = 5.43  # Si lattice constant [Å]; a/2 scales the fcc primitive cell
    h = a / 2
    body = f"""
structure:
  cell: [[0.0, {h}, {h}], [{h}, 0.0, {h}], [{h}, {h}, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [Si, Si]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Si: {SI_ONCV}}}
ecut: {15 * RY}
xc: pbe
kpoints:
  mesh: [2, 2, 2]
smearing:
  type: none
bader:
  enabled: true
  nna_tol: 0.5
output:
  dir: {tmp_path}
  checkpoint: false
error_estimate: false
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


def test_bader_input_runs_and_reports(tmp_path):
    import torch

    from gradwave.api import run

    torch.set_num_threads(8)
    inp = _si_input(tmp_path)
    assert inp.bader.enabled and inp.bader.nna_tol == 0.5

    summary = run(inp, verbose=False)
    assert summary["scf"]["converged"]
    bader = summary["bader"]
    assert bader["available"]
    assert len(bader["atoms"]) == 2

    # invariants that hold for ANY density: charge is conserved (Σ electrons ==
    # ∫ρ) and the cell is neutral (Σ charge ≈ 0). Si carries 4 valence e⁻/atom.
    elec = sum(a["electrons"] for a in bader["atoms"])
    charge = sum(a["charge"] for a in bader["atoms"])
    assert elec == pytest.approx(8.0, abs=1e-2)
    assert bader["total_electrons"] == pytest.approx(8.0, abs=1e-2)
    assert abs(charge) < 1e-2
    for atom in bader["atoms"]:
        assert atom["valence"] == pytest.approx(4.0, abs=1e-6)

    # the human report carries the Bader section with a per-atom charge column
    out = (tmp_path / "scf.out").read_text()
    assert "Bader charges" in out
