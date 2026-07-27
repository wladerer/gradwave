"""Hybrid functional reachable from the YAML input (xc: pbe0), end to end.

The Γ-only PBE0 SCF is validated at the machinery level in
tests/integration/test_hybrid_scf.py; this gate exercises the input → api
plumbing added for the io catch-up: `xc: pbe0` resolves to a hybrid, the api
dispatches to `hybrid_scf` (the Fock term is live, < 0), and the summary/report
label the run `pbe0`.
"""

from pathlib import Path

import pytest

from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard  # a small self-consistent hybrid SCF


def _input(tmp_path: Path):
    from gradwave.inputs import load_input

    body = f"""
structure:
  cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [Si, Si]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {14 * RY}
xc: pbe0
hybrid:
  alpha: 0.25
kpoints:
  mesh: [1, 1, 1]
nbands: 8
scf:
  etol: 1.0e-8
  rhotol: 1.0e-7
output:
  dir: {tmp_path}
  checkpoint: false
error_estimate: false
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


def test_pbe0_input_runs_end_to_end(tmp_path):
    import torch

    from gradwave.api import run

    torch.set_num_threads(4)
    inp = _input(tmp_path)
    assert inp.hybrid.enabled and inp.hybrid.name == "pbe0"

    summary = run(inp, verbose=False)
    scf = summary["scf"]
    assert scf["converged"]
    # the label propagates to the machine summary and the human report
    assert summary["parameters"]["xc"] == "pbe0"
    # the Fock hook actually acted: α·E_x^Fock is present and negative
    assert scf["energies_eV"]["fock"] < 0.0
    report = (tmp_path / "scf.out").read_text()
    assert "Fock exchange" in report
    assert "PBE0".lower() in report.lower() or "pbe0" in report
