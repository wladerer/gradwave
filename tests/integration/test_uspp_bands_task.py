"""USPP/PAW band structure reachable from `task: bands`, end to end.

`postscf.uspp_bands.bands_uspp` was implemented and unit-tested, but the
`task: bands` driver only called the norm-conserving `bands_along_ase_path`,
which cannot consume a USPP result (a USPPResult carries no `v_eff`). This gate
covers the io-catch-up dispatch: `_bands_extra` now routes a USPP/PAW run through
`bands_uspp`, and the eigenvalues are finite, sorted, and reproduce the SCF
spectrum at the coincident Γ point.
"""

import numpy as np
import pytest

from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard  # ultrasoft SCF + a short band solve


def _input(tmp_path):
    from gradwave.inputs import load_input

    body = f"""
structure:
  cell: [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
  positions:
    frac: [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
  species: [Si, Si]
pseudopotentials:
  dir: {PSEUDOS}
  map: {{Si: Si.pbe-n-rrkjus_psl.1.0.0.UPF}}
ecut: {18 * RY}
xc: pbe
kpoints:
  mesh: [1, 1, 1]
smearing:
  type: gaussian
  width: 0.05
nbands: 8
task: bands
bands:
  path: "GX"
  npoints: 3
output:
  dir: {tmp_path}
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


def test_uspp_task_bands_dispatch(tmp_path):
    import torch

    from gradwave.api import _bands_extra, run_scf

    torch.set_num_threads(8)
    inp = _input(tmp_path)
    res = run_scf(inp, verbose=False)   # USPPResult (dict-bridge)
    assert res["converged"]

    block = _bands_extra(inp, res, verbose=False)["bands"]
    eig = np.asarray(block["eigenvalues_eV"], dtype=float)
    assert eig.shape == (3, 8)          # (path points, bands), nspin=1
    assert np.isfinite(eig).all()
    # each k-point's bands are sorted ascending
    assert np.all(np.diff(eig, axis=1) >= -1e-6)
    assert np.isfinite(block["reference_eV"])

    # Γ is path point 0 and the (only) SCF k-point: the frozen-potential band
    # solve there must reproduce the SCF eigenvalues
    scf_gamma = res["eigenvalues"][0].detach().cpu().numpy()
    assert np.max(np.abs(eig[0] - scf_gamma[:8])) < 5e-3
