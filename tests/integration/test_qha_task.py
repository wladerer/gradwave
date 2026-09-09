"""Quasi-harmonic thermodynamics reachable from the YAML input, end to end.

`postscf.qha.qha` fit G(T)/V(T)/α(T) from a phonon-DOS-per-volume sweep but had
no task surface. This gate covers the `qha` task wiring: a volume sweep runs a
static SCF + a phonon DOS at each volume and fits the quasi-harmonic Gibbs free
energy, and the summary/report carry V(T), G(T), Cv(T) and the thermal-expansion
coefficient.

Uses a tiny Si cell with a 1×1×1 supercell and a coarse DOS mesh so the sweep of
per-volume phonon calculations stays affordable; the QHA physics itself is
validated in tests/unit/test_qha.py (not touched here).
"""

from pathlib import Path

import pytest

from tests.helpers import PSEUDOS, RY, SI_ONCV

pytestmark = pytest.mark.standard  # a small phonon-per-volume QHA sweep


def _qha_input(tmp_path: Path):
    from gradwave.inputs import load_input

    a = 5.43
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
scf:
  max_iter: 60
  etol: 1.0e-6
  rhotol: 1.0e-5
phonons:
  supercell: [1, 1, 1]
  dos_mesh: [2, 2, 2]
  npoints: 20
task: qha
qha:
  scales: [0.98, 0.99, 1.00, 1.01, 1.02]
  temperatures: [0, 150, 300, 450, 600]
  energy: total
output:
  dir: {tmp_path}
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


def test_qha_sweep_runs_and_fits(tmp_path):
    import numpy as np
    import torch

    from gradwave.api import run

    torch.set_num_threads(8)
    inp = _qha_input(tmp_path)
    assert inp.task == "qha" and len(inp.qha.scales) == 5

    summary = run(inp, verbose=False)
    qha = summary["qha"]
    nt = len(inp.qha.temperatures)
    # per-temperature curves are all present and the right length
    for key in ("volume_T_ang3", "gibbs_T_eV", "bulk_modulus_T_GPa",
                "cv_T_eV_per_K", "cp_T_eV_per_K", "thermal_expansion_T_per_K",
                "gruneisen_T"):
        assert len(qha[key]) == nt, key
    # the fitted equilibrium volumes are finite and near the scanned window
    v = np.asarray(qha["volume_T_ang3"])
    assert np.all(np.isfinite(v))
    vmin = min(qha["volumes_ang3"])
    vmax = max(qha["volumes_ang3"])
    # V(T) should sit within (a little beyond) the scanned volume window
    assert (v > 0.9 * vmin).all() and (v < 1.1 * vmax).all()
    # heat capacity rises from ~0 at T=0 toward the Dulong-Petit plateau
    cv = np.asarray(qha["cv_T_eV_per_K"])
    assert cv[0] == pytest.approx(0.0, abs=1e-4)
    assert cv[-1] > cv[0]
    # the human report carries the QHA section
    assert "quasi-harmonic" in (tmp_path / "qha.out").read_text()
