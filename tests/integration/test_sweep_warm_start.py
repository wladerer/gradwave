"""Cross-SCF warm-starting in the multi-SCF sweep drivers must be a pure
iteration accelerator: the converged E(V) — and everything fit from it — is
identical whether each point cold-starts or warm-starts from its neighbour.

The warm arm seeds each volume's SCF from the previous converged density (a
near-exact donor under uniform scaling on the shared FFT grid). A converged
fixed point does not depend on its seed, so warm and cold must agree to SCF
tolerance while the warm arm takes fewer iterations. These are the correctness
gates for the `eos.warm_start` / `qha.warm_start` toggles; the physics of the
fits themselves is covered by test_eos.py / test_qha_task.py.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from gradwave.api import run_eos
from gradwave.inputs import (
    EOSParams,
    Input,
    KPointsParams,
    SCFParams,
    SmearingParams,
)
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard  # multi-SCF sweeps, tightly converged


def _si_eos_input(warm_start: bool) -> Input:
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    atoms = Atoms("Si2", positions=pos, cell=cell, pbc=True)
    return Input(
        atoms=atoms, pseudo_dir=Path(PSEUDOS),
        pseudo_map={"Si": "Si_ONCV_PBE-1.2.upf"}, ecut=20 * RY, xc="pbe",
        kpoints=KPointsParams(mesh=(4, 4, 4)),
        smearing=SmearingParams(type="none"),
        scf=SCFParams(etol=1e-10, rhotol=1e-9, max_iter=100),
        eos=EOSParams(scales=(0.96, 0.98, 1.00, 1.02, 1.04),
                      warm_start=warm_start))


def test_eos_warm_start_matches_cold_and_cuts_iterations():
    torch.set_num_threads(8)
    warm = run_eos(_si_eos_input(True), verbose=False)
    cold = run_eos(_si_eos_input(False), verbose=False)

    # the seed engaged on every point past the first, and the cold arm on none
    assert warm["warm_start"] is True
    assert warm["warm_start_applied"] == [False, True, True, True, True]
    assert cold["warm_start_applied"] == [False] * 5

    # The converged E(V) is seed-independent to SCF tolerance. This is the
    # correctness statement: the SCF outputs (the per-volume energies) are what
    # the warm start must not perturb, and they agree to the convergence floor.
    # (The downstream BM3 fit amplifies that ~1e-8 floor into a few 1e-3
    # relative on the soft B0 of this deliberately off-equilibrium toy window,
    # so the fit is not a precision gate — the energies are.)
    ew = np.asarray(warm["energies_eV_per_atom"])
    ec = np.asarray(cold["energies_eV_per_atom"])
    assert np.max(np.abs(ew - ec)) < 1e-8, (
        f"E(V) moved with the seed: {np.max(np.abs(ew - ec)):.2e} eV/atom")

    # the point of the whole exercise: fewer total iterations
    assert warm["n_iter_total"] < cold["n_iter_total"]


def _qha_input(tmp_path: Path, warm_start: bool):
    from gradwave.inputs import load_input

    tmp_path.mkdir(parents=True, exist_ok=True)
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
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {20 * RY}
xc: pbe
kpoints:
  mesh: [2, 2, 2]
scf:
  max_iter: 100
  etol: 1.0e-10
  rhotol: 1.0e-9
phonons:
  supercell: [1, 1, 1]
  dos_mesh: [2, 2, 2]
  npoints: 10
task: qha
qha:
  scales: [0.98, 0.99, 1.00, 1.01, 1.02]
  temperatures: [0, 300, 600]
  energy: total
  warm_start: {str(warm_start).lower()}
output:
  dir: {tmp_path}
"""
    p = tmp_path / "in.yaml"
    p.write_text(body)
    return load_input(p)


def test_qha_warm_start_matches_cold(tmp_path):
    from gradwave.api import run

    torch.set_num_threads(8)
    warm = run(_qha_input(tmp_path / "w", warm_start=True), verbose=False)["qha"]
    cold = run(_qha_input(tmp_path / "c", warm_start=False), verbose=False)["qha"]

    # the static-energy chain seeds every volume past the first (same FFT grid
    # across this ±2% window); the cold arm seeds none
    assert warm["warm_start"] is True
    assert warm["warm_start_applied"][0] is False
    assert all(warm["warm_start_applied"][1:])
    assert cold["warm_start_applied"] == [False] * 5

    # qha.warm_start only seeds the static E(V) chain (the phonon DOS at each
    # volume is computed identically in both arms), so the correctness gate is
    # that the static energies — the seed-affected SCF outputs — are seed-
    # independent to the convergence floor. (The per-temperature quasi-harmonic
    # G(T) fit over this coarse toy DOS is ill-conditioned and amplifies that
    # 1e-8 floor by orders of magnitude, so it is not a precision gate here.)
    ew = np.asarray(warm["static_energies_eV"])
    ec = np.asarray(cold["static_energies_eV"])
    assert np.max(np.abs(ew - ec)) < 1e-8, (
        f"static E(V) moved with the seed: {np.max(np.abs(ew - ec)):.2e} eV")

    # fewer static iterations with the seed on
    assert sum(warm["n_iter_static_per_volume"]) < sum(
        cold["n_iter_static_per_volume"])
