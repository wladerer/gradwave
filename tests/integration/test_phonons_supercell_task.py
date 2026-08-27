"""Supercell phonon task: diamond Si, PBE.

Slow tier: a small supercell finite-displacement run (13 warm-started SCFs).
Validates the full `run_phonons` path — the folded Γ frequencies must match the
primitive optical phonon (~521 cm⁻¹, experiment ~520), the acoustic modes must
sit near zero, and no branch may go strongly imaginary. The converged full-cell
dispersion vs QE ph.x / neutron data runs as a benchmark on asus, not here.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from gradwave.api import run_phonons
from gradwave.inputs import Input, KPointsParams, PhononParams, SmearingParams
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def si_phonons():
    torch.set_num_threads(8)
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    atoms = Atoms("Si2", positions=pos, cell=cell, pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=Path(PSEUDOS),
        pseudo_map={"Si": "Si_ONCV_PBE-1.2.upf"}, ecut=20 * RY, xc="pbe",
        kpoints=KPointsParams(mesh=(4, 4, 4)),
        smearing=SmearingParams(type="none"),
        phonons=PhononParams(supercell=(2, 2, 2), displacement=0.01,
                             npoints=60, dos_mesh=(6, 6, 6)))
    return run_phonons(inp, verbose=False)


def test_phonon_gamma_and_stability(si_phonons):
    ph = si_phonons
    freqs = np.array(ph["frequencies_cm1"])          # (nq, 6)
    x = np.array(ph["x"])
    labels = ph["labels"]
    # locate Γ (label "G") on the path
    gx = next(xt for xt, lab in labels if lab in ("G", "Γ"))
    ig = int(np.argmin(np.abs(x - gx)))
    at_gamma = np.sort(freqs[ig])
    # 3 acoustic near zero, 3 optical near the Si Γ optical phonon (~520 cm⁻¹)
    assert np.abs(at_gamma[:3]).max() < 10.0            # acoustic ≈ 0
    assert 490.0 < at_gamma[3:].mean() < 545.0          # optical ~521
    # optical branch is threefold degenerate at Γ
    assert at_gamma[3:].std() < 8.0
    # no strongly imaginary branch anywhere on the path
    assert ph["min_frequency_cm1"] > -15.0


def test_phonon_dos_present(si_phonons):
    dos = si_phonons["dos"]
    g = np.array(dos["frequency_cm1"])
    d = np.array(dos["dos"])
    assert g.shape == d.shape and (d >= 0).all()
    assert d.sum() > 0.0
    # spectral weight extends up to the optical band, none far above it
    assert g[d > d.max() * 0.01].max() < 600.0


def test_phonon_thermo_present(si_phonons):
    """The DOS is now bridged to harmonic thermodynamics (F, U, Cv, S vs T)."""
    th = si_phonons["thermo"]
    # Si primitive cell has 2 atoms → 6 modes; the DOS integrates to that.
    assert th["mode_count"] == pytest.approx(6.0, rel=0.05)
    assert th["zero_point_energy_eV"] > 0.0
    assert th["debye_temperature_K"] > 0.0
    assert th["imaginary_modes_present"] is False
    temps = th["temperatures_K"]
    cv = th["heat_capacity_eV_K"]
    assert cv[0] == 0.0                       # Cv(0 K) = 0
    assert cv[-1] > cv[1] > 0.0               # rises with T toward Dulong–Petit
    # F = U − T·S at every tabulated temperature
    for T, f, u, s in zip(temps, th["free_energy_eV"], th["internal_energy_eV"],
                          th["entropy_eV_K"], strict=True):
        assert f == pytest.approx(u - T * s, abs=1e-9)


def _si_phonon_input(nspin: int):
    """Diamond Si supercell phonons, coarse and shared between nspin 1 and 2.
    A 1×1×1 supercell keeps the SCF count small (Γ dynamical matrix only); the
    nspin=2 seed is pinned to zero moment so it is the nonmagnetic limit of
    nspin=1 — the two force paths must agree."""
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    atoms = Atoms("Si2", positions=pos, cell=cell, pbc=True)
    return Input(
        atoms=atoms, pseudo_dir=Path(PSEUDOS),
        pseudo_map={"Si": "Si_ONCV_PBE-1.2.upf"}, ecut=14 * RY, xc="pbe",
        kpoints=KPointsParams(mesh=(2, 2, 2)),
        smearing=SmearingParams(type="none"), nspin=nspin,
        start_mag={"Si": 0.0},
        tot_magnetization=0.0 if nspin == 2 else None,
        phonons=PhononParams(supercell=(1, 1, 1), displacement=0.01,
                             npoints=8, dos_mesh=(0, 0, 0)))


@pytest.mark.slow
def test_run_phonons_nspin2_matches_nspin1():
    """The ungated nspin=2 supercell-phonon path (issue #147): a nonmagnetic Si
    run through the nspin=2 driver folds the same force constants as nspin=1, so
    the frequencies are identical. Proves the api.py gate was stale — the FD
    force fold calls postscf.forces.forces, which already sums per spin channel
    (#45); the driver just hardcoded the nspin=1 SCF call."""
    torch.set_num_threads(8)
    f1 = np.array(run_phonons(_si_phonon_input(1), verbose=False)["frequencies_cm1"])
    f2 = np.array(run_phonons(_si_phonon_input(2), verbose=False)["frequencies_cm1"])
    assert f2.shape == f1.shape
    # nonmagnetic limit: the nspin=2 forces equal the nspin=1 forces to SCF
    # convergence, so the folded frequencies agree to well under 1 cm⁻¹
    assert np.allclose(f1, f2, atol=1.0), np.abs(f1 - f2).max()
