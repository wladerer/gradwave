"""EOS task: PBE bulk modulus of diamond Si vs the WIEN2k reference, plus an
ASE cross-check of the fit on the driver's own E(V) points.

Slow tier: seven warm-started SCFs on one FFT grid (run once, shared across the
tests via a module fixture). The WIEN2k all-electron reference is
V0 = 20.4530 Å³/atom, B0 = 88.545 GPa, B1 = 4.31 (Lejaeghere et al., Science
351, aad3000, 2016). At 30 Ry / 8³ this driver lands within ~1% (V0 20.57,
B0 87.8 GPa, B0' 4.21); the tolerances below are wider so the gate is robust
across platforms, catching a broken driver or fitter rather than a sub-GPa
convergence drift.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from gradwave.api import run_eos
from gradwave.inputs import EOSParams, Input, KPointsParams, SmearingParams
from tests.helpers import PSEUDOS, RY

WIEN2K_SI = dict(v0=20.4530, b0_GPa=88.545, b1=4.31)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def si_eos():
    """One PBE volume scan of diamond Si; shared by every assertion below."""
    torch.set_num_threads(8)
    a = 5.47  # ~PBE equilibrium so the default 94–106% window brackets V0
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    atoms = Atoms("Si2", positions=pos, cell=cell, pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=Path(PSEUDOS),
        pseudo_map={"Si": "Si_ONCV_PBE-1.2.upf"}, ecut=30 * RY, xc="pbe",
        kpoints=KPointsParams(mesh=(8, 8, 8)),
        smearing=SmearingParams(type="none"), eos=EOSParams())
    return run_eos(inp, verbose=False)


def test_eos_si_bulk_modulus_vs_wien2k(si_eos):
    assert si_eos["all_converged"]
    # the minimum of the sampled E(V) lands inside the scan (V0 is bracketed)
    e = np.array(si_eos["energies_eV_per_atom"])
    assert 0 < int(np.argmin(e)) < len(e) - 1
    # physics: within a wide band of the all-electron reference
    assert si_eos["v0_ang3_per_atom"] == pytest.approx(WIEN2K_SI["v0"], abs=1.0)
    assert si_eos["b0_GPa"] == pytest.approx(WIEN2K_SI["b0_GPa"], abs=15.0)
    assert 3.0 < si_eos["b0_prime"] < 5.5
    assert si_eos["rms_residual_eV_per_atom"] < 1e-3  # BM3 describes the points well


def test_eos_fit_agrees_with_ase_on_driver_points(si_eos):
    """Fit the driver's own (V, E) with ASE's independent 3rd-order BM: V0/B0
    must agree with our fitter (same data, independent implementation)."""
    from ase.eos import EquationOfState

    eos = EquationOfState(si_eos["volumes_ang3_per_atom"],
                          si_eos["energies_eV_per_atom"], eos="birchmurnaghan")
    v0_ase, _e0_ase, b_ase_ev_a3 = eos.fit()
    assert si_eos["v0_ang3_per_atom"] == pytest.approx(v0_ase, rel=1e-3)
    assert si_eos["b0_eV_ang3"] == pytest.approx(b_ase_ev_a3, rel=5e-3)
