"""nspin=2 degenerate limit for PAW forces/stress (displaced Si, cheap ecut).

A spin-polarized run with a broken-symmetry start must relax to m=0 and
reproduce the nspin=1 forces and stress at the SAME settings to near machine
precision — every per-spin term (becsum channels, S-orthogonality, one-center
ddd, spin XC with its σ_tot cross gradients) collapses onto the unpolarized
path. The one-center GGA factor-2 bug shifted stress by 0.67 kbar here while
leaving all energies intact.

QE anchoring is inherited: the nspin=1 path is held to the si_paw_force_ci
reference (1.3e-4 eV/Å, 0.13 kbar) by test_paw_derivatives_vs_qe.py.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.paw_forces import forces_uspp
from gradwave.postscf.paw_stress import stress_uspp
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS_DISP = np.array([[0.0, 0.0, 0.0], [1.4075, 1.3175, 1.3775]])


@pytest.mark.torture
def test_spin_degenerate_forces_stress_match_nspin1():
    torch.set_num_threads(8)
    from gradwave.pseudo.upf_paw import parse_upf_paw

    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")

    def make():
        return setup_uspp(SI_CELL, SI_POS_DISP, [0, 0], [paw], ecut=20 * RY,
                          kmesh=(2, 2, 2), ecutrho=80 * RY)

    r1 = scf_uspp(make(), PBE(), nspin=1, smearing="gaussian", width=0.05,
                  etol=1e-10, rhotol=1e-9, verbose=False, max_iter=60)
    assert r1["converged"]
    r2 = scf_uspp(make(), SpinPBE(), nspin=2, start_mag=[0.3],
                  smearing="gaussian", width=0.05,
                  etol=1e-10, rhotol=1e-9, verbose=False, max_iter=60)
    assert r2["converged"]
    assert abs(r2["mag_total"]) < 1e-8

    de = abs(float(r2["energies"].free_energy) - float(r1["energies"].free_energy))
    assert de < 1e-6, f"free energy split by {de:.3e} eV at zeta=0"

    f1 = forces_uspp(r1, PBE()).numpy()
    f2 = forces_uspp(r2, SpinPBE()).numpy()
    assert np.abs(f2 - f1).max() < 1e-6, np.abs(f2 - f1).max()

    s1 = stress_uspp(r1, PBE()).numpy()
    s2 = stress_uspp(r2, SpinPBE()).numpy()
    assert np.abs(s2 - s1).max() * 1602.176634 < 1e-3, "stress split [kbar]"
