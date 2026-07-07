"""Collinear spin validation.

- Nonmagnetic limit: nspin=2 with zero starting moment must reproduce the
  spin-restricted result exactly (Al).
- Ferromagnetic bcc Fe vs QE at matched settings: free energy, total and
  absolute magnetization, Fermi level. SG15 Fe (3s3p semicore, Z=16) needs
  60 Ry — at 45 Ry even QE collapses to a spurious nonmagnetic state.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994


def test_nonmagnetic_limit_matches_spin_restricted():
    torch.set_num_threads(4)
    FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    al = parse_upf(FIX / "pseudos" / "Al_ONCV_PBE-1.2.upf")

    def make():
        return setup_system(4.05 / 2 * FCC, np.zeros((1, 3)), [0], [al],
                            ecut=20 * RY, kmesh=(2, 2, 2), nbands=10)

    r1 = scf(make(), PBE(), smearing="gaussian", width=0.1,
             etol=1e-10, rhotol=1e-9, verbose=False)
    r2 = scf(make(), SpinPBE(), smearing="gaussian", width=0.1,
             etol=1e-10, rhotol=1e-9, verbose=False, nspin=2, start_mag=[0.0])
    assert r1.converged and r2.converged
    diff = abs(float(r2.energies.free_energy) - float(r1.energies.free_energy))
    assert diff < 1e-6  # eV
    assert abs(r2.mag_total) < 1e-8


@pytest.mark.slow
def test_ferromagnetic_iron_vs_qe():
    torch.set_num_threads(8)
    ref = json.loads((FIX / "fe_pbe_ci" / "reference.json").read_text())
    a = 2.87
    cell = a / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
    fe = parse_upf(FIX / "pseudos" / "Fe_ONCV_PBE-1.2.upf")
    system = setup_system(cell, np.zeros((1, 3)), [0], [fe], ecut=60 * RY,
                          kmesh=(6, 6, 6), nbands=12, fft_shape=ref["fft_dims"])
    res = scf(system, SpinPBE(), smearing="gaussian", width=0.1,
              etol=1e-9, rhotol=1e-8, verbose=False, nspin=2, start_mag=[0.4])
    assert res.converged
    f = float(res.energies.free_energy)
    assert abs(f - ref["etot_eV"]) * 1000 < 1.0  # meV/atom (1 atom)
    assert abs(res.mag_total - ref["total_magnetization"]) < 0.02  # μB
    assert abs(res.mag_abs - ref["absolute_magnetization"]) < 0.02
    assert abs(res.fermi - ref["fermi_eV"]) < 0.010
