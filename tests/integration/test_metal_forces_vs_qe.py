"""Metal validation: smeared forces vs QE, and the Cu3Al intermetallic.

Smeared Hellmann–Feynman forces are gradients of the FREE energy F = E − σS
at fixed occupations (the entropy term cancels the occupation response at
self-consistency); this test is the end-to-end check of that statement
against QE tprnfor on a displaced-atom aluminum cell.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

pytestmark = pytest.mark.standard

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994

AL_A = 4.05
AL_CELL = AL_A * np.eye(3)
AL_FRAC = np.array(
    [[0.03, 0.02, -0.015], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
)

CU3AL_CELL = 3.70 * np.eye(3)
CU3AL_FRAC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


def test_smeared_metal_forces_vs_qe():
    torch.set_num_threads(4)
    ref = json.loads((FIX / "al_forces_ci" / "reference.json").read_text())
    upf = parse_upf(FIX / "pseudos" / "Al_ONCV_PBE-1.2.upf")
    system = setup_system(AL_CELL, AL_FRAC @ AL_CELL, [0] * 4, [upf],
                          ecut=20 * RY, kmesh=(2, 2, 2), nbands=32,
                          fft_shape=ref.get("fft_dims"))
    res = scf(system, PBE(), smearing="gaussian", width=0.1,
              etol=1e-10, rhotol=1e-9, verbose=False)
    assert res.converged
    e = float(res.energies.free_energy)
    assert abs(e - ref["etot_eV"]) / 4 * 1000 < 0.01  # meV/atom, grid-matched

    f_us = forces(res).cpu().numpy()
    f_qe = np.array(ref["forces_eV_ang"])
    # net force removed on both sides (QE does the same) → tight agreement
    assert np.abs(f_us - f_qe).max() < 5e-4, f"\nqe:\n{f_qe}\nus:\n{f_us}"
    assert np.abs(f_us.sum(axis=0)).max() < 1e-10  # exact by construction


@pytest.mark.parametrize("smearing", ["mp1", "cold"])
def test_smeared_forces_match_free_energy_fd(smearing):
    # The rigorous per-scheme check: F_a = −dF/dτ_a of that scheme's OWN
    # free energy (fixed-occupation Hellmann–Feynman is exact because F is
    # stationary in the occupations for a consistent (f, s) pair).
    # Raw forces here (remove_net=False): the FD probes the same raw dF/dτ.
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "Al_ONCV_PBE-1.2.upf")

    def run(pos):
        system = setup_system(AL_CELL, pos, [0] * 4, [upf],
                              ecut=20 * RY, kmesh=(2, 2, 2), nbands=32)
        res = scf(system, PBE(), smearing=smearing, width=0.1,
                  etol=1e-11, rhotol=1e-10, verbose=False)
        assert res.converged
        return res

    base = AL_FRAC @ AL_CELL
    res = run(base)
    f = forces(res, remove_net=False)

    h = 1e-4
    for comp in (0, 1):
        dp = np.zeros((4, 3))
        dp[0, comp] = h
        fp = float(run(base + dp).energies.free_energy)
        fm = float(run(base - dp).energies.free_energy)
        fd = -(fp - fm) / (2 * h)
        assert abs(fd - float(f[0, comp])) < 2e-4, (smearing, comp, fd, float(f[0, comp]))


def test_cu3al_vs_qe():
    torch.set_num_threads(4)
    ref = json.loads((FIX / "cu3al_pbe_ci" / "reference.json").read_text())
    al = parse_upf(FIX / "pseudos" / "Al_ONCV_PBE-1.2.upf")
    cu = parse_upf(FIX / "pseudos" / "Cu_ONCV_PBE-1.2.upf")
    system = setup_system(CU3AL_CELL, CU3AL_FRAC @ CU3AL_CELL, [0, 1, 1, 1],
                          [al, cu], ecut=40 * RY, kmesh=(2, 2, 2), nbands=45)
    res = scf(system, PBE(), smearing="gaussian", width=0.1,
              etol=1e-9, rhotol=1e-8, verbose=False)
    assert res.converged
    e = float(res.energies.free_energy)
    diff = abs(e - ref["etot_eV"]) / 4 * 1000
    assert diff < 1.0, f"Cu3Al: {diff:.4f} meV/atom"
    if "fermi_eV" in ref:
        assert abs(res.fermi - ref["fermi_eV"]) < 0.010
