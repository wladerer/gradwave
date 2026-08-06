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
from tests.helpers import RY

# Per-test tiers: the Al smeared-force checks are standard-tier (seconds);
# the Cu3Al intermetallic (ecut=40 Ry, Cu d-states, ~12 min) is slow-tier and
# runs nightly, not on every standard CI shard, where it was the split floor.


FIX = Path(__file__).parents[1] / "fixtures" / "qe"
AL_A = 4.05
AL_CELL = AL_A * np.eye(3)
AL_FRAC = np.array(
    [[0.03, 0.02, -0.015], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
)

CU3AL_CELL = 3.70 * np.eye(3)
CU3AL_FRAC = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])


@pytest.mark.standard
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


# Committed finite-difference reference for the free-energy FD check below.
# It holds the ±h-displaced free energies (the *reference* the analytic force
# is compared against), NOT the analytic force itself. Caching it turns each
# scheme from 5 SCFs (1 base + 4 displaced) into 1 live SCF. Regenerate after
# any change to the free-energy model or the Al test cell with:
#   uv run python scripts/gen_al_forces_fd.py
FD_FIX = FIX / "al_forces_fd" / "fd_reference.json"


@pytest.mark.standard
@pytest.mark.parametrize("smearing", ["mp1", "cold"])
def test_smeared_forces_match_free_energy_fd(smearing):
    # The rigorous per-scheme check: F_a = −dF/dτ_a of that scheme's OWN
    # free energy (fixed-occupation Hellmann–Feynman is exact because F is
    # stationary in the occupations for a consistent (f, s) pair).
    # Raw forces here (remove_net=False): the FD probes the same raw dF/dτ.
    # The analytic force is computed LIVE; only the FD reference free energies
    # are read from the committed fixture (see gen_al_forces_fd.py).
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "Al_ONCV_PBE-1.2.upf")

    base = AL_FRAC @ AL_CELL
    system = setup_system(AL_CELL, base, [0] * 4, [upf],
                          ecut=20 * RY, kmesh=(2, 2, 2), nbands=32)
    res = scf(system, PBE(), smearing=smearing, width=0.1,
              etol=1e-9, rhotol=1e-8, verbose=False)
    assert res.converged
    f = forces(res, remove_net=False)

    fd_ref = json.loads(FD_FIX.read_text())[smearing]
    h = fd_ref["h"]
    for comp in (0, 1):
        c = fd_ref[str(comp)]
        fd = -(c["fp"] - c["fm"]) / (2 * h)
        assert abs(fd - float(f[0, comp])) < 2e-4, (smearing, comp, fd, float(f[0, comp]))


@pytest.mark.slow
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
