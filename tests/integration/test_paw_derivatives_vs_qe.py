"""PAW forces and stress vs Quantum ESPRESSO (displaced-Si kjpaw fixture).

The USPP/PAW derivative terms beyond norm-conserving — augmentation-density
force ∫v ∂ρ_aug/∂τ, the S-orthogonality term, the one-center chain
Σ ddd·∂ρ_ij, NLCC — all come from one autograd backward (postscf/paw_forces,
postscf/paw_stress). Observed agreement: forces 1.3e-4 eV/Å on 0.9 eV/Å
components; stress asserted at the same fixture.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS_DISP = np.array([[0.0, 0.0, 0.0], [1.4075, 1.3175, 1.3775]])


@pytest.fixture(scope="module")
def displaced_si_paw():
    torch.set_num_threads(8)
    ref = json.loads((FIX / "si_paw_force_ci" / "reference.json").read_text())
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    system = setup_uspp(SI_CELL, SI_POS_DISP, [0, 0], [paw], ecut=45 * RY,
                        kmesh=(2, 2, 2), ecutrho=180 * RY,
                        fft_shape=ref["fft_dims"])
    res = scf_uspp(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
                   verbose=False, max_iter=40)
    assert res["converged"]
    de = abs(float(res["energies"].total) - ref["etot_eV"]) / 2 * 1000
    assert de < 1.0, f"energy off by {de:.3f} meV/atom"
    return res, ref


@pytest.mark.slow
def test_paw_forces_vs_qe(displaced_si_paw):
    from gradwave.postscf.paw_forces import forces_uspp

    res, ref = displaced_si_paw
    f = forces_uspp(res, PBE()).cpu().numpy()
    qe = np.array(ref["forces_eV_A"])
    assert np.abs(f - qe).max() < 1e-3, np.abs(f - qe).max()


@pytest.mark.slow
def test_paw_stress_vs_qe(displaced_si_paw):
    from gradwave.postscf.paw_stress import stress_uspp

    res, ref = displaced_si_paw
    sig = stress_uspp(res, PBE()).cpu().numpy() * 1602.176634
    qe = np.array(ref["stress_ev_a3"]) * 1602.176634  # QE sign: −(1/Ω)∂E/∂ε
    # observed 0.13 kbar on ~100 kbar components — the same quadrature-level
    # residual as the one-center energy (0.3 meV/atom), not a convention error
    assert np.abs(-sig - qe).max() < 0.3, np.abs(-sig - qe).max()
