"""Triplet O2 in a box vs QE — the real-moment (zeta != 0) PAW spin gate.

Stretched, slightly skewed O2 (d = 1.351 A) in a 6.0 A cube: m = 2 muB with
integer occupations in both codes, forces ~5.7 eV/A along the bond. Verifies
the spin-asymmetric machinery end to end: per-spin becsum/augmentation,
S-orthogonality, spin one-center (Hartree + XC with sigma_tot cross terms),
NLCC core split.

Force tolerance is representation-level, not exactness-level: gradwave's ddd
is the exact dE_1c/drho_ij (autograd through the quadrature) while QE's comes
from the divergence-form v_xc, inexact at lm-truncation level — the two
codes' forces legitimately differ by ~1e-2 eV/A on 5.7 eV/A components (the
force also moves ~3% per 15 Ry of cutoff here). gradwave's own force/energy
consistency is held to FD separately (becsum-space in
test_paw_onsite_spin.py; tau-space checked at 1e-3 during development).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
@pytest.mark.torture
def test_o2_triplet_forces_stress_vs_qe():
    torch.set_num_threads(8)
    ref = json.loads((FIX / "o2_paw_spin_ci" / "reference.json").read_text())
    paw = parse_upf_paw(FIX / "pseudos" / "O.pbe-n-kjpaw_psl.1.0.0.UPF")
    cell = 6.0 * np.eye(3)
    pos = np.array([[3.0, 3.0, 2.40], [3.06, 3.0, 3.75]])
    system = setup_uspp(cell, pos, [0, 0], [paw], ecut=50 * RY, kmesh=(1, 1, 1),
                        ecutrho=400 * RY, nbands=10, fft_shape=ref["fft_dims"])
    res = scf_uspp(system, SpinPBE(), nspin=2, start_mag=[0.5],
                   smearing="gaussian", width=0.01 * RY,
                   etol=1e-8, rhotol=1e-6, verbose=False, max_iter=40)
    assert res["converged"]
    assert abs(res["mag_total"] - ref["mag_muB"]) < 1e-3

    de = abs(float(res["energies"].free_energy) - ref["etot_eV"]) * 1000
    assert de < 3.0, f"energy off by {de:.2f} meV"

    from gradwave.postscf.paw_forces import forces_uspp
    from gradwave.postscf.paw_stress import stress_uspp

    f = forces_uspp(res, SpinPBE()).numpy()
    qe = np.array(ref["forces_eV_A"])
    assert np.abs(f - qe).max() < 3e-2, np.abs(f - qe).max()

    sig = stress_uspp(res, SpinPBE()).numpy()
    qe_sig = np.array(ref["stress_ev_a3"])
    dk = np.abs(-sig - qe_sig).max() * 1602.176634
    assert dk < 0.4, f"stress off by {dk:.3f} kbar"
