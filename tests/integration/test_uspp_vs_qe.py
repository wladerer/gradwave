"""Ultrasoft (stage-1 PAW) SCF vs Quantum ESPRESSO.

A bare ultrasoft dataset (psl rrkjus) has no one-center corrections, so the
plane-wave machinery — augmentation Q_ij(G) with real-Gaunt/(−i)^L phases,
the S-metric generalized Davidson, screened D_ij, ρ = ρ_s + ρ_aug — is
validated against QE's total energy DIRECTLY (observed agreement 0.1 µeV,
QE's own print resolution). PAW adds the one-center terms on top (stage 2).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.hamiltonian import becp, projectors
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994
SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS = np.array([[0.0, 0, 0], [5.43 / 4] * 3])


def test_uspp_internal_identities():
    """Charge conservation and S-orthonormality at small cutoff (fast)."""
    torch.set_num_threads(4)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-rrkjus_psl.1.0.0.UPF")
    system = setup_uspp(SI_CELL, SI_POS, [0, 0], [paw], ecut=20 * RY, kmesh=(2, 2, 2))
    res = scf_uspp(system, PBE(), smearing="none", etol=1e-8, rhotol=1e-7,
                   verbose=False, max_iter=30)
    assert res["converged"]
    # total charge (asserted each iteration inside scf_uspp too). The floor is
    # the UPF's own internal precision: PP_Q and ∫q⁰_ij dr agree only to ~5e-8
    # per pair (S uses PP_Q, the augmentation table uses the radial functions —
    # QE pairs them the same way)
    n = float(res["rho"].sum()) * system.grid.volume / system.grid.n_points
    assert abs(n - system.n_electrons) < 1e-5
    # S-orthonormality of the converged states at every k
    q = system.q_full.to(torch.complex128)
    for ik, c in enumerate(res["coeffs"]):
        p = projectors(system.proj_data[ik], system.positions)
        b = becp(p, c)
        s_c = c + (b @ q) @ p
        ovl = c.conj() @ s_c.T
        err = float((ovl - torch.eye(c.shape[0], dtype=torch.complex128)).abs().max())
        assert err < 1e-8, (ik, err)
    # the augmentation carries real charge: S-normalized states have
    # plane-wave-only norm ≠ 1 (Si's q_ij < 0 ⇒ Σ|c|² > 1)
    pw_norm = float((res["coeffs"][0][0].abs() ** 2).sum())
    assert abs(pw_norm - 1.0) > 1e-3


@pytest.mark.slow
def test_uspp_vs_qe_total_energy():
    torch.set_num_threads(8)
    ref = json.loads((FIX / "si_uspp_ci" / "reference.json").read_text())
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-rrkjus_psl.1.0.0.UPF")
    system = setup_uspp(SI_CELL, SI_POS, [0, 0], [paw], ecut=45 * RY,
                        kmesh=(2, 2, 2), ecutrho=180 * RY,
                        fft_shape=ref["fft_dims"])
    res = scf_uspp(system, PBE(), smearing="none", etol=1e-9, rhotol=1e-8,
                   verbose=False, max_iter=30)
    assert res["converged"]
    diff_mev = abs(float(res["energies"].total) - ref["etot_eV"]) * 1000
    assert diff_mev < 0.05, f"USPP total off by {diff_mev:.5f} meV"
