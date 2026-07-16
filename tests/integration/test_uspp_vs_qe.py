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
def test_paw_vs_qe_total_energy():
    """Full PAW (ultrasoft + one-center) vs pw.x, psl Si kjpaw. Observed:
    total +0.31 meV/atom, one-center contribution +1.0 meV vs QE's printout —
    the residual is quadrature-level (angular XC grid / radial Poisson
    scheme differences), not a convention error."""
    torch.set_num_threads(8)
    RY_ = RY
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    system = setup_uspp(SI_CELL, SI_POS, [0, 0], [paw], ecut=45 * RY_,
                        kmesh=(2, 2, 2), ecutrho=180 * RY_, fft_shape=(32, 32, 32))
    res = scf_uspp(system, PBE(), smearing="none", etol=1e-9, rhotol=1e-8,
                   verbose=False, max_iter=40)
    assert res["converged"]
    e = res["energies"]
    qe_tot, qe_onec = -93.26524951 * RY_, -71.19134543 * RY_
    assert abs(float(e.total) - qe_tot) / 2 * 1000 < 1.0, (
        f"PAW total off by {(float(e.total) - qe_tot) * 1000:+.3f} meV"
    )
    assert abs(float(e.onecenter) - qe_onec) * 1000 < 3.0


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


@pytest.mark.slow
def test_paw_spin_vs_qe():
    """Ferromagnetic fcc Ni kjpaw (spn semicore) — spin-polarized USPP/PAW SCF
    with spin one-center XC. Observed: F +1.6 meV/atom, m 0.594 vs QE 0.59 μB.
    The mixer needs the full stability stack for this system (becsum in the
    Pulay vector, Kerker on the total, trust-region resets, α = 0.3); the
    residual plateaus at ~1e-3 (metallic occupation noise), so the gate is
    the energy/moment, not the formal rhotol."""
    from gradwave.core.xc.spin import SpinPBE

    torch.set_num_threads(8)
    ref = json.loads((FIX / "ni_paw_spin_ci" / "reference.json").read_text())
    paw = parse_upf_paw(FIX / "pseudos" / "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF")
    cell = np.array([[0.0, 1.76, 1.76], [1.76, 0.0, 1.76], [1.76, 1.76, 0.0]])
    system = setup_uspp(cell, np.zeros((1, 3)), [0], [paw], ecut=50 * RY,
                        kmesh=(4, 4, 4), ecutrho=400 * RY, nbands=18,
                        fft_shape=ref["fft_dims"])
    res = scf_uspp(system, SpinPBE(), nspin=2, start_mag=[0.5],
                   smearing="gaussian", width=0.1, etol=1e-5, rhotol=5e-4,
                   mixing_alpha=0.3, verbose=False, max_iter=80)
    e = res["energies"]
    dF = abs(float(e.free_energy) - ref["etot_Ry"] * RY) * 1000
    assert dF < 5.0, f"F off by {dF:.2f} meV"
    # ±m branches are degenerate without SOC — which one the SCF lands on
    # depends on the mixing trajectory; gate the magnitude
    assert abs(abs(res["mag_total"]) - ref["mag_muB"]) < 0.02
    assert abs(float(e.onecenter) / RY - ref["onecenter_Ry"]) < 0.005


@pytest.mark.slow
def test_paw_symmetry_ibz_equals_full_mesh():
    """IBZ + rho/becsum symmetrization vs the full TR mesh (observed equality
    to 1e-7 eV; 36 -> 8 k for diamond Si, ~5x faster)."""
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    vals = {}
    for sym in (False, True):
        system = setup_uspp(SI_CELL, SI_POS, [0, 0], [paw], ecut=25 * RY,
                            kmesh=(4, 4, 4), use_symmetry=sym)
        r = scf_uspp(system, PBE(), smearing="none", etol=1e-9, rhotol=1e-8,
                     verbose=False, max_iter=30)
        assert r["converged"]
        vals[sym] = float(r["energies"].total)
    assert abs(vals[True] - vals[False]) < 1e-6


@pytest.mark.slow
def test_paw_bands_vs_qe():
    """Frozen-potential generalized solves at arbitrary k: Si PAW Γ eigenvalues
    vs QE's printed bands from the si_paw_ci reference run (observed 0.65 meV,
    the one-center quadrature scale)."""
    from gradwave.postscf.uspp_bands import bands_uspp

    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    system = setup_uspp(SI_CELL, SI_POS, [0, 0], [paw], ecut=45 * RY,
                        kmesh=(2, 2, 2), ecutrho=180 * RY, fft_shape=(32, 32, 32))
    res = scf_uspp(system, PBE(), smearing="none", etol=1e-9, rhotol=1e-8,
                   verbose=False, max_iter=40)
    assert res["converged"]
    eigs = bands_uspp(res, PBE(), [[0.0, 0.0, 0.0]], nbands=4)[0]
    qe_gamma = [-5.5159, 6.5075, 6.5075, 6.5075]  # pw.x verbosity=high printout
    assert np.abs(eigs.numpy() - np.array(qe_gamma)).max() < 0.005


@pytest.mark.slow
def test_paw_metal_pt_vs_qe():
    """Nonmagnetic PAW metal (fcc Pt kjpaw, gaussian-smeared) vs pw.x — the one
    PAW-metal cell, since the other PAW-vs-QE tests are a Si insulator and a Ni
    spin metal. Observed F +0.24 meV, one-center -1.9 meV vs QE's printout
    (quadrature scale). Pins the fcc-Pt reference: a stale -10167.30 eV QE figure
    once suggested a 0.23 eV offset that fresh QE (this fixture) shows is not real.
    Grid pinned to QE's dense FFT dims per the reference-grid rule in wisdom."""
    torch.set_num_threads(8)
    ref = json.loads((FIX / "pt_paw_metal_ci" / "reference.json").read_text())
    paw = parse_upf_paw(FIX / "pseudos" / "Pt.pbe-n-kjpaw_psl.1.0.0.UPF")
    cell = 0.5 * 3.97 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    system = setup_uspp(cell, np.zeros((1, 3)), [0], [paw], ecut=40 * RY,
                        kmesh=tuple(ref["kmesh"]), ecutrho=400 * RY, nbands=14,
                        fft_shape=tuple(ref["fft_dims"]))
    res = scf_uspp(system, PBE(), smearing="gaussian", width=0.2, etol=1e-8,
                   rhotol=1e-7, verbose=False, max_iter=80)
    assert res["converged"]
    e = res["energies"]
    dF = abs(float(e.free_energy) - ref["free_energy_Ry"] * RY) * 1000
    assert dF < 3.0, f"F off by {dF:.3f} meV"
    d_onec = abs(float(e.onecenter) / RY - ref["onecenter_Ry"]) * RY * 1000
    assert d_onec < 5.0, f"one-center off by {d_onec:.3f} meV"
