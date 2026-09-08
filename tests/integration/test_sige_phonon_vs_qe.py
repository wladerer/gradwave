"""SiGe Γ-optical phonon vs Quantum ESPRESSO ph.x DFPT.

Cross-code anchor for the analytic Γ-phonon path (``postscf.phonons.gamma_hessian``
→ ``gamma_frequencies``, built from the ``uspp_position.hessian_column`` Sternheimer
response). The companion ``test_gamma_phonons_self_oracle`` cross-validates that
analytic path against finite displacement on the *same* discretized Si cell, but
both routes there share this code's approximations — it is a self-oracle, and its
absolute frequency is deliberately k-under-converged (2×2×2, 15 Ry). This test is
the external anchor: it reproduces the exact QE fixture cell/pseudo/ecut/k-mesh and
asserts the Γ-optical frequency against ph.x, promoting phonons from self-oracle to
a real cross-code comparison.

Fixture: ``fixtures/qe/sige_phonon`` — zincblende SiGe, a = 5.545 Å (Vegard), PBE,
psl 1.0.0 kjpaw Si + dn-kjpaw Ge, 45/240 Ry, 4×4×4 Γ-centred, 40³ FFT (pinned
glide-commensurate), fixed occupations. QE 7.5 ph.x Γ DFPT, tr2_ph 1e-16, no NAC.
The QE Γ-optical branch is threefold degenerate at 419.141947 cm⁻¹; the acoustic
branch sits at a small unenforced −6.2 cm⁻¹ (ph.x applies no acoustic sum rule),
whereas ``gamma_hessian`` enforces the ASR so its acoustic modes are ~0.

The same pseudopotentials QE used are committed under ``fixtures/qe/pseudos``, so no
pseudo substitution is needed and the tolerance is not widened on that account. The
observed gradwave−QE offset is ~0.2–0.3 cm⁻¹ (~0.07 %); the 1 % (~4.2 cm⁻¹) bound
carries ~13× margin while still catching any sign/factor/unit/convergence breakage.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.constants import BOHR_ANG
from gradwave.core.xc.pbe import PBE
from gradwave.postscf.phonons import gamma_frequencies, gamma_hessian
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
SIGE = FIX / "sige_phonon"

# QE ibrav=2 celldm(1) = 10.47853136 Bohr → an FCC primitive cell. Γ phonons are
# invariant to the choice of primitive vectors, so the standard diamond/zincblende
# convention (rows a/2·{(0,1,1),(1,0,1),(1,1,0)}) describes the same physical
# lattice as QE's ibrav=2 and yields the same Γ dynamical matrix.
A = 10.47853136 * BOHR_ANG  # ≈ 5.545 Å
CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS = np.array([[0.0, 0.0, 0.0], [A / 4, A / 4, A / 4]])  # Si, Ge (crystal ¼¼¼)
SPECIES = [0, 1]  # Si, Ge
MASSES = np.array([28.0855, 72.63])  # amu, matching QE amass()
SHAPE = (40, 40, 40)  # QE nr1=nr2=nr3=40, glide-commensurate


@pytest.mark.slow
def test_sige_gamma_optical_vs_qe():
    torch.set_num_threads(8)
    ref = json.loads((SIGE / "reference.json").read_text())

    si = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    ge = parse_upf_paw(FIX / "pseudos" / "Ge.pbe-dn-kjpaw_psl.1.0.0.UPF")
    system = setup_uspp(CELL, POS, SPECIES, [si, ge], ecut=45 * RY,
                        kmesh=(4, 4, 4), ecutrho=240 * RY, fft_shape=SHAPE)
    res = scf_uspp(system, PBE(), smearing="none", etol=1e-12, rhotol=1e-10,
                   verbose=False, max_iter=80)
    assert res["converged"]

    # setup sanity: total energy tracks QE to well under 1 meV/atom
    etot_ry = float(res["energies"].total) / RY
    assert abs(etot_ry - ref["etot_Ry"]) < 1e-3, (
        f"etot {etot_ry:.6f} Ry vs QE {ref['etot_Ry']:.6f}")

    freqs = np.sort(gamma_frequencies(gamma_hessian(res, PBE()), MASSES))
    acoustic, optical = freqs[:3], freqs[3:]

    # gamma_hessian enforces the acoustic sum rule → acoustic modes ~0
    assert np.abs(acoustic).max() < 1.0, f"acoustic not ~0: {acoustic}"
    # Γ-optical branch is threefold degenerate
    assert optical.std() < 1.0, f"optical branch not degenerate: {optical}"

    qe_opt = ref["gamma_optical_cm1"]  # 419.141947
    tol = 0.01 * qe_opt  # 1 % ≈ 4.2 cm⁻¹ (observed offset ~0.3 cm⁻¹)
    assert np.abs(optical - qe_opt).max() < tol, (
        f"Γ-optical gradwave={optical} vs QE={qe_opt} cm⁻¹")
