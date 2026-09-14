"""M4: Γ-point phonons from force constants (FD of analytic forces).

Acceptance: force-constant Hessian matches energy-FD Hessian to 0.5%,
acoustic modes vanish after ASR, and the Si optical triplet is degenerate.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.hessian import force_constants_gamma, gamma_phonons
from gradwave.postscf.phonons_supercell import asr_residual
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
CELL, POS = si_fcc()
M_SI = 28.0855


def make_scf(pos):
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(CELL, pos, [0, 0], [upf], ecut=10 * RY, kmesh=(2, 2, 2))
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-11, rhotol=1e-10, verbose=False)
    assert res.converged
    return res


@pytest.mark.standard
def test_gamma_raw_asr_residual_small():
    """Gate translational invariance of the RAW force constants directly.

    The acoustic sum rule is enforced by construction (each atom's self block is
    subtracted so Σ_b Φ[a,i,b,j] = 0 exactly), after which the acoustic modes are
    ~0 no matter what — asserting the ENFORCED modes are small is a tautology that
    masks a Hessian violating translational invariance. The convention-free,
    reference-free check is the residual of the RAW (pre-enforcement) Hessian:
    max_{a,i,j} |Σ_b Φ[a,i,b,j]|, which is 0 exactly for the true Hessian (a rigid
    translation exerts no net force). This is the phonon analogue of
    ``postscf.born``'s ``asr_max``.

    Measured on this Si cell (asus, 6 threads, 12 SCFs ~3.3 s): raw residual
    5.84e-4 eV/Å², max|Φ| = 18.3 eV/Å² → relative 3.2e-5. The 1e-4 relative bound
    keeps ~3× margin while still catching a real translational-invariance defect
    (a broken force term, wrong grid symmetry) that the enforced-mode check would
    hide entirely.
    """
    torch.set_num_threads(6)
    phi_raw = force_constants_gamma(make_scf, POS, h=5e-3, acoustic_sum_rule=False)
    na = POS.shape[0]
    scale = float(np.abs(phi_raw).max())  # typical force-constant magnitude
    resid = asr_residual(phi_raw.reshape(na, 3, na, 3))
    assert resid / scale < 1e-4, (
        f"raw ASR residual {resid:.3e} eV/Å² (rel {resid / scale:.3e}) — the RAW "
        f"Hessian violates translational invariance even though the enforced "
        f"acoustic modes look clean")


@pytest.mark.slow
def test_gamma_phonons_si():
    torch.set_num_threads(8)
    phi = force_constants_gamma(make_scf, POS, h=5e-3)
    freqs = gamma_phonons(phi, np.array([M_SI, M_SI]))

    # 3 acoustic ~ 0 (ASR enforced; residual from egg-box), 3 degenerate optical
    assert np.abs(freqs[:3]).max() < 15.0  # cm⁻¹
    optical = freqs[3:]
    assert optical.min() > 300.0
    assert np.ptp(optical) < 0.02 * optical.mean()

    # cross-validate one diagonal force-constant against energy second difference
    h = 5e-3
    e = {}
    for s in (+1, 0, -1):
        pos = POS.copy()
        pos[1, 0] += s * h
        e[s] = float(make_scf(pos).energies.total)
    d2e = (e[1] - 2 * e[0] + e[-1]) / h**2
    # compare against the pre-ASR force-constant entry
    phi_raw = force_constants_gamma(make_scf, POS, h=5e-3, acoustic_sum_rule=False)
    assert abs(phi_raw[3, 3] - d2e) < 0.005 * abs(d2e)
