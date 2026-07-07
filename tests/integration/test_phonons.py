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
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994
A = 5.43
CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS = np.array([[0.0, 0, 0], [A / 4] * 3])
M_SI = 28.0855


def make_scf(pos):
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(CELL, pos, [0, 0], [upf], ecut=10 * RY, kmesh=(2, 2, 2))
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-11, rhotol=1e-10, verbose=False)
    assert res.converged
    return res


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
