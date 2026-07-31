"""Smoke coverage for the H ONCV PBE fixture.

Guards two things at once: the newly-committed ``H_ONCV_PBE-1.2.upf`` parses,
and the lightest end-to-end norm-conserving path (an H2 molecule in a box,
Gamma-only) reaches self-consistency. Fast tier — a ~25³ grid with one occupied
band, seconds of CPU.
"""

import math

import numpy as np

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo

H_ONCV = "H_ONCV_PBE-1.2.upf"


def test_h_pseudo_parses():
    u = parse_upf(pseudo(H_ONCV))
    assert u.element == "H"
    assert u.z_valence == 1.0


def test_h2_in_a_box_gamma_scf_converges():
    h = parse_upf(pseudo(H_ONCV))
    a = 6.0  # Å cubic box, ~5 Å vacuum around the molecule
    cell = np.diag([a, a, a])
    d = 0.74  # H2 bond length (Å)
    pos = np.array([[a / 2 - d / 2, a / 2, a / 2],
                    [a / 2 + d / 2, a / 2, a / 2]])
    system = setup_system(cell, pos, [0, 0], [h], ecut=12 * RY,
                          kmesh=(1, 1, 1), use_symmetry=False)
    r = scf(system, LDA_PW92(), smearing="none", etol=1e-6, rhotol=1e-5,
            max_iter=60, verbose=False)
    assert r.converged
    assert r.n_iter < 60
    # closed-shell H2: a bound total energy, not a diverged/NaN one
    assert math.isfinite(float(r.energies.total))
