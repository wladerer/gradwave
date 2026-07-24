"""USPP/PAW band structure for collinear spin (nspin=2 unblock).

bands_uspp now runs the frozen-potential generalized solve once per spin
channel (frozen_veff / screened_dscr already return per-spin lists), returning
(2, nk, nbands). Validated in the nonmagnetic limit: nspin=2 with zero starting
moment reproduces the spin-restricted USPP bands, both channels identical.
Genuine V↑≠V↓ is covered by the norm-conserving magnetic bands test and the
per-spin USPP path in discretization_error.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.uspp_bands import bands_uspp
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import PSEUDOS, RY, si_fcc

pytestmark = pytest.mark.standard  # ultrasoft SCF; not a fast-gate test


def test_uspp_bands_nspin2_nonmagnetic_limit_matches_restricted():
    torch.set_num_threads(8)
    cell, pos = si_fcc()
    paw = parse_upf_paw(PSEUDOS / "Si.pbe-n-rrkjus_psl.1.0.0.UPF")

    def make():
        return setup_uspp(cell, pos, [0, 0], [paw], ecut=20 * RY, kmesh=(2, 2, 2))

    r1 = scf_uspp(make(), PBE(), smearing="gaussian", width=0.05,
                  etol=1e-9, rhotol=1e-8, verbose=False, max_iter=40)
    r2 = scf_uspp(make(), SpinPBE(), nspin=2, start_mag=[0.0, 0.0],
                  smearing="gaussian", width=0.05, etol=1e-9, rhotol=1e-8,
                  verbose=False, max_iter=40)
    assert r1["converged"] and r2["converged"]
    assert abs(r2["mag_total"]) < 1e-6

    kpath = [[0.0, 0, 0], [0.5, 0, 0], [0.25, 0.25, 0.25]]
    b1 = bands_uspp(r1, PBE(), kpath, nbands=8)
    b2 = bands_uspp(r2, SpinPBE(), kpath, nbands=8)
    b1 = b1.cpu().numpy()
    b2 = b2.cpu().numpy()
    assert b1.shape == (len(kpath), 8)        # nspin=1 stays (nk, nbands)
    assert b2.shape == (2, len(kpath), 8)     # nspin=2 gains a spin axis
    # both spin channels coincide and match the spin-restricted bands
    assert np.max(np.abs(b2[0] - b2[1])) < 1e-5
    assert np.max(np.abs(b2[0] - b1)) < 2e-3
