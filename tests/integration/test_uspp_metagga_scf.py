"""USPP/PAW meta-GGA SCF end-to-end: an r2SCAN PAW run converges and is
genuinely τ-dependent (smooth generalized-KS operator in H + one-center τ
augmentation), i.e. it is NOT the silently-GGA result the un-wired path used
to produce.

A QE r2SCAN-PAW reference fixture (absolute-energy comparison) is a follow-up
(the pseudos here are PBE-generated, so the absolute number is not physical);
this test locks in convergence and the presence of the τ terms.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.r2scan import R2SCAN
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994


@pytest.mark.standard
def test_uspp_paw_r2scan_converges_and_is_tau_dependent():
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Al.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 4.04
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0]])

    def run(xc):
        s = setup_uspp(cell, pos, [0], [paw], ecut=20 * RY, kmesh=(2, 2, 2),
                       ecutrho=100 * RY, nbands=8)
        r = scf_uspp(s, xc, smearing="gaussian", width=0.5, verbose=False,
                     max_iter=120, etol=1e-9, rhotol=1e-8, criterion="drho")
        assert r["converged"]
        return float(r["energies"].free_energy)

    f_pbe = run(PBE())
    f_r2 = run(R2SCAN())
    # τ is genuinely in the Hamiltonian and the one-center: r2SCAN must not
    # collapse to the PBE number (the pre-wiring bug silently dropped τ).
    assert abs(f_r2 - f_pbe) > 1e-2


@pytest.mark.slow
def test_uspp_paw_r2scan_forces_match_fd():
    """Analytic r2SCAN PAW forces (forces_uspp) vs central finite difference of
    the SCF total energy — the meta-GGA force is complete: the one-center τ term
    rides ddd, the smooth τ̃ is a detached (position-free) constant."""
    from gradwave.postscf.paw_forces import forces_uspp

    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    cell = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [1.42, 1.30, 1.38]])  # displaced off ideal

    def scf_at(p):
        s = setup_uspp(cell, p, [0, 0], [paw], ecut=25 * RY, kmesh=(1, 1, 1),
                       ecutrho=100 * RY)
        r = scf_uspp(s, R2SCAN(), smearing="none", etol=1e-11, rhotol=1e-10,
                     verbose=False, max_iter=100)
        assert r["converged"]
        return r

    res = scf_at(pos)
    f = forces_uspp(res, R2SCAN(), remove_net=False).cpu().numpy()
    delta = 0.005
    for a, d in [(1, 0), (1, 2)]:
        pp, pm = pos.copy(), pos.copy()
        pp[a, d] += delta
        pm[a, d] -= delta
        ep = float(scf_at(pp)["energies"].total)
        em = float(scf_at(pm)["energies"].total)
        fd = -(ep - em) / (2 * delta)
        assert abs(fd - f[a, d]) < 2e-3, f"F[{a},{d}]: {f[a, d]} vs FD {fd}"
