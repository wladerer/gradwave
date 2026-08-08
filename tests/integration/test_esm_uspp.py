"""End-to-end: open-boundary (ESM) SCF on the USPP/PAW path.

CO is a dipolar molecule (and the CO/Au motivation), so the open-vs-periodic ESM
correction is a real signal. Confirms the ESM term is wired through the USPP loop
(potential, energy) and the PAW forces, and that the loop's ΔE matches a
standalone esm_energy on the converged FULL (smooth + aug) density.
"""

from __future__ import annotations

import numpy as np
import pytest

from gradwave.core.energies.esm import esm_energy, esm_mode_of
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.paw_forces import forces_uspp
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp_loop import scf_uspp
from gradwave.scf.uspp_setup import setup_uspp
from tests.helpers import RY, pseudo


@pytest.mark.standard
@pytest.mark.parametrize(("boundary", "bias"), [("open_z", 0.0), ("open_z_metal", 2.0)])
def test_uspp_open_boundary_scf_energy_and_forces(boundary, bias):
    """The ESM correction (vacuum and metal/capacitor+bias) is wired through the
    USPP/PAW loop — potential, energy, and PAW forces — and the loop's ΔE matches
    a standalone esm_energy on the converged FULL (smooth + aug) density. CO is
    dipolar, so the correction is a real signal."""
    c = parse_upf_paw(pseudo("C.pbe-n-kjpaw_psl.1.0.0.UPF"))
    o = parse_upf_paw(pseudo("O.pbe-n-kjpaw_psl.1.0.0.UPF"))
    cell = np.diag([7.0, 7.0, 16.0])
    pos = np.array([[3.5, 3.5, 7.0], [3.5, 3.5, 8.13]])  # C–O dipole along z
    common = dict(smearing="fermi-dirac", width=0.1, etol=1e-7, rhotol=1e-6,
                  max_iter=100)
    mode = esm_mode_of(boundary)

    sys_o = setup_uspp(cell, pos, [0, 1], [c, o], ecut=40 * RY)
    res_o = scf_uspp(sys_o, LDA_PW92(), boundary=boundary, esm_bias=bias, **common)
    assert res_o.converged, f"{boundary} USPP SCF did not converge"

    # the loop's ΔE equals the standalone correction on the converged full density
    de = esm_energy(res_o.rho, sys_o.positions, sys_o.charges, sys_o.grid,
                    mode=mode, bias=bias)
    assert float(res_o.energies.esm) == pytest.approx(float(de), abs=1e-6)
    assert abs(float(res_o.energies.esm)) > 1e-3  # CO dipole → a real correction

    # PAW forces include the ESM term (toggle it off at the same density — cheap)
    f = forces_uspp(res_o, LDA_PW92(), remove_net=False)
    res_o.boundary = "periodic"
    f_no = forces_uspp(res_o, LDA_PW92(), remove_net=False)
    res_o.boundary = boundary
    assert abs(float(f[1, 2]) - float(f_no[1, 2])) > 1e-3
