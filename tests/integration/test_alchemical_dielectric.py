"""Composition derivative of the dielectric response (the second-order tier).

Two things: the E-field DFPT (`dielectric_born`) runs on the blended-projector
alchemical System at all (the `_shifted_projectors` stacked-rebuild fix), and the
composition derivative dε∞/dλ by finite difference is h-convergent — the reliable
ground truth for this mixed third-order quantity.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.alchemical_response import alchemical_dielectric_gradient
from gradwave.postscf.dielectric import dielectric_born
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.alchemical import setup_alchemical_substitution
from gradwave.scf.loop import scf
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard


def _sic_converger():
    si = parse_upf(PSEUDOS / "Si_ONCV_PBE_sr.upf")
    c = parse_upf(PSEUDOS / "C_ONCV_PBE_sr.upf")
    a = 4.36
    cell = a * np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    pos = np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
    kw = dict(smearing="none", etol=1e-10, rhotol=1e-9, max_iter=300, verbose=False)

    def converge(lam):
        return scf(setup_alchemical_substitution(
            cell, pos, [si, c], [0, 1], {0: c}, lam, ecut=35 * RY, kmesh=(2, 2, 2),
            use_symmetry=False), PBE(), **kw)

    return converge


def test_dielectric_born_runs_on_alchemical_system():
    """The E-field DFPT works on the alchemical blended-projector System (the
    stacked shifted-projector rebuild). eps and Born charges come out finite and
    correctly shaped."""
    torch.set_num_threads(4)
    res = _sic_converger()(0.5)
    d = dielectric_born(res, PBE())
    assert d["eps"].shape == (3, 3)
    assert d["born"].shape == (2, 3, 3)
    assert torch.isfinite(d["eps"]).all() and torch.isfinite(d["born"]).all()
    assert d["eps_iso"] > 1.0  # a real dielectric response


def test_alchemical_dielectric_gradient_h_convergent():
    """dε∞/dλ by central FD of the E-field DFPT is h-convergent — the FD ground
    truth for the mixed third-order derivative is well-behaved."""
    torch.set_num_threads(4)
    converge = _sic_converger()
    g4 = alchemical_dielectric_gradient(converge, PBE(), 0.5, h=0.04)
    g2 = alchemical_dielectric_gradient(converge, PBE(), 0.5, h=0.02)
    assert abs(g4["d_eps_iso"]) > 1.0                      # a real, nonzero slope
    assert abs(g4["d_eps_iso"] - g2["d_eps_iso"]) < 0.2    # h-convergent
    assert g2["d_eps"].shape == (3, 3)
    assert g2["d_born"].shape == (2, 3, 3)
