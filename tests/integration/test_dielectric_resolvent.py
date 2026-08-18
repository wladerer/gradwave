"""ε∞ / Born from the resolvent Sternheimer solver must equal the iterative-CG
result — same conduction-projected linear solves, so both the full-mesh and the
IBZ-symmetrized field-response paths agree to solver precision.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.dielectric import dielectric_born
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

pytestmark = pytest.mark.standard
FIX = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.mark.parametrize("use_sym", [False, True])
def test_dielectric_resolvent_matches_cg(use_sym):
    torch.set_num_threads(8)
    a = 5.43
    si = parse_upf(str(FIX / "Si_ONCV_PBE-1.2.upf"))
    cell = a / 2 * FCC
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    system = setup_system(cell, pos, [0, 0], [si], ecut=12 * RY,
                          kmesh=(2, 2, 2), use_symmetry=use_sym)
    res = scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9, verbose=False)
    assert res.converged

    kw = dict(cg_tol=1e-8, outer_tol=1e-6, max_outer=100)
    cg = dielectric_born(res, PBE(), solver="cg", **kw)
    rv = dielectric_born(res, PBE(), solver="resolvent", **kw)

    assert abs(cg["eps_iso"] - rv["eps_iso"]) < 1e-5
    assert torch.allclose(cg["eps"], rv["eps"], atol=1e-4, rtol=0), \
        f"eps Δ={ (cg['eps'] - rv['eps']).abs().max().item():.2e}"
    assert torch.allclose(cg["born"], rv["born"], atol=1e-4, rtol=0), \
        f"born Δ={ (cg['born'] - rv['born']).abs().max().item():.2e}"
