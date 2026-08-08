"""End-to-end: the norm-conserving SCF runs with open-boundary (ESM)
electrostatics (boundary="open_z") and it changes the result vs periodic.

This exercises the full wiring — inputs → scf → effective_potentials (the ESM
potential ΔV added to v_eff, whose internal autograd runs inside the loop's
no_grad) and _assemble_scf_energies (EnergyBreakdown.esm = ΔE) — on a real
pseudopotential system, not just the standalone correction functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo


@pytest.mark.standard
def test_open_z_scf_runs_and_differs_from_periodic():
    upf = parse_upf(pseudo("H_ONCV_PBE-1.2.upf"))
    # slab box: c ⊥ a,b, vacuum along z; an asymmetric H2 gives a real z-dipole.
    cell = np.diag([6.0, 6.0, 14.0])
    pos = np.array([[3.0, 3.0, 6.2], [3.0, 3.0, 8.0]])
    common = dict(smearing="none", etol=1e-7, rhotol=1e-6, max_iter=60, verbose=False)

    sys_p = setup_system(cell, pos, [0, 0], [upf], ecut=18 * RY)
    res_p = scf(sys_p, LDA_PW92(), boundary="periodic", **common)
    sys_o = setup_system(cell, pos, [0, 0], [upf], ecut=18 * RY)
    res_o = scf(sys_o, LDA_PW92(), boundary="open_z", **common)

    assert res_p.converged, "periodic SCF did not converge"
    assert res_o.converged, "open_z SCF did not converge"

    # ESM term present only for open_z; periodic path is untouched.
    assert float(res_p.energies.esm) == 0.0
    assert abs(float(res_o.energies.esm)) > 1e-4

    # the open-boundary correction shifts the total energy
    assert abs(float(res_o.energies.total) - float(res_p.energies.total)) > 1e-4
