"""End-to-end: the norm-conserving SCF runs with open-boundary (ESM)
electrostatics (boundary="open_z") and it changes the result vs periodic.

This exercises the full wiring — inputs → scf → effective_potentials (the ESM
potential ΔV added to v_eff, whose internal autograd runs inside the loop's
no_grad) and _assemble_scf_energies (EnergyBreakdown.esm = ΔE) — on a real
pseudopotential system. NaH is an ionic dimer with a genuine z-dipole, so the
open-vs-periodic correction is a robust ~76 meV (a homonuclear H2 has no dipole
and a near-zero correction).
"""

from __future__ import annotations

import numpy as np
import pytest

from gradwave.core.energies.esm import esm_energy
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo


@pytest.mark.standard
def test_open_z_scf_matches_periodic_plus_correction():
    na = parse_upf(pseudo("Na_ONCV_PBE_sr.upf"))
    h = parse_upf(pseudo("H_ONCV_PBE-1.2.upf"))
    # slab box: c ⊥ a,b, vacuum along z; NaH points its ionic dipole along z.
    cell = np.diag([7.0, 7.0, 16.0])
    pos = np.array([[3.5, 3.5, 6.5], [3.5, 3.5, 9.0]])
    common = dict(smearing="fermi-dirac", width=0.1, etol=1e-6, rhotol=1e-5,
                  max_iter=80, verbose=False)

    res_p = scf(setup_system(cell, pos, [0, 1], [na, h], ecut=24 * RY),
                LDA_PW92(), boundary="periodic", **common)
    sys_o = setup_system(cell, pos, [0, 1], [na, h], ecut=24 * RY)
    res_o = scf(sys_o, LDA_PW92(), boundary="open_z", **common)

    assert res_p.converged, "periodic SCF did not converge"
    assert res_o.converged, "open_z SCF did not converge"

    # periodic path is byte-for-byte untouched (no ESM term)
    assert float(res_p.energies.esm) == 0.0

    # the loop's ΔE equals the standalone correction recomputed on the converged
    # density + ions — proof the loop passes the right density/positions/charges
    de_direct = esm_energy(res_o.rho, sys_o.positions, sys_o.charges, sys_o.grid)
    assert float(res_o.energies.esm) == pytest.approx(float(de_direct), abs=1e-6)

    # a real, sizable ionic-dipole correction (not numerical noise), and it
    # shifts the converged total energy
    assert abs(float(res_o.energies.esm)) > 1e-2
    assert abs(float(res_o.energies.total) - float(res_p.energies.total)) > 1e-2
