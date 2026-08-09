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
from gradwave.postscf.forces import forces
from gradwave.postscf.stress import stress
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


@pytest.mark.standard
def test_open_z_forces_match_total_energy_finite_difference():
    """The analytic force (now including the ESM term) equals −dE_total/dR by
    finite difference — the Hellmann-Feynman consistency that holds only because
    the SCF potential is δE/δρ. Also checks the ESM contribution is non-trivial."""
    na = parse_upf(pseudo("Na_ONCV_PBE_sr.upf"))
    h = parse_upf(pseudo("H_ONCV_PBE-1.2.upf"))
    cell = np.diag([7.0, 7.0, 16.0])
    z0 = 9.0
    common = dict(smearing="fermi-dirac", width=0.1, etol=1e-8, rhotol=1e-7,
                  max_iter=150, verbose=False)

    def total_at(zh):
        pos = np.array([[3.5, 3.5, 6.5], [3.5, 3.5, zh]])
        sysx = setup_system(cell, pos, [0, 1], [na, h], ecut=24 * RY)
        return scf(sysx, LDA_PW92(), boundary="open_z", **common)

    res = total_at(z0)
    f = forces(res, remove_net=False)  # includes the ESM term via res.boundary
    hs = 0.01
    e_plus = float(total_at(z0 + hs).energies.total)
    e_minus = float(total_at(z0 - hs).energies.total)
    f_fd = -(e_plus - e_minus) / (2 * hs)  # −dE/dz on the H atom
    assert float(f[1, 2]) == pytest.approx(f_fd, abs=3e-2)

    # the ESM force term is significant (toggle it off at the same density)
    res.boundary = "periodic"
    f_no_esm = forces(res, remove_net=False)
    res.boundary = "open_z"
    assert abs(float(f[1, 2]) - float(f_no_esm[1, 2])) > 1e-2


@pytest.mark.standard
def test_open_z_metal_capacitor_scf_and_bias():
    """The metal/capacitor mode (boundary=open_z_metal) runs end-to-end: the SCF
    converges, the loop ΔE matches the standalone capacitor energy on the
    converged density, and an applied bias does work on the charge (nonzero ΔE)
    and shifts the forces — the field effect."""
    upf = parse_upf(pseudo("H_ONCV_PBE-1.2.upf"))
    cell = np.diag([5.0, 5.0, 14.0])
    pos = np.array([[2.5, 2.5, 6.0], [2.5, 2.5, 8.0]])
    common = dict(smearing="none", etol=1e-6, rhotol=1e-5, max_iter=60, verbose=False)

    def run(bias):
        sysx = setup_system(cell, pos, [0, 0], [upf], ecut=16 * RY)
        return scf(sysx, LDA_PW92(), boundary="open_z_metal", esm_bias=bias, **common), sysx

    r0, s0 = run(0.0)
    rb, sb = run(2.0)
    assert r0.converged and rb.converged

    de0 = esm_energy(r0.rho, s0.positions, s0.charges, s0.grid, mode="capacitor", bias=0.0)
    deb = esm_energy(rb.rho, sb.positions, sb.charges, sb.grid, mode="capacitor", bias=2.0)
    assert float(r0.energies.esm) == pytest.approx(float(de0), abs=1e-6)
    assert float(rb.energies.esm) == pytest.approx(float(deb), abs=1e-6)

    assert abs(float(rb.energies.esm)) > 1e-3  # bias does work on the charge
    f0 = forces(r0, remove_net=False)
    fb = forces(rb, remove_net=False)
    assert abs(float(fb[1, 2]) - float(f0[1, 2])) > 1e-3  # bias shifts the force


@pytest.mark.standard
def test_open_z_stress_is_in_plane_only():
    """The ESM stress contribution lands only on the in-plane (surface-stress)
    components — the open z-axis has no lattice to strain, so σ's z-row/col are
    left to the periodic terms. Toggle boundary at the same converged result."""
    na = parse_upf(pseudo("Na_ONCV_PBE_sr.upf"))
    h = parse_upf(pseudo("H_ONCV_PBE-1.2.upf"))
    cell = np.diag([7.0, 7.0, 16.0])
    pos = np.array([[3.5, 3.5, 6.5], [3.5, 3.5, 9.0]])
    common = dict(smearing="fermi-dirac", width=0.1, etol=1e-7, rhotol=1e-6,
                  max_iter=100, verbose=False)
    res = scf(setup_system(cell, pos, [0, 1], [na, h], ecut=24 * RY),
              LDA_PW92(), boundary="open_z", **common)

    s_open = stress(res, LDA_PW92(), symmetrize=False)
    res.boundary = "periodic"
    s_per = stress(res, LDA_PW92(), symmetrize=False)
    res.boundary = "open_z"
    d = s_open - s_per  # the ESM stress contribution

    assert abs(float(d[0, 0])) + abs(float(d[1, 1])) > 1e-4  # in-plane surface stress
    for i in range(3):  # open z-axis untouched (gated)
        assert abs(float(d[2, i])) < 1e-8
        assert abs(float(d[i, 2])) < 1e-8
