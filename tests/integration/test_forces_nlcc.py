"""NLCC core-correction contribution to the Hellmann–Feynman forces.

The nonlinear-core-correction pseudopotential adds a frozen pseudo-core charge
ρ_core(r−R_I) to the XC argument. Its force term is −∫ v_xc(r) ∂ρ_core/∂R_I,
which the suite obtains as the autograd gradient of E_xc(ρ + ρ_core(τ)) with the
SCF density ρ detached (Hellmann–Feynman: stationary at convergence). Tier-0
gate: total forces on a rattled, low-symmetry cell built with a norm-conserving
pseudo that HAS an NLCC match a central finite difference of the total energy to
the FD floor. A valence-only (no-NLCC) system is unaffected — passing xc is a
no-op there.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard  # full SCF + FD re-runs; not a fast-gate test

# Low-symmetry (triclinic, rattled) 2-atom carbon cell — no special positions,
# so every force component is nonzero and the core-correction term is exercised
# in all three Cartesian directions.
A = 3.2
CELL = A * np.array([[1.0, 0.0, 0.0], [0.12, 1.0, 0.0], [0.05, 0.08, 1.05]])
FRAC = np.array([[0.02, 0.01, 0.0], [0.27, 0.31, 0.24]])
POS = FRAC @ CELL


def _run(upf, pos, etol=1e-11, rhotol=1e-10):
    system = setup_system(CELL, pos, [0, 0], [upf], ecut=35 * RY, kmesh=(2, 2, 2))
    return scf(system, PBE(), smearing="gaussian", width=0.05,
               etol=etol, rhotol=rhotol, verbose=False)


def test_nlcc_force_matches_finite_difference():
    """C_ONCV_PBE_sr carries a core charge; the analytic force (which now
    includes −∫ v_xc ∂ρ_core/∂τ) matches FD of the total energy. Compared with
    remove_net=False so the analytic derivative is the raw −dE/dτ that FD gives
    (the mean-force removal would shift single components by the egg-box term)."""
    torch.set_num_threads(4)
    upf = parse_upf(PSEUDOS / "C_ONCV_PBE_sr.upf")
    assert upf.core_rho is not None, "fixture must have an NLCC core charge"

    res = _run(upf, POS)
    assert res.converged
    assert res.system.rho_core is not None  # NLCC active in the SCF
    f = forces(res, remove_net=False, xc=PBE()).cpu().numpy()

    h = 1e-4
    for ia, ic in [(1, 0), (1, 1), (0, 2)]:
        vals = []
        for sign in (+1, -1):
            pos = POS.copy()
            pos[ia, ic] += sign * h
            vals.append(float(_run(upf, pos).energies.total))
        fd = -(vals[0] - vals[1]) / (2 * h)
        assert abs(fd - float(f[ia, ic])) < 1e-6, (
            f"comp ({ia},{ic}): analytic={f[ia, ic]:.9f} fd={fd:.9f}"
        )


def test_nlcc_requires_xc():
    """Forces on an NLCC system raise a clear error when xc is omitted, since the
    core-correction term needs the functional to evaluate v_xc."""
    torch.set_num_threads(4)
    upf = parse_upf(PSEUDOS / "C_ONCV_PBE_sr.upf")
    res = _run(upf, POS, etol=1e-8, rhotol=1e-7)
    with pytest.raises(ValueError, match="NLCC"):
        forces(res)


def test_non_nlcc_forces_unchanged_by_xc():
    """A valence-only pseudo has no core charge; passing xc must be a no-op, and
    forces must not require it (the NLCC branch stays dormant)."""
    torch.set_num_threads(4)
    upf = parse_upf(PSEUDOS / "Si_ONCV_PBE-1.2.upf")
    assert upf.core_rho is None
    res = _run(upf, POS, etol=1e-10, rhotol=1e-9)
    assert res.system.rho_core is None
    f_no_xc = forces(res).cpu().numpy()
    f_xc = forces(res, xc=PBE()).cpu().numpy()
    assert np.array_equal(f_no_xc, f_xc)
