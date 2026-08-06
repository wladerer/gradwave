"""NLCC core-correction force for a meta-GGA (r2SCAN).

The nonlinear-core-correction pseudopotential adds a frozen pseudo-core charge
ρ_core(r−R_I) to the XC argument. For a meta-GGA the XC energy also depends on
the kinetic-energy density τ, so v_xc = δE_xc/δρ carries a τ-dependence and the
core-correction force −∫ v_xc ∂ρ_core/∂R_I differs from its GGA value. The suite
obtains it as the autograd gradient of E_xc(ρ + ρ_core(τ), σ, τ_valence) with ρ
and τ detached (Hellmann–Feynman: stationary at convergence, so τ is a fixed
orbital field and only ρ_core's structure factor carries the position gradient).
τ_valence is rebuilt from the converged orbitals via the same pad_coeffs → tau_b
path the SCF/ELF/stress code uses.

Tier-0 self-consistency gate (no QE reference needed): total forces on a
rattled, low-symmetry cell built with a norm-conserving pseudo that HAS an NLCC,
run under r2SCAN, match a central finite difference of the r2SCAN total energy to
the FD floor. This is the oracle that catches an implementation that merely
agrees with another code — it is a derivative of gradwave's own energy.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.r2scan import R2SCAN
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard  # full meta-GGA SCF + FD re-runs; not a fast gate

# Low-symmetry (triclinic, rattled) 2-atom carbon cell — no special positions, so
# every force component is nonzero and the core-correction term is exercised in
# all three Cartesian directions. Same geometry as the GGA NLCC-force gate.
A = 3.2
CELL = A * np.array([[1.0, 0.0, 0.0], [0.12, 1.0, 0.0], [0.05, 0.08, 1.05]])
FRAC = np.array([[0.02, 0.01, 0.0], [0.27, 0.31, 0.24]])
POS = FRAC @ CELL

# Committed finite-difference reference: the six ±h-displaced r2SCAN total
# energies the analytic NLCC force is checked against. Caching them turns this
# from 7 live meta-GGA SCFs (1 base + 6 displaced) into 1. The analytic force
# is still computed live; only the FD reference energies are read from disk.
# Regenerate after any change to the r2SCAN model, the NLCC force path, or the
# carbon cell with:  uv run python scripts/gen_metagga_nlcc_fd.py
FD_FIX = (Path(__file__).parents[1] / "fixtures" / "qe" / "metagga_nlcc_fd"
          / "fd_reference.json")


def _run(upf, pos, xc, etol=1e-11, rhotol=1e-10):
    system = setup_system(CELL, pos, [0, 0], [upf], ecut=30 * RY, kmesh=(2, 2, 2))
    return scf(system, xc, smearing="gaussian", width=0.05,
               etol=etol, rhotol=rhotol, verbose=False)


def test_metagga_nlcc_force_matches_finite_difference():
    """C_ONCV_PBE_sr carries a core charge; under r2SCAN the analytic force
    (which now includes the τ-dependent −∫ v_xc ∂ρ_core/∂τ) matches central FD of
    the r2SCAN total energy. Compared with remove_net=False so the analytic
    derivative is the raw −dE/dτ that FD gives."""
    torch.set_num_threads(4)
    upf = parse_upf(PSEUDOS / "C_ONCV_PBE_sr.upf")
    assert upf.core_rho is not None, "fixture must have an NLCC core charge"

    res = _run(upf, POS, R2SCAN())
    assert res.converged
    assert res.system.rho_core is not None  # NLCC active in the SCF
    f = forces(res, remove_net=False, xc=R2SCAN()).cpu().numpy()

    fd_ref = json.loads(FD_FIX.read_text())
    h = fd_ref["h"]
    for ia, ic in [(1, 0), (1, 1), (0, 2)]:
        rec = fd_ref[f"{ia},{ic}"]
        fd = -(rec["ep"] - rec["em"]) / (2 * h)
        assert abs(fd - float(f[ia, ic])) < 1e-6, (
            f"comp ({ia},{ic}): analytic={f[ia, ic]:.9f} fd={fd:.9f}"
        )


def test_metagga_nlcc_force_differs_from_gga():
    """Sanity: the meta-GGA core-correction force is genuinely τ-coupled, so it is
    not numerically identical to the GGA (PBE) NLCC force on the same converged
    r2SCAN state. Guards against the τ argument being silently dropped (which
    would collapse the meta-GGA path onto the GGA one)."""
    torch.set_num_threads(4)
    upf = parse_upf(PSEUDOS / "C_ONCV_PBE_sr.upf")
    res = _run(upf, POS, R2SCAN())
    assert res.converged
    f_mgga = forces(res, remove_net=False, xc=R2SCAN()).cpu().numpy()
    f_gga = forces(res, remove_net=False, xc=PBE()).cpu().numpy()
    # Same detached ρ, but the meta-GGA v_xc sees τ; the core-correction term
    # differs by well above the FD floor.
    assert np.abs(f_mgga - f_gga).max() > 1e-4
