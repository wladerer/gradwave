"""Hydrostatic (pressure) plane-wave stress-error estimate.

The estimator differentiates the frozen-state energy error at a fixed Miller set
(ecut/s² scaling) to get P_error = -d(dE_error)/dV. Validated against the true
basis-set pressure error (a low-cutoff run vs a converged reference) on the
sheared-silicon cell: correctly signed and right order of magnitude (a
first-order indicator, ~0.5-0.75x in the meaningful regime -- see
postscf/stress_error.py). Both naive recipes (fixed-δP, and the volume
derivative at fixed ecut) come out anti-correlated, so the sign is the load-
bearing check here.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.stress import stress
from gradwave.postscf.stress_error import estimate_pressure_error
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo

pytestmark = pytest.mark.standard

# sheared Si cell (full anisotropic stress; here we probe its trace)
CELL = np.array([
    [0.01086, 2.70414, 2.7285750],
    [2.74215, 0.01629, 2.7231450],
    [2.75301, 2.70957, 0.0054300],
])
POS = np.array([[0.0, 0.0, 0.0], [1.426505, 1.3275, 1.3842875]])


def _run(ecut, sym=False):
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    system = setup_system(CELL, POS, [0, 0], [upf], ecut=ecut, kmesh=(2, 2, 2),
                          use_symmetry=sym)
    res = scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
              verbose=False)
    assert res.converged
    return res


@pytest.mark.slow
def test_pressure_error_matches_true_basis_error():
    """The estimate has the true pressure error's sign and right scale.

    Low-cutoff (12 Ry) vs a converged 45 Ry reference; the estimate reuses only
    the 12 Ry frozen state (no reference run). Both naive forms are anti-
    correlated, so a positive ratio is the primary assertion.
    """
    torch.set_num_threads(4)
    KB = 1602.176634
    res_ref = _run(45 * RY)
    sig_ref = stress(res_ref, PBE(), symmetrize=False).cpu().numpy()

    res = _run(12 * RY)
    out = estimate_pressure_error(res, PBE(), ecut_large=45 * RY)
    sig = stress(res, PBE(), symmetrize=False).cpu().numpy()
    p_true = -np.trace(sig_ref - sig) / 3.0 * KB      # kbar
    p_est = out["pressure_error_kbar"]

    assert p_true > 5.0                                # a real, resolvable error
    assert p_est > 0.0                                 # correctly signed (not the
    #                                                    anti-correlated naive form)
    ratio = p_est / p_true
    assert 0.3 < ratio < 1.2, f"ratio {ratio:.3f} (P_est {p_est:.1f}, P_true {p_true:.1f})"
    # finite-difference half-step is not load-bearing: flat from 0.005 to 0.02
    out2 = estimate_pressure_error(res, PBE(), ecut_large=45 * RY, strain=0.005)
    assert abs(out2["pressure_error_kbar"] - p_est) < 0.05 * abs(p_est)


@pytest.mark.slow
def test_pressure_error_rejects_symmetry():
    """The frozen strained rebuild needs the full k-point set (use_symmetry=False)."""
    torch.set_num_threads(4)
    res = _run(12 * RY, sym=True)
    if getattr(res.system, "sym", None) is None:
        pytest.skip("sheared cell has no symmetry to reduce")
    with pytest.raises(NotImplementedError, match="use_symmetry=False"):
        estimate_pressure_error(res, PBE())
