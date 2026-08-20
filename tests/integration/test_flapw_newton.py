"""Newton-Krylov polish of the FLAPW fixed point (flapw/newton.py).

Measured context: Anderson wins easy fixed points; on the production-hard TiO2 config it
stalls (r_v plateau at 0.46) while Newton-Krylov converges in 3 steps to r_nsph ~1e-9
(experiments/autoapw/newton_probe.py, asus 2026-08-19). These tests pin the shipped
machinery at BCT-O scale: the state round-trip layout, the stall detector's semantics,
and (standard tier) an end-to-end polish beating the plain iteration's own residual.
"""

import numpy as np
import pytest

from gradwave.flapw import anderson_stalled, crystal_scf_multi, newton_polish
from gradwave.flapw.newton import StateLayout
from gradwave.flapw.recorder import FLAPWRecorder

BCT_A = [5.4, 5.4, 6.6]
ATOMS = [((0.0, 0.0, 0.0), "O"), ((0.5, 0.5, 0.5), "O")]
RADII = {"O": 0.75}
CFG = dict(ecut=100.0, lmax=2, kmesh=(1, 1, 1), smearing=0.0, efg=False, fullpot=True,
           fullpot_lmax=2, use_symmetry=True, kworkers=1, subspace_reuse=False)


def _rec(r_vs, exact=True):
    rec = FLAPWRecorder()
    for r in r_vs:
        rec.record(it=0, r_v=r, exact_solve=exact)
    return rec


def test_stall_detector_semantics():
    # plateau above the gate with no 2x improvement over the earlier best -> stalled
    assert anderson_stalled(_rec([2.0, 1.0, 0.8] + [0.5] * 8))
    # healthy geometric decay dipping under the gate -> not stalled
    assert not anderson_stalled(_rec([1.0 * 0.5**i for i in range(12)]))
    # still improving (tail best beats half the earlier best) -> not stalled
    assert not anderson_stalled(_rec([10.0, 5.0, 2.0, 1.5, 1.2, 1.0, 0.9, 0.85,
                                      0.5, 0.45, 0.42, 0.4]))
    # too little history -> never stalled
    assert not anderson_stalled(_rec([1.0, 1.0, 1.0]))
    # subspace iterations are ignored
    assert not anderson_stalled(_rec([0.5] * 12, exact=False))


def test_state_layout_roundtrip_exact():
    _, info = crystal_scf_multi(BCT_A, ATOMS, RADII, iters=22, tol=0.0, **CFG)
    st = info["state"]
    lay = StateLayout(st)
    x = lay.flatten(st)
    x2 = lay.flatten(lay.unflatten(x, st))
    assert np.abs(x - x2).max() == 0.0
    assert st["v_nsph"], "22 iterations must clear the MT->fullpot flip (state has v_nsph)"


def test_polish_rejects_partial_state():
    with pytest.raises(ValueError):
        newton_polish(BCT_A, ATOMS, RADII, {"v_by_key": {}, "v_grid": None},
                      scf_kwargs=CFG)


@pytest.mark.standard
def test_polish_converges_bcto():
    """End-to-end: from a modestly-warm state the polish reaches a residual the same
    number of plain iterations cannot, and reports its own diagnostics honestly."""
    _, info = crystal_scf_multi(BCT_A, ATOMS, RADII, iters=22, tol=0.0, **CFG)
    st, ninfo = newton_polish(BCT_A, ATOMS, RADII, info["state"], scf_kwargs=CFG,
                              maxiter=3, inner_maxiter=8, f_tol=1e-6)
    assert ninfo["residual_norm"] < 1e-5
    assert ninfo["f_evals"] < 45
    # the polished state resumes as a fixed point: one plain iteration barely moves
    _, i1 = crystal_scf_multi(BCT_A, ATOMS, RADII, iters=1, tol=0.0,
                              v_start={"__full_state__": st}, **CFG)
    assert i1["recorder"].iters[-1]["r_v"] < 1e-4
