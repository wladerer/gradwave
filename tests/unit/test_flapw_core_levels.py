"""Initial-state core-level (XPS) chemical shifts from all-electron FLAPW
(``flapw.core_levels``). Absolute core levels wander with the interstitial zero,
so the physics is in the within-cell SHIFT: equivalent sites must be exactly
equal, and an inequivalent site's shift must have the right sign/magnitude."""

from __future__ import annotations

import numpy as np
import pytest

from gradwave.flapw.atom import NIST_LDA_EV, atomic_scf
from gradwave.flapw.core_levels import core_level_shifts, core_levels_from_state
from gradwave.flapw.radial import log_mesh
from gradwave.flapw.scf import _CORE

# The fixed FLAPW radial mesh (flapw.scf._multi_setup); v_by_key lives on it.
_R, _DX = log_mesh(1e-5, 28.0, 2500)


def test_core_levels_atomic_anchor():
    # Units/sign anchor: re-solving the core in the isolated-ATOM converged
    # potential recovers the known LDA core eigenvalues. This pins that
    # v_by_key is a true all-electron core potential and the solver returns eV.
    for sym, tol in (("Ne", 2.0), ("O", 3.0)):
        _, v = atomic_scf(sym, _R, _DX)
        lv = core_levels_from_state({"a0": v}, [sym], ["a0"], _CORE, _R, _DX)
        assert lv["a0"]["symbol"] == sym
        e1s = lv["a0"]["levels"]["1s"]
        assert e1s < 0  # a bound core state (binding energy convention)
        ref = NIST_LDA_EV.get(sym, {}).get("1s")
        if ref is not None:
            assert abs(e1s - ref) < tol  # Ne measured +1.1 eV vs NIST-LDA


def test_core_level_shifts_helper_pairs():
    # The shift helper: same-element sites are paired against the first; a
    # lone element yields nothing; equal levels give exactly zero.
    levels = {
        "a0": {"symbol": "O", "levels": {"1s": -510.0}},
        "a1": {"symbol": "O", "levels": {"1s": -508.0}},
        "a2": {"symbol": "Ti", "levels": {"1s": -4800.0}},
    }
    sh = core_level_shifts(levels)
    assert len(sh) == 1  # one O pair; the lone Ti is skipped
    (s,) = sh
    assert s["species"] == "O" and s["orbital"] == "1s"
    assert s["delta_eV"] == pytest.approx(2.0)  # -508 − (−510)


@pytest.mark.standard
def test_core_level_shift_equivalent_sites_null():
    # Two symmetry-equivalent Ne share one interstitial reference: the shift is
    # exactly zero (clean by construction — no symmetrization needed).
    from gradwave.flapw import crystal_scf_multi

    atoms = [((0.25, 0.25, 0.25), "Ne"), ((0.75, 0.75, 0.75), "Ne")]
    _, info = crystal_scf_multi(12.0, atoms, {"Ne": 2.0}, ecut=120.0, iters=15,
                                kmesh=(1, 1, 1), use_symmetry=False)
    cl = info["core_levels"]
    assert cl["a0"]["levels"]["1s"] == pytest.approx(cl["a1"]["levels"]["1s"], abs=1e-9)
    (sh,) = info["core_level_shifts"]
    assert sh["species"] == "Ne" and abs(sh["delta_eV"]) < 1e-9


@pytest.mark.slow
def test_core_level_shift_inequivalent_oxygen():
    # A real initial-state O 1s chemical shift: Ti(4+) with two O at DIFFERENT
    # Ti-O distances. The O nearer the cation sits in a more positive potential
    # -> its 1s is LESS bound (higher, less-negative) -> the short−long shift is
    # POSITIVE, ~1-3 eV (measured ~+1.7 eV at 1.75 vs 2.30 A).
    import numpy as _np

    from gradwave.flapw import crystal_scf_multi

    bohr = 0.529177210903
    ll = 13.0
    ti = _np.array([0.5, 0.5, 0.5]) * ll
    o_short = ti + _np.array([1.75 / bohr, 0.0, 0.0])
    o_long = ti + _np.array([0.0, 2.30 / bohr, 0.0])
    atoms = [(tuple(ti / ll), "Ti"), (tuple(o_short / ll), "O"), (tuple(o_long / ll), "O")]
    _, info = crystal_scf_multi(float(ll), atoms, {"Ti": 0.90, "O": 0.70},
                                ecut=200.0, iters=40, kmesh=(1, 1, 1),
                                use_symmetry=False, smearing=0.10)
    o = [s for s in info["core_level_shifts"] if s["species"] == "O" and s["orbital"] == "1s"]
    assert len(o) == 1
    d = o[0]["delta_eV"]  # a1 (short) is the ref; a2 (long) − short
    # long is MORE bound than short -> (long − short) is negative, ~1-3 eV
    assert -3.5 < d < -0.5
    assert not np.isnan(d)
