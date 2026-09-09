"""Initial-state core-level (XPS) chemical shifts from all-electron FLAPW
(``flapw.core_levels``). Absolute core levels wander with the interstitial zero,
so the physics is in the within-cell SHIFT: equivalent sites must be exactly
equal, and an inequivalent site's shift must have the right sign/magnitude."""

from __future__ import annotations

import numpy as np
import pytest

from gradwave.flapw.atom import NIST_LDA_EV, atomic_scf
from gradwave.flapw.core_levels import (
    core_level_shifts,
    core_levels_from_state,
    cross_cell_binding_shift,
    referenced_binding_energies,
)
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


def test_referenced_binding_energy_wander_invariance():
    # The cross-cell reference cancels the interstitial-zero wander: adding a
    # constant c to EVERY level in a cell AND to that cell's reference (VBM/E_F)
    # — exactly what an interstitial-zero shift does — leaves the referenced
    # binding energies unchanged. BE = e_ref - e_core is positive (bound core).
    levels = {"a0": {"symbol": "Si", "levels": {"2p": -1830.0, "2s": -1870.0}}}
    e_ref = -5.0
    be0 = referenced_binding_energies(levels, e_ref)
    assert be0["a0"]["binding"]["2p"] == pytest.approx(-5.0 - (-1830.0))
    assert be0["a0"]["binding"]["2p"] > 0  # bound core -> positive binding energy
    c = 12.3
    shifted = {"a0": {"symbol": "Si",
                      "levels": {o: v + c for o, v in levels["a0"]["levels"].items()}}}
    be1 = referenced_binding_energies(shifted, e_ref + c)
    for orb in ("2p", "2s"):
        assert be1["a0"]["binding"][orb] == pytest.approx(be0["a0"]["binding"][orb])


def test_cross_cell_binding_shift_sign_and_invariance():
    # ΔBE(b - a) > 0 means MORE bound in cell b (the Si-2p-in-SiO2 shift). The
    # shift is invariant under an independent per-cell wander (each cell's core
    # AND its reference shift together), and averages over inequivalent sites.
    si = {"a0": {"symbol": "Si", "levels": {"2p": -1830.0}}}
    ef_si = -5.0
    sio2 = {"b0": {"symbol": "Si", "levels": {"2p": -1836.0}},
            "b1": {"symbol": "Si", "levels": {"2p": -1834.0}}}
    ef_sio2 = -3.0
    out = cross_cell_binding_shift(si, ef_si, sio2, ef_sio2, "Si", "2p")
    # BE_Si = -5-(-1830)=1825 ; BE_SiO2 = -3 - mean(-1836,-1834)= -3+1835=1832
    assert out["delta_BE_eV"] == pytest.approx(1832.0 - 1825.0)  # +7 (b more bound)
    assert out["n_sites_b"] == 2 and out["n_sites_a"] == 1
    # independent wander of each cell cancels
    si2 = {"a0": {"symbol": "Si", "levels": {"2p": -1830.0 + 4.0}}}
    sio2b = {k: {"symbol": "Si", "levels": {o: v - 2.5 for o, v in r["levels"].items()}}
             for k, r in sio2.items()}
    out2 = cross_cell_binding_shift(si2, ef_si + 4.0, sio2b, ef_sio2 - 2.5, "Si", "2p")
    assert out2["delta_BE_eV"] == pytest.approx(out["delta_BE_eV"])


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
    # Two chemically-identical O at different Ti-O distances (1.75/2.30 A) in a vacuum
    # box: a within-cell shift whose ONLY source is the inter-site electrostatic
    # (Madelung) field. Elk 11.0.2 gives the O NEARER the net-positive Ti MORE bound,
    # ΔC0_ext(long-short) = +2.90/+4.79 eV at O R_MT 1.00/1.40 Bohr (matched-basis
    # decomposition, experiments/autoapw/xps_madelung_stage2a.md).
    #
    # Reproducing that — and flipping the RAW muffin-tin eigenvalue `delta_eV` to the
    # correct positive sign — needs the two levers of the stage-2 fix:
    #   (b) mask_interstitial=True: band-limited interstitial ρ_I mask (Elk rhoir) — removes
    #       the small-Ti-sphere catastrophic-cancellation corruption of the Weinert v_hart.
    #   (a) Ti 3s as a valence semicore local orbital (los/val_e/core) instead of frozen
    #       core, so the Ti in-sphere charge matches Elk's (18.75 e, not 19.52 e); a correct
    #       cation charge is the DOMINANT lever for the Madelung field difference.
    # With both, delta_madelung_eV → Elk's ballpark and the raw delta_eV flips POSITIVE
    # (measured on the Ti+2O demo: -1.19 → +1.19 at O R_MT 1.40 Bohr; the corundum ²⁷Al
    # V_zz EFG is unchanged by masking, -6.996 → -6.992 eV/Å²). Left at the defaults both
    # numbers are wrong-signed — see xps_madelung_stage2a.md for the attribution.
    import numpy as _np

    from gradwave.flapw import crystal_scf_multi

    bohr = 0.529177210903
    ll = 13.0
    ti = _np.array([0.5, 0.5, 0.5]) * ll
    o_short = ti + _np.array([1.75 / bohr, 0.0, 0.0])
    o_long = ti + _np.array([0.0, 2.30 / bohr, 0.0])
    atoms = [(tuple(ti / ll), "Ti"), (tuple(o_short / ll), "O"), (tuple(o_long / ll), "O")]
    _, info = crystal_scf_multi(
        float(ll), atoms, {"Ti": 0.90, "O": 0.70},
        ecut=200.0, iters=40, kmesh=(1, 1, 1), use_symmetry=False, smearing=0.10,
        mask_interstitial=True,                                    # lever (b)
        los={"Ti": [(0, "3s")]}, val_e={"Ti": 12},                 # lever (a): Ti 3s LO
        core={"Ti": [(0, 1, 2), (0, 2, 2), (1, 1, 6)]})            # drop 3s from frozen core
    o = [s for s in info["core_level_shifts"] if s["species"] == "O" and s["orbital"] == "1s"]
    assert len(o) == 1
    # Physically-correct initial-state shift: POSITIVE, in Elk's ballpark (a2 long − a1
    # short). At O R_MT 0.70 A (1.32 Bohr) gradwave gives ~+4.3 (Elk interp ~+4.4).
    d_mad = o[0]["delta_madelung_eV"]
    assert d_mad is not None and not np.isnan(d_mad)
    assert 3.0 < d_mad < 5.5  # Elk interp ~+4.4 eV at this R_MT, O nearer Ti more bound
    # The raw muffin-tin eigenvalue shift now FLIPS POSITIVE with the two levers (was the
    # documented ~-1.67 eV wrong-signed artifact at the defaults).
    d_eig = o[0]["delta_eV"]
    assert 0.5 < d_eig < 2.0
    assert not np.isnan(d_eig)
