"""End-to-end FLAPW electric-field-gradient (EFG) magnitude, cross-validated against Elk 11.0.2.

The synthetic EFG unit tests (``test_flapw_efg*``, ``test_efg_paw``) pin the operator algebra and a
frozen-density gradcheck, but carry NO external magnitude anchor — the real gradwave-vs-Elk numbers
lived only in ``experiments/autoapw/`` (``efg_multimaterial_validation.md``). This test promotes the
strongest of those to a committed cross-code assertion: α-Al₂O₃ corundum's ²⁷Al site, the
multi-material study's headline (axial η≈0, and gradwave's C_Q lands within 4 % of experiment).

Elk 11.0.2 reference (all-electron FLAPW, LDA PW92 / xctype 3, muffin-tin radii forced to gradwave's
spheres R_Al=0.97 Å R_O=0.824 Å, rgkmax matched, ngridk — the ``experiments/autoapw/corundum_elk``
run): Al V_zz = −0.0607456 a.u. × 97.174 eV/Å²·a.u.⁻¹ = **−5.90 eV/Å²**, η = 0.005 (axial ∥c).

gradwave reproduces V_zz ≈ −7 eV/Å² (~118 % of Elk, same sign, η≈0) — an all-electron FLAPW EFG
magnitude that lands within ~20 % of an independent all-electron code, with the Al 2p semicore
frozen. This is a genuine cross-code magnitude check (it catches sign flips and the documented
"runaway fixed point" magnitude blow-ups), asserted with a defensible band, not a tight pin: the
FLAPW fullpot EFG is basin/lmax-sensitive (see ``experiments/autoapw/TIO2_NMR.md``).

TORTURE tier: the staged muffin-tin → fullpot converger is minutes-long and not run in CI (Elk is
never in CI either). Run manually when the FLAPW EFG subsystem changes:
``uv run python -m pytest tests/integration/test_flapw_efg_vs_elk.py -m torture -n0``.
"""

from __future__ import annotations

import numpy as np
import pytest

# 1 a.u. (Ha/Bohr²) → eV/Å²; and the committed Elk 11.0.2 corundum Al reference.
_AU_TO_EV_ANG2 = 97.174
ELK_CORUNDUM_AL_VZZ_EV_ANG2 = -0.0607456 * _AU_TO_EV_ANG2  # = −5.903 eV/Å²
ELK_CORUNDUM_AL_ETA = 0.005

# α-Al₂O₃ corundum primitive cell (R-3c #167, a=4.7602 Å c=12.9933 Å; Al z=0.35216, O x=0.30624).
_CELL_BOHR = np.array([[4.497737, 2.596770, 8.184592],
                       [-4.497737, 2.596770, 8.184592],
                       [-0.000000, -5.193539, 8.184592]])
_ATOMS = [((0.352160, 0.352160, 0.352160), "Al"), ((0.147840, 0.147840, 0.147840), "Al"),
          ((0.647840, 0.647840, 0.647840), "Al"), ((0.852160, 0.852160, 0.852160), "Al"),
          ((0.250000, 0.556240, 0.943760), "O"), ((0.056240, 0.750000, 0.443760), "O"),
          ((0.556240, 0.943760, 0.250000), "O"), ((0.443760, 0.056240, 0.750000), "O"),
          ((0.943760, 0.250000, 0.556240), "O"), ((0.750000, 0.443760, 0.056240), "O")]
_RADII = {"Al": 0.97, "O": 0.824}


def _inject_aluminium() -> None:
    """Inject the Al atomic species at runtime (gradwave's FLAPW tables ship Ti/O/Ne/Be/He/Si/Ar,
    not Al). The reference was validated vs NIST LDA in the multi-material study; core partition is
    frozen [Ne] (valence 3s²3p¹), mirroring ``experiments/autoapw/corundum_efg.py`` exactly."""
    from gradwave.flapw import atom as _atom
    from gradwave.flapw import scf as _scf

    _atom.CONFIG["Al"] = (13.0, [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 1)])
    _scf._CORE["Al"] = [(0, 1, 2), (0, 2, 2), (1, 1, 6)]      # freeze 1s, 2s, 2p ([Ne])
    _scf._VAL_E["Al"] = 3                                      # valence 3s² 3p¹
    _scf._N_VAL_BANDS["Al"] = 2
    _scf._VALENCE_NL["Al"] = {0: "3s", 1: "3p"}
    _al_ha = {"1s": -55.5, "2s": -3.93, "2p": -2.56, "3s": -0.286, "3p": -0.10}
    _atom.NIST_LDA_EV["Al"] = {k: v * 27.211386 for k, v in _al_ha.items()}


def _converged_corundum_efg():
    """Staged muffin-tin → warm-fullpot → newton_polish converger (a compact inline of
    ``experiments/autoapw/_efgrun.converge_efg``), then one exact EFG pass. Returns the info."""
    from gradwave.flapw import crystal_scf_multi, newton_polish

    cfg = dict(ecut=300.0, lmax=4, fullpot=True, fullpot_lmax=4, smearing=0.0,
               use_symmetry=True, subspace_reuse=False, kerker=0.7, shift_invert=True,
               kworkers=4, los={"O": [(0, "2s")]},
               el_override={"O": {0: "2p"}})
    k = (2, 2, 2)

    # Stage A — muffin-tin (robust spherical/interstitial fixed point).
    mtcfg = {kk: vv for kk, vv in cfg.items() if kk != "fullpot_lmax"}
    mtcfg["fullpot"] = False
    _, iwA = crystal_scf_multi(_CELL_BOHR, _ATOMS, _RADII, iters=90, tol=1e-5, efg=False,
                               kmesh=k, **mtcfg)

    # Stage B — short fullpot continuation, warm-started, chunked with a divergence guard.
    best_state, best_rv, warm, n, r = None, np.inf, {"__full_state__": iwA["state"]}, 0, None
    while n < 48:
        _, iw = crystal_scf_multi(_CELL_BOHR, _ATOMS, _RADII, iters=6, tol=1e-3, efg=False,
                                  kmesh=k, v_start=warm, **cfg)
        r = iw["recorder"].summarize()
        n += r["n_iter"]
        warm = {"__full_state__": iw["state"]}
        if r["r_v"] < best_rv:
            best_rv, best_state = r["r_v"], iw["state"]
        if r["r_nsph"] < 1e-3 and r["r_v"] < 1.2e-2:
            break
        if r["r_v"] > 20 * best_rv and r["r_v"] > 1.0:
            break
    gated = r is not None and r["r_nsph"] < 1e-3 and r["r_v"] < 1.2e-2
    final_state = iw["state"] if gated else best_state

    # Stage C — newton_polish the best fullpot state to the coupled fixed point if B did not gate.
    if not gated and best_state is not None:
        st, ni = newton_polish(_CELL_BOHR, _ATOMS, _RADII, best_state,
                               scf_kwargs=dict(cfg, kmesh=k, efg=False),
                               maxiter=6, inner_maxiter=14, f_tol=1e-6, rounds=3)
        if ni["residual_norm"] < 5e-3:
            final_state = st

    _, ie = crystal_scf_multi(_CELL_BOHR, _ATOMS, _RADII, iters=1, tol=0.0, efg=True, kmesh=k,
                              v_start={"__full_state__": final_state}, **cfg)
    return ie


@pytest.mark.torture
def test_corundum_al_efg_vs_elk():
    """gradwave FLAPW ²⁷Al V_zz in α-Al₂O₃ matches Elk 11.0.2 in sign and magnitude (~118 %).

    Asserts: same (negative) sign, |V_zz| within ±35 % of Elk's −5.90 eV/Å² (bracketing the
    documented ~118 % with margin for the fullpot loop's basin sensitivity), and an axial η. This is
    the committed cross-code magnitude anchor for the all-electron FLAPW EFG."""
    _inject_aluminium()
    ie = _converged_corundum_efg()
    al = ie["efg"]["a0"]           # first Al site
    v_zz = float(al["V_zz"])
    elk = ELK_CORUNDUM_AL_VZZ_EV_ANG2

    assert np.sign(v_zz) == np.sign(elk), f"Al V_zz sign {v_zz:+.2f} vs Elk {elk:+.2f}"
    assert abs(v_zz - elk) < 0.35 * abs(elk), f"Al V_zz {v_zz:+.3f} vs Elk {elk:+.3f} eV/Å²"
    assert float(al["eta"]) < 0.15, f"Al site is axial; eta={al['eta']:.3f}"
