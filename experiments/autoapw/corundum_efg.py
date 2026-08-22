"""gradwave FLAPW EFG for alpha-Al2O3 CORUNDUM (R-3c, primitive 10-atom cell) vs Elk + experiment.

gradwave's flapw ships no Al species data (CONFIG/_CORE/_VAL_E/_VALENCE_NL cover only Ti/O/Ne/Be).
This runner INJECTS Al at runtime (no src change) — the injected Al atomic reference was validated
against NIST LDA eigenvalues (1s -55.06, 2p -2.54, 3s -0.295, 3p -0.092 Ha, all within ~1%).
Core partition: frozen [Ne] (1s2s2p), valence 3s^2 3p^1 (the simple, stable setup). This freezes
the Al 2p semicore, so — exactly as the rutile campaign found for the Ti 3p semicore — the Al
Sternheimer antishielding is NOT captured; expect gradwave to under-shoot the Elk (all-electron,
2p-in-valence) V_zz. That comparison is the point.

Injection is at MODULE TOP LEVEL so the spawned k-worker pool (spawn re-imports this module)
re-applies it. Recipe mirrors rutile/anatase: aug4/fp4/k222 ecut300 kerker=0.7, O 2s LO.
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np
from _efgrun import converge_efg

from gradwave.flapw import atom as _atom
from gradwave.flapw import scf as _scf
from gradwave.flapw.nmr import quadrupolar_coupling

# ---- runtime Al species injection (validated vs NIST LDA; no src file is modified). The dicts are
# read at call time by crystal_scf_multi, so injecting after import is sufficient; done at module
# top level so the spawned k-worker pool (spawn re-imports this module) re-applies it. ----
_atom.CONFIG["Al"] = (13.0, [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 1)])
_scf._CORE["Al"] = [(0, 1, 2), (0, 2, 2), (1, 1, 6)]       # freeze 1s,2s,2p ([Ne])
_scf._VAL_E["Al"] = 3                                       # valence 3s^2 3p^1
_scf._N_VAL_BANDS["Al"] = 2
_scf._VALENCE_NL["Al"] = {0: "3s", 1: "3p"}                # l>=2 falls back to default
_al_ha = {"1s": -55.5, "2s": -3.93, "2p": -2.56, "3s": -0.286, "3p": -0.10}
_atom.NIST_LDA_EV["Al"] = {k: v * 27.211386 for k, v in _al_ha.items()}

# ---- corundum primitive cell (ASE spacegroup 167, a=4.7602 c=12.9933 A; Al z=0.35216, O x=0.30624)
CELL_BOHR = np.array([[4.497737, 2.596770, 8.184592],
                      [-4.497737, 2.596770, 8.184592],
                      [-0.000000, -5.193539, 8.184592]])
ATOMS = [((0.352160, 0.352160, 0.352160), "Al"),
         ((0.147840, 0.147840, 0.147840), "Al"),
         ((0.647840, 0.647840, 0.647840), "Al"),
         ((0.852160, 0.852160, 0.852160), "Al"),
         ((0.250000, 0.556240, 0.943760), "O"),
         ((0.056240, 0.750000, 0.443760), "O"),
         ((0.556240, 0.943760, 0.250000), "O"),
         ((0.443760, 0.056240, 0.750000), "O"),
         ((0.943760, 0.250000, 0.556240), "O"),
         ((0.750000, 0.443760, 0.056240), "O")]
RADII = {"Al": float(os.environ.get("RAL", "0.97")), "O": float(os.environ.get("RO", "0.824"))}

AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0])}    # Cartesian z = trigonal c-axis of the primitive cell


def cfg_from_env():
    kerker = os.environ.get("KERKER", "0.7")
    c = dict(
        ecut=float(os.environ.get("ECUT", "300")),
        lmax=int(os.environ.get("LMAX", "4")),
        fullpot=True, fullpot_lmax=int(os.environ.get("FPLMAX", "4")),
        smearing=0.0, use_symmetry=True, subspace_reuse=False,
        kerker=(None if kerker.lower() in ("none", "0", "") else float(kerker)),
        shift_invert=True, kworkers=int(os.environ.get("KWORKERS", "4")),
    )
    if int(os.environ.get("OLO", "1")):
        c["los"] = {"O": [(0, "2s")]}
        c["el_override"] = {"O": {0: "2p"}}
    return c


def report_site(site, name, isotope):
    t = np.array(site["tensor"])
    w, vecs = np.linalg.eigh(t)
    order = np.argsort(-np.abs(w))
    w = w[order]
    vecs = vecs[:, order]
    proj = {ax: float(n @ t @ n) for ax, n in AXES.items()}
    cq = quadrupolar_coupling(site["V_zz"], site["eta"], isotope)
    print(f"{name} FULL: eig=[{w[0]:+.3f},{w[1]:+.3f},{w[2]:+.3f}] eta={site['eta']:.3f} "
          f"V_zz={site['V_zz']:+.3f} C_Q({isotope})={cq['abs_C_Q_MHz']:.3f} MHz", flush=True)
    print(f"     proj x/y/c = {proj['x']:+.3f}/{proj['y']:+.3f}/{proj['c']:+.3f}  "
          f"V_zz eigvec=[{vecs[0,0]:+.2f},{vecs[1,0]:+.2f},{vecs[2,0]:+.2f}]", flush=True)
    print(f"     ONSITE(valence): V_zz={site['V_zz_valence']:+.3f} eta={site['eta_valence']:.3f}",
          flush=True)


def main():
    cfg = cfg_from_env()
    out = os.environ.get("OUT", os.path.expanduser("~/efg_multimat/corundum.pkl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    state_env = os.environ.get("STATE", "")
    print(f"# CORUNDUM Al2O3 ecut={cfg['ecut']} lmax={cfg['lmax']} fp{cfg['fullpot_lmax']} "
          f"k222 kerker={cfg['kerker']} R_Al={RADII['Al']} R_O={RADII['O']} "
          f"kworkers={cfg['kworkers']} OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"start={'WARM' if state_env else 'COLD'} (Al frozen-[Ne], no antishielding)", flush=True)
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
    t0 = time.time()
    try:
        ie, status, state, _ = converge_efg(CELL_BOHR, ATOMS, RADII, cfg, (2, 2, 2),
                                             rv_gate=rv_gate)
        with open(out, "wb") as f:
            pickle.dump(state, f)
        print(f"=== RESULT CORUNDUM {status} ({time.time()-t0:.0f}s) state: {out} ===", flush=True)
        report_site(ie["efg"]["a0"], "Al", "27Al")
        report_site(ie["efg"]["a4"], "O", "17O")
    except Exception as e:      # noqa: BLE001
        print(f"=== FAILED CORUNDUM after {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===",
              flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
