"""gradwave FLAPW EFG for MgF2 (rutile P4_2/mnm — the non-oxide analog of rutile TiO2) vs Elk.

The controlled anion test: MgF2 has the SAME rutile structure as TiO2, and the F site sits at the
SAME crystallographic position as rutile's O. Comparing the F-site EFG (gradwave vs Elk) isolates
whether the "anion at ~73% of Elk" deficit is O-chemistry-specific or a general property of the
rutile anion site. ¹⁹F is I=½ (no quadrupole → no experimental C_Q); the F check is gradwave-vs-Elk
tensor only. ²⁵Mg (I=5/2) has a Q, compared to experiment where data exists.

Mg and F are injected at runtime (no src change); both atomic references were validated vs NIST LDA
(Mg 1s −45.89/2p −1.70/3s −0.165 Ha; F 1s −24.16/2s −1.07/2p −0.39 Ha, all within ~1%). Mg 2p
frozen ([Ne] core); F frozen 1s (2s2p valence, like O), with the conditioned F 2s LO. Recipe mirrors
the TiO2 runs: aug4/fp4/k222 ecut300 kerker=0.7.
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

# ---- runtime Mg + F species injection (validated vs NIST LDA; no src file modified). Top level so
# the spawned k-worker pool re-applies it. ----
_atom.CONFIG["Mg"] = (12.0, [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2)])
_atom.CONFIG["F"] = (9.0, [(1, 0, 2), (2, 0, 2), (2, 1, 5)])
_scf._CORE["Mg"] = [(0, 1, 2), (0, 2, 2), (1, 1, 6)]     # freeze [Ne], valence 3s^2
_scf._CORE["F"] = [(0, 1, 2)]                            # freeze 1s, valence 2s^2 2p^5
_scf._VAL_E["Mg"] = 2
_scf._VAL_E["F"] = 7
_scf._N_VAL_BANDS["Mg"] = 1
_scf._N_VAL_BANDS["F"] = 4
_scf._VALENCE_NL["Mg"] = {0: "3s"}                       # l=1 falls back to default
_scf._VALENCE_NL["F"] = {0: "2s", 1: "2p"}

# rutile MgF2 (P4_2/mnm): a=4.621 A c=3.016 A u=0.303. Same topology as rutile TiO2 (Ti->Mg, O->F).
U = 0.303
CELL_BOHR = np.array([8.73242, 8.73242, 5.69941])        # orthorhombic edges (Bohr)
ATOMS = [((0.0, 0.0, 0.0), "Mg"), ((0.5, 0.5, 0.5), "Mg"),
         ((U, U, 0.0), "F"), ((1 - U, 1 - U, 0.0), "F"),
         ((0.5 + U, 0.5 - U, 0.5), "F"), ((0.5 - U, 0.5 + U, 0.5), "F")]
RADII = {"Mg": float(os.environ.get("RMG", "1.0")), "F": float(os.environ.get("RF", "0.80"))}

Q_BARN = {"25Mg": 0.1994}      # Pyykko 2018; 19F has no quadrupole moment
AXES = {"a": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        "b": np.array([1.0, -1.0, 0.0]) / np.sqrt(2),
        "c": np.array([0.0, 0.0, 1.0])}


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
    if int(os.environ.get("FLO", "1")):
        c["los"] = {"F": [(0, "2s")]}
        c["el_override"] = {"F": {0: "2p"}}
    return c


def report_site(site, name, isotope, elk=None):
    t = np.array(site["tensor"])
    w, vecs = np.linalg.eigh(t)
    order = np.argsort(-np.abs(w))
    w = w[order]
    vecs = vecs[:, order]
    proj = {ax: float(n @ t @ n) for ax, n in AXES.items()}
    cq = ""
    if isotope in Q_BARN:
        cq = f" C_Q({isotope})={abs(2.4180 * Q_BARN[isotope] * site['V_zz']):.3f} MHz"
    print(f"{name} FULL: eig=[{w[0]:+.3f},{w[1]:+.3f},{w[2]:+.3f}] eta={site['eta']:.3f} "
          f"V_zz={site['V_zz']:+.3f}{cq}", flush=True)
    print(f"     proj [110]/[1-10]/c = {proj['a']:+.3f}/{proj['b']:+.3f}/{proj['c']:+.3f}  "
          f"V_zz eigvec=[{vecs[0,0]:+.2f},{vecs[1,0]:+.2f},{vecs[2,0]:+.2f}]", flush=True)
    print(f"     ONSITE(valence): V_zz={site['V_zz_valence']:+.3f} eta={site['eta_valence']:.3f}",
          flush=True)
    if elk:
        print(f"     Elk {name}: {elk}", flush=True)


def main():
    cfg = cfg_from_env()
    out = os.environ.get("OUT", os.path.expanduser("~/efg_multimat/mgf2.pkl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"# MgF2 rutile ecut={cfg['ecut']} lmax={cfg['lmax']} fp{cfg['fullpot_lmax']} k222 "
          f"kerker={cfg['kerker']} R_Mg={RADII['Mg']} R_F={RADII['F']} "
          f"kworkers={cfg['kworkers']} OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
    t0 = time.time()
    try:
        ie, status, state, _ = converge_efg(CELL_BOHR, ATOMS, RADII, cfg, (2, 2, 2),
                                             rv_gate=rv_gate)
        with open(out, "wb") as f:
            pickle.dump(state, f)
        print(f"=== RESULT MgF2 {status} ({time.time()-t0:.0f}s) state: {out} ===", flush=True)
        report_site(ie["efg"]["a0"], "Mg", "25Mg")
        report_site(ie["efg"]["a2"], "F", "19F")
    except Exception as e:      # noqa: BLE001
        print(f"=== FAILED MgF2 after {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===",
              flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
