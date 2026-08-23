"""Converged-k EFG re-read runner (AutoAPW task #53).

Unified k-parameterized FLAPW-EFG driver over the #350/#352/#353 material set, so the
k222-vs-Elk-k444 mismatch found in efg_lightcation_diagnosis.md can be re-taken at matched,
converged k. One MAT per invocation; KMESH sets the isotropic n (n,n,n) fed to BOTH gradwave
and Elk. Reuses converge_efg (muffin-tin -> warm fullpot -> newton -> exact efg pass) verbatim.

Per site it prints the SAME term-decomposition the campaign standardized on:
  full V_zz / eta / C_Q,  on-site V_zz_valence (l=2 sphere-Poisson),  boundary = full - on-site
  (a scalar; strictly the same principal axis only for the axial eta=0 sites),
  sphere_charge (valence l=0 in-sphere charge, the partition check vs Elk).

HELO (the #353 production basis): unconfined l=1 anion HELO (e=90, confine=False) on O/F; a
confined l=1 Al LO (e=90) on corundum Al. Light cations (B, Li) get NO cation HELO (the #353
recipe does not apply one) -> hbn/li3n default to baseline. Env HELO=0/1 overrides per run.

Env: MAT, KMESH (int n), HELO (0/1), ECUT, LMAX, FPLMAX, KERKER, KWORKERS, RV, OUT.
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np
from _efgrun import converge_efg
from _mgroup import hex_cell_bohr  # noqa: F401  (import injects B/N/Na/Li/Al species)

from gradwave.flapw import atom as _atom
from gradwave.flapw import scf as _scf

# ---- Mg + F injection (validated vs NIST LDA in efg_multimaterial_validation.md); B/N/Na/Li/Al
# come from importing _mgroup above. Module top level so the spawn k-worker pool re-applies it. ----
_atom.CONFIG["Mg"] = (12.0, [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2)])
_atom.CONFIG["F"] = (9.0, [(1, 0, 2), (2, 0, 2), (2, 1, 5)])
_scf._CORE["Mg"] = [(0, 1, 2), (0, 2, 2), (1, 1, 6)]
_scf._CORE["F"] = [(0, 1, 2)]
_scf._VAL_E["Mg"] = 2
_scf._VAL_E["F"] = 7
_scf._N_VAL_BANDS["Mg"] = 1
_scf._N_VAL_BANDS["F"] = 4
_scf._VALENCE_NL["Mg"] = {0: "3s"}
_scf._VALENCE_NL["F"] = {0: "2s", 1: "2p"}


def _hex(a, c):
    return hex_cell_bohr(a, c)


# ---- material table: cell, atoms, radii, anion, ecut default, sites [(akey,label,iso,Q_barn)],
# helo_default (1 => apply anion HELO / Al LO), al_confined_helo flag ----
U_RUT = 0.3048
U_MGF = 0.303
MATS = {
    "rutile": dict(
        cell=np.array([8.68083, 8.68083, 5.59096]),
        atoms=[((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
               ((U_RUT, U_RUT, 0.0), "O"), ((1 - U_RUT, 1 - U_RUT, 0.0), "O"),
               ((0.5 + U_RUT, 0.5 - U_RUT, 0.5), "O"), ((0.5 - U_RUT, 0.5 + U_RUT, 0.5), "O")],
        radii={"Ti": 1.098, "O": 0.824}, anion="O", ecut=300.0,
        sites=[("a0", "Ti", "49Ti", 0.247), ("a2", "O", "17O", -0.02558)],
        helo=1),
    "anatase": dict(
        cell=np.array([[-3.575551, 3.575551, 8.989993],
                       [3.575551, -3.575551, 8.989993],
                       [3.575551, 3.575551, -8.989993]]),
        atoms=[((0.0, 0.0, 0.0), "Ti"), ((0.75, 0.25, 0.5), "Ti"),
               ((0.2081, 0.2081, 0.0), "O"), ((0.9581, 0.4581, 0.5), "O"),
               ((0.5419, 0.0419, 0.5), "O"), ((0.7919, 0.7919, 0.0), "O")],
        radii={"Ti": 1.06, "O": 0.824}, anion="O", ecut=300.0,
        sites=[("a0", "Ti", "49Ti", 0.247), ("a2", "O", "17O", -0.02558)],
        helo=1),
    "corundum": dict(
        cell=np.array([[4.497737, 2.596770, 8.184592],
                       [-4.497737, 2.596770, 8.184592],
                       [-0.000000, -5.193539, 8.184592]]),
        atoms=[((0.352160, 0.352160, 0.352160), "Al"), ((0.147840, 0.147840, 0.147840), "Al"),
               ((0.647840, 0.647840, 0.647840), "Al"), ((0.852160, 0.852160, 0.852160), "Al"),
               ((0.250000, 0.556240, 0.943760), "O"), ((0.056240, 0.750000, 0.443760), "O"),
               ((0.556240, 0.943760, 0.250000), "O"), ((0.443760, 0.056240, 0.750000), "O"),
               ((0.943760, 0.250000, 0.556240), "O"), ((0.750000, 0.443760, 0.056240), "O")],
        radii={"Al": 0.97, "O": 0.824}, anion="O", ecut=300.0,
        sites=[("a0", "Al", "27Al", 0.1466), ("a4", "O", "17O", -0.02558)],
        helo=1, al_confined=True),
    "mgf2": dict(
        cell=np.array([8.73242, 8.73242, 5.69941]),
        atoms=[((0.0, 0.0, 0.0), "Mg"), ((0.5, 0.5, 0.5), "Mg"),
               ((U_MGF, U_MGF, 0.0), "F"), ((1 - U_MGF, 1 - U_MGF, 0.0), "F"),
               ((0.5 + U_MGF, 0.5 - U_MGF, 0.5), "F"), ((0.5 - U_MGF, 0.5 + U_MGF, 0.5), "F")],
        radii={"Mg": 1.0, "F": 0.80}, anion="F", ecut=300.0,
        sites=[("a0", "Mg", "25Mg", 0.1994), ("a2", "F", "19F", 0.0)],
        helo=1),
    "hbn": dict(
        cell=_hex(2.504, 6.661),
        atoms=[((1 / 3, 2 / 3, 1 / 4), "B"), ((2 / 3, 1 / 3, 3 / 4), "B"),
               ((2 / 3, 1 / 3, 1 / 4), "N"), ((1 / 3, 2 / 3, 3 / 4), "N")],
        radii={"B": 0.70, "N": 0.70}, anion="N", ecut=400.0,
        sites=[("a0", "B", "11B", 0.04059), ("a2", "N", "14N", 0.02044)],
        helo=0),
    "li3n": dict(
        cell=_hex(3.648, 3.875),
        atoms=[((0.0, 0.0, 0.5), "Li"), ((1 / 3, 2 / 3, 0.0), "Li"),
               ((2 / 3, 1 / 3, 0.0), "Li"), ((0.0, 0.0, 0.0), "N")],
        radii={"Li": 0.90, "N": 0.90}, anion="N", ecut=250.0,
        sites=[("a0", "Li1", "7Li", -0.04010), ("a1", "Li2", "7Li", -0.04010),
               ("a3", "N", "14N", 0.02044)],
        helo=0),
}


def build_cfg(m, helo):
    kerker = os.environ.get("KERKER", "0.7")
    c = dict(ecut=float(os.environ.get("ECUT", str(m["ecut"]))),
             lmax=int(os.environ.get("LMAX", "4")), fullpot=True,
             fullpot_lmax=int(os.environ.get("FPLMAX", "4")), smearing=0.0,
             use_symmetry=True, subspace_reuse=False,
             kerker=(None if kerker.lower() in ("none", "0", "") else float(kerker)),
             shift_invert=True, kworkers=int(os.environ.get("KWORKERS", "4")))
    anion = m["anion"]
    los = {anion: [(0, "2s")]}
    el = {anion: {0: "2p"}}
    if helo:
        los[anion].append((1, {"e": 90.0, "confine": False}))   # unconfined anion l=1 HELO
        if m.get("al_confined"):
            los["Al"] = [(1, {"e": 90.0, "confine": True})]      # confined Al l=1 LO (#353)
    c["los"] = los
    c["el_override"] = el
    return c


def cq(vzz, q):
    return abs(2.4180 * q * vzz) if q else 0.0


def report(site, label, iso, q):
    on = site["V_zz_valence"]
    full = site["V_zz"]
    bnd = full - on
    ch = site["sphere_charge"]
    c = cq(full, q)
    ctxt = f" C_Q({iso})={c:.4f}MHz" if q else ""
    print(f"[{label}] full V_zz={full:+.4f} eta={site['eta']:.4f}{ctxt}", flush=True)
    print(f"       ONSITE V_zz={on:+.4f} eta_val={site['eta_valence']:.4f}  "
          f"BOUNDARY={bnd:+.4f}  sphere_charge(val,l0)={ch:.4f} e", flush=True)


def main():
    mat = os.environ["MAT"]
    m = MATS[mat]
    n = int(os.environ.get("KMESH", "6"))
    kmesh = (n, n, n)
    helo = int(os.environ.get("HELO", str(m["helo"])))
    cfg = build_cfg(m, helo)
    tag = f"{mat}_k{n}_{'helo' if helo else 'base'}"
    out = os.path.expanduser(os.environ.get("OUT", f"~/efg_kconv/{tag}.pkl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"# {tag} MAT={mat} kmesh={kmesh} HELO={helo} los={cfg['los']} "
          f"ecut={cfg['ecut']} R={m['radii']} kerker={cfg['kerker']} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')} kworkers={cfg['kworkers']}", flush=True)
    t0 = time.time()
    try:
        ie, status, state, _ = converge_efg(m["cell"], m["atoms"], m["radii"], cfg, kmesh,
                                            rv_gate=float(os.environ.get("RV", "1.2e-2")))
        with open(out, "wb") as f:
            pickle.dump(state, f)
        print(f"=== RESULT {tag} {status} ({time.time()-t0:.0f}s) ===", flush=True)
        for akey, label, iso, q in m["sites"]:
            report(ie["efg"][akey], label, iso, q)
    except Exception as e:  # noqa: BLE001
        print(f"=== FAILED {tag} {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
