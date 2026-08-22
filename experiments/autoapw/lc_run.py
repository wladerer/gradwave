"""Light-cation EFG diagnostic driver: term-decomposed on-site vs boundary + in-sphere charge.

MAT=hbn|li3n selects the material. Baseline (no cation lever) unless LEVER is set. Prints, per
site: full V_zz/eta, on-site V_zz_valence (the l=2 sphere-Poisson term the bias lives in),
sphere_charge (gw valence l=0 in-sphere charge, for the partition check vs Elk), and boundary =
full - on-site (valid for the axial eta=0 cation sites, same principal axis).

LEVER examples (opt-in per species,l — added to the cation's LO list, exactly like the HELO spec):
  LEVER="B:1:e=90:c=0"    high-E unconfined p-HELO on B (the anion-style lever; expected to WORSEN)
  LEVER="B:1:e=-8:c=0"    low-E / contracted unconfined p radial on B (the hypothesized opposite)
  LEVER="B:1:e=5:c=1"     confined LO at low E on B
Multiple levers separated by ';'.
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np
from _efgrun import converge_efg
from _mgroup import hex_cell_bohr, report_site  # noqa: F401 (import injects species)

MAT = os.environ.get("MAT", "hbn")
AXES = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0])}

if MAT == "hbn":
    CELL = hex_cell_bohr(2.504, 6.661)
    ATOMS = [((1 / 3, 2 / 3, 1 / 4), "B"), ((2 / 3, 1 / 3, 3 / 4), "B"),
             ((2 / 3, 1 / 3, 1 / 4), "N"), ((1 / 3, 2 / 3, 3 / 4), "N")]
    RADII = {"B": 0.70, "N": 0.70}
    SITES = [("a0", "B", "11B"), ("a2", "N", None)]
    ANION, ECUT_D = "N", 400.0
elif MAT == "li3n":
    CELL = hex_cell_bohr(3.648, 3.875)
    ATOMS = [((0.0, 0.0, 0.5), "Li"), ((1 / 3, 2 / 3, 0.0), "Li"),
             ((2 / 3, 1 / 3, 0.0), "Li"), ((0.0, 0.0, 0.0), "N")]
    RADII = {"Li": float(os.environ.get("RLI", "0.90")), "N": 0.90}
    SITES = [("a0", "Li1", "7Li"), ("a1", "Li2", "7Li"), ("a3", "N", None)]
    ANION, ECUT_D = "N", 250.0
else:
    raise SystemExit(f"unknown MAT={MAT}")


def parse_levers():
    out = {}
    spec = os.environ.get("LEVER", "").strip()
    if not spec:
        return out
    for chunk in spec.split(";"):
        parts = chunk.split(":")
        sym, l = parts[0], int(parts[1])
        kv = dict(p.split("=") for p in parts[2:])
        e = float(kv["e"])
        conf = bool(int(kv.get("c", "0")))
        out.setdefault(sym, []).append((l, {"e": e, "confine": conf}))
    return out


def cfg_from_env():
    kerker = os.environ.get("KERKER", "0.7")
    c = dict(ecut=float(os.environ.get("ECUT", str(ECUT_D))),
             lmax=int(os.environ.get("LMAX", "4")), fullpot=True,
             fullpot_lmax=int(os.environ.get("FPLMAX", "4")), smearing=0.0,
             use_symmetry=True, subspace_reuse=False,
             kerker=(None if kerker.lower() in ("none", "0", "") else float(kerker)),
             shift_invert=True, kworkers=int(os.environ.get("KWORKERS", "4")))
    los = {ANION: [(0, "2s")]}
    el = {ANION: {0: "2p"}}
    for sym, lst in parse_levers().items():
        los.setdefault(sym, []).extend(lst)
    c["los"] = los
    c["el_override"] = el
    return c


def report(site, name, iso):
    on = site["V_zz_valence"]
    full = site["V_zz"]
    bnd = full - on            # axial eta=0 -> same principal axis
    ch = site["sphere_charge"]
    cq = ""
    if iso is not None:
        from gradwave.flapw.nmr import quadrupolar_coupling
        cq = f" C_Q({iso})={quadrupolar_coupling(full, site['eta'], iso)['abs_C_Q_MHz']:.4f}MHz"
    print(f"[{name}] full V_zz={full:+.3f} eta={site['eta']:.3f}{cq}", flush=True)
    print(f"       ONSITE V_zz={on:+.3f}  BOUNDARY={bnd:+.3f}  sphere_charge(val,l0)={ch:.4f} e",
          flush=True)


def main():
    cfg = cfg_from_env()
    out = os.path.expanduser(os.environ.get("OUT", f"~/efg_lc/{MAT}.pkl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    km = int(os.environ.get("KMESH", "2"))
    kmesh = (km, km, km)
    print(f"# {MAT} lever={os.environ.get('LEVER','NONE')} los={cfg['los']} "
          f"ecut={cfg['ecut']} R={RADII} kerker={cfg['kerker']} kmesh={kmesh} "
          f"fplmax={cfg['fullpot_lmax']}", flush=True)
    t0 = time.time()
    try:
        ie, status, state, _ = converge_efg(CELL, ATOMS, RADII, cfg, kmesh,
                                            rv_gate=float(os.environ.get("RV", "1.2e-2")))
        with open(out, "wb") as f:
            pickle.dump(state, f)
        print(f"=== RESULT {MAT} {status} ({time.time()-t0:.0f}s) ===", flush=True)
        for key, name, iso in SITES:
            report(ie["efg"][key], name, iso)
    except Exception as e:  # noqa: BLE001
        print(f"=== FAILED {MAT} {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
