"""DECISIVE Step-1 test: does an l=1 HELO move the on-site EFG gw/Elk ratio -> 1 for BOTH Al and O?

Warm-starts from the converged no-HELO corundum state (~/efg_multimat/corundum.pkl carries the
potentials; the LAPW+LO coefficients are recomputed fresh in the HELO-extended basis), adds an
unconfined l=1 HELO on the requested species, re-gates a short fullpot continuation, and reports
the on-site (valence l=2 sphere-Poisson) V_zz for Al and O vs Elk's on-site reference.

Elk on-site (nominal R, from efg_partition_diagnosis, validated vs EFG.OUT):
  Al on-site V_zz = -6.185 eV/A^2   O on-site V_zz = +27.082 eV/A^2
gradwave baseline on-site (no HELO, gated): Al -7.26 (ratio 1.17)  O +17.76 (ratio 0.66)

Env:
  HELO_E=90       HELO E2 (absolute atomic eV). "" => no HELO (baseline control).
  HELO_CONF=0     confine flag (1 => confined LO, 0 => unconfined HELO).
  HELO_SYMS=Al,O  which species get the l=1 HELO.
  KERKER=0.7 ECUT=300 LMAX=4 FPLMAX=4 KWORKERS=4 MAXIT=60 RV=1.2e-2
  STATE=~/efg_multimat/corundum.pkl  OUT=~/efg_multimat/corundum_helo.pkl
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np

from gradwave.flapw import atom as _atom
from gradwave.flapw import crystal_scf_multi
from gradwave.flapw import scf as _scf
from gradwave.flapw.nmr import quadrupolar_coupling

# ---- runtime Al injection (validated vs NIST LDA; module top level for the spawn pool) ----
_atom.CONFIG["Al"] = (13.0, [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 1)])
_scf._CORE["Al"] = [(0, 1, 2), (0, 2, 2), (1, 1, 6)]
_scf._VAL_E["Al"] = 3
_scf._N_VAL_BANDS["Al"] = 2
_scf._VALENCE_NL["Al"] = {0: "3s", 1: "3p"}
_al_ha = {"1s": -55.5, "2s": -3.93, "2p": -2.56, "3s": -0.286, "3p": -0.10}
_atom.NIST_LDA_EV["Al"] = {k: v * 27.211386 for k, v in _al_ha.items()}

CELL_BOHR = np.array([[4.497737, 2.596770, 8.184592],
                      [-4.497737, 2.596770, 8.184592],
                      [-0.000000, -5.193539, 8.184592]])
ATOMS = [((0.352160, 0.352160, 0.352160), "Al"), ((0.147840, 0.147840, 0.147840), "Al"),
         ((0.647840, 0.647840, 0.647840), "Al"), ((0.852160, 0.852160, 0.852160), "Al"),
         ((0.250000, 0.556240, 0.943760), "O"), ((0.056240, 0.750000, 0.443760), "O"),
         ((0.556240, 0.943760, 0.250000), "O"), ((0.443760, 0.056240, 0.750000), "O"),
         ((0.943760, 0.250000, 0.556240), "O"), ((0.750000, 0.443760, 0.056240), "O")]
RADII = {"Al": 0.97, "O": 0.824}

ELK_ON = {"Al": -6.185, "O": +27.082}
BASE_ON = {"Al": -7.26, "O": +17.76}


def cfg_from_env():
    kerker = os.environ.get("KERKER", "0.7")
    c = dict(ecut=float(os.environ.get("ECUT", "300")), lmax=int(os.environ.get("LMAX", "4")),
             fullpot=True, fullpot_lmax=int(os.environ.get("FPLMAX", "4")), smearing=0.0,
             use_symmetry=True, subspace_reuse=False,
             kerker=(None if kerker.lower() in ("none", "0", "") else float(kerker)),
             shift_invert=True, kworkers=int(os.environ.get("KWORKERS", "4")))
    los = {"O": [(0, "2s")]}
    el = {"O": {0: "2p"}}
    he = os.environ.get("HELO_E", "90")
    # HELO_CONF: a scalar "0"/"1" applied to all HELO species, or a per-species map
    # "Al=1,O=0" (Al confined LO, O unconfined HELO).
    conf_env = os.environ.get("HELO_CONF", "0")
    if "=" in conf_env:
        conf_map = {kv.split("=")[0]: bool(int(kv.split("=")[1])) for kv in conf_env.split(",")}
    else:
        conf_map = None
        conf_default = bool(int(conf_env))
    if he:
        for sym in os.environ.get("HELO_SYMS", "Al,O").split(","):
            conf = conf_map[sym] if conf_map is not None else conf_default
            los.setdefault(sym, []).append((1, {"e": float(he), "confine": conf}))
    c["los"] = los
    c["el_override"] = el
    return c, he


def report_site(site, name, isotope):
    t = np.array(site["tensor"])
    w = np.sort(np.abs(np.linalg.eigvalsh(t)))[::-1]
    cq = quadrupolar_coupling(site["V_zz"], site["eta"], isotope)
    on = site["V_zz_valence"]
    ratio = on / ELK_ON[name]
    print(f"{name} FULL: V_zz={site['V_zz']:+.3f} eta={site['eta']:.3f} "
          f"C_Q({isotope})={cq['abs_C_Q_MHz']:.3f} MHz  eig|{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}|",
          flush=True)
    print(f"     ONSITE: V_zz={on:+.3f} (Elk {ELK_ON[name]:+.3f}, base {BASE_ON[name]:+.2f})  "
          f"gw/Elk={ratio:.3f}  eta_val={site['eta_valence']:.3f}", flush=True)


def main():
    cfg, he = cfg_from_env()
    state_env = os.path.expanduser(os.environ.get("STATE", "~/efg_multimat/corundum.pkl"))
    out = os.path.expanduser(os.environ.get("OUT", "~/efg_multimat/corundum_helo.pkl"))
    maxit = int(os.environ.get("MAXIT", "60"))
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
    conf = os.environ.get("HELO_CONF", "0")
    syms = os.environ.get("HELO_SYMS", "Al,O")
    print(f"# CORUNDUM HELO decisive: HELO_E={he or 'NONE'} confine={conf} syms={syms} "
          f"kerker={cfg['kerker']} los={cfg['los']} warm={state_env}", flush=True)
    with open(state_env, "rb") as f:
        state = pickle.load(f)
    t0 = time.time()
    try:
        _, iw = crystal_scf_multi(CELL_BOHR, ATOMS, RADII, iters=20, tol=1e-3, efg=False,
                                  kmesh=(2, 2, 2), v_start={"__full_state__": state}, **cfg)
        r = iw["recorder"].summarize()
        n = r["n_iter"]
        while (r["r_nsph"] >= 1e-3 or r["r_v"] >= rv_gate) and n < maxit:
            _, iw = crystal_scf_multi(CELL_BOHR, ATOMS, RADII, iters=12, tol=1e-3, efg=False,
                                      kmesh=(2, 2, 2),
                                      v_start={"__full_state__": iw["state"]}, **cfg)
            r = iw["recorder"].summarize()
            n += r["n_iter"]
            print(f"  cont: n_it={n} r_v={r['r_v']:.2e} r_nsph={r['r_nsph']:.2e}", flush=True)
        gated = r["r_nsph"] < 1e-3 and r["r_v"] < rv_gate
        with open(out, "wb") as f:
            pickle.dump(iw["state"], f)
        print(f"SCF ({time.time()-t0:.0f}s): n_it={n} r_v={r['r_v']:.2e} r_nsph={r['r_nsph']:.2e} "
              f"gated={gated} state={out}", flush=True)
        _, ie = crystal_scf_multi(CELL_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                                  kmesh=(2, 2, 2),
                                  v_start={"__full_state__": iw["state"]}, **cfg)
        print(f"=== RESULT HELO_E={he or 'NONE'} {'GATED' if gated else 'MARGINAL'} ===",
              flush=True)
        report_site(ie["efg"]["a0"], "Al", "27Al")
        report_site(ie["efg"]["a4"], "O", "17O")
    except Exception as e:  # noqa: BLE001
        print(f"=== FAILED after {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
