"""Warm-tighten a saved main-group fullpot state, then re-measure the EFG.

For the harder cells (AlN's marginal interstitial mode diverges the chunked continuation before it
gates, exactly like anatase in the prior study), warm-start from the saved best state and run warm
PLAIN Anderson (kerker=None — the doc's post-fix converger from a good basin), keep the lowest-r_v
state, newton_polish (more rounds), then one exact EFG pass. MOD selects the driver module (its
CELL/ATOMS/RADII/AXES/cfg_from_env are reused). NO src change.

Env: MOD=aln_efg STATE=~/efg_mgroup/aln.pkl KERKER=none NEWTON=1
"""
import importlib
import os
import pickle
import sys
import time

import numpy as np
from _mgroup import report_site

from gradwave.flapw import crystal_scf_multi, newton_polish

MOD = os.environ.get("MOD", "aln_efg")
m = importlib.import_module(MOD)
cfg = m.cfg_from_env()
cfg["kerker"] = None if os.environ.get("KERKER", "none").lower() in ("none", "0", "") \
    else float(os.environ["KERKER"])
STATE = os.environ.get("STATE", os.path.expanduser(f"~/efg_mgroup/{MOD.split('_')[0]}.pkl"))
K = (2, 2, 2)
# (efg-dict key, label, isotope) to report for each module
SITES = {"aln_efg": [("a0", "Al", "27Al")], "hbn_efg": [("a0", "B", "11B")],
         "nano2_efg": [("a0", "Na", "23Na")],
         "li3n_efg": [("a0", "Li1(1b)", "7Li"), ("a1", "Li2(2c)", "7Li")]}


def main():
    with open(STATE, "rb") as f:
        st0 = pickle.load(f)
    print(f"# refine {MOD} from {STATE} kerker={cfg['kerker']} ecut={cfg['ecut']} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    warm = {"__full_state__": st0}
    best, best_rv, best_rn = st0, np.inf, np.inf
    n, t0 = 0, time.time()
    for _ in range(int(os.environ.get("ROUNDS", "12"))):
        _, iw = crystal_scf_multi(m.CELL_BOHR, m.ATOMS, m.RADII, iters=8, tol=1e-3, efg=False,
                                  kmesh=K, v_start=warm, **cfg)
        r = iw["recorder"].summarize()
        n += r["n_iter"]
        warm = {"__full_state__": iw["state"]}
        if r["r_v"] < best_rv:
            best_rv, best_rn, best = r["r_v"], r["r_nsph"], iw["state"]
        print(f"[warm] n_it={n} r_v={r['r_v']:.2e} r_nsph={r['r_nsph']:.2e} best_rv={best_rv:.2e}",
              flush=True)
        if r["r_nsph"] < 5e-4 and r["r_v"] < 1.2e-2:
            break
        if r["r_v"] > 20 * best_rv and r["r_v"] > 1.0:
            print("[warm] diverging; stop", flush=True)
            break
    print(f"[warm {time.time()-t0:.0f}s] best_rv={best_rv:.2e} best_rnsph={best_rn:.2e}", flush=True)
    final, gated = best, (best_rv < 1.2e-2 and best_rn < 1e-3)
    if int(os.environ.get("NEWTON", "1")):
        t0 = time.time()
        try:
            st, ni = newton_polish(m.CELL_BOHR, m.ATOMS, m.RADII, best,
                                   scf_kwargs=dict(cfg, kmesh=K, efg=False),
                                   maxiter=6, inner_maxiter=14, f_tol=1e-7, rounds=3)
            print(f"[newton {time.time()-t0:.0f}s] res={ni['residual_norm']:.2e} "
                  f"f_evals={ni['f_evals']} converged={ni['converged']}", flush=True)
            if ni["residual_norm"] < 5e-3:
                final = st
                gated = ni["residual_norm"] < 1e-3
        except Exception as e:      # noqa: BLE001
            print(f"[newton FAILED] {type(e).__name__}: {e}", flush=True)
    out = os.path.expanduser(f"~/efg_mgroup/{MOD.split('_')[0]}_refined.pkl")
    with open(out, "wb") as f:
        pickle.dump(final, f)
    _, ie = crystal_scf_multi(m.CELL_BOHR, m.ATOMS, m.RADII, iters=1, tol=0.0, efg=True, kmesh=K,
                              v_start={"__full_state__": final}, **cfg)
    print(f"=== REFINED {MOD} {'GATED' if gated else 'MARGINAL'} state:{out} ===", flush=True)
    for key, label, iso in SITES[MOD]:
        report_site(ie["efg"][key], label, iso, m.AXES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
