"""Tighten the anatase fullpot fixed point from the saved best state, then re-measure the EFG.

The first pass reached r_nsph~3e-3 (kerker=0.7) but newton_polish diverged (chaotic-sensitive) and
the EFG came out at ~40% of Elk. The rutile parity was at r_nsph~2e-4. This refinement warm-starts
from the saved state and tries warm PLAIN Anderson (kerker=None — the doc's post-fix converger from
a good basin) in small chunks, keeps the best (lowest r_v) state, then newton_polish with more
rounds, then the EFG.
"""
import os
import pickle
import sys
import time

import numpy as np
from anatase_efg import ATOMS, AXES, CELL_BOHR, RADII, cfg_from_env, report_site  # noqa: F401

from gradwave.flapw import crystal_scf_multi, newton_polish

cfg, _ = cfg_from_env()
cfg["kerker"] = None if os.environ.get("KERKER", "none").lower() in ("none", "0", "") \
    else float(os.environ["KERKER"])
STATE = os.environ.get("STATE", os.path.expanduser("~/efg_multimat/anatase_OLO1.pkl"))
K = (2, 2, 2)


def main():
    with open(STATE, "rb") as f:
        st0 = pickle.load(f)
    print(f"# refine anatase from {STATE} kerker={cfg['kerker']} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    warm = {"__full_state__": st0}
    best, best_rv = st0, np.inf
    n = 0
    t0 = time.time()
    for _ in range(10):
        _, iw = crystal_scf_multi(CELL_BOHR, ATOMS, RADII, iters=8, tol=1e-3, efg=False,
                                  kmesh=K, v_start=warm, **cfg)
        r = iw["recorder"].summarize()
        n += r["n_iter"]
        warm = {"__full_state__": iw["state"]}
        if r["r_v"] < best_rv:
            best_rv, best = r["r_v"], iw["state"]
        print(f"[warm] n_it={n} r_v={r['r_v']:.2e} r_nsph={r['r_nsph']:.2e} best={best_rv:.2e}",
              flush=True)
        if r["r_nsph"] < 3e-4 and r["r_v"] < 1.2e-2:
            break
        if r["r_v"] > 20 * best_rv and r["r_v"] > 1.0:
            print("[warm] diverging; stop", flush=True)
            break
    print(f"[warm {time.time()-t0:.0f}s] best_rv={best_rv:.2e}", flush=True)
    final = best
    if int(os.environ.get("NEWTON", "1")):
        t0 = time.time()
        try:
            st, ni = newton_polish(CELL_BOHR, ATOMS, RADII, best,
                                   scf_kwargs=dict(cfg, kmesh=K, efg=False),
                                   maxiter=6, inner_maxiter=12, f_tol=1e-7, rounds=2)
            print(f"[newton {time.time()-t0:.0f}s] res={ni['residual_norm']:.2e} "
                  f"f_evals={ni['f_evals']}", flush=True)
            if ni["residual_norm"] < 5e-3:
                final = st
        except Exception as e:      # noqa: BLE001
            print(f"[newton FAILED] {type(e).__name__}: {e}", flush=True)
    with open(os.path.expanduser("~/efg_multimat/anatase_refined.pkl"), "wb") as f:
        pickle.dump(final, f)
    _, ie = crystal_scf_multi(CELL_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True, kmesh=K,
                              v_start={"__full_state__": final}, **cfg)
    print("=== REFINED ANATASE ===", flush=True)
    report_site(ie["efg"]["a0"], "Ti", "49Ti",
                "Elk: V_zz=-9.48 eta=0.00 (axial c) C_Q(49Ti)=5.66 MHz")
    report_site(ie["efg"]["a2"], "O", "17O", "Elk: V_zz=-12.42 eta=0.096 C_Q(17O)=0.77 MHz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
