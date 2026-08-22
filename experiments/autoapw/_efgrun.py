"""Shared robust FLAPW-EFG driver for the multi-material validation study.

The cold fullpot k222 trajectory is run-to-run fragile (marginal interstitial mode |rho|~1.02;
see TIO2_NMR.md). This driver converges robustly by:
  A. muffin-tin SCF (fullpot=False) to a tight spherical/interstitial fixed point (robust);
  B. a short fullpot continuation warm-started from A (coupled aspherical channel switched on near
     the basin), run in small chunks that track the best (lowest r_v) state and bail on divergence;
  C. newton_polish of the best fullpot state to the coupled fixed point (converges from r_v~2-3,
     r_nsph~0.2 per the measured hard-config probe).
Then one exact efg=True pass. Returns the efg info dict + a status string.
"""
import time

import numpy as np

from gradwave.flapw import crystal_scf_multi, newton_polish


def converge_efg(cell, atoms, radii, cfg, kmesh, *, rv_gate=1.2e-2, mt_iters=90,
                 fp_chunk=6, fp_max=48, newton_rounds=3):
    k = kmesh
    log = []

    def p(msg):
        print(msg, flush=True)
        log.append(msg)

    # ---- Stage A: muffin-tin ----
    t0 = time.time()
    mtcfg = {kk: vv for kk, vv in cfg.items() if kk != "fullpot_lmax"}
    mtcfg["fullpot"] = False
    _, iwA = crystal_scf_multi(cell, atoms, radii, iters=mt_iters, tol=1e-5, efg=False,
                               kmesh=k, **mtcfg)
    rA = iwA["recorder"].summarize()
    p(f"[A muffin-tin {time.time()-t0:.0f}s] n_it={rA['n_iter']} r_v={rA['r_v']:.2e}")

    # ---- Stage B: short fullpot warm from A, chunked + divergence-guarded ----
    t0 = time.time()
    best_state, best_rv = None, np.inf
    warm = {"__full_state__": iwA["state"]}
    n = 0
    r = None
    while n < fp_max:
        _, iw = crystal_scf_multi(cell, atoms, radii, iters=fp_chunk, tol=1e-3, efg=False,
                                  kmesh=k, v_start=warm, **cfg)
        r = iw["recorder"].summarize()
        n += r["n_iter"]
        warm = {"__full_state__": iw["state"]}
        rv = r["r_v"]
        if rv < best_rv:
            best_rv, best_state = rv, iw["state"]
        p(f"[B fullpot] n_it={n} r_v={rv:.2e} r_nsph={r['r_nsph']:.2e} best_rv={best_rv:.2e}")
        if r["r_nsph"] < 1e-3 and rv < rv_gate:
            break
        if rv > 20 * best_rv and rv > 1.0:            # diverging — stop, polish the best
            p("[B] diverging; stop and hand best state to newton")
            break
    p(f"[B fullpot {time.time()-t0:.0f}s] best_rv={best_rv:.2e}")

    gated = (r is not None and r["r_nsph"] < 1e-3 and r["r_v"] < rv_gate)
    final_state = iw["state"] if gated else best_state

    # ---- Stage C: newton_polish ----
    if not gated and best_state is not None:
        t0 = time.time()
        try:
            st, ni = newton_polish(cell, atoms, radii, best_state,
                                   scf_kwargs=dict(cfg, kmesh=k, efg=False),
                                   maxiter=6, inner_maxiter=14, f_tol=1e-6, rounds=newton_rounds)
            p(f"[C newton {time.time()-t0:.0f}s] res={ni['residual_norm']:.2e} "
              f"f_evals={ni['f_evals']} converged={ni['converged']}")
            if ni["residual_norm"] < 5e-3:
                final_state = st
                gated = ni["residual_norm"] < 1e-3
        except Exception as e:      # noqa: BLE001
            p(f"[C newton FAILED] {type(e).__name__}: {e}")

    # ---- EFG ----
    _, ie = crystal_scf_multi(cell, atoms, radii, iters=1, tol=0.0, efg=True, kmesh=k,
                              v_start={"__full_state__": final_state}, **cfg)
    status = "GATED" if gated else f"MARGINAL(best_rv={best_rv:.1e})"
    return ie, status, final_state, log
