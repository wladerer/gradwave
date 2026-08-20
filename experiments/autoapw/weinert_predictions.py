"""Falsifiable predictions of the exact own-field + high-L Weinert fix (one runner per mode).

WP_MODE:
  fp6       — cold TiO2 fullpot_lmax=6 (the old runaway config): O EFG must land between
              the fp4 value (~9 eV/A^2 leading) and the old 6x runaway (~115), eta ~0.74-0.77.
              Jackpot = near Elk O [-19.1,+16.6,+2.5] on [110]/[-110]/[001].
  hard_cold — cold hard config (ecut300 lmax3 fp4 k222) with PLAIN Anderson (kerker=None):
              previously the |rho|=1.02 interstitial mode diverged it; should now converge
              or at least not diverge.
  ab        — round-3 A/B: EFG at aug4/fp4 and aug4/fp6 (cold chains, gated) vs Elk.
Env: WP_KWORKERS (default 5), WP_ECUT (300), WP_ITERS chunk cap.
"""
import math
import os
import pickle
import sys
import time

import numpy as np
from _common import A_BOHR, ATOMS, RADII, efg_eigs, env_float, env_int

from gradwave.flapw import crystal_scf_multi

AX = {"[110]": np.array([1.0, 1.0, 0.0]) / math.sqrt(2),
      "[-110]": np.array([-1.0, 1.0, 0.0]) / math.sqrt(2),
      "[001]": np.array([0.0, 0.0, 1.0])}
ELK = {"O ": {"[110]": -19.1, "[-110]": +16.6, "[001]": +2.5},
       "Ti": None}  # Ti axis-resolved not recorded; compare eigenvalues/eta


def report(info, tag):
    np.set_printoptions(precision=3, suppress=True)
    for key, name in (("a0", "Ti"), ("a2", "O ")):
        t = info["efg"][key]["tensor"]
        w = efg_eigs(t)
        ax = {lab: float(e @ t @ e) for lab, e in AX.items()}
        print(f"{tag} {name}: evals [{w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f}] "
              f"eta={info['efg'][key]['eta']:.3f}  axis-resolved "
              + " ".join(f"{lab}={v:+.2f}" for lab, v in ax.items()), flush=True)
        if ELK.get(name):
            print(f"{tag} {name}: Elk axis-resolved "
                  + " ".join(f"{lab}={v:+.1f}" for lab, v in ELK[name].items()), flush=True)


def gated_run(cfg, kmesh=(2, 2, 2), cap=120):
    """Cold chunked Anderson chain until r_nsph<1e-3 and r_v<0.15 (the validated recipe)."""
    t0 = time.time()
    _, iw = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=40, tol=1e-3, efg=False,
                              kmesh=kmesh, **cfg)
    r = iw["recorder"].summarize()
    n_tot = r["n_iter"]
    while (r["r_nsph"] >= 1e-3 or r["r_v"] >= 0.15) and n_tot < cap:
        _, iw = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=20, tol=1e-3, efg=False,
                                  kmesh=kmesh, v_start={"__full_state__": iw["state"]}, **cfg)
        r = iw["recorder"].summarize()
        n_tot += r["n_iter"]
        print(f"  cont: n_it={n_tot} r_v={r['r_v']:.2e} r_nsph={r['r_nsph']:.2e}", flush=True)
    print(f"chain ({time.time()-t0:.0f}s): n_it={n_tot} r_v={r['r_v']:.2e} "
          f"r_nsph={r['r_nsph']:.2e} gated={r['r_nsph'] < 1e-3 and r['r_v'] < 0.15}",
          flush=True)
    return iw, r


def efg_of(state, cfg, kmesh):
    _, i = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                             v_start={"__full_state__": state}, kmesh=kmesh, **cfg)
    return i


def base_cfg(**over):
    cfg = dict(ecut=env_float("WP", "ECUT", 300.0), lmax=3, smearing=0.0, fullpot=True,
               fullpot_lmax=4, use_symmetry=True, subspace_reuse=False,
               kworkers=env_int("WP", "KWORKERS", 5), kerker=0.7, verbose=False)
    cfg.update(over)
    return cfg


def save_state(state, tag):
    d = os.path.expanduser("~/tio2_states")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"weinert_{tag}.pkl"), "wb") as f:
        pickle.dump(state, f)


def main():
    mode = os.environ.get("WP_MODE", "fp6")
    print(f"mode={mode}", flush=True)

    if mode == "fp6":
        cfg = base_cfg(fullpot_lmax=6)
        iw, r = gated_run(cfg)
        save_state(iw["state"], "fp6_k222")
        i = efg_of(iw["state"], cfg, (2, 2, 2))
        report(i, "fp6-k222")
        return 0

    if mode == "hard_cold":
        cfg = base_cfg(kerker=None, verbose=True)
        _, iw = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=60, tol=1e-3, efg=False,
                                  kmesh=(2, 2, 2), **cfg)
        r = iw["recorder"].summarize()
        print(f"hard_cold plain-Anderson: n_it={r['n_iter']} r_v={r['r_v']:.2e} "
              f"r_nsph={r['r_nsph']:.2e}", flush=True)
        return 0

    if mode == "ab":
        for tag, over in (("aug4-fp4", dict(lmax=4, fullpot_lmax=4)),
                          ("aug4-fp6", dict(lmax=4, fullpot_lmax=6))):
            print(f"--- {tag} ---", flush=True)
            cfg = base_cfg(**over)
            iw, r = gated_run(cfg)
            save_state(iw["state"], tag.replace("-", "_") + "_k222")
            i = efg_of(iw["state"], cfg, (2, 2, 2))
            report(i, f"{tag}-k222")
        return 0

    print(f"unknown mode {mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
