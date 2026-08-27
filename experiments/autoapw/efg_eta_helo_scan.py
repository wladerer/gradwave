"""Front B — differentiable-basis finite-difference scan of the l=1 anion HELO energy vs the
anion EFG (V_zz/C_Q AND eta), on rutile-TiO2 O and (optionally) corundum-Al2O3 O.

Motivation (efg_accuracy_plan.md step 6): the FLAPW EFG forward path is numpy/float end-to-end,
so V_zz/eta are NOT autograd-differentiable w.r.t. basis parameters. The one scalar the anion
recipe exposes is the unconfined l=1 HELO energy E2 (``efg_anion_basis(helo_e=...)`` /
``los={"O":[(1,{"e":E2,"confine":False})]}``). This scans E2 by finite differences: for each E2
we re-converge the fullpot SCF (warm-started from the no-HELO O-2s-LO state) and run one exact
efg pass, printing BOTH eta and V_zz so the known FLAPW magnitude<->eta trade-off (aug-lmax-6
lifts |V_zz| but drops eta) is visible on the E2 axis.

Rutile Elk 11: O full V_zz -19.10 eta 0.740 (C_Q(17O) via quadrupolar_coupling).
Baseline E2=90 eV: V_zz -15.07 (79%) eta 0.654.

Env:
  SYS=rutile|corundum  (default rutile)
  HELO_ES="60,75,90,110,130,160"   comma list of E2 in eV to scan
  KWORKERS=4 (set OMP_NUM_THREADS=1; kworkers gives the k parallelism)
  MAXIT=80 RV=1.2e-2 KERKER=0.7 ECUT=300 LMAX=4 FPLMAX=4
  STATE=~/tio2_states/aug4olo_k222.pkl   (rutile no-HELO warm start)

Run on asus (respecting the 5-core budget: kworkers*OMP ~ 4):
  OMP_NUM_THREADS=1 KWORKERS=4 uv run python experiments/autoapw/efg_eta_helo_scan.py | tee scan.log
"""
from __future__ import annotations

import os
import pickle
import sys
import time
import traceback

import numpy as np

from gradwave.flapw import crystal_scf_multi, newton_polish
from gradwave.flapw.nmr import quadrupolar_coupling

# ---- rutile TiO2 (same cell/atoms/radii as helo_rutile.py) ----
RUTILE_U = 0.3048
RUTILE_A = [8.68083, 8.68083, 5.59096]
RUTILE_ATOMS = [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
                ((RUTILE_U, RUTILE_U, 0.0), "O"), ((1 - RUTILE_U, 1 - RUTILE_U, 0.0), "O"),
                ((0.5 + RUTILE_U, 0.5 - RUTILE_U, 0.5), "O"),
                ((0.5 - RUTILE_U, 0.5 + RUTILE_U, 0.5), "O")]
RUTILE_RADII = {"Ti": 1.098, "O": 0.824}
RUTILE_ELK = dict(full=-19.10, eta=0.740)


def cfg_from_env(helo_e: float):
    kerker = os.environ.get("KERKER", "0.7")
    # SHIFT_INVERT default 0 -> force the exact dense eigensolve: the shift-invert certificate
    # fails on this hard fullpot config and its silent dense fallback added run-to-run EFG noise
    # in the coarse scan (56 certificate failures). Dense is deterministic.
    si_env = os.environ.get("SHIFT_INVERT", "0").lower()
    shift_invert = si_env not in ("0", "none", "", "false")
    c = dict(ecut=float(os.environ.get("ECUT", "300")), lmax=int(os.environ.get("LMAX", "4")),
             fullpot=True, fullpot_lmax=int(os.environ.get("FPLMAX", "4")), smearing=0.0,
             use_symmetry=True, subspace_reuse=False,
             kerker=(None if kerker.lower() in ("none", "0", "") else float(kerker)),
             shift_invert=shift_invert, kworkers=int(os.environ.get("KWORKERS", "4")))
    los = {"O": [(0, "2s")]}
    if helo_e > 0:
        los["O"].append((1, {"e": float(helo_e), "confine": False}))
    c["los"] = los
    c["el_override"] = {"O": {0: "2p"}}
    return c


def run_point(helo_e: float, state, maxit: int, rv_gate: float, newton_rounds: int):
    """Robustly re-converge fullpot from the no-HELO warm state with this HELO energy (chunked
    continuation tracking the best/lowest-r_v state + a newton_polish fallback per the
    efg_status.md recipe), then one exact efg pass. Returns a dict with V_zz/eta/C_Q, the actual
    iteration count, the converged flag, and the final residual."""
    cfg = cfg_from_env(helo_e)
    t0 = time.time()
    best_state, best_rv = None, np.inf
    warm = {"__full_state__": state}
    n, chunk = 0, 20
    r = None
    while n < maxit:
        _, iw = crystal_scf_multi(RUTILE_A, RUTILE_ATOMS, RUTILE_RADII, iters=chunk, tol=1e-3,
                                  efg=False, kmesh=(2, 2, 2), v_start=warm, **cfg)
        r = iw["recorder"].summarize()
        n += r["n_iter"]
        warm = {"__full_state__": iw["state"]}
        if r["r_v"] < best_rv:
            best_rv, best_state = r["r_v"], iw["state"]
        chunk = 12
        if r["r_nsph"] < 1e-3 and r["r_v"] < rv_gate:
            break
        if r["r_v"] > 20 * best_rv and r["r_v"] > 1.0:      # diverging -> polish the best
            break
    gated = r is not None and r["r_nsph"] < 1e-3 and r["r_v"] < rv_gate
    final_state = iw["state"] if gated else best_state
    resid = best_rv
    if not gated and newton_rounds > 0 and best_state is not None:
        try:
            st, ni = newton_polish(RUTILE_A, RUTILE_ATOMS, RUTILE_RADII, best_state,
                                   scf_kwargs=dict(cfg, kmesh=(2, 2, 2), efg=False),
                                   maxiter=6, inner_maxiter=14, f_tol=1e-6, rounds=newton_rounds)
            resid = ni["residual_norm"]
            if resid < 5e-3:
                final_state = st
                gated = resid < 1e-3
        except Exception as ex:
            print(f"    [newton failed E2={helo_e}: {type(ex).__name__}: {ex}]", flush=True)
    _, ie = crystal_scf_multi(RUTILE_A, RUTILE_ATOMS, RUTILE_RADII, iters=1, tol=0.0, efg=True,
                              kmesh=(2, 2, 2), v_start={"__full_state__": final_state}, **cfg)
    o = ie["efg"]["a2"]
    cq = quadrupolar_coupling(o["V_zz"], o["eta"], "17O")["abs_C_Q_MHz"]
    return dict(vzz=o["V_zz"], eta=o["eta"], cq=cq, gated=gated, resid=resid,
                secs=time.time() - t0, nit=n)


def main():
    es = [float(x) for x in os.environ.get("HELO_ES", "110,120,130,140,150").split(",")]
    maxit = int(os.environ.get("MAXIT", "200"))
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
    newton_rounds = int(os.environ.get("NEWTON", "3"))
    state_env = os.path.expanduser(os.environ.get("STATE", "~/tio2_states/aug4olo_k222.pkl"))
    print(f"# FRONT B rutile-O HELO_E scan  es={es}  maxit={maxit} newton={newton_rounds} "
          f"warm={state_env} kw={os.environ.get('KWORKERS')} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    print(f"# Elk 11: O V_zz={RUTILE_ELK['full']:+.2f} eta={RUTILE_ELK['eta']:.3f}", flush=True)
    with open(state_env, "rb") as f:
        state = pickle.load(f)
    rows = []
    for e2 in es:
        try:
            d = run_point(e2, state, maxit, rv_gate, newton_rounds)
            frac = d["vzz"] / RUTILE_ELK["full"]
            tag = "CONVERGED" if d["gated"] else "MARGINAL"
            print(f"E2={e2:6.1f} eV | V_zz={d['vzz']:+8.3f} ({frac*100:5.1f}% Elk) "
                  f"eta={d['eta']:.3f} C_Q(17O)={d['cq']:.3f} MHz | {tag} "
                  f"resid={d['resid']:.1e} n_it={d['nit']} ({d['secs']:.0f}s)", flush=True)
            rows.append((e2, d))
        except Exception as ex:
            print(f"E2={e2:6.1f} eV | FAILED: {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()
    print("# --- summary (E2, V_zz, %Elk, eta, C_Q, resid, conv) ---", flush=True)
    for e2, d in rows:
        print(f"#  {e2:6.1f}  {d['vzz']:+8.3f}  {d['vzz']/RUTILE_ELK['full']*100:5.1f}%  "
              f"{d['eta']:.3f}  {d['cq']:.3f}  {d['resid']:.1e}  "
              f"{'C' if d['gated'] else 'M'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
