"""Acceptance harness for the conditioned O 2s LO overlap (rutile TiO2, aug4/fp4/k222).

kerker=0.7 SCF to the aspherical gate (cold, or warm from STATE), then the EFG from the exact
finalize pass. Reports Ti and O: full tensor axis-resolved ([110]/[-110]/[001]) + V_zz/eta, and
the on-site (valence l=2 sphere-Poisson) V_zz/eta. Elk 11 reference alongside. NaN/singular
failures are caught and reported (the unconditioned O-LO reproduces the pre-fix divergence).

Env:
  OLO=1        include the O 2s LO (0 = Ti-only baseline)
  CONDTOL=-1   lo_cond_tol; -1 => module default (0.18). 0 => conditioning OFF (divergence repro).
  LMAX=4 ECUT=300 KWORKERS=4 MAXIT=120
  STATE=path   warm-start full state (pickle of info["state"]); default cold
  OUT=path     where to persist the converged state (default derived from OLO/CONDTOL)
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np
from _common import A_BOHR, ATOMS, RADII

from gradwave.flapw import crystal_scf_multi

# Elk 11.0.2 reference (eV/Å²).
ELK_O_FULL = (-19.10, +16.60, +2.50, -19.10, 0.740)      # [110],[−110],[001], V_zz, eta
ELK_O_ONSITE_VZZ, ELK_O_ONSITE_ETA = -10.27, 0.099
ELK_TI_VZZ, ELK_TI_ETA = 19.34, 0.36

AXES = {"[110]": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        "[-110]": np.array([1.0, -1.0, 0.0]) / np.sqrt(2),
        "[001]": np.array([0.0, 0.0, 1.0])}


def cfg_from_env():
    # Plain aug4 no-LO is the diagnosis's Ti +18 / O eta 0.92 baseline (default Ti: 3p semicore
    # linearized in the valence, 3s frozen). The O 2s LO is added on its own — it needs no Ti-LO
    # bundle, so Ti's treatment is bit-identical between OLO=0 and OLO=1 (gate 4 is then exact up
    # to the self-consistency coupling through the O density).
    olo = int(os.environ.get("OLO", "1"))
    condtol = float(os.environ.get("CONDTOL", "-1"))
    c = dict(
        ecut=float(os.environ.get("ECUT", "300")),
        lmax=int(os.environ.get("LMAX", "4")),
        fullpot=True, fullpot_lmax=4, smearing=0.0, use_symmetry=True,
        subspace_reuse=False, kerker=0.7, shift_invert=True,
        kworkers=int(os.environ.get("KWORKERS", "4")),
    )
    if olo:
        c["los"] = {"O": [(0, "2s")]}
        olel = os.environ.get("OLEL", "")          # linearize O l=0 away from the 2s (semicore-LO)
        if olel:
            try:
                c["el_override"] = {"O": {0: float(olel)}}
            except ValueError:
                c["el_override"] = {"O": {0: olel}}   # orbital label ("2p")
    if condtol >= 0.0:
        c["lo_cond_tol"] = condtol
    return c, olo, condtol


def report_site(site, name, ref_full, onsite_ref):
    t = site["tensor"]
    w = np.linalg.eigvalsh(t)
    w = w[np.argsort(-np.abs(w))]
    proj = {ax: float(n @ t @ n) for ax, n in AXES.items()}
    vzz_axis = max(AXES, key=lambda ax: abs(proj[ax]))
    print(f"{name} FULL : eig=[{w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f}] eta={site['eta']:.3f} "
          f"[110]/[-110]/[001]={proj['[110]']:+.2f}/{proj['[-110]']:+.2f}/{proj['[001]']:+.2f} "
          f"V_zz={proj[vzz_axis]:+.2f} on {vzz_axis}", flush=True)
    print(f"{name} ONSITE: V_zz_val={site['V_zz_valence']:+.2f} eta_val={site['eta_valence']:.3f}",
          flush=True)
    if ref_full is not None:
        print(f"    Elk {name} full: [110]/[-110]/[001]={ref_full[0]:+.2f}/{ref_full[1]:+.2f}/"
              f"{ref_full[2]:+.2f} V_zz={ref_full[3]:+.2f} eta={ref_full[4]:.3f}", flush=True)
    if onsite_ref is not None:
        print(f"    Elk {name} onsite: V_zz={onsite_ref[0]:+.2f} eta={onsite_ref[1]:.3f}",
              flush=True)


def main():
    cfg, olo, condtol = cfg_from_env()
    ct = cfg.get("lo_cond_tol", "default")
    olel = cfg.get("el_override", {}).get("O", {}).get(0, "2s(default)")
    tag = f"OLO={olo} condtol={ct} lmax={cfg['lmax']} O-l0-El={olel}"
    state_env = os.environ.get("STATE", "")
    out = os.environ.get("OUT", os.path.expanduser(
        f"~/tio2_states/olo_OLO{olo}_ct{ct}.pkl"))
    print(f"# olo_accept {tag} ecut={cfg['ecut']} fp4 k222 kerker=0.7 SI=1 "
          f"kworkers={cfg['kworkers']} OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"start={'WARM:'+state_env if state_env else 'COLD'}", flush=True)
    maxit = int(os.environ.get("MAXIT", "120"))
    rv_gate = float(os.environ.get("RV", "1.2e-2"))   # tight r_v for trustworthy EFG (diag ~8e-3)
    v_start = None
    if state_env:
        with open(os.path.expanduser(state_env), "rb") as f:
            v_start = {"__full_state__": pickle.load(f)}
    iters0 = int(os.environ.get("ITERS", "40"))
    t0 = time.time()
    try:
        _, iw = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=iters0, tol=1e-3, efg=False,
                                  kmesh=(2, 2, 2), v_start=v_start, **cfg)
        r = iw["recorder"].summarize()
        n = r["n_iter"]
        while (r["r_nsph"] >= 1e-3 or r["r_v"] >= rv_gate) and n < maxit:
            _, iw = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=20, tol=1e-3, efg=False,
                                      kmesh=(2, 2, 2), v_start={"__full_state__": iw["state"]},
                                      **cfg)
            r = iw["recorder"].summarize()
            n += r["n_iter"]
            print(f"  cont: n_it={n} r_v={r['r_v']:.2e} r_nsph={r['r_nsph']:.2e}", flush=True)
        gated = r["r_nsph"] < 1e-3 and r["r_v"] < rv_gate
        # persist immediately — never lose a converged state to a downstream reporting error
        with open(out, "wb") as f:
            pickle.dump(iw["state"], f)
        print(f"SCF ({time.time()-t0:.0f}s): n_it={n} r_v={r['r_v']:.2e} "
              f"r_nsph={r['r_nsph']:.2e} gated={gated}  state saved: {out}", flush=True)
        _, ie = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                                  kmesh=(2, 2, 2), v_start={"__full_state__": iw["state"]}, **cfg)
        print(f"=== RESULT {tag} {'GATED' if gated else 'MARGINAL'} ===", flush=True)
        report_site(ie["efg"]["a0"], "Ti", None, (ELK_TI_VZZ, ELK_TI_ETA))
        report_site(ie["efg"]["a2"], "O", ELK_O_FULL, (ELK_O_ONSITE_VZZ, ELK_O_ONSITE_ETA))
    except Exception as e:
        print(f"=== FAILED {tag} after {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===",
              flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
