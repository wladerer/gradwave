"""Acceptance harness for the l=1 (O 2p) local orbital on rutile TiO2 (aug4/fp4/k222).

The diagnosis (oxygen_efg_diagnosis.md) localized the residual O EFG deficit to the on-site
p×p (l=1) density: too biaxial (on-site eta ~0.92 vs Elk 0.10) and radially too diffuse. The
l=0 O 2s LO (#344) enriched the l=0 density and total |V_zz| but could not relax the l=1 2p
radial that sets the biaxiality. This harness adds an l=1 O LO (a second 2p radial degree of
freedom, phi = a u(E_2p) + b u_dot + c u(E2), confined phi(R)=phi'(R)=0, reusing #344's
overlap conditioning) and measures the decisive number: on-site O eta.

kerker=0.7 SCF to the aspherical gate (cold, or warm from STATE), then EFG from the exact
finalize pass. Reports Ti + O: full tensor axis-resolved ([110]/[-110]/[001]) + V_zz/eta, and
the on-site (valence l=2 sphere-Poisson) V_zz/eta. Elk 11 reference alongside.

Env:
  OL1E=-20     the l=1 O LO's E2: absolute atomic energy (eV) or an orbital label ("2p").
               "" => no l=1 LO (baseline). A label / an E2 equal to the valence 2p (-11.72)
               is degenerate with the l=1 linearization unless OL1EL moves the valence off it.
  OL1EL=       el_override for O l=1: move the valence l=1 linearization off 2p (float eV or
               label) so an E2="2p" LO carries a distinct radial (semicore-LO setup). Usually
               unneeded when OL1E is a deeper absolute energy (then E2 != E_l already).
  OLO=1        also include the l=0 O 2s LO from #344 (combined O 2s+2p LO config). 0 => l=1 only.
  CONDTOL=-1   lo_cond_tol; -1 => module default (0.18). 0 => conditioning OFF.
  LMAX=4 ECUT=300 KWORKERS=4 MAXIT=120 ITERS=40 RV=1.2e-2
  STATE=path   warm-start full state (pickle of info["state"]); default cold
  OUT=path     where to persist the converged state
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np
from _common import A_BOHR, ATOMS, RADII

from gradwave.flapw import crystal_scf_multi

# Elk 11.0.2 reference (eV/A^2).
ELK_O_FULL = (-19.10, +16.60, +2.50, -19.10, 0.740)      # [110],[-110],[001], V_zz, eta
ELK_O_ONSITE_VZZ, ELK_O_ONSITE_ETA = -10.27, 0.099
ELK_TI_VZZ, ELK_TI_ETA = 19.34, 0.36

AXES = {"[110]": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        "[-110]": np.array([1.0, -1.0, 0.0]) / np.sqrt(2),
        "[001]": np.array([0.0, 0.0, 1.0])}


def _spec(val):
    """Parse an env energy spec: float eV, or an orbital label ("2p")."""
    try:
        return float(val)
    except ValueError:
        return val


def cfg_from_env():
    ol1e = os.environ.get("OL1E", "")
    ol1el = os.environ.get("OL1EL", "")
    olo = int(os.environ.get("OLO", "1"))
    condtol = float(os.environ.get("CONDTOL", "-1"))
    c = dict(
        ecut=float(os.environ.get("ECUT", "300")),
        lmax=int(os.environ.get("LMAX", "4")),
        fullpot=True, fullpot_lmax=4, smearing=0.0, use_symmetry=True,
        subspace_reuse=False, kerker=float(os.environ.get("KERKER", "0.7")),
        shift_invert=True,
        kworkers=int(os.environ.get("KWORKERS", "4")),
    )
    los = []
    el = {}
    if olo:
        # #344's usable l=0 O 2s LO: anchor the LO at the deep 2s, linearize the l=0 valence at
        # 2p so the LO carries a distinct radial (else E2==E_l → degeneracy guard). resid_frac
        # 0.12 < 0.18 → conditioning triggers. This reproduces the 73%-Elk / on-site η 0.913 base.
        los.append((0, "2s"))
        el[0] = _spec(os.environ.get("OLEL0", "2p"))
    if ol1e:
        los.append((1, _spec(ol1e)))
    if los:
        c["los"] = {"O": los}
    if ol1el:
        el[1] = _spec(ol1el)
    if el:
        c["el_override"] = {"O": el}
    if condtol >= 0.0:
        c["lo_cond_tol"] = condtol
    return c, olo, ol1e, ol1el, condtol


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
    cfg, olo, ol1e, ol1el, condtol = cfg_from_env()
    ct = cfg.get("lo_cond_tol", "default")
    tag = f"OL1E={ol1e or 'none'} OL1EL={ol1el or 'none'} OLO={olo} condtol={ct} lmax={cfg['lmax']}"
    state_env = os.environ.get("STATE", "")
    out = os.environ.get("OUT", os.path.expanduser(
        f"~/tio2_states/ol1_e{ol1e or 'none'}_olo{olo}.pkl"))
    print(f"# ol1_accept {tag} ecut={cfg['ecut']} fp4 k222 kerker=0.7 SI=1 "
          f"kworkers={cfg['kworkers']} OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"start={'WARM:'+state_env if state_env else 'COLD'}", flush=True)
    maxit = int(os.environ.get("MAXIT", "120"))
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
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
