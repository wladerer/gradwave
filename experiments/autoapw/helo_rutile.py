"""Generalization test: unconfined l=1 O HELO on rutile TiO2 (the system the CONFINED l=1 LO
FAILED on: on-site O eta stuck 0.911->0.889 vs Elk 0.099, and it diverged at kerker=0.7).

Warm-starts from the #344 O-2s-LO TiO2 state (~/tio2_states/aug4olo_k222.pkl), adds the unconfined
O l=1 HELO, re-gates at kerker=0.7, reports Ti + O full/on-site vs Elk. The decisive comparison:
does the unconfined HELO (a) STAY GATED at kerker=0.7 where the confined LO diverged, and (b) move
the on-site O toward Elk?

Elk 11 (eV/A^2): O full V_zz -19.10 eta 0.740; O on-site V_zz -10.27 eta 0.099; Ti full V_zz 19.34.

Env: HELO_E=90 HELO_CONF=0 HELO_SYMS=O  KERKER=0.7 KWORKERS=4 MAXIT=80 RV=1.2e-2
     STATE=~/tio2_states/aug4olo_k222.pkl
"""
import os
import pickle
import sys
import time
import traceback

from gradwave.flapw import crystal_scf_multi
from gradwave.flapw.nmr import quadrupolar_coupling

U = 0.3048
A_BOHR = [8.68083, 8.68083, 5.59096]
ATOMS = [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
         ((U, U, 0.0), "O"), ((1 - U, 1 - U, 0.0), "O"),
         ((0.5 + U, 0.5 - U, 0.5), "O"), ((0.5 - U, 0.5 + U, 0.5), "O")]
RADII = {"Ti": 1.098, "O": 0.824}
ELK = {"Ti": dict(full=19.34, on=None), "O": dict(full=-19.10, on=-10.27, on_eta=0.099)}


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
    if he:
        conf = bool(int(os.environ.get("HELO_CONF", "0")))
        for sym in os.environ.get("HELO_SYMS", "O").split(","):
            los.setdefault(sym, []).append((1, {"e": float(he), "confine": conf}))
    c["los"] = los
    c["el_override"] = el
    return c, he


def report(site, name, isotope=None):
    on = site["V_zz_valence"]
    ref = ELK[name]
    rtxt = (f"gw/Elk={on/ref['on']:.3f}" if ref["on"] else "")
    cq = ""
    if isotope:
        c = quadrupolar_coupling(site["V_zz"], site["eta"], isotope)["abs_C_Q_MHz"]
        cq = f" C_Q({isotope})={c:.3f}MHz"
    print(f"{name} FULL: V_zz={site['V_zz']:+.3f} eta={site['eta']:.3f}{cq} "
          f"(Elk full {ref['full']:+.2f})", flush=True)
    print(f"     ONSITE: V_zz={on:+.3f} eta={site['eta_valence']:.3f}  "
          f"(Elk on {ref['on']} eta {ref.get('on_eta')})  {rtxt}", flush=True)


def main():
    cfg, he = cfg_from_env()
    state_env = os.path.expanduser(os.environ.get("STATE", "~/tio2_states/aug4olo_k222.pkl"))
    out = os.path.expanduser(os.environ.get("OUT", "~/tio2_states/rutile_helo.pkl"))
    maxit = int(os.environ.get("MAXIT", "80"))
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
    print(f"# RUTILE HELO: HELO_E={he or 'NONE'} conf={os.environ.get('HELO_CONF','0')} "
          f"syms={os.environ.get('HELO_SYMS','O')} kerker={cfg['kerker']} los={cfg['los']} "
          f"warm={state_env}", flush=True)
    with open(state_env, "rb") as f:
        state = pickle.load(f)
    t0 = time.time()
    try:
        _, iw = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=20, tol=1e-3, efg=False,
                                  kmesh=(2, 2, 2), v_start={"__full_state__": state}, **cfg)
        r = iw["recorder"].summarize()
        n = r["n_iter"]
        while (r["r_nsph"] >= 1e-3 or r["r_v"] >= rv_gate) and n < maxit:
            _, iw = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=12, tol=1e-3, efg=False,
                                      kmesh=(2, 2, 2),
                                      v_start={"__full_state__": iw["state"]}, **cfg)
            r = iw["recorder"].summarize()
            n += r["n_iter"]
            print(f"  cont: n_it={n} r_v={r['r_v']:.2e} r_nsph={r['r_nsph']:.2e}", flush=True)
        gated = r["r_nsph"] < 1e-3 and r["r_v"] < rv_gate
        with open(out, "wb") as f:
            pickle.dump(iw["state"], f)
        print(f"SCF ({time.time()-t0:.0f}s): n_it={n} r_v={r['r_v']:.2e} r_nsph={r['r_nsph']:.2e} "
              f"gated={gated}", flush=True)
        _, ie = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                                  kmesh=(2, 2, 2), v_start={"__full_state__": iw["state"]}, **cfg)
        print(f"=== RESULT RUTILE HELO_E={he or 'NONE'} {'GATED' if gated else 'MARGINAL'} ===",
              flush=True)
        report(ie["efg"]["a0"], "Ti")
        report(ie["efg"]["a2"], "O", "17O")
    except Exception as e:  # noqa: BLE001
        print(f"=== FAILED after {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
