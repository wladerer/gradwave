"""The first D5 (basis-richness) EFG probe for rutile TiO2, with both affordability enablers on
(Gaunt multipole contraction + opt-in shift-invert secular solves).

Post-#335 the O-EFG magnitude gap vs Elk (~0.4× at every gated point) was pinned to D5 — density /
basis richness (Elk: lmaxapw 8, lmaxo 6, Ti 3s/3p + O 2s semicore local orbitals). This probe runs
the richest in-sphere basis affordable: aug-lmax 6 (``lmax=6``), Ti 3s/3p semicore LOs, an O l=0
(2s-anchored) LO, fullpot_lmax 4, k222. Warm-started from a saved exact-chain aug4-fp4 k222 state,
continued with plain warm Anderson (kerker off, the post-#335 recipe) until the aspherical channel
gates, then the EFG is read from the exact (dense) finalize density pass.

THE question: does basis richness move the O-EFG magnitude toward Elk, or does it saturate (a
decisive negative)? Reports O and Ti axis-resolved vs Elk and the aug4 baseline.

Env: D5_STATE (warm state path), D5_LMAX (aug-lmax, default 6), D5_KWORKERS (default 5),
D5_MAXIT (k222 gate cap, default 90), D5_SI (1=shift_invert on, default 1), D5_ECUT (default 300).
"""
import os
import pickle
import sys
import time

import numpy as np
from _common import A_BOHR, ATOMS, RADII, env_float, env_int

from gradwave.flapw import crystal_scf_multi
from gradwave.flapw.scf import _multi_setup

# Elk 11.0.2 reference (eV/Å²): O axis-resolved on [110]/[-110]/[001]; Ti V_zz/eta.
ELK_O = (-19.10, +16.60, +2.50, 0.740)
ELK_TI_VZZ, ELK_TI_ETA = 19.34, 0.36
# aug4-fp6 exact-chain baseline (TIO2_NMR round-3): O [110]/[-110]/[001], eta.
BASE_O = (-7.27, +8.17, -0.90, 0.781)

AXES = {"[110]": np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
        "[-110]": np.array([1.0, -1.0, 0.0]) / np.sqrt(2),
        "[001]": np.array([0.0, 0.0, 1.0])}


def d5_cfg():
    return dict(
        ecut=env_float("D5", "ECUT", 300.0),
        lmax=env_int("D5", "LMAX", 6),
        fullpot=True, fullpot_lmax=4, smearing=0.0, use_symmetry=True,
        subspace_reuse=False,
        # plain Anderson (kerker=None) diverged on the new lmax6+LO basis warm-started from the
        # aug4 (no-LO) state — a basis mismatch the bare mixer can't absorb. kerker=0.7 is the
        # campaign's proven recipe for a fresh basis (tio2_basis_ab); override with D5_KERKER.
        kerker=env_float("D5", "KERKER", 0.7),
        shift_invert=bool(env_int("D5", "SI", 1)),
        kworkers=env_int("D5", "KWORKERS", 5),
        los={"Ti": [(0, "3s"), (1, "3p")], "O": [(0, "2s")]},
        core={"Ti": [(0, 1, 2), (0, 2, 2), (1, 1, 6)]},    # freeze only 1s,2s,2p (3s→valence)
        val_e={"Ti": 12},                                  # 3s² 3p⁶ 3d² 4s²
        el_override={"Ti": {1: "3d"}},                     # l=1 param moved to the valence
    )


def report(info, tag):
    for key, name, elk in (("a0", "Ti", None), ("a2", "O", ELK_O)):
        site = info["efg"][key]
        t = site["tensor"]
        w = np.linalg.eigvalsh(t)
        w = w[np.argsort(-np.abs(w))]                       # V_zz, V_yy, V_xx
        proj = {ax: float(n @ t @ n) for ax, n in AXES.items()}
        print(f"{tag} {name}: eig=[{w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f}] eta={site['eta']:.3f} "
              f"axis[110]/[-110]/[001]={proj['[110]']:+.2f}/{proj['[-110]']:+.2f}/"
              f"{proj['[001]']:+.2f}", flush=True)
        if elk is not None:
            print(f"    Elk O : {elk[0]:+.2f}/{elk[1]:+.2f}/{elk[2]:+.2f} eta={elk[3]:.3f} | "
                  f"base O: {BASE_O[0]:+.2f}/{BASE_O[1]:+.2f}/{BASE_O[2]:+.2f} eta={BASE_O[3]:.3f}",
                  flush=True)
        else:
            print(f"    Elk Ti: V_zz {ELK_TI_VZZ:+.2f} eta {ELK_TI_ETA:.3f}  "
                  f"(gradwave V_zz {w[0]:+.2f} eta {site['eta']:.3f})", flush=True)


def main():
    cfg = d5_cfg()
    print(f"# d5_probe lmax={cfg['lmax']} fp={cfg['fullpot_lmax']} ecut={cfg['ecut']} k222 "
          f"SI={cfg['shift_invert']} kworkers={cfg['kworkers']} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)

    # confirm the LO linearization energies (the "2s" label must resolve near O's −0.87 Ha ⇒
    # −23.75 eV, Elk's O 2s LO anchor; Ti 3s/3p are the semicore anchors)
    ctx = _multi_setup(a_bohr=A_BOHR, atoms=ATOMS, radii=RADII, **cfg)
    o2s, ti3s, ti3p = ctx.at_by_sym["O"]["2s"], ctx.at_by_sym["Ti"]["3s"], ctx.at_by_sym["Ti"]["3p"]
    print(f"LO anchors (eV): O 2s={o2s:.2f} (Elk −23.75)  Ti 3s={ti3s:.2f}  Ti 3p={ti3p:.2f}  "
          f"[Ha: O 2s={o2s/27.2114:.4f} (Elk −0.8727)]", flush=True)

    # The D5 basis (Ti 3s in valence, O 2s LO, val_e 12) has a different electron partition than
    # the saved aug4 (no-LO) states, so warm-starting their potentials is only approximate and
    # plain Anderson diverged; default to a COLD kerker=0.7 run (tio2_basis_ab's fresh-basis
    # recipe). Set D5_STATE to warm-start a same-basis continuation.
    state_env = os.environ.get("D5_STATE", "")
    v_start = None
    if state_env:
        with open(os.path.expanduser(state_env), "rb") as f:
            v_start = {"__full_state__": pickle.load(f)}
        print(f"warm state: {state_env}", flush=True)
    else:
        print("cold start (kerker-stabilized)", flush=True)

    t0 = time.time()
    _, iw = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=40, tol=1e-3, efg=False,
                              kmesh=(2, 2, 2), v_start=v_start, **cfg)
    r = iw["recorder"].summarize()
    n_tot = r["n_iter"]
    cap = env_int("D5", "MAXIT", 90)
    while (r["r_nsph"] >= 1e-3 or r["r_v"] >= 0.15) and n_tot < cap:
        _, iw = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=20, tol=1e-3, efg=False,
                                  kmesh=(2, 2, 2), v_start={"__full_state__": iw["state"]}, **cfg)
        r = iw["recorder"].summarize()
        n_tot += r["n_iter"]
        print(f"  k222 continuation: n_it={n_tot} r_v={r['r_v']:.2e} r_nsph={r['r_nsph']:.2e}",
              flush=True)
    gated = r["r_nsph"] < 1e-3 and r["r_v"] < 0.15
    print(f"k222 ({time.time()-t0:.0f}s): n_it={n_tot} r_v={r['r_v']:.2e} r_nsph={r['r_nsph']:.2e} "
          f"gated={gated}", flush=True)

    # EFG from the exact (dense) finalize density pass at the converged potential
    _, ie = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                              kmesh=(2, 2, 2), v_start={"__full_state__": iw["state"]}, **cfg)
    tag = f"D5-aug{cfg['lmax']}{'-GATED' if gated else '-MARGINAL'}"
    report(ie, tag)
    if not gated:
        print("WARNING: ungated — EFG is indicative only (ungated states are unquotable)",
              flush=True)
    # persist the D5 state for a possible aug8 warm continuation
    out = os.path.expanduser(f"~/tio2_states/d5_aug{cfg['lmax']}_k222.pkl")
    with open(out, "wb") as f:
        pickle.dump(iw["state"], f)
    print(f"state saved: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
