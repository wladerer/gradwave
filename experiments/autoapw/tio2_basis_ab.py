"""Task-7 basis A/B: close the EFG magnitude gap vs Elk now that convergence is eliminated.

The trustworthy-EFG pipeline (asus job 128) proved the k222+kerker -> newton_polish(k333)
recipe converges and the EFG is k-stable — but O's |V_zz| is ~44% of Elk's (eta matches,
0.77 vs 0.74) and Ti's ~5%. The remaining suspects are all BASIS: in-sphere augmentation
lmax, O local orbitals (radial flexibility), and their combination. Each variant here runs
the same validated recipe self-contained (cold -> k222 gate -> newton k333 -> EFG at both
stages) and — unlike the pipeline — pickles its gated states to ~/tio2_states/<tag>_*.pkl
so later probes can warm-start instead of repeating the Anderson leg.

One variant per queued job: BA_TAG selects from VARIANTS; BA_KWORKERS/BA_ECUT override.
fullpot_lmax stays 4 (the Weinert matched-L cap above 4 is unimplemented — known gap).
"""
import os
import pickle
import sys
import time

from _common import A_BOHR, ATOMS, RADII, efg_eigs, env_float, env_int

from gradwave.flapw import crystal_scf_multi, newton_polish

# aug-lmax raises the in-sphere angular response; the O LOs add radial freedom in the
# valence region (l=0,1 at +5 eV above the muffin-tin zero — the run_tio2.py precedent).
O_LO = {"O": [(0, 5.0), (1, 5.0)]}
VARIANTS = {
    "base": dict(lmax=3, los=None),        # control == job 128, now with saved states
    "aug4": dict(lmax=4, los=None),
    "aug5": dict(lmax=5, los=None),
    "olo": dict(lmax=3, los=O_LO),
    "aug4olo": dict(lmax=4, los=O_LO),
}

STATE_DIR = os.path.expanduser("~/tio2_states")


def efg_at(state, kmesh, tag, cfg):
    _, i = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                             v_start={"__full_state__": state}, kmesh=kmesh, **cfg)
    for key, name in (("a0", "Ti"), ("a2", "O ")):
        w = efg_eigs(i["efg"][key]["tensor"])
        print(f"{tag} {name}: [{w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f}] "
              f"eta={i['efg'][key]['eta']:.3f}", flush=True)


def save_state(state, tag, stage):
    os.makedirs(STATE_DIR, exist_ok=True)
    path = os.path.join(STATE_DIR, f"{tag}_{stage}.pkl")
    with open(path, "wb") as f:
        pickle.dump(state, f)
    print(f"state saved: {path}", flush=True)


def main():
    tag = os.environ["BA_TAG"]
    var = VARIANTS[tag]
    cfg = dict(ecut=env_float("BA", "ECUT", 300.0), smearing=0.0, fullpot=True,
               fullpot_lmax=4, use_symmetry=True, subspace_reuse=False,
               kworkers=env_int("BA", "KWORKERS", 5), kerker=0.7, **var)
    print(f"variant {tag}: {var}", flush=True)

    t0 = time.time()
    _, iw = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=40, tol=1e-3, efg=False,
                              kmesh=(2, 2, 2), **cfg)
    r = iw["recorder"].summarize()
    print(f"k222 ({time.time()-t0:.0f}s): n_it={r['n_iter']} r_nsph={r['r_nsph']:.2e}",
          flush=True)
    if r["r_nsph"] > 1e-2:
        print(f"{tag}: k222 did NOT gate — EFG untrustworthy, stopping", flush=True)
        return 1
    save_state(iw["state"], tag, "k222")
    efg_at(iw["state"], (2, 2, 2), f"{tag}-k222", cfg)

    t0 = time.time()
    st3, ni = newton_polish(A_BOHR, ATOMS, RADII, iw["state"],
                            scf_kwargs=dict(cfg, kmesh=(3, 3, 3)),
                            maxiter=5, inner_maxiter=12, f_tol=1e-6)
    print(f"newton k333 ({time.time()-t0:.0f}s): {ni}", flush=True)
    if ni["residual_norm"] >= 1e-3:
        print(f"{tag}: k333 polish did not converge; EFG skipped", flush=True)
        return 1
    save_state(st3, tag, "k333")
    efg_at(st3, (3, 3, 3), f"{tag}-k333", cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
