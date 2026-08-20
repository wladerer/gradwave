"""Full trustworthy-EFG pipeline: k222+kerker converge -> EFG -> newton_polish k333 -> EFG."""
import sys
import time

from _common import A_BOHR, ATOMS, RADII, efg_eigs

from gradwave.flapw import crystal_scf_multi, newton_polish

# no kmesh here: each stage of the pipeline supplies its own (k222 warm, k333 polish)
CFG = dict(ecut=300.0, lmax=3, smearing=0.0, fullpot=True, fullpot_lmax=4,
           use_symmetry=True, subspace_reuse=False, kworkers=5, kerker=0.7)


def efg_at(state, kmesh, tag):
    _, i = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=1, tol=0.0, efg=True,
                             v_start={"__full_state__": state}, kmesh=kmesh, **CFG)
    for key, name in (("a0", "Ti"), ("a2", "O ")):
        w = efg_eigs(i["efg"][key]["tensor"])
        print(f"{tag} {name}: [{w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f}] "
              f"eta={i['efg'][key]['eta']:.3f}", flush=True)


def main():
    t0 = time.time()
    _, iw = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=40, tol=1e-3, efg=False,
                              kmesh=(2, 2, 2), **CFG)
    r = iw["recorder"].summarize()
    print(f"k222 ({time.time()-t0:.0f}s): n_it={r['n_iter']} r_nsph={r['r_nsph']:.2e}",
          flush=True)
    efg_at(iw["state"], (2, 2, 2), "k222-gated")
    t0 = time.time()
    st3, ni = newton_polish(A_BOHR, ATOMS, RADII, iw["state"],
                            scf_kwargs=dict(CFG, kmesh=(3, 3, 3), efg=False),
                            maxiter=5, inner_maxiter=12, f_tol=1e-6)
    print(f"newton k333 ({time.time()-t0:.0f}s): {ni}", flush=True)
    if ni["residual_norm"] < 1e-3:
        efg_at(st3, (3, 3, 3), "k333-newton")
    else:
        print("k333 polish did not converge; EFG skipped", flush=True)


if __name__ == "__main__":
    sys.exit(main())
