"""Ti search matrix from the converged baseline: aug-lmax, semicore LOs, k, fp_lmax."""
import numpy as np

from gradwave.flapw import crystal_scf_multi

U = 0.3048
ATOMS = [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
         ((U, U, 0.0), "O"), ((1 - U, 1 - U, 0.0), "O"),
         ((0.5 + U, 0.5 - U, 0.5), "O"), ((0.5 - U, 0.5 + U, 0.5), "O")]
BASE = dict(ecut=300.0, smearing=0.0, efg=True, fullpot=True, use_symmetry=True, kworkers=5)
LO = dict(los={"Ti": [(0, "3s"), (1, "3p")]},
          core={"Ti": [(0, 1, 2), (0, 2, 2), (1, 1, 6)]},
          val_e={"Ti": 12}, el_override={"Ti": {1: "3d"}})


def run(tag, **kw):
    try:
        b, i = crystal_scf_multi([8.68083, 8.68083, 5.59096], ATOMS,
                                 {"Ti": 1.098, "O": 0.824}, **BASE, **kw)
        rec = i["recorder"].summarize()
        out = [f"{tag}: n_it={rec['n_iter']} r_nsph={rec['r_nsph']:.1e} "
               f"symdev={rec['symmetry_dev']:.1e}"]
        for key, name in (("a0", "Ti"), ("a2", "O ")):
            s = i["efg"][key]
            w = np.linalg.eigvalsh(s["tensor"]); w = w[np.argsort(-np.abs(w))]
            out.append(f"  {name} [{w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f}] eta={s['eta']:.2f}")
        print("\n".join(out), flush=True)
        return i
    except Exception as e:
        print(f"{tag}: FAILED {e!r}", flush=True)
        return None


def main():
    i0 = run("base k333 aug3 fp4", lmax=3, fullpot_lmax=4, iters=40, kmesh=(3, 3, 3))
    v = i0["v_by_key"] if i0 else None
    run("aug4 fp4 k333 warm", lmax=4, fullpot_lmax=4, iters=30, kmesh=(3, 3, 3), v_start=v)
    run("aug3 fp6 k333 warm", lmax=3, fullpot_lmax=6, iters=30, kmesh=(3, 3, 3), v_start=v)
    run("k444 aug3 fp4 warm", lmax=3, fullpot_lmax=4, iters=30, kmesh=(4, 4, 4), v_start=v)
    run("TiLO aug3 fp4 k333", lmax=3, fullpot_lmax=4, iters=40, kmesh=(3, 3, 3), **LO)
    print("done", flush=True)


if __name__ == "__main__":
    main()
