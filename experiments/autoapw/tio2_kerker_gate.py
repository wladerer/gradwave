"""Production TiO2 EFG point with the validated convergence recipe (kerker=0.7).

The acceptance matrix's base config (k333 aug3 fp4) never gated in 40 iterations with the
plain joint Anderson; the hard-config A/B showed kerker=0.7 reaching r_nsph ~1e-3 in 40.
This runs the full production point cold with the screen on, gates via tol, and reports
the EFG against Elk. Env: TK_K (default 3), TK_ITERS (default 60), TK_KERKER (default 0.7),
TK_KWORKERS (default 5).
"""
import os
import sys
import time

import numpy as np
from newton_probe import A_BOHR, ATOMS, RADII

from gradwave.flapw import crystal_scf_multi

ELK = {"a0": ("Ti", "[+19.34,-13.16,-6.18] eta 0.36"),
       "a2": ("O ", "[-19.10,+16.60,+2.50] eta 0.74")}


def main():
    k = int(os.environ.get("TK_K", "3"))
    cfg = dict(ecut=300.0, lmax=3, kmesh=(k,) * 3, smearing=0.0, efg=True, fullpot=True,
               fullpot_lmax=4, use_symmetry=True, subspace_reuse=False,
               kworkers=int(os.environ.get("TK_KWORKERS", "5")),
               kerker=float(os.environ.get("TK_KERKER", "0.7")))
    t0 = time.time()
    bands, info = crystal_scf_multi(A_BOHR, ATOMS, RADII, tol=1e-3,
                                    iters=int(os.environ.get("TK_ITERS", "60")), **cfg)
    rec = info["recorder"].summarize()
    print(f"({time.time()-t0:.0f}s) n_it={rec['n_iter']} r_v={rec['r_v']:.2e} "
          f"r_nsph={rec['r_nsph']:.2e} symdev={rec['symmetry_dev']:.1e} "
          f"span={bands['span']:.4f}", flush=True)
    for key, (name, elk) in ELK.items():
        s = info["efg"][key]
        w = np.linalg.eigvalsh(s["tensor"])
        w = w[np.argsort(-np.abs(w))]
        print(f"  {name}: [{w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f}] eta={s['eta']:.3f} | Elk {elk}",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
