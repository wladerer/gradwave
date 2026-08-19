"""Warm-chained TiO2 rutile EFG campaign on the corrected stack (degeneracy-aware occupations,
IBZ star-unfolding, k-pool). Every comparison is basin-anchored: the k-chain warm-starts each mesh
from the previous one's converged potential, and the aug-lmax / local-orbital probes warm-start
from the chain's fixed point — so differences measure the knob, not SCF basin selection.

Env: GW_KWORKERS = k-point pool size (divide OMP_NUM_THREADS to match).
"""
import os
import time

import numpy as np

from gradwave.flapw import crystal_scf_multi

U = 0.3048
A_BOHR = [8.68083, 8.68083, 5.59096]
ATOMS = [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
         ((U, U, 0.0), "O"), ((1 - U, 1 - U, 0.0), "O"),
         ((0.5 + U, 0.5 - U, 0.5), "O"), ((0.5 - U, 0.5 + U, 0.5), "O")]
RADII = {"Ti": 1.098, "O": 0.824}          # Elk's muffin tins
BASE = dict(ecut=300.0, smearing=0.0, efg=True, fullpot=True, fullpot_lmax=4,
            use_symmetry=True)
LO_TI = dict(los={"Ti": [(0, "3s"), (1, "3p")]},
             core={"Ti": [(0, 1, 2), (0, 2, 2), (1, 1, 6)]},
             val_e={"Ti": 12}, el_override={"Ti": {1: "3d"}})


def report(tag, bands, info):
    out = [f"{tag}: span={bands['span']:.4f} symdev={info.get('symmetry_dev', -1):.2e}"]
    for key, name, elk in (("a0", "Ti", "[+19.34,-13.16,-6.18] 0.36"),
                           ("a2", "O ", "[-19.10,+16.60,+2.50] 0.74")):
        s = info["efg"][key]
        w = np.linalg.eigvalsh(s["tensor"])
        w = w[np.argsort(-np.abs(w))]
        out.append(f"  {name}: [{w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f}] eta={s['eta']:.3f}"
                   f" | Elk {elk}")
    print("\n".join(out), flush=True)


def run(tag, kmesh, iters, v_start=None, lmax=3, extra=None):
    t0 = time.time()
    b, i = crystal_scf_multi(A_BOHR, ATOMS, RADII, lmax=lmax, iters=iters, kmesh=kmesh,
                             v_start=v_start, kworkers=int(os.environ.get("GW_KWORKERS", "1")),
                             **BASE, **(extra or {}))
    report(f"{tag} ({time.time() - t0:.0f}s)", b, i)
    return b, i


def main():
    # warm-chained k-convergence at aug-lmax=3
    _, i222 = run("k222 cold  aug3", (2, 2, 2), 30)
    _, i333 = run("k333 warm  aug3", (3, 3, 3), 15, v_start=i222["v_by_key"])
    _, i444 = run("k444 warm  aug3", (4, 4, 4), 12, v_start=i333["v_by_key"])
    # basin-anchored aug-lmax stability at the k333 fixed point
    run("k333 warm  aug4", (3, 3, 3), 10, v_start=i333["v_by_key"], lmax=4)
    # basin-anchored LO probe (Ti semicore only, no O LO) at the same fixed point
    run("k333 warm  aug3 +TiLO", (3, 3, 3), 12, v_start=i333["v_by_key"], extra=LO_TI)
    print("done", flush=True)


if __name__ == "__main__":
    main()
