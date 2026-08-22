"""gradwave FLAPW EFG for hexagonal BN (P6_3/mmc) — ¹¹B quadrupolar NMR vs Elk + experiment.

B is a light closed-valence main-group site (2s²2p¹, no frozen semicore), 3-coordinate trigonal-
planar → axial EFG (η=0) along c. Experimental ¹¹B C_Q(h-BN) = 2.934 MHz, η≈0 (Jeschke et al.,
Solid State NMR 1998). B and N injected at runtime via _mgroup (validated vs NIST-LSD). Recipe
mirrors corundum: aug4/fp4/k222 kerker=0.7, conditioned anion(N) 2s LO. NO src change.
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np
from _efgrun import converge_efg
from _mgroup import hex_cell_bohr, report_site

A, C = 2.504, 6.661        # Å (h-BN, P6_3/mmc)
CELL_BOHR = hex_cell_bohr(A, C)
ATOMS = [((1 / 3, 2 / 3, 1 / 4), "B"), ((2 / 3, 1 / 3, 3 / 4), "B"),      # 2c
         ((2 / 3, 1 / 3, 1 / 4), "N"), ((1 / 3, 2 / 3, 3 / 4), "N")]      # 2d
RADII = {"B": float(os.environ.get("RB", "0.70")), "N": float(os.environ.get("RN", "0.70"))}
AXES = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0])}


def cfg_from_env():
    kerker = os.environ.get("KERKER", "0.7")
    c = dict(ecut=float(os.environ.get("ECUT", "400")), lmax=int(os.environ.get("LMAX", "4")),
             fullpot=True, fullpot_lmax=int(os.environ.get("FPLMAX", "4")),
             smearing=0.0, use_symmetry=True, subspace_reuse=False,
             kerker=(None if kerker.lower() in ("none", "0", "") else float(kerker)),
             shift_invert=True, kworkers=int(os.environ.get("KWORKERS", "4")))
    if int(os.environ.get("NLO", "1")):
        c["los"] = {"N": [(0, "2s")]}
        c["el_override"] = {"N": {0: "2p"}}
    return c


def main():
    cfg = cfg_from_env()
    out = os.environ.get("OUT", os.path.expanduser("~/efg_mgroup/hbn.pkl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"# h-BN ecut={cfg['ecut']} lmax={cfg['lmax']} fp{cfg['fullpot_lmax']} k222 "
          f"kerker={cfg['kerker']} R_B={RADII['B']} R_N={RADII['N']} "
          f"kworkers={cfg['kworkers']} OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
    t0 = time.time()
    try:
        ie, status, state, _ = converge_efg(CELL_BOHR, ATOMS, RADII, cfg, (2, 2, 2),
                                             rv_gate=rv_gate)
        with open(out, "wb") as f:
            pickle.dump(state, f)
        print(f"=== RESULT h-BN {status} ({time.time()-t0:.0f}s) state: {out} ===", flush=True)
        report_site(ie["efg"]["a0"], "B", "11B", AXES)
        report_site(ie["efg"]["a2"], "N", None, AXES)
    except Exception as e:      # noqa: BLE001
        print(f"=== FAILED h-BN {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
