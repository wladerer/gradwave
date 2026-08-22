"""gradwave FLAPW EFG for α-Li₃N (P6/mmm) — ⁷Li quadrupolar NMR (two Li sites) vs Elk + experiment.

Two crystallographic Li: Li1 (1b, 0,0,½; between N-layers) and Li2 (2c, ⅓,⅔,0; in the Li₂N plane),
both on the hexagonal axis → axial EFG (η=0). Experimental ⁷Li: C_Q(Li1)=0.258, C_Q(Li2)=0.585 MHz,
η=0 (Differ/Bastow static ⁷Li NMR). Li⁺ has no p semicore → its EFG is pure lattice + tiny 1s
Sternheimer (an intrinsically small, noise-floor-adjacent observable). Li and N injected. NO src
change.
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np
from _efgrun import converge_efg
from _mgroup import hex_cell_bohr, report_site

A, C = 3.648, 3.875        # Å (α-Li3N, P6/mmm)
CELL_BOHR = hex_cell_bohr(A, C)
ATOMS = [((0.0, 0.0, 0.5), "Li"),                                   # a0 = Li1 (1b)
         ((1 / 3, 2 / 3, 0.0), "Li"), ((2 / 3, 1 / 3, 0.0), "Li"),  # a1,a2 = Li2 (2c)
         ((0.0, 0.0, 0.0), "N")]                                    # a3 = N (1a)
RADII = {"Li": float(os.environ.get("RLI", "0.90")), "N": float(os.environ.get("RN", "0.90"))}
AXES = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0])}


def cfg_from_env():
    kerker = os.environ.get("KERKER", "0.7")
    c = dict(ecut=float(os.environ.get("ECUT", "250")), lmax=int(os.environ.get("LMAX", "4")),
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
    out = os.environ.get("OUT", os.path.expanduser("~/efg_mgroup/li3n.pkl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"# Li3N ecut={cfg['ecut']} lmax={cfg['lmax']} fp{cfg['fullpot_lmax']} k222 "
          f"kerker={cfg['kerker']} R_Li={RADII['Li']} R_N={RADII['N']} "
          f"kworkers={cfg['kworkers']} OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
    t0 = time.time()
    try:
        ie, status, state, _ = converge_efg(CELL_BOHR, ATOMS, RADII, cfg, (2, 2, 2),
                                             rv_gate=rv_gate)
        with open(out, "wb") as f:
            pickle.dump(state, f)
        print(f"=== RESULT Li3N {status} ({time.time()-t0:.0f}s) state: {out} ===", flush=True)
        report_site(ie["efg"]["a0"], "Li1(1b)", "7Li", AXES)
        report_site(ie["efg"]["a1"], "Li2(2c)", "7Li", AXES)
        report_site(ie["efg"]["a3"], "N", None, AXES)
    except Exception as e:      # noqa: BLE001
        print(f"=== FAILED Li3N {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
