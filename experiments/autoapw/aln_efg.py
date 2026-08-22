"""gradwave FLAPW EFG for wurtzite AlN (P6_3mc) — a 2nd ²⁷Al site (tetrahedral nitride) vs Elk.

Different Al environment than corundum: 4-coordinate tetrahedral nitride vs 6-coordinate octahedral
oxide. Wurtzite c/a distortion → axial EFG (η=0) along c. Experimental ²⁷Al C_Q(AlN) = 1.914 MHz,
η=0 (single-crystal ²⁷Al NMR, Molecules 2020; Bastow ~1.9 MHz). Al frozen [Ne] (the corundum recipe
that reached experiment); N injected. NO src change.
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np
from _efgrun import converge_efg
from _mgroup import hex_cell_bohr, report_site

A, C, U = 3.111, 4.978, 0.3821        # Å, Å, internal (AlN wurtzite)
CELL_BOHR = hex_cell_bohr(A, C)
ATOMS = [((1 / 3, 2 / 3, 0.0), "Al"), ((2 / 3, 1 / 3, 0.5), "Al"),
         ((1 / 3, 2 / 3, U), "N"), ((2 / 3, 1 / 3, 0.5 + U), "N")]
RADII = {"Al": float(os.environ.get("RAL", "0.95")), "N": float(os.environ.get("RN", "0.85"))}
AXES = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0])}


def cfg_from_env():
    kerker = os.environ.get("KERKER", "0.7")
    c = dict(ecut=float(os.environ.get("ECUT", "300")), lmax=int(os.environ.get("LMAX", "4")),
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
    out = os.environ.get("OUT", os.path.expanduser("~/efg_mgroup/aln.pkl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"# AlN ecut={cfg['ecut']} lmax={cfg['lmax']} fp{cfg['fullpot_lmax']} k222 "
          f"kerker={cfg['kerker']} R_Al={RADII['Al']} R_N={RADII['N']} "
          f"kworkers={cfg['kworkers']} OMP={os.environ.get('OMP_NUM_THREADS')} (Al frozen-[Ne])",
          flush=True)
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
    t0 = time.time()
    try:
        ie, status, state, _ = converge_efg(CELL_BOHR, ATOMS, RADII, cfg, (2, 2, 2),
                                             rv_gate=rv_gate)
        with open(out, "wb") as f:
            pickle.dump(state, f)
        print(f"=== RESULT AlN {status} ({time.time()-t0:.0f}s) state: {out} ===", flush=True)
        report_site(ie["efg"]["a0"], "Al", "27Al", AXES)
        report_site(ie["efg"]["a2"], "N", None, AXES)
    except Exception as e:      # noqa: BLE001
        print(f"=== FAILED AlN {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
