"""h-BN ¹¹B EFG with an optional unconfined l=1 B HELO (cold converge_efg).

Baseline gradwave B: C_Q 3.60 MHz (+22% over Elk 2.95 / exp 2.934) — the cation-overshoot
direction. Tests whether the B HELO pulls C_Q down toward experiment, as it moved Al 1.17->1.10.

Env: HELO_E=90  (="" => baseline)  HELO_CONF=0  ECUT=400 KWORKERS=4 KERKER=0.7
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np
from _efgrun import converge_efg
from _mgroup import hex_cell_bohr, report_site  # noqa: F401  (import injects B/N species)

A, C = 2.504, 6.661
CELL_BOHR = hex_cell_bohr(A, C)
ATOMS = [((1 / 3, 2 / 3, 1 / 4), "B"), ((2 / 3, 1 / 3, 3 / 4), "B"),
         ((2 / 3, 1 / 3, 1 / 4), "N"), ((1 / 3, 2 / 3, 3 / 4), "N")]
RADII = {"B": 0.70, "N": 0.70}
AXES = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0])}


def cfg_from_env():
    kerker = os.environ.get("KERKER", "0.7")
    c = dict(ecut=float(os.environ.get("ECUT", "400")), lmax=int(os.environ.get("LMAX", "4")),
             fullpot=True, fullpot_lmax=int(os.environ.get("FPLMAX", "4")), smearing=0.0,
             use_symmetry=True, subspace_reuse=False,
             kerker=(None if kerker.lower() in ("none", "0", "") else float(kerker)),
             shift_invert=True, kworkers=int(os.environ.get("KWORKERS", "4")))
    los = {"N": [(0, "2s")]}
    el = {"N": {0: "2p"}}
    he = os.environ.get("HELO_E", "90")
    if he:
        conf = bool(int(os.environ.get("HELO_CONF", "0")))
        los["B"] = [(1, {"e": float(he), "confine": conf})]
    c["los"] = los
    c["el_override"] = el
    return c, he


def main():
    cfg, he = cfg_from_env()
    out = os.path.expanduser(os.environ.get("OUT", "~/efg_mgroup/hbn_helo.pkl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"# h-BN HELO_E={he or 'NONE'} conf={os.environ.get('HELO_CONF','0')} "
          f"kerker={cfg['kerker']} los={cfg['los']} (Elk B -30.02/eta0/2.95 MHz, exp 2.934)",
          flush=True)
    t0 = time.time()
    try:
        ie, status, state, _ = converge_efg(CELL_BOHR, ATOMS, RADII, cfg, (2, 2, 2),
                                             rv_gate=float(os.environ.get("RV", "1.2e-2")))
        with open(out, "wb") as f:
            pickle.dump(state, f)
        print(f"=== RESULT h-BN HELO_E={he or 'NONE'} {status} ({time.time()-t0:.0f}s) ===",
              flush=True)
        report_site(ie["efg"]["a0"], "B", "11B", AXES)
        report_site(ie["efg"]["a2"], "N", None, AXES)
    except Exception as e:  # noqa: BLE001
        print(f"=== FAILED h-BN {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
