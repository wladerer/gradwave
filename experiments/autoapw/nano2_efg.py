"""gradwave FLAPW EFG for ferroelectric NaNO₂ (Imm2) — ²³Na quadrupolar NMR vs Elk + experiment.

Na⁺ sits in a distorted NaO₆ site with a C₂ axis along b (the polar axis) → biaxial EFG (η≠0).
Experimental ²³Na C_Q(NaNO₂) ≈ 1.09 MHz, η ≈ 0.11 (RT; Freude survey / temperature-dependent ²³Na
NMR). Na is isoelectronic with Mg²⁺ ([Ne]); frozen 2p → the same antishielding omission that failed
(wrong sign) for Mg²⁺ in MgF₂. This run tests whether Na⁺ needs its 2p semicore in valence. The
N–O bond is very short (1.27 Å) → small anion spheres; only the Na site (large sphere)
is a trustworthy report. Conventional I-centered orthorhombic cell (8 atoms). Na/N injected via
_mgroup, O ships. NO src change.

Coordinates: Ekuma et al., arXiv:1208.5710 (ferroelectric NaNO₂); RT lattice a=3.5653, b=5.5728,
c=5.3846 Å (Imm2, #44).
"""
import os
import pickle
import sys
import time
import traceback

import numpy as np
from _efgrun import converge_efg
from _mgroup import BOHR, report_site

A, B, C = 3.5653, 5.5728, 5.3846        # Å
CELL_BOHR = np.array([A / BOHR, B / BOHR, C / BOHR])       # orthorhombic edges
ATOMS = [((0.0, 0.5881, 0.0), "Na"), ((0.5, 0.0881, 0.5), "Na"),
         ((0.0, 0.1224, 0.0), "N"), ((0.5, 0.6224, 0.5), "N"),
         ((0.0, 0.0, 0.1962), "O"), ((0.5, 0.5, 0.6962), "O"),
         ((0.0, 0.0, 0.8038), "O"), ((0.5, 0.5, 0.3038), "O")]
RADII = {"Na": float(os.environ.get("RNA", "1.05")), "N": float(os.environ.get("RN", "0.62")),
         "O": float(os.environ.get("RO", "0.62"))}
AXES = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([0.0, 0.0, 1.0])}


def cfg_from_env():
    kerker = os.environ.get("KERKER", "0.7")
    c = dict(ecut=float(os.environ.get("ECUT", "350")), lmax=int(os.environ.get("LMAX", "4")),
             fullpot=True, fullpot_lmax=int(os.environ.get("FPLMAX", "4")),
             smearing=0.0, use_symmetry=True, subspace_reuse=False,
             kerker=(None if kerker.lower() in ("none", "0", "") else float(kerker)),
             shift_invert=True, kworkers=int(os.environ.get("KWORKERS", "4")))
    if int(os.environ.get("OLO", "1")):
        c["los"] = {"O": [(0, "2s")], "N": [(0, "2s")]}
        c["el_override"] = {"O": {0: "2p"}, "N": {0: "2p"}}
    return c


def main():
    cfg = cfg_from_env()
    out = os.environ.get("OUT", os.path.expanduser("~/efg_mgroup/nano2.pkl"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"# NaNO2 ecut={cfg['ecut']} lmax={cfg['lmax']} fp{cfg['fullpot_lmax']} k222 "
          f"kerker={cfg['kerker']} R_Na={RADII['Na']} R_N={RADII['N']} R_O={RADII['O']} "
          f"kworkers={cfg['kworkers']} OMP={os.environ.get('OMP_NUM_THREADS')} (Na frozen-[Ne])",
          flush=True)
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
    t0 = time.time()
    try:
        ie, status, state, _ = converge_efg(CELL_BOHR, ATOMS, RADII, cfg, (2, 2, 2),
                                             rv_gate=rv_gate)
        with open(out, "wb") as f:
            pickle.dump(state, f)
        print(f"=== RESULT NaNO2 {status} ({time.time()-t0:.0f}s) state: {out} ===", flush=True)
        report_site(ie["efg"]["a0"], "Na", "23Na", AXES)
    except Exception as e:      # noqa: BLE001
        print(f"=== FAILED NaNO2 {time.time()-t0:.0f}s: {type(e).__name__}: {e} ===", flush=True)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
