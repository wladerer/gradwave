"""Front B transferability control — l=1 HELO-energy scan on CORUNDUM Al2O3 O.

Rutile O (efg_eta_helo_scan.py) showed E2 is a clean, monotonic η lever (η 0.69->0.72->0.82 as
E2 110->120->140) that reaches Elk's 0.74 near E2~123. Corundum O is a DIFFERENT structure whose
η is already good (gw 0.48 vs Elk 0.51, efg_converged_k_validation.md). The decisive
transferability test: does the SAME raised E2 that fixes rutile push corundum-O η UP past its
already-correct 0.51 (=> optimum is structure-specific => per-material tunable, keep default 90),
or does corundum stay put (=> a moderate raise is safe everywhere => ship a raised default)?

Same robust setup as corundum_efg.py: runtime Al injection (validated vs NIST LDA), converge_efg
(muffin-tin -> chunked fullpot -> newton_polish -> one exact efg pass), dense eigensolve.

Env: HELO_ES="90,120"  RAL/RO radii  ECUT=300 LMAX=4 FPLMAX=4 KERKER=0.7 KWORKERS OMP
Run on asus:  OMP_NUM_THREADS=8 HELO_ES="90,120" \
  uv run python experiments/autoapw/efg_eta_helo_corundum.py 2>&1 | tee scanBc.log
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import numpy as np
from _efgrun import converge_efg

from gradwave.flapw import atom as _atom
from gradwave.flapw import scf as _scf
from gradwave.flapw.nmr import quadrupolar_coupling

# ---- runtime Al injection (identical to corundum_efg.py; validated vs NIST LDA, no src edit) ----
_atom.CONFIG["Al"] = (13.0, [(1, 0, 2), (2, 0, 2), (2, 1, 6), (3, 0, 2), (3, 1, 1)])
_scf._CORE["Al"] = [(0, 1, 2), (0, 2, 2), (1, 1, 6)]
_scf._VAL_E["Al"] = 3
_scf._N_VAL_BANDS["Al"] = 2
_scf._VALENCE_NL["Al"] = {0: "3s", 1: "3p"}
_al_ha = {"1s": -55.5, "2s": -3.93, "2p": -2.56, "3s": -0.286, "3p": -0.10}
_atom.NIST_LDA_EV["Al"] = {k: v * 27.211386 for k, v in _al_ha.items()}

CELL_BOHR = np.array([[4.497737, 2.596770, 8.184592],
                      [-4.497737, 2.596770, 8.184592],
                      [-0.000000, -5.193539, 8.184592]])
ATOMS = [((0.352160, 0.352160, 0.352160), "Al"), ((0.147840, 0.147840, 0.147840), "Al"),
         ((0.647840, 0.647840, 0.647840), "Al"), ((0.852160, 0.852160, 0.852160), "Al"),
         ((0.250000, 0.556240, 0.943760), "O"), ((0.056240, 0.750000, 0.443760), "O"),
         ((0.556240, 0.943760, 0.250000), "O"), ((0.443760, 0.056240, 0.750000), "O"),
         ((0.943760, 0.250000, 0.556240), "O"), ((0.750000, 0.443760, 0.056240), "O")]
RADII = {"Al": float(os.environ.get("RAL", "0.97")), "O": float(os.environ.get("RO", "0.824"))}
ELK = dict(O_full=35.64, O_eta=0.51, O_cq=2.20)   # efg_converged_k_validation.md (k6/k8)


def cfg_from_env(helo_e: float):
    kerker = os.environ.get("KERKER", "0.7")
    si_env = os.environ.get("SHIFT_INVERT", "0").lower()
    c = dict(ecut=float(os.environ.get("ECUT", "300")), lmax=int(os.environ.get("LMAX", "4")),
             fullpot=True, fullpot_lmax=int(os.environ.get("FPLMAX", "4")), smearing=0.0,
             use_symmetry=True, subspace_reuse=False,
             kerker=(None if kerker.lower() in ("none", "0", "") else float(kerker)),
             shift_invert=si_env not in ("0", "none", "", "false"),
             kworkers=int(os.environ.get("KWORKERS", "4")))
    los = {"O": [(0, "2s")]}
    if helo_e > 0:
        los["O"].append((1, {"e": float(helo_e), "confine": False}))
    c["los"] = los
    c["el_override"] = {"O": {0: "2p"}}
    return c


def main() -> int:
    es = [float(x) for x in os.environ.get("HELO_ES", "90,120").split(",")]
    rv_gate = float(os.environ.get("RV", "1.2e-2"))
    print(f"# FRONT B corundum-O HELO_E control  es={es}  R_O={RADII['O']} "
          f"kw={os.environ.get('KWORKERS')} OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    print(f"# Elk 11: O full V_zz={ELK['O_full']:.2f} eta={ELK['O_eta']:.3f} "
          f"C_Q(17O)={ELK['O_cq']} MHz", flush=True)
    rows = []
    for e2 in es:
        cfg = cfg_from_env(e2)
        t0 = time.time()
        try:
            ie, status, _state, _log = converge_efg(CELL_BOHR, ATOMS, RADII, cfg, (2, 2, 2),
                                                    rv_gate=rv_gate)
            o = ie["efg"]["a4"]   # first O site (index 4)
            cq = quadrupolar_coupling(o["V_zz"], o["eta"], "17O")["abs_C_Q_MHz"]
            frac = abs(o["V_zz"]) / ELK["O_full"]
            print(f"E2={e2:6.1f} eV | V_zz={o['V_zz']:+8.3f} (|r|={frac:.3f}) eta={o['eta']:.3f} "
                  f"C_Q(17O)={cq:.3f} MHz | {status} ({time.time()-t0:.0f}s)", flush=True)
            rows.append((e2, o["V_zz"], o["eta"], cq, status))
        except Exception as ex:
            print(f"E2={e2:6.1f} eV | FAILED: {type(ex).__name__}: {ex}", flush=True)
            traceback.print_exc()
    print("# --- summary (E2, V_zz, |r|, eta, C_Q, status) ---", flush=True)
    for e2, vzz, eta, cq, status in rows:
        print(f"#  {e2:6.1f}  {vzz:+8.3f}  {abs(vzz)/ELK['O_full']:.3f}  {eta:.3f}  "
              f"{cq:.3f}  {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
