"""Phase-2 FLAPW-DFPT: converge the rutile TiO2 base fixed point ONCE and checkpoint it.

The base SCF is the single expensive/fragile step of the chi_0 validation (the fullpot
k222 fixed point is chaotically multistable — see efg_gradcheck.md). We converge it here,
tightly, with the campaign's stabilised recipe (kerker=0.7, smearing=0, cold), and pickle
the complete fixed-point state (converged spherical + aspherical potentials, interstitial
grid) plus the reference EFG. The chi_0 validator (dfpt_validate.py) then restores this
state cheaply and non-fragilely (one iterate AT the fixed point) — so the chi_0 machinery
can be iterated on without ever re-paying the fragile convergence.

Run on asus. Config matches experiments/autoapw/efg_fd_gradient.py (the FD reference).
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np

from gradwave.flapw import crystal_scf_multi

U0 = 0.3048
A = 8.68083          # a = b (Bohr)
C0 = 5.59096         # c (Bohr)
RADII = {"Ti": 1.098, "O": 0.824}

ECUT = float(sys.argv[1]) if len(sys.argv) > 1 else 250.0
LMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 3
FP_LMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 4
KK = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 80
TOL = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-6
KW = int(sys.argv[7]) if len(sys.argv) > 7 else 4
KERKER = float(sys.argv[8]) if len(sys.argv) > 8 else 0.7
OUT = sys.argv[9] if len(sys.argv) > 9 else "dfpt_base_state.pkl"


def atoms_for(u):
    return [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
            ((u, u, 0.0), "O"), ((1 - u, 1 - u, 0.0), "O"),
            ((0.5 + u, 0.5 - u, 0.5), "O"), ((0.5 - u, 0.5 + u, 0.5), "O")]


def main():
    print(f"CONFIG ecut={ECUT} lmax={LMAX} fp_lmax={FP_LMAX} k={KK} tol={TOL} "
          f"kerker={KERKER} kworkers={KW}", flush=True)
    t0 = time.time()
    bands, info = crystal_scf_multi(
        [A, A, C0], atoms_for(U0), RADII, ecut=ECUT, lmax=LMAX, iters=ITERS, tol=TOL,
        kmesh=(KK, KK, KK), smearing=0.0, efg=True, fullpot=True, fullpot_lmax=FP_LMAX,
        kworkers=KW, v_start=None, kerker=(KERKER if KERKER > 0 else None), verbose=False)
    dt = time.time() - t0
    rec = info.get("recorder")
    last = rec.iters[-1] if rec is not None else {}
    r_v = last.get("r_v")
    r_nsph = last.get("r_nsph")
    ti = info["efg"]["a0"]
    o = info["efg"]["a2"]
    niter = len(rec.iters) if rec is not None else -1
    print(f"base: {dt:.0f}s niter={niter} r_v={r_v:.2e} r_nsph={r_nsph} "
          f"Ti.Vzz={ti['V_zz']:+.5f} O.Vzz={o['V_zz']:+.5f} symdev={info.get('symmetry_dev'):.1e}",
          flush=True)

    payload = {
        "config": dict(A=A, C0=C0, U0=U0, RADII=RADII, ECUT=ECUT, LMAX=LMAX,
                       FP_LMAX=FP_LMAX, KK=KK, KERKER=KERKER),
        "state": info["state"],                       # __full_state__ for warm restore
        "efg": {"Ti": {kk: (float(vv) if np.isscalar(vv) or isinstance(vv, float) else vv)
                       for kk, vv in ti.items() if kk in ("V_zz", "eta", "V_zz_valence")},
                "O": {kk: (float(vv) if np.isscalar(vv) or isinstance(vv, float) else vv)
                      for kk, vv in o.items() if kk in ("V_zz", "eta", "V_zz_valence")}},
        "conv": {"r_v": r_v, "r_nsph": r_nsph, "niter": niter},
    }
    with open(OUT, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"saved -> {OUT}  (r_v={r_v:.2e})", flush=True)
    ok = (r_v is not None and r_v < 1e-3 and (r_nsph is None or r_nsph < 1e-2))
    print("CONVERGED_OK" if ok else "CONVERGENCE_SUSPECT", flush=True)


if __name__ == "__main__":
    main()
