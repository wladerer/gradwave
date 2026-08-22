"""Phase-3 FLAPW-DFPT: converge the rutile TiO2 base fixed point ROBUSTLY and checkpoint it.

Same base state as ``dfpt_base.py``, but converged with the trustworthy-EFG recipe instead of a
single symmetry-free kerker Anderson run: the k222 fullpot fixed point is chaotically multistable
and, with ``use_symmetry=False``, the symmetry-breaking modes grow and the SCF never settles
(measured: r_v stuck ~0.4, symdev oscillating to 1.0 at 80-150 iterations). The fix is the same as
the shipped trustworthy-EFG pipeline (``tio2_newton_pipeline.py``): converge with
``use_symmetry=True`` (the space-group symmetrization kills the runaway modes), then
``newton_polish`` to a TIGHT coupled fixed point (f_tol 1e-6) — the density response is a
derivative, so the base must be a genuine fixed point, not a stalled iterate.

The saved STATE is the real-space potentials (spherical + aspherical + interstitial grid), which
are k-mesh / symmetry AGNOSTIC — so ``dfpt_validate.py`` restores it into its ``use_symmetry=False``
full-mesh context (one iterate at the fixed point, no re-convergence) and the per-k response sum is
the full-BZ density directly. The pickle format is identical to ``dfpt_base.py``.

Run on asus.
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np

from gradwave.flapw import crystal_scf_multi, newton_polish

U0 = 0.3048
A = 8.68083          # a = b (Bohr)
C0 = 5.59096         # c (Bohr)
RADII = {"Ti": 1.098, "O": 0.824}

ECUT = float(sys.argv[1]) if len(sys.argv) > 1 else 250.0
LMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 3
FP_LMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 4
KK = int(sys.argv[4]) if len(sys.argv) > 4 else 2
WARM_ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 300
KW = int(sys.argv[6]) if len(sys.argv) > 6 else 4
KERKER = float(sys.argv[7]) if len(sys.argv) > 7 else 0.7
OUT = sys.argv[8] if len(sys.argv) > 8 else "dfpt_base_state.pkl"
DO_NEWTON = int(sys.argv[9]) if len(sys.argv) > 9 else 1


def atoms_for(u):
    return [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
            ((u, u, 0.0), "O"), ((1 - u, 1 - u, 0.0), "O"),
            ((0.5 + u, 0.5 - u, 0.5), "O"), ((0.5 - u, 0.5 + u, 0.5), "O")]


def main():
    print(f"CONFIG ecut={ECUT} lmax={LMAX} fp_lmax={FP_LMAX} k={KK} warm_iters={WARM_ITERS} "
          f"kerker={KERKER} kworkers={KW}", flush=True)
    atoms = atoms_for(U0)
    cfg = dict(ecut=ECUT, lmax=LMAX, smearing=0.0, fullpot=True, fullpot_lmax=FP_LMAX,
               use_symmetry=True, subspace_reuse=False, kworkers=KW,
               kerker=(KERKER if KERKER > 0 else None))

    t0 = time.time()
    _, iw = crystal_scf_multi([A, A, C0], atoms, RADII, iters=WARM_ITERS, tol=1e-5, efg=False,
                              kmesh=(KK, KK, KK), **cfg)
    r = iw["recorder"].summarize()
    last = iw["recorder"].iters[-1]
    print(f"warm k{KK}22 ({time.time()-t0:.0f}s): n_it={r['n_iter']} "
          f"r_v={last.get('r_v')} r_nsph={last.get('r_nsph')}", flush=True)

    state = iw["state"]
    if DO_NEWTON:
        t1 = time.time()
        # newton_polish returns a state NEVER WORSE than the input (monotone acceptance), so
        # it can only tighten the long warm fixed point; skip via DO_NEWTON=0 if it stalls.
        state, ni = newton_polish([A, A, C0], atoms, RADII, iw["state"],
                                  scf_kwargs=dict(cfg, kmesh=(KK, KK, KK), efg=False),
                                  maxiter=8, inner_maxiter=20, f_tol=1e-6, rounds=4, verbose=True)
        print(f"newton ({time.time()-t1:.0f}s): converged={ni['converged']} "
              f"residual_norm={ni['residual_norm']:.2e} rounds={ni['rounds']}", flush=True)
    else:
        ni = {"converged": False, "residual_norm": float("nan")}

    # EFG at the polished fixed point (one efg=True iterate; state unchanged).
    _, ei = crystal_scf_multi([A, A, C0], atoms, RADII, iters=1, tol=0.0, efg=True,
                              v_start={"__full_state__": state}, kmesh=(KK, KK, KK), **cfg)
    ti, o = ei["efg"]["a0"], ei["efg"]["a2"]
    print(f"polished EFG: Ti.Vzz={ti['V_zz']:+.5f} O.Vzz={o['V_zz']:+.5f} "
          f"symdev={ei.get('symmetry_dev'):.1e}", flush=True)

    payload = {
        "config": dict(A=A, C0=C0, U0=U0, RADII=RADII, ECUT=ECUT, LMAX=LMAX,
                       FP_LMAX=FP_LMAX, KK=KK, KERKER=KERKER),
        "state": state,
        "efg": {"Ti": {kk: (float(vv) if np.isscalar(vv) or isinstance(vv, float) else vv)
                       for kk, vv in ti.items() if kk in ("V_zz", "eta", "V_zz_valence")},
                "O": {kk: (float(vv) if np.isscalar(vv) or isinstance(vv, float) else vv)
                      for kk, vv in o.items() if kk in ("V_zz", "eta", "V_zz_valence")}},
        "conv": {"residual_norm": ni["residual_norm"], "converged": ni["converged"],
                 "warm_r_nsph": last.get("r_nsph"), "warm_r_v": last.get("r_v")},
    }
    with open(OUT, "wb") as fh:
        pickle.dump(payload, fh)
    rn = last.get("r_nsph")
    ok = rn is not None and rn < 1e-2
    print(f"saved -> {OUT}  (warm r_nsph={rn})", flush=True)
    print("CONVERGED_OK" if ok else "CONVERGENCE_SUSPECT", flush=True)


if __name__ == "__main__":
    main()
