"""Phase-3 FLAPW-DFPT: the FULLY SELF-CONSISTENT finite-difference dV_zz/du reference.

The screened analytic dV_zz/du (``dfpt_validate.py``) must match a central finite difference of
the self-consistently RECONVERGED V_zz at u ± delta — this is the ground truth the Dyson screening
reproduces (the bare, frozen-v_eff gradient is Gate B; this is the screened reference). Each
displaced geometry is warm-started from the base fixed point (``dfpt_base_state.pkl``) and
reconverged with the same trustworthy recipe (symmetry-on kerker + ``newton_polish``); the small
displacement keeps it in the base's basin so the reconvergence is cheap and robust.

Computed AT THE SAME BASE as the analytic gradient (not a hardcoded literal), so the comparison is
self-contained and independent of any other run's fixed point. Run on asus.
"""
from __future__ import annotations

import pickle
import sys
import time

from gradwave.flapw import crystal_scf_multi, newton_polish

PKL = sys.argv[1] if len(sys.argv) > 1 else "dfpt_base_state.pkl"
DELTA = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-3
WARM = int(sys.argv[3]) if len(sys.argv) > 3 else 60
DO_NEWTON = int(sys.argv[4]) if len(sys.argv) > 4 else 1

with open(PKL, "rb") as fh:
    BASE = pickle.load(fh)
CFG = BASE["config"]
A, C0, U0 = CFG["A"], CFG["C0"], CFG["U0"]
RADII, ECUT, LMAX, FP_LMAX, KK = (CFG["RADII"], CFG["ECUT"], CFG["LMAX"], CFG["FP_LMAX"], CFG["KK"])
KERKER = CFG["KERKER"]

cfg = dict(ecut=ECUT, lmax=LMAX, smearing=0.0, fullpot=True, fullpot_lmax=FP_LMAX,
           use_symmetry=True, subspace_reuse=False, kworkers=4,
           kerker=(KERKER if KERKER > 0 else None))


def atoms_for(u):
    return [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
            ((u, u, 0.0), "O"), ((1 - u, 1 - u, 0.0), "O"),
            ((0.5 + u, 0.5 - u, 0.5), "O"), ((0.5 - u, 0.5 + u, 0.5), "O")]


def converged_vzz(u, tag):
    atoms = atoms_for(u)
    _, iw = crystal_scf_multi([A, A, C0], atoms, RADII, iters=WARM, tol=1e-5, efg=False,
                              kmesh=(KK, KK, KK), v_start={"__full_state__": BASE["state"]}, **cfg)
    state = iw["state"]
    rn = float("nan")
    if DO_NEWTON:
        state, ni = newton_polish([A, A, C0], atoms, RADII, iw["state"],
                                  scf_kwargs=dict(cfg, kmesh=(KK, KK, KK), efg=False),
                                  maxiter=8, inner_maxiter=20, f_tol=1e-6, rounds=4)
        rn = ni["residual_norm"]
    _, ei = crystal_scf_multi([A, A, C0], atoms, RADII, iters=1, tol=0.0, efg=True,
                              v_start={"__full_state__": state}, kmesh=(KK, KK, KK), **cfg)
    ti, o = ei["efg"]["a0"]["V_zz"], ei["efg"]["a2"]["V_zz"]
    print(f"  {tag} u={u:.5f}: Ti.Vzz={ti:+.5f} O.Vzz={o:+.5f} "
          f"(newton_res={rn:.2e} symdev={ei.get('symmetry_dev'):.1e})", flush=True)
    return ti, o


def main():
    t0 = time.time()
    print(f"CONFIG ecut={ECUT} lmax={LMAX} fp_lmax={FP_LMAX} k={KK} delta={DELTA} warm={WARM}",
          flush=True)
    ti_p, o_p = converged_vzz(U0 + DELTA, "+")
    ti_m, o_m = converged_vzz(U0 - DELTA, "-")
    dti = (ti_p - ti_m) / (2 * DELTA)
    do = (o_p - o_m) / (2 * DELTA)
    print("=== self-consistent FD dV_zz/du (screened reference, this base) ===", flush=True)
    print(f"  Ti: dV_zz/du = {dti:+.1f}   O: dV_zz/du = {do:+.1f}   "
          f"(delta={DELTA}, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
