"""End-to-end whole-SCF wall A/B for the auto-default shift-invert secular solve.

For one config, runs a FIXED number of warm SCF iterations twice from the SAME start state —
once with ``shift_invert="auto"`` (the new default: engages per-solve above the measured crossover
dim) and once with ``shift_invert=False`` (forced exact dense) — and reports the whole-SCF wall of
each plus the final Γ eigenvalue agreement. Both runs take identical iters/tol from identical
v_start, so the wall difference isolates the solver.

Configs (SE_CFG):
    base  production rutile TiO2 (ecut 300, lmax 3, fp 4, k2²²): pencil dim ~737, ABOVE D*=400 →
          auto engages, the whole-SCF win should show. Warm-started from perf_iter's cached state.
    small ecut 150 TiO2 (dim ~265, BELOW D*): auto must NOT engage → wall-neutral vs exact (the
          no-regression control). Cold MT staging (no fullpot), a handful of iterations.
    d5    D5 aug6 (ecut 300, lmax 6, fp 4, k2²²), warm-started from a saved aug6 state: a richer
          in-sphere basis at the same dim ~737.

Env: SE_CFG (base|small|d5), SE_N (iterations, default 6), SE_KWORKERS (default 5),
     SE_D5_STATE (path to the D5 warm state; default ~/tio2_states/d5_aug6_noLO_k222.pkl).
"""
import os
import pickle
import time

import numpy as np
from _common import A_BOHR, ATOMS, RADII, env_int
from perf_iter import cfg, load_state

from gradwave.flapw import crystal_scf_multi


def _base_cfg():
    c = cfg()
    c.pop("verbose", None)
    c["kworkers"] = env_int("SE", "KWORKERS", 5)
    return c, {"__full_state__": load_state(c)}


def _small_cfg():
    c = dict(ecut=150.0, lmax=2, kmesh=(2, 2, 2), smearing=0.0, fullpot=False,
             use_symmetry=True, kworkers=env_int("SE", "KWORKERS", 5), subspace_reuse=False)
    return c, None                                     # cold MT staging, no warm state


def _d5_cfg():
    c = dict(ecut=300.0, lmax=6, fullpot=True, fullpot_lmax=4, smearing=0.0, use_symmetry=True,
             kerker=0.7, subspace_reuse=False, kworkers=env_int("SE", "KWORKERS", 5))
    path = os.path.expanduser(os.environ.get(
        "SE_D5_STATE", "~/tio2_states/d5_aug6_noLO_k222.pkl"))
    with open(path, "rb") as f:
        warm = {"__full_state__": pickle.load(f)}
    return c, warm


def _run(c, warm, n, mode):
    t0 = time.time()
    bands, _info = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=n, tol=0.0, efg=False,
                                     v_start=warm, shift_invert=mode, **c)
    return time.time() - t0, np.asarray(bands["ev"])


def main():
    which = os.environ.get("SE_CFG", "base")
    n = env_int("SE", "N", 6)
    c, warm = {"base": _base_cfg, "small": _small_cfg, "d5": _d5_cfg}[which]()
    dimtag = {"base": "~737 (>D*)", "small": "~265 (<D*)", "d5": "~737 (>D*)"}[which]
    print(f"# si_e2e cfg={which} dim~{dimtag} ecut={c['ecut']} lmax={c['lmax']} n={n} "
          f"kworkers={c['kworkers']} OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    # warm the import/BLAS/pool once with a throwaway single iteration before timing
    _run(c, warm, 1, False)
    t_exact, ev_exact = _run(c, warm, n, False)
    t_auto, ev_auto = _run(c, warm, n, "auto")
    dev = float(np.abs(ev_auto - ev_exact).max())
    print(f"exact(False)  {t_exact:8.1f} s   (whole-SCF, {n} iters)", flush=True)
    print(f"auto          {t_auto:8.1f} s   x{t_exact/t_auto:.2f}   max|Δev_Γ|={dev:.2e} eV",
          flush=True)
    verdict = ("ENGAGED-WIN" if which != "small" else "NEUTRAL-EXPECTED")
    print(f"SUMMARY cfg={which} speedup={t_exact/t_auto:.2f}x parity={dev:.2e}eV {verdict}",
          flush=True)


if __name__ == "__main__":
    main()
