"""A/B for the Gaunt-contraction ``sphere_density_multipoles_bands`` vs the angular-grid
reference (``_sphere_density_multipoles_bands_grid``), on the production TiO2 warm fullpot state.

The rewrite is ulp-different (different summation order), and this map chaotically amplifies ulp
noise (documented in TIO2_NMR.md), so two divergent trajectories are NOT the right gate. Instead:

    parity  run N warm iterations along the Gaunt trajectory; at every density-pass call also
            evaluate the grid reference on the SAME amps and record max|Δρ_LM|/scale — the
            "documented-ulp over several iterations" gate on the real, evolving state.
    bench   per-iteration wall, both paths, medians of N warm iterations from the shared cached
            state (whole-iteration A/B), plus the contention-immune in-process per-call ratio on
            the captured production amps.

Env: PI_* (shared with perf_iter — ECUT/LMAX/FPLMAX/K/KERKER/KWORKERS), GA_N (iters, default 8).
"""
import os
import sys
import time

import numpy as np
from _common import A_BOHR, ATOMS, RADII, env_int
from perf_iter import _state_sha, cfg, load_state

import gradwave.flapw.efg as efg
from gradwave.flapw.efg import (
    _sphere_density_multipoles_bands_grid as grid_ref,
)
from gradwave.flapw.efg import (
    sphere_density_multipoles_bands as gaunt,
)
from gradwave.flapw.scf import (
    _multi_finalize,
    _multi_init_state,
    _multi_iterate,
    _multi_setup,
)

GRABBED = []               # captured (amps_all, occ, us, lmax, lset) of the first density call


def make_ctx_state(c, warm):
    ctx = _multi_setup(a_bohr=A_BOHR, atoms=ATOMS, radii=RADII, **c)
    st = _multi_init_state(ctx, {"__full_state__": warm})
    return ctx, st


def mode_parity(c):
    n = env_int("GA", "N", 8)
    warm = load_state(c)
    ctx, st = make_ctx_state(c, warm)
    worst = []

    def comparator(amps_all, occ, us, lmax, lset, nx=None, nphi=None):
        rf = gaunt(amps_all, occ, us, lmax, lset, nx, nphi)
        rg = grid_ref(amps_all, occ, us, lmax, lset, nx, nphi)
        scale = max(max(float(np.abs(v).max()) for v in rg.values()), 1e-30)
        d = max(float(np.abs(rf[lm] - rg[lm]).max()) for lm in lset)
        worst.append(d / scale)
        return rf

    efg.sphere_density_multipoles_bands = comparator
    try:
        for it in range(n):
            _multi_iterate(ctx, st, it, iters=200, tol=0.0)
            print(f"it={it} max_rel|Δρ_LM|(all density calls this iter)="
                  f"{max(worst):.3e}", flush=True)
    finally:
        efg.sphere_density_multipoles_bands = gaunt
    print(f"PARITY worst_rel={max(worst):.3e} over {len(worst)} density-pass calls "
          f"({'PASS' if max(worst) < 1e-10 else 'FAIL'} @1e-10)", flush=True)


def _bench_path(c, fn, n):
    warm = load_state(c)
    efg.sphere_density_multipoles_bands = fn
    ctx, st = make_ctx_state(c, warm)
    _multi_iterate(ctx, st, 0, iters=200, tol=0.0)                  # warmup
    walls = []
    for it in range(1, 1 + n):
        t0 = time.perf_counter()
        _multi_iterate(ctx, st, it, iters=200, tol=0.0)
        walls.append(time.perf_counter() - t0)
    conv, info = _multi_finalize(ctx, st, efg=False)
    return walls, _state_sha(info["state"])


def _capture_amps(c):
    warm = load_state(c)
    ctx, st = make_ctx_state(c, warm)

    def grab(amps_all, occ, us, lmax, lset, nx=None, nphi=None):
        if not GRABBED:
            GRABBED.append((amps_all, occ, us, lmax, lset))
        return gaunt(amps_all, occ, us, lmax, lset, nx, nphi)

    efg.sphere_density_multipoles_bands = grab
    _multi_iterate(ctx, st, 0, iters=200, tol=0.0)
    efg.sphere_density_multipoles_bands = gaunt


def _timeit(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def mode_bench(c):
    n = env_int("GA", "N", 8)
    # isolated per-call ratio on real production amps (contention-immune)
    _capture_amps(c)
    amps_all, occ, us, lmax, lset = GRABBED[0]
    t_g = _timeit(lambda: gaunt(amps_all, occ, us, lmax, lset), 25)
    t_r = _timeit(lambda: grid_ref(amps_all, occ, us, lmax, lset), 25)
    print(f"per-call (dim lset={len(lset)}, lmax={lmax}, nb={len(occ)}): "
          f"grid {t_r*1e3:.2f} ms  gaunt {t_g*1e3:.2f} ms  x{t_r/t_g:.1f}", flush=True)
    # whole-iteration wall A/B
    wg, sg = _bench_path(c, gaunt, n)
    wr, sr = _bench_path(c, grid_ref, n)
    print(f"gaunt  walls_s={[round(w,3) for w in wg]} median={np.median(wg):.3f} sha={sg}",
          flush=True)
    print(f"grid   walls_s={[round(w,3) for w in wr]} median={np.median(wr):.3f} sha={sr}",
          flush=True)
    print(f"WHOLE-ITER median grid {np.median(wr):.3f}s -> gaunt {np.median(wg):.3f}s "
          f"x{np.median(wr)/np.median(wg):.2f}", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "parity"
    c = cfg()
    c.pop("verbose", None)
    print(f"# gaunt_ab mode={mode} OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"ecut={c['ecut']} lmax={c['lmax']} fp={c['fullpot_lmax']} k={c['kmesh'][0]}",
          flush=True)
    {"parity": mode_parity, "bench": mode_bench}[mode](c)


if __name__ == "__main__":
    main()
