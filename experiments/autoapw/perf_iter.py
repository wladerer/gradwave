"""Per-iteration wall/profile harness for the production TiO2 fullpot config.

The unit under measurement is one ``_multi_iterate`` from a WARM full state (the
production regime: fullpot on, aspherical channel live), so numbers are per-iteration
walls of the coupled fixed-point map, not cold muffin-tin staging iterations.

Modes (argv[1]):
    warm     converge a warm full state once and cache it (pickle) — every other
             mode resumes from this cache so A/B runs share their starting point
    profile  cProfile N iterations (kworkers=1 for a clean serial profile), print
             the top cumulative/tottime lines
    bench    wall-clock N iterations (kworkers as configured), print each + median
    coldwarmup  time a fresh 22-iteration run (the warmup number in the perf table)
    gate     short cold run; print sha256 of every info["state"] array — compare
             across code versions for the bit-identity gate on pure hoists/caches

Env knobs (prefix PI_): ECUT LMAX FPLMAX K KWORKERS ITERS N SUBSPACE KERKER.
State cache: ~/gwq-logs/flapw_perf_state_<tag>.pkl (keyed by config).
"""
import cProfile
import hashlib
import io
import os
import pickle
import pstats
import sys
import time
from pathlib import Path

import numpy as np
from _common import A_BOHR, ATOMS, RADII, env_float, env_int

from gradwave.flapw.scf import (
    _full_state,
    _multi_finalize,
    _multi_init_state,
    _multi_iterate,
    _multi_setup,
    crystal_scf_multi,
)


def cfg():
    return dict(
        ecut=env_float("PI", "ECUT", 300.0),
        lmax=env_int("PI", "LMAX", 3),
        fullpot=True,
        fullpot_lmax=env_int("PI", "FPLMAX", 4),
        kmesh=(env_int("PI", "K", 2),) * 3,
        smearing=0.0,
        use_symmetry=True,
        kerker=env_float("PI", "KERKER", 0.7),
        subspace_reuse=bool(env_int("PI", "SUBSPACE", 0)),
        kworkers=env_int("PI", "KWORKERS", 1),
        verbose=True,
    )


def tag(c):
    return (f"e{int(c['ecut'])}_l{c['lmax']}_fp{c['fullpot_lmax']}_k{c['kmesh'][0]}"
            f"_kk{c['kerker']}")


def state_path(c):
    return Path.home() / "gwq-logs" / f"flapw_perf_state_{tag(c)}.pkl"


def load_state(c):
    p = state_path(c)
    if not p.exists():
        raise SystemExit(f"no cached warm state at {p} — run `perf_iter.py warm` first")
    with open(p, "rb") as f:
        return pickle.load(f)


def make_ctx_state(c, warm):
    ctx = _multi_setup(a_bohr=A_BOHR, atoms=ATOMS, radii=RADII, **c)
    st = _multi_init_state(ctx, {"__full_state__": warm})
    return ctx, st


def mode_warm(c):
    t0 = time.perf_counter()
    _, info = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=env_int("PI", "ITERS", 30),
                                tol=1e-3, efg=False, **c)
    r = info["recorder"].summarize()
    print(f"warm run: {time.perf_counter() - t0:.1f}s n_iter={r['n_iter']} "
          f"r_nsph={r.get('r_nsph')}", flush=True)
    p = state_path(c)
    p.parent.mkdir(exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(info["state"], f)
    print(f"state cached: {p}")


def mode_profile(c):
    n = env_int("PI", "N", 3)
    ctx, st = make_ctx_state(c, load_state(c))
    _multi_iterate(ctx, st, 0, iters=200, tol=0.0)                  # warmup (JIT/caches)
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    for it in range(1, 1 + n):
        _multi_iterate(ctx, st, it, iters=200, tol=0.0)
    prof.disable()
    wall = time.perf_counter() - t0
    print(f"profiled {n} iterations: {wall:.2f}s total, {wall / n:.2f}s/iter", flush=True)
    for sort in ("cumulative", "tottime"):
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats(sort).print_stats(35)
        print(f"===== sorted by {sort} =====")
        print(s.getvalue())


def mode_bench(c):
    n = env_int("PI", "N", 12)
    ctx, st = make_ctx_state(c, load_state(c))
    _multi_iterate(ctx, st, 0, iters=200, tol=0.0)                  # warmup
    walls = []
    for it in range(1, 1 + n):
        t0 = time.perf_counter()
        _multi_iterate(ctx, st, it, iters=200, tol=0.0)
        walls.append(time.perf_counter() - t0)
    print("walls_s = [" + ", ".join(f"{w:.3f}" for w in walls) + "]")
    print(f"median {np.median(walls):.3f}s  mean {np.mean(walls):.3f}s  "
          f"min {min(walls):.3f}s  n={n}  kworkers={c['kworkers']}")
    # final-state hash: two builds benched from the same cached state must agree
    conv, info = _multi_finalize(ctx, st, efg=False)
    print(f"span={conv['span']:.6f}  state_sha={_state_sha(info['state'])}")


def mode_coldwarmup(c):
    t0 = time.perf_counter()
    _, info = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=22, tol=0.0, efg=False, **c)
    wall = time.perf_counter() - t0
    r = info["recorder"].summarize()
    print(f"cold 22-iter warmup: {wall:.1f}s  n_iter={r['n_iter']}  "
          f"state_sha={_state_sha(info['state'])}")


def _state_sha(state):
    h = hashlib.sha256()
    for k in sorted(state["v_by_key"]):
        h.update(np.ascontiguousarray(state["v_by_key"][k]).tobytes())
    if state.get("v_nsph"):
        for k in sorted(state["v_nsph"]):
            for lm in sorted(state["v_nsph"][k]):
                h.update(np.ascontiguousarray(state["v_nsph"][k][lm]).tobytes())
    if state.get("v_grid") is not None:
        h.update(np.ascontiguousarray(state["v_grid"]).tobytes())
    return h.hexdigest()[:16]


def mode_gate(c):
    """Bit-identity gate: N cold iterations, print the state hash after each."""
    n = env_int("PI", "N", 4)
    ctx = _multi_setup(a_bohr=A_BOHR, atoms=ATOMS, radii=RADII, **c)
    st = _multi_init_state(ctx, None)
    for it in range(n):
        _multi_iterate(ctx, st, it, iters=200, tol=0.0)
        print(f"it={it} state_sha={_state_sha(_full_state(ctx, st))}", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    c = cfg()
    print(f"# perf_iter mode={mode} cfg={tag(c)} kworkers={c['kworkers']} "
          f"OMP={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    {"warm": mode_warm, "profile": mode_profile, "bench": mode_bench,
     "coldwarmup": mode_coldwarmup, "gate": mode_gate}[mode](c)


if __name__ == "__main__":
    main()
