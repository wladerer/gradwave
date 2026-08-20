"""Newton-on-the-potential convergence probe (the "autograd-Newton" idea, FD-JVP stage).

Question: does Newton on the full FLAPW potential state collapse the iteration count of the
fullpot SCF, whose telemetry shows the aspherical channel (r_nsph) as the last, slowest gate?

Design: F(x) = one damped SCF iteration (``crystal_scf_multi(iters=1)`` with the full-state
in/out added for this probe), x = flatten(v_sph ⊕ v_nsph ⊕ interstitial grid ⊕ v_i0).
Newton-Krylov solves R(x) = F(x) − x = 0 matrix-free; every directional derivative is a
finite-difference of F, so each JVP costs exactly one F evaluation. That makes this probe an
UPPER BOUND on the cost of the real autograd version (an autograd JVP is exact and no more
expensive than F), while requiring zero torch surgery today. The honest cost unit is
F-evaluations, compared against the Anderson baseline's iteration count FROM THE SAME STATE.

Each F call re-runs the geometry/basis setup (~constant overhead); a production Newton would
hoist it. Wall time is reported but F-evals are the verdict.

Spawn-safe: nothing here uses process pools (kworkers=1), but keep the __main__ guard anyway.
"""
import os
import sys
import time

import numpy as np

from gradwave.flapw import crystal_scf_multi

U = 0.3048
A_BOHR = [8.68083, 8.68083, 5.59096]
ATOMS = [((0.0, 0.0, 0.0), "Ti"), ((0.5, 0.5, 0.5), "Ti"),
         ((U, U, 0.0), "O"), ((1 - U, 1 - U, 0.0), "O"),
         ((0.5 + U, 0.5 - U, 0.5), "O"), ((0.5 - U, 0.5 + U, 0.5), "O")]
RADII = {"Ti": 1.098, "O": 0.824}
CFG = dict(ecut=float(os.environ.get("NP_ECUT", "150")), lmax=2, kmesh=(1, 1, 1),
           smearing=0.0, efg=False, fullpot=True, fullpot_lmax=2, use_symmetry=True,
           kworkers=1, subspace_reuse=False)
GATE, TIGHT = 5e-2, 1e-3


def run(iters, v_start=None, tol=0.0):
    b, i = crystal_scf_multi(A_BOHR, ATOMS, RADII, iters=iters, tol=tol,
                             v_start=v_start, **CFG)
    return b, i


class Layout:
    """Flatten/unflatten the full state dict <-> one real vector."""

    def __init__(self, st):
        self.skeys = sorted(st["v_by_key"])
        self.nkeys = sorted(st["v_nsph"] or {})
        self.lms = {k: sorted(st["v_nsph"][k]) for k in self.nkeys}
        self.gshape = st["v_grid"].shape

    def flatten(self, st):
        parts = [np.asarray(st["v_by_key"][k], dtype=float).ravel() for k in self.skeys]
        for k in self.nkeys:
            for lm in self.lms[k]:
                v = np.asarray(st["v_nsph"][k][lm])
                parts += [v.real.ravel(), v.imag.ravel()]
        parts.append(np.asarray(st["v_grid"], dtype=float).ravel())
        parts.append(np.array([st["v_i0"]], dtype=float))
        return np.concatenate(parts)

    def unflatten(self, x, ref):
        st, off = {"v_by_key": {}, "v_nsph": {}, "v_grid": None, "v_i0": None}, 0
        for k in self.skeys:
            n = np.asarray(ref["v_by_key"][k]).size
            st["v_by_key"][k] = x[off:off + n].copy()
            off += n
        for k in self.nkeys:
            st["v_nsph"][k] = {}
            for lm in self.lms[k]:
                n = np.asarray(ref["v_nsph"][k][lm]).size
                st["v_nsph"][k][lm] = x[off:off + n] + 1j * x[off + n:off + 2 * n]
                off += 2 * n
        n = int(np.prod(self.gshape))
        st["v_grid"] = x[off:off + n].reshape(self.gshape).copy()
        off += n
        st["v_i0"] = float(x[off])
        return st


def main():
    t_all = time.time()
    # warmup must clear the MT->fullpot flip (it >= max(20, iters//2) at tol=0)
    # plus a few coupled iterations so v_nsph exists in the start state
    n0 = int(os.environ.get("NP_WARM", "24"))
    nbase = int(os.environ.get("NP_BASE", "40"))
    print(f"config: {CFG}", flush=True)

    # --- shared start state: n0 iterations from cold (MT staging + fullpot switch-on)
    t0 = time.time()
    _, i0 = run(n0)
    st0 = i0["state"]
    rec0 = i0["recorder"].iters[-1]
    print(f"warmup {n0} it ({time.time()-t0:.0f}s): r_v={rec0['r_v']:.2e} "
          f"r_nsph={rec0['r_nsph']:.2e}", flush=True)
    lay = Layout(st0)
    x0 = lay.flatten(st0)
    print(f"state dof = {x0.size}", flush=True)

    # --- baseline: the Anderson loop continued from st0
    t0 = time.time()
    _, ib = run(nbase, v_start={"__full_state__": st0})
    tr = ib["recorder"].to_trace_dict()["columns"]
    rn, rv = tr["r_nsph"], tr["r_v"]
    def first(cond):
        return next((j + 1 for j in range(len(rn)) if cond(j)), None)
    b_gate = first(lambda j: rn[j] is not None and rn[j] < GATE and rv[j] < 0.1)
    b_tight = first(lambda j: rn[j] is not None and rn[j] < TIGHT and rv[j] < 1e-2)
    print(f"baseline ({time.time()-t0:.0f}s): iters to gate={b_gate} tight={b_tight} "
          f"final r_nsph={rn[-1]:.2e} r_v={rv[-1]:.2e}", flush=True)

    # --- Newton-Krylov from the same state
    nev = {"n": 0}

    def F(x):
        nev["n"] += 1
        st = lay.unflatten(x, st0)
        _, ii = run(1, v_start={"__full_state__": st})
        return lay.flatten(ii["state"])

    def R(x):
        return F(x) - x

    from scipy.optimize import newton_krylov
    from scipy.optimize._nonlin import NoConvergence
    t0 = time.time()
    sol, err = None, None
    try:
        sol = newton_krylov(R, x0, method="lgmres", maxiter=8, inner_maxiter=12,
                            f_tol=1e-6, verbose=True)
    except NoConvergence as e:  # carries the last iterate; still report F-evals
        err, sol = e, np.asarray(e.args[0]) if e.args else None
    except Exception as e:
        err = e
    t_nk = time.time() - t0
    print(f"newton-krylov: {nev['n']} F-evals in {t_nk:.0f}s"
          + (f" (stopped: {type(err).__name__}: {err})" if err else ""), flush=True)

    # --- diagnose the NK solution with one plain iteration (its recorder = true residuals)
    if sol is not None and np.all(np.isfinite(sol)):
        stn = lay.unflatten(np.asarray(sol, dtype=float), st0)
        _, idg = run(1, v_start={"__full_state__": stn})
        d = idg["recorder"].iters[-1]
        print(f"NK solution residuals: r_v={d['r_v']:.2e} r_nsph={d['r_nsph']:.2e} "
              f"span={d['span']:.4f}", flush=True)
    print(f"total {time.time()-t_all:.0f}s", flush=True)
    print(f"VERDICT-INPUTS: baseline gate={b_gate} tight={b_tight} vs NK F-evals={nev['n']}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
