"""Newton-Krylov polish of the FLAPW fixed point (the validated sledgehammer).

Measured verdict (experiments/autoapw/newton_probe.py, asus 2026-08-19): on easy fixed
points the joint Anderson loop wins (3-7 iterations vs ~30 F-evaluations) — but on the
production-hard TiO2 config (ecut 300, lmax 3, fullpot_lmax 4, k 2³, 590k dof) Anderson
STALLS (r_v plateau at 0.46 for 50 iterations) while Newton-Krylov converges in 3 Newton
steps / 32 F-evaluations to r_nsph ~1e-9. So the production pattern is:

    bands, info = crystal_scf_multi(..., iters=N)          # Anderson as far as it goes
    if anderson_stalled(info["recorder"]):
        state, ninfo = newton_polish(a, atoms, radii, info["state"], scf_kwargs={...})
        bands, info = crystal_scf_multi(..., iters=1, efg=True,
                                        v_start={"__full_state__": state})  # observables

F is one damped SCF iteration chained through the full state (``__full_state__`` in/out),
so this needs no new physics: Newton solves R(x) = F(x) − x = 0 matrix-free with
finite-difference directional derivatives (each JVP = one F evaluation; an autograd JVP
would be exact and cheaper — planned once the residual map is torch-clean end to end).
The geometry/basis setup (atomic solves, space group, Θ_I mask) is paid ONCE per polish:
F calls the hoisted iterate phase (``scf._multi_iterate``) on a shared ``scf._multi_setup``
context instead of re-running ``crystal_scf_multi`` end to end — same map, bit-identical
values, with the per-evaluation setup cost removed.

Determinism: F must be deterministic for FD JVPs to be meaningful — keep ``kworkers``
modest (the pool accumulates in mesh order, so it is bit-stable) and do not enable
``subspace_reuse`` in scf_kwargs (exact solves only).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gradwave.flapw.recorder import FLAPWRecorder
from gradwave.flapw.scf import (
    _full_state,
    _multi_init_state,
    _multi_iterate,
    _multi_setup,
    _shutdown_ctx_pool,
)

__all__ = ["StateLayout", "anderson_stalled", "newton_polish"]

# The full fixed-point state dict (info["state"] from crystal_scf_multi):
# v_by_key / v_nsph / v_grid / v_i0. Heterogeneous by construction.
State = dict[str, Any]


class StateLayout:
    """Flatten/unflatten the ``info["state"]`` dict <-> one real vector.

    The layout is frozen from a reference state: spherical radial potentials per
    symmetry-distinct atom key, then re/im of each aspherical (l,m) channel, then
    the interstitial grid, then the interstitial reference scalar.
    """

    def __init__(self, st: State) -> None:
        self.skeys: list[str] = sorted(st["v_by_key"])
        self.nkeys: list[str] = sorted(st["v_nsph"] or {})
        self.lms: dict[str, list[Any]] = {k: sorted(st["v_nsph"][k]) for k in self.nkeys}
        self.gshape: tuple[int, ...] = st["v_grid"].shape

    def flatten(self, st: State) -> np.ndarray:
        parts = [np.asarray(st["v_by_key"][k], dtype=float).ravel() for k in self.skeys]
        for k in self.nkeys:
            for lm in self.lms[k]:
                v = np.asarray(st["v_nsph"][k][lm])
                parts += [v.real.ravel(), v.imag.ravel()]
        parts.append(np.asarray(st["v_grid"], dtype=float).ravel())
        parts.append(np.array([st["v_i0"]], dtype=float))
        return np.concatenate(parts)

    def unflatten(self, x: np.ndarray, ref: State) -> State:
        st: State = {"v_by_key": {}, "v_nsph": {}, "v_grid": None, "v_i0": None}
        off = 0
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

    def segments(self, ref: State) -> list[tuple[str, int, int]]:
        """(label, start, stop) spans of the flattened vector, one per channel.

        Labels: ``sph:<key>``, ``nsph:<key>:<lm>`` (re+im, hence 2n wide),
        ``interstitial``, ``v_i0`` — in the frozen ``flatten`` order, so a mode
        vector can be attributed to physical channels by slicing.
        """
        spans: list[tuple[str, int, int]] = []
        off = 0
        for k in self.skeys:
            n = np.asarray(ref["v_by_key"][k]).size
            spans.append((f"sph:{k}", off, off + n))
            off += n
        for k in self.nkeys:
            for lm in self.lms[k]:
                n = np.asarray(ref["v_nsph"][k][lm]).size
                spans.append((f"nsph:{k}:{lm}", off, off + 2 * n))
                off += 2 * n
        n = int(np.prod(self.gshape))
        spans.append(("interstitial", off, off + n))
        off += n
        spans.append(("v_i0", off, off + 1))
        return spans


def anderson_stalled(recorder: FLAPWRecorder, window: int = 8, gate: float = 0.1) -> bool:
    """True when the Anderson loop's r_v has plateaued above the gate.

    The measured stall signature (hard TiO2) is r_v hovering at O(0.1-1) with no
    downward trend: the last ``window`` exact-solve iterations neither reach the
    gate nor improve the running best by 2x.
    """
    rows = [it for it in recorder.iters if it.get("exact_solve") and it.get("r_v") is not None]
    if len(rows) < window:
        return False
    tail = [float(it["r_v"]) for it in rows[-window:]]
    if min(tail) < gate:
        return False
    best_before = min(float(it["r_v"]) for it in rows[:-window])
    return min(tail) > 0.5 * best_before


def newton_polish(a_bohr: Any, atoms: Any, radii: dict[str, float], state: State, *,
                  scf_kwargs: dict[str, Any], maxiter: int = 6, inner_maxiter: int = 12,
                  f_tol: float = 1e-8, rounds: int = 3,
                  verbose: bool = False) -> tuple[State, dict[str, Any]]:
    """Polish a full FLAPW state to the coupled fixed point by Newton-Krylov.

    ``a_bohr``/``atoms``/``radii`` are passed through to ``crystal_scf_multi``
    unchanged (same accepted forms). ``state`` is ``info["state"]`` from a
    (possibly stalled) ``crystal_scf_multi`` run; ``scf_kwargs`` are that run's
    keyword arguments (ecut/lmax/kmesh/fullpot/...). ``iters``/``tol``/``v_start``
    are overridden internally. Returns ``(polished_state, info)`` with
    info = {"f_evals", "converged", "residual_norm", "rounds"}. Raises ValueError
    for a non-fullpot state with no interstitial grid (nothing to polish that
    Anderson cannot already do).

    Robustness (measured, TiO2 hard config, 2026-08-20): a single newton_krylov
    call on this map is CHAOTIC-SENSITIVE — ulp-level rounding differences in F
    flipped outcomes between residual 5e+3 (divergence) and 1.4e-3 (near-converged)
    on the same input state. So the polish runs up to ``rounds`` restarts with
    MONOTONE ACCEPTANCE: each round restarts newton_krylov from the best iterate
    so far (never a worse one), later rounds jitter the FD step (``rdiff``) to
    redraw the trajectory, and the returned state is never worse than the input.
    A near-miss round (like the measured 1.4e-3) becomes the next round's start,
    which is exactly the regime where the quadratic phase re-engages.
    """
    if state.get("v_grid") is None:
        raise ValueError("newton_polish needs a full (fullpot) state with v_grid")
    from scipy.optimize import NoConvergence, newton_krylov

    kw = dict(scf_kwargs)
    kw.pop("iters", None)
    kw.pop("tol", None)
    kw.pop("v_start", None)
    kw.setdefault("subspace_reuse", False)
    # F consumes only info["state"]; efg is a post-loop diagnostic that cannot change the
    # state, so it is dropped rather than recomputed ~30 times.
    kw.pop("efg", None)
    # Fast path: pay the state-independent setup (atomic solves, space group, Θ_I mask, …)
    # ONCE, then each F evaluation is init-state + one iterate on the shared context —
    # bit-identical to a full crystal_scf_multi(iters=1, v_start={"__full_state__": ...}).
    ctx = _multi_setup(a_bohr=a_bohr, atoms=atoms, radii=radii, **kw)
    lay = StateLayout(state)
    x0 = lay.flatten(state)
    n_evals = 0

    def F(x: np.ndarray) -> np.ndarray:
        nonlocal n_evals
        n_evals += 1
        st = _multi_init_state(ctx, {"__full_state__": lay.unflatten(x, state)})
        _multi_iterate(ctx, st, 0, iters=1, tol=0.0)
        return lay.flatten(_full_state(ctx, st))

    def residual(x: np.ndarray) -> np.ndarray:
        return F(x) - x

    # None = scipy's default FD step; later rounds redraw the trajectory with a
    # different rdiff (the measured chaos means the step size is a dice roll, so
    # reroll it rather than repeating the same throw).
    rdiffs = [None, 1e-6, 3e-8, 1e-5, 1e-7]
    try:
        best_x, best_r = x0, float(np.linalg.norm(residual(x0)))
        converged, n_rounds = False, 0
        for rd in rdiffs[:max(1, rounds)]:
            n_rounds += 1
            kw_rd = {} if rd is None else {"rdiff": rd}
            try:
                sol = newton_krylov(residual, best_x, method="lgmres", maxiter=maxiter,
                                    inner_maxiter=inner_maxiter, f_tol=f_tol,
                                    verbose=verbose, **kw_rd)
                converged = True
            except NoConvergence as e:
                sol = np.asarray(e.args[0]) if e.args else best_x
            rn = float(np.linalg.norm(residual(sol)))
            if converged:                   # scipy hit f_tol: this iterate IS the answer
                best_x, best_r = np.asarray(sol), rn
                break
            if rn < best_r:
                best_x, best_r = np.asarray(sol), rn
    finally:
        _shutdown_ctx_pool(ctx)             # the F evaluations share one persistent pool
    return lay.unflatten(best_x, state), {
        "f_evals": n_evals, "converged": converged, "residual_norm": best_r,
        "rounds": n_rounds}
