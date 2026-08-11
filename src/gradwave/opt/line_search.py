"""Parallel / speculative line search for the nested geometry relax.

A quasi-Newton relax computes a search direction ``dₖ`` at each ionic step and
then takes a *fixed* step along it — BFGS uses ``α = 1`` (capped by ``maxstep``),
without ever searching the step length. That fixed step is a sequential
bottleneck: a single guess that a soft or overshoot-prone mode routinely gets
wrong, wasting the whole ionic step's SCF.

This module replaces that fixed step (opt-in, positions-only, bfgs) with ONE
*parallel* round. It evaluates ``Rₖ + αᵢ·dₖ`` for a spread of step lengths
``αᵢ`` CONCURRENTLY — each a forward SCF warm-started from ``Rₖ``'s checkpoint,
returning both the energy ``Eᵢ`` and the force ``Fᵢ`` — fits a cubic through the
``(αᵢ, Eᵢ, gᵢ)`` samples (``gᵢ = −Fᵢ·dₖ`` the projected gradient) and steps to
the interpolated minimum ``α*``. It is the relax member of the campaign-parallel
family: the candidates fan out over :func:`gradwave.postscf.seedpool.map_spokes`
(a spawn-based process pool), so it is FORWARD-ONLY — PyTorch autograd graphs are
process-local and never cross the boundary, and a differentiable relax must stay
on the serial path.

Exactness. The line search only changes the PATH, never the minimum: the accepted
geometry is re-evaluated by the *main* calculator at full SCF tolerance on the
next ionic step's force call (which also gates convergence and records the frame),
so a ``parallel``/``adaptive`` relax lands on the same minimum as a plain serial
BFGS relax to the geometry/energy tolerance. The parallel candidate SCFs are used
ONLY to pick ``α*``.

Hessian consistency. The wrapper subclasses ASE's :class:`~ase.optimize.BFGS` and
overrides only :meth:`~ParallelLineSearchBFGS.step`; it reuses the base
``prepare_step``/``determine_step`` verbatim to obtain the BFGS-proposed step and
to record ``(pos0, forces0)``. The base's next-step ``update`` therefore forms the
curvature pair from the ACTUAL displacement taken (``α*·dₖ``) and the force change,
so the quasi-Newton Hessian stays consistent regardless of the line-searched
length.

Everything here is pure/light except :meth:`ParallelLineSearchBFGS.step`, which
dispatches the candidate SCFs. The cubic-fit math (:func:`fit_cubic`,
:func:`cubic_line_min`, :func:`choose_alpha`) and the adaptive trigger
(:func:`struggle_trigger`) are dependency-free (numpy only) and unit-tested
without any SCF.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Pure math: alpha schedule, cubic fit from (E, g), interpolated minimum.
# ---------------------------------------------------------------------------


def default_alpha_schedule(n_samples: int, max_alpha: float) -> tuple[float, ...]:
    """A geometric α ladder around the proposed step (α = 1).

    ``n_samples`` points spaced by a factor of two and centred so ``1.0`` is
    always included (e.g. ``n=3 → {0.5, 1, 2}``, ``n=4 → {0.25, 0.5, 1, 2}``),
    each clamped to ``max_alpha`` (a large α crosses a bigger geometry change →
    a worse warm-start seed → a costlier candidate SCF). Duplicates that clamping
    collapses are dropped, order preserved.
    """
    n_samples = max(2, int(n_samples))
    exps = [i - (n_samples // 2) for i in range(n_samples)]
    out: list[float] = []
    for e in exps:
        a = min(2.0**e, float(max_alpha))
        if a > 0.0 and a not in out:
            out.append(a)
    return tuple(out)


def fit_cubic(
    alphas: Any, energies: Any, grads: Any
) -> np.ndarray:
    """Least-squares cubic ``p(α) = c₀ + c₁α + c₂α² + c₃α³`` through the samples.

    Each sample contributes BOTH a value row (``[1, α, α², α³]·c = E``) and a
    derivative row (``[0, 1, 2α, 3α²]·c = g``), so two samples already
    over-determine the four coefficients and more samples improve the fit. Uses
    the projected gradient ``g = dE/dα`` as first-derivative data — the reason the
    candidate SCFs return forces, not just energies. Returns ``(c₀, c₁, c₂, c₃)``.
    """
    a = np.asarray(alphas, dtype=float).ravel()
    e = np.asarray(energies, dtype=float).ravel()
    g = np.asarray(grads, dtype=float).ravel()
    val = np.stack([np.ones_like(a), a, a**2, a**3], axis=1)
    der = np.stack([np.zeros_like(a), np.ones_like(a), 2.0 * a, 3.0 * a**2], axis=1)
    amat = np.concatenate([val, der], axis=0)
    bvec = np.concatenate([e, g], axis=0)
    coeffs, *_ = np.linalg.lstsq(amat, bvec, rcond=None)
    return coeffs


def cubic_line_min(
    coeffs: Any, lo: float, hi: float
) -> tuple[float | None, bool]:
    """Interpolated minimiser of the cubic on the open bracket ``(lo, hi)``.

    Returns ``(α*, True)`` for a genuine interior local minimum (a stationary
    point with ``p''>0`` strictly inside the bracket), else ``(None, False)`` —
    the fit is a line, non-convex, or its minimum lies outside the sampled
    bracket, in which case the caller falls back to the best sampled α. This is
    the GUARD that keeps a bad extrapolation from taking an unsampled step.
    """
    c0, c1, c2, c3 = (float(c) for c in coeffs)
    if abs(c3) < 1e-14:  # degenerate cubic → parabola / line
        if c2 <= 1e-14:  # non-convex or linear: no reliable interior minimum
            return None, False
        a_star = -c1 / (2.0 * c2)
        return (a_star, True) if lo < a_star < hi else (None, False)
    # p'(α) = c1 + 2c2 α + 3c3 α² = 0 ; p''(α) = 2c2 + 6c3 α
    disc = (2.0 * c2) ** 2 - 4.0 * (3.0 * c3) * c1
    if disc < 0.0:
        return None, False
    sq = float(np.sqrt(disc))
    roots = ((-2.0 * c2 + sq) / (6.0 * c3), (-2.0 * c2 - sq) / (6.0 * c3))
    best: tuple[float, float] | None = None
    for r in roots:
        if lo < r < hi and (2.0 * c2 + 6.0 * c3 * r) > 0.0:  # interior local min
            p = c0 + c1 * r + c2 * r**2 + c3 * r**3
            if best is None or p < best[1]:
                best = (r, p)
    return (best[0], True) if best is not None else (None, False)


def choose_alpha(
    samples: list[tuple[float, float, float]],
    *,
    max_alpha: float,
) -> tuple[float, str]:
    """Pick the step length α from ``(α, E, g)`` samples (α=0 anchor allowed).

    Fits the cubic and returns its interior minimum when reliable and inside
    ``(0, min(max_sampled_α, max_alpha)]``; otherwise falls back to the best
    ACTUALLY-SAMPLED positive α (lowest energy). The α=0 anchor (the current
    point, free) is never taken — the relax must move. Returns ``(α, reason)``
    with ``reason ∈ {"cubic", "fallback"}``.
    """
    positive = [s for s in samples if s[0] > 0.0]
    if not positive:
        return 1.0, "fallback"
    hi = min(max(s[0] for s in positive), float(max_alpha))
    coeffs = fit_cubic(
        [s[0] for s in samples], [s[1] for s in samples], [s[2] for s in samples]
    )
    a_star, ok = cubic_line_min(coeffs, 0.0, hi)
    if ok and a_star is not None and 0.0 < a_star <= hi:
        return float(a_star), "cubic"
    best = min(positive, key=lambda s: s[1])
    return float(best[0]), "fallback"


# ---------------------------------------------------------------------------
# Adaptive trigger: is the optimizer struggling?
# ---------------------------------------------------------------------------


def struggle_trigger(
    progress: list[tuple[float, float]], *, patience: int = 1
) -> bool:
    """Whether the optimizer is STRUGGLING and should fan a line search out.

    ``progress`` is the accepted-step history as ``(energy, max|F|)`` pairs,
    oldest→newest. Fires when EITHER the energy increased on the last accepted
    step (a rejected/overshot step), OR ``max|F|`` failed to decrease across the
    last ``patience`` steps (a stall). Stays dormant while the relax makes
    monotone progress (an easy bulk relax), so ``adaptive`` pays for the parallel
    round only where a fixed step is likely wrong.
    """
    if len(progress) < 2:
        return False
    e_prev, _ = progress[-2]
    e_cur, _ = progress[-1]
    if e_cur > e_prev + 1e-12:  # energy increased on the last accepted step
        return True
    window = progress[-(int(patience) + 1):]
    if len(window) >= 2 and window[-1][1] >= window[0][1]:  # max|F| stalled
        return True
    return False


# ---------------------------------------------------------------------------
# Candidate worker (runs in a spawned worker process, picklable).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """One line-search candidate: a forward SCF at ``positions``, warm-started
    from the reference ``checkpoint`` (Rₖ). All fields are picklable plain data —
    the live calculator/System is rebuilt inside the worker, nothing heavy
    crosses the process boundary (see seedpool)."""

    inp: Any  # gradwave.inputs.Input
    positions: np.ndarray  # (natoms, 3) Cartesian [Å]
    checkpoint: str  # path to Rₖ's checkpoint


def _evaluate_candidate(cand: _Candidate) -> tuple[float, np.ndarray]:
    """Forward SCF at ``cand.positions`` warm-started from Rₖ's checkpoint;
    return ``(energy_eV, forces)`` with ``forces`` shaped ``(natoms, 3)``.

    Runs in a spawned worker: imports gradwave fresh, rebuilds the GradWave
    calculator from the (picklable) Input, injects the checkpoint-derived
    warm-start seed for the single candidate SCF, and returns only small arrays.
    """
    from gradwave.api import _build_relax_calc
    from gradwave.io.checkpoint import as_start_from, load_checkpoint

    inp = cand.inp
    atoms = inp.atoms.copy()
    atoms.set_positions(np.asarray(cand.positions, dtype=float))
    if inp.fixed is not None:  # mirror _relax_nested's selective-dynamics mask
        from ase.constraints import FixScaled

        atoms.set_constraint([
            FixScaled(i, mask=tuple(bool(b) for b in inp.fixed[i]))
            for i in range(len(atoms)) if bool(inp.fixed[i].any())
        ])
    calc = _build_relax_calc(inp, verbose=False)
    # one-shot seed: the candidate SCF warm-starts from Rₖ's converged density
    calc._warm_start_override = as_start_from(load_checkpoint(cand.checkpoint))
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return energy, forces


def _ls_init_worker(n_threads: int) -> None:
    """Pool initializer: pin this candidate worker to ``n_threads`` intra-op
    threads so N concurrent candidate SCFs don't each grab every core (the same
    thread-budget split as ``seedpool._init_worker``). Module-level so it is
    picklable for the spawn context."""
    import os

    os.environ["GRADWAVE_NUM_THREADS"] = str(max(1, int(n_threads)))
    try:
        import torch

        torch.set_num_threads(max(1, int(n_threads)))
    except Exception:  # pragma: no cover - torch is always present in the relax path
        pass


# ---------------------------------------------------------------------------
# The ASE optimizer wrapper.
# ---------------------------------------------------------------------------


def make_line_search_bfgs(target: Any, *, inp: Any, verbose: bool) -> Any:
    """Construct a :class:`ParallelLineSearchBFGS` over ``target`` configured
    from ``inp.relax``. The default path (``line_search == "off"``) never reaches
    here — the caller uses a plain :class:`~ase.optimize.BFGS` — so this is only
    built for ``parallel``/``adaptive``."""
    opt = ParallelLineSearchBFGS(target, logfile=None)
    r = inp.relax
    opt._ls_configure(
        inp=inp,
        mode=r.line_search,
        n_samples=r.line_search_n_samples,
        n_workers=r.line_search_n_workers,
        alphas=tuple(r.line_search_alphas),
        max_alpha=r.line_search_max_alpha,
        patience=r.line_search_patience,
        warmup=r.line_search_warmup,
        warmup_samples=r.line_search_warmup_samples,
        verbose=verbose,
    )
    return opt


try:  # ASE is a hard dependency of the relax path; import lazily-safe for tools
    from ase.optimize import BFGS as _BFGS
except Exception:  # pragma: no cover - ASE always present in the relax context
    _BFGS = object  # type: ignore[assignment,misc]


class ParallelLineSearchBFGS(_BFGS):  # type: ignore[valid-type,misc]
    """BFGS whose per-step fixed length is replaced by a parallel line search.

    Overrides only :meth:`step`; ``prepare_step``/``determine_step``/``update``
    and the Hessian state are the stock BFGS, so the quasi-Newton update is
    unchanged and consistent with the actual (line-searched) displacement. Gate
    the feature at construction (see :func:`make_line_search_bfgs`); with the
    feature off the caller uses plain BFGS and this class is never instantiated,
    keeping the default relax byte-identical.
    """

    def _ls_configure(
        self,
        *,
        inp: Any,
        mode: str,
        n_samples: int,
        n_workers: int,
        alphas: tuple[float, ...],
        max_alpha: float,
        patience: int,
        verbose: bool,
        warmup: int = 0,
        warmup_samples: int = 0,
    ) -> None:
        self.ls_inp = inp
        self.ls_mode = mode
        self.ls_n_samples = int(n_samples)
        self.ls_n_workers = int(n_workers)
        self.ls_alphas = tuple(alphas)
        self.ls_max_alpha = float(max_alpha)
        self.ls_patience = int(patience)
        self.ls_verbose = bool(verbose)
        self.ls_warmup = int(warmup)
        self.ls_warmup_samples = int(warmup_samples)
        self._ls_progress: list[tuple[float, float]] = []
        self._ls_events: list[dict[str, Any]] = []
        self._ls_pool: Any = None  # persistent candidate pool, created on first use

    def _ls_in_warmup(self) -> bool:
        return bool(self.ls_warmup) and self.nsteps < self.ls_warmup

    def _ls_should_search(self) -> bool:
        # PREDICTIVE: force a search on the first `warmup` steps (Hessian least
        # informed → overshoot likeliest), regardless of mode.
        if self._ls_in_warmup():
            return True
        if self.ls_mode == "parallel":
            return True
        if self.ls_mode == "adaptive":
            return struggle_trigger(self._ls_progress, patience=self.ls_patience)
        return False

    def _ls_map(self, cands: list[Any]) -> list[tuple[float, np.ndarray]]:
        """Evaluate the candidate SCFs. ``n_workers <= 1`` (or a single candidate)
        runs them serially in-process; otherwise they fan out over a PERSISTENT
        worker pool that lives for the whole relax — created once, reused every
        ionic step — so spawn + torch re-import is paid once, not per step (the
        old per-step ``map_spokes`` pool made ``n_workers>1`` a net loss on cheap
        SCFs)."""
        from gradwave.postscf.seedpool import resolve_workers

        w = resolve_workers(self.ls_n_workers, len(cands))
        if w <= 1 or len(cands) <= 1:
            return [_evaluate_candidate(c) for c in cands]
        pool = self._ls_get_pool(w)
        futs = [pool.submit(_evaluate_candidate, c) for c in cands]
        return [f.result() for f in futs]

    def _ls_get_pool(self, w: int) -> Any:
        """Lazily create — and thereafter reuse — the spawn-based candidate pool.
        Each worker's intra-op thread budget is split off the parent's (matching
        seedpool) so N candidate SCFs don't each grab every core."""
        if self._ls_pool is not None:
            return self._ls_pool
        import concurrent.futures as cf
        import multiprocessing as mp

        import torch

        from gradwave.postscf.seedpool import worker_thread_cap

        cap = worker_thread_cap(torch.get_num_threads(), w)
        self._ls_pool = cf.ProcessPoolExecutor(
            max_workers=w, mp_context=mp.get_context("spawn"),
            initializer=_ls_init_worker, initargs=(cap,))
        if self.ls_verbose:
            print(f"    line-search: persistent pool of {w} workers "
                  f"({cap} threads each)", flush=True)
        return self._ls_pool

    def _ls_close_pool(self) -> None:
        """Shut the persistent candidate pool down (idempotent). The relax driver
        calls this when the optimization finishes; ``__del__`` is a safety net."""
        pool = getattr(self, "_ls_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
            self._ls_pool = None

    def __del__(self) -> None:
        try:
            self._ls_close_pool()
        except Exception:  # pragma: no cover - interpreter-teardown races
            pass

    def step(self, gradient=None):  # type: ignore[override]
        gradient = self._get_gradient(gradient)
        optimizable = self.optimizable
        pos = optimizable.get_x()
        # stock BFGS: update the Hessian, record (pos0, forces0), form the step
        dpos, steplengths = self.prepare_step(pos, gradient)
        dpos = self.determine_step(dpos, steplengths)  # BFGS proposed step (α=1)

        # record (energy, max|F|) for the adaptive trigger before deciding
        forces = -np.asarray(gradient, dtype=float).ravel()
        fmax = float(np.linalg.norm(forces.reshape(-1, 3), axis=1).max())
        energy = float(self.atoms.get_potential_energy())
        self._ls_progress.append((energy, fmax))

        alpha, how = 1.0, "serial"
        if self._ls_should_search():
            alpha, how = self._line_search(pos, dpos, forces, energy)
        self._ls_events.append(
            {"nstep": int(self.nsteps), "searched": how != "serial",
             "alpha": float(alpha), "how": how})
        if self.ls_verbose and how != "serial":
            print(f"    line-search: α = {alpha:.4f} ({how})", flush=True)

        optimizable.set_x(pos + alpha * dpos)
        from ase.optimize.optimize import UnitCellFilter

        if isinstance(self.atoms, UnitCellFilter):
            self.dump((self.state.hessian, self.pos0, self.forces0,
                       self.maxstep, self.atoms.orig_cell))
        else:
            self.dump((self.state.hessian, self.pos0, self.forces0,
                       self.maxstep))

    def _line_search(
        self, pos: np.ndarray, dpos: np.ndarray, forces: np.ndarray, energy: float
    ) -> tuple[float, str]:
        """Fan the candidates out, fit the cubic, return ``(α*, reason)``.

        Any failure (no reference checkpoint, a worker error, an empty schedule)
        falls back to the ordinary BFGS step (``α = 1``) so the relax always
        makes progress; the reason is recorded for the summary/parent."""
        res = getattr(self.atoms.calc, "last_result", None)
        if res is None:
            return 1.0, "serial(no-ref)"
        na = len(self.atoms)
        d = np.asarray(dpos, dtype=float).reshape(na, 3)
        base = np.asarray(pos, dtype=float).reshape(na, 3)
        n_samples = self.ls_n_samples
        if self._ls_in_warmup() and self.ls_warmup_samples:
            n_samples = self.ls_warmup_samples  # denser bracket while overshoot-prone
        alphas = self.ls_alphas or default_alpha_schedule(
            n_samples, self.ls_max_alpha)
        if not alphas:
            return 1.0, "serial(no-alphas)"

        from gradwave.io.checkpoint import save_checkpoint

        tmpdir = tempfile.mkdtemp(prefix="gw_ls_")
        try:
            ckpt = os.path.join(tmpdir, "ref.pt")
            save_checkpoint(res, ckpt)
            cands = [_Candidate(self.ls_inp, base + a * d, ckpt) for a in alphas]
            try:
                out = self._ls_map(cands)
            except Exception:  # noqa: BLE001 - worker/pool failure → serial step
                return 1.0, "serial(worker-error)"
            # α=0 anchor is free: the current point's E and projected gradient
            #   g(α) = dE/dα = ∇E·d = −F·d  (descent direction ⇒ g(0) < 0)
            samples: list[tuple[float, float, float]] = [
                (0.0, float(energy), -float(np.dot(forces, np.asarray(dpos).ravel())))
            ]
            for a, (e_i, f_i) in zip(alphas, out, strict=True):
                g_i = -float(np.dot(np.asarray(f_i, dtype=float).ravel(),
                                    np.asarray(dpos).ravel()))
                samples.append((float(a), float(e_i), g_i))
            return choose_alpha(samples, max_alpha=self.ls_max_alpha)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
