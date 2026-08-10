"""Formalism-neutral soft-mode deflation core (SoftModeDeflate).

The linear-algebra half of SoftModeDeflate, decoupled from any DFT formalism:
every function here operates on an abstract matrix-free linear operator
``LinOp = Callable[[Field], Field]`` and a reference field, and knows nothing
about χ₀, the plane-wave basis, or norm-conserving vs USPP/PAW. It provides

- power iteration for the dominant / max-real eigenpair of ``M``
  (``_power_iterate``, the estimator the ``*_screening_eigenvalue`` wrappers use),
- non-symmetric Krylov extraction of the soft subspace — single-vector and
  block Arnoldi (``arnoldi_factorization``, ``soft_subspace_from_operator``),
- the deflated solve of ``(1 − M) u = v̄`` (Anderson smoothing of the
  well-conditioned complement + an exact coarse solve on the soft subspace,
  ``anderson_solve`` / ``deflated_solve``).

Because it lives in ``solvers`` (which may not import ``scf``/``postscf``), it is
importable by BOTH the norm-conserving wiring (``scf.soft_mode``) and the
USPP/PAW wiring (``postscf.uspp_softmode``): each supplies its own concrete
screening operator ``M`` as a ``LinOp`` over its own ``Field`` — a grid tensor
for norm-conserving, a flat composite (ρ, becsum, +U-occupation) vector for
USPP/PAW — and hands it to the same extractor and deflated solve unchanged.

``M = K·χ`` is non-symmetric (a product of symmetric operators, and non-normal
near criticality), so the extraction is Arnoldi, not symmetric Lanczos, and the
deflated solve makes no symmetry assumption on ``M``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from gradwave.core._anderson import AndersonMixer

__all__ = [
    "Field",
    "LinOp",
    "SoftModeEstimate",
    "SoftSubspace",
    "SolveResult",
    "arnoldi_factorization",
    "soft_subspace_from_operator",
    "anderson_solve",
    "deflated_solve",
]

# A response field is whatever the concrete operator acts on: a real grid tensor
# (norm-conserving), a stacked per-spin pair, or a flat composite vector
# (USPP/PAW). The core only requires elementwise ops, a Euclidean inner product,
# and a norm — so any real tensor works.
Field = torch.Tensor
LinOp = Callable[[Field], Field]


def _inner(a: Field, b: Field) -> float:
    return float((a * b).sum())


def _norm(a: Field) -> float:
    return float(torch.linalg.vector_norm(a))


def _rand_field_like(ref: Field, seed: int) -> Field:
    """Deterministic mean-zero random field shaped like ``ref``.

    Mean-subtracted because the response kernels pin the G=0 (constant) component
    to zero — K_Hxc annihilates a uniform shift — so the constant mode is a
    trivial null vector of ``M`` that would otherwise contaminate the power start.
    (For composite USPP/PAW vectors the mean-subtraction is harmless — the extra
    becsum/occupation blocks are near-zero-mean and the constant grid mode is
    still the one to suppress.)"""
    g = torch.Generator(device=ref.device).manual_seed(seed)
    v = torch.randn(ref.shape, generator=g, dtype=ref.dtype, device=ref.device)
    return v - v.mean()


@dataclass
class SoftModeEstimate:
    """An eigenpair of the screening operator ``M = K·χ``."""
    eigenvalue: float          # Rayleigh quotient ⟨v, Mv⟩/⟨v, v⟩ at convergence
    n_iter: int                # power iterations used
    residual: float            # ‖Mv − λv‖ / ‖v‖ (eigenpair quality)
    eigenvector: Field         # the approximate mode of M, unit norm


def _power_iterate(apply: LinOp, ref: Field, *, seed: int, n_iter: int,
                   tol: float) -> tuple[float, Field, float, int]:
    """Power iteration for the dominant (largest-magnitude) eigenpair of ``apply``.

    Returns ``(rayleigh, v, residual, used)`` where ``residual = ‖A v − λ v‖``
    (the un-normalised eigenpair defect, ``v`` being unit norm)."""
    v = _rand_field_like(ref, seed)
    v = v / torch.linalg.norm(v)
    lam = 0.0
    lam_prev = float("inf")
    resid = float("inf")
    used = 0
    while used < n_iter:
        used += 1
        av = apply(v)
        lam = float((v * av).sum())            # ⟨v, Av⟩, v is unit norm
        resid = float(torch.linalg.norm(av - lam * v))
        nrm = float(torch.linalg.norm(av))
        if nrm == 0.0:
            break
        v = av / nrm
        if abs(lam - lam_prev) < tol * max(1.0, abs(lam)):
            break
        lam_prev = lam
    return lam, v, resid, used


# ---------------------------------------------------------------------------
# Soft-subspace extraction (Arnoldi / block-Arnoldi).
#
# M = K·χ is non-symmetric (a product of two symmetric operators), and the
# matrix-free χ has no cheap square root, so a symmetric Lanczos on |χ|^½ K |χ|^½
# is impractical. Arnoldi extracts the same near-null / top-of-spectrum invariant
# subspace directly from the non-symmetric matvec, and — unlike a forced
# symmetrisation — it *exposes* non-normality as complex Ritz pairs rather than
# hiding it (the defective-mode signal the deflation must handle).
# ---------------------------------------------------------------------------


def arnoldi_factorization(apply: LinOp, v0: Field, *, krylov: int,
                          reorth: bool = True, breakdown: float = 1e-12
                          ) -> tuple[list[Field], torch.Tensor, int]:
    """``krylov``-step Arnoldi factorisation of ``apply`` from start field ``v0``.

    Modified Gram-Schmidt with one reorthogonalisation pass (``reorth``) for
    numerical stability at tight extraction tolerances. Returns ``(V, H, m)``:
    ``V`` the length-``m`` list of orthonormal basis fields, ``H`` the
    ``(krylov+1, krylov)`` upper-Hessenberg matrix with ``A V_m = V_m H[:m,:m]
    + h_{m+1,m} v_{m+1} e_m^T``, and ``m`` the reached dimension (``< krylov``
    only on a lucky breakdown, i.e. an exact invariant subspace)."""
    beta0 = _norm(v0)
    v_list = [v0 / beta0]
    h = torch.zeros(krylov + 1, krylov, dtype=torch.float64)
    reached = krylov
    for j in range(krylov):
        w = apply(v_list[j])
        for i in range(j + 1):
            hij = _inner(v_list[i], w)
            h[i, j] = hij
            w = w - hij * v_list[i]
        if reorth:
            for i in range(j + 1):
                s = _inner(v_list[i], w)
                h[i, j] += s
                w = w - s * v_list[i]
        hnext = _norm(w)
        h[j + 1, j] = hnext
        if hnext < breakdown:
            reached = j + 1
            break
        v_list.append(w / hnext)
    return v_list, h, reached


@dataclass
class SoftSubspace:
    """The extracted soft subspace of ``M = K·χ`` (top of the real spectrum).

    Selected Ritz pairs sorted by descending real part — the modes nearest (and
    past) the +1 instability that deflation removes. ``values`` are the (generally
    complex) Ritz values; ``residuals`` are true operator residuals ``‖M q − θ q‖``
    at the realified Ritz vectors; ``vectors`` are those unit-norm real fields;
    ``max_imag`` is the largest ``|Im|`` among the *selected* values — the
    non-normality warning (≈ 0 for a benign, near-normal spectrum)."""
    values: list[complex]
    residuals: list[float]
    vectors: list[Field]
    n_krylov: int
    max_imag: float


def _block_krylov_basis(apply: LinOp, ref: Field, *, block: int, steps: int,
                        seed: int, reorth: bool = True,
                        breakdown: float = 1e-10) -> list[Field]:
    """Orthonormal basis of the block-Krylov space ``span{V0, MV0, …, M^{s-1}V0}``.

    A block start of ``block`` vectors lets the extractor span an *exactly*
    ``block``-fold degenerate eigenspace, which a single Krylov start cannot: inside
    a scalar eigenblock ``M`` acts as ``λ·I``, so the powers ``M^j v0`` stay parallel
    and a single-vector Krylov space intersects the eigenspace in one dimension."""
    basis: list[Field] = []

    def _add(w: Field) -> None:
        for u in basis:
            w = w - _inner(u, w) * u
        if reorth:
            for u in basis:
                w = w - _inner(u, w) * u
        if _norm(w) > breakdown:
            basis.append(w / _norm(w))

    for b in range(block):
        _add(_rand_field_like(ref, seed + b))
    lo, hi = 0, len(basis)
    for _ in range(steps - 1):
        new_lo = len(basis)
        for i in range(lo, hi):
            _add(apply(basis[i]))
        lo, hi = new_lo, len(basis)
        if lo == hi:  # invariant subspace reached
            break
    return basis


def _projection(apply: LinOp, basis: list[Field]) -> torch.Tensor:
    """Rayleigh-Ritz projection ``T = Vᵀ M V`` over an orthonormal basis list."""
    mv = [apply(v) for v in basis]
    n = len(basis)
    t = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            t[i, j] = _inner(basis[i], mv[j])
    return t


def _select_ritz(v_list: list[Field], proj: torch.Tensor, apply: LinOp, *,
                 n_modes: int, real_tol: float) -> SoftSubspace:
    """Top-``n_modes`` real Ritz pairs of ``proj = Vᵀ M V`` by descending real part.

    Ritz vectors for real-ish values are realified (the real/imag part of larger
    norm) and their residuals recomputed against the true operator, not the cheap
    Arnoldi bound."""
    reached = proj.shape[0]  # single-vector Arnoldi returns v_list of len reached+1
    evals, evecs = torch.linalg.eig(proj)
    order = sorted(range(reached), key=lambda k: float(evals[k].real), reverse=True)

    values: list[complex] = []
    residuals: list[float] = []
    vectors: list[Field] = []
    max_imag = 0.0
    for k in order:
        if len(values) >= n_modes:
            break
        theta = complex(evals[k])
        if abs(theta.imag) > real_tol * (1.0 + abs(theta.real)):
            max_imag = max(max_imag, abs(theta.imag))
            continue
        y = evecs[:, k]
        yv = y.real if _norm(y.real) >= _norm(y.imag) else y.imag  # real e-vector
        q = sum((float(yv[i]) * v_list[i] for i in range(reached)),
                torch.zeros_like(v_list[0]))
        q = q / _norm(q)
        mq = apply(q)
        theta_r = _inner(q, mq)  # Rayleigh quotient at the realified vector
        values.append(complex(theta_r, 0.0))
        residuals.append(_norm(mq - theta_r * q))
        vectors.append(q)
        max_imag = max(max_imag, abs(theta.imag))

    return SoftSubspace(values=values, residuals=residuals, vectors=vectors,
                        n_krylov=reached, max_imag=max_imag)


def soft_subspace_from_operator(apply: LinOp, ref: Field, *, krylov: int = 24,
                                n_modes: int = 1, seed: int = 0,
                                real_tol: float = 1e-4,
                                block: int = 1) -> SoftSubspace:
    """Extract the top-``n_modes`` real eigenpairs of a linear operator ``apply``.

    Decoupled from any DFT operator so the linear algebra can be exercised on a
    synthetic operator with a planted spectrum, and reused by every formalism's
    screening operator. "Top" is by descending real part — the soft end that
    reaches +1 at an instability. A benign (near-normal) spectrum returns modes
    with ``max_imag ≈ 0`` and small residuals; a defective/near-critical spectrum
    shows a growing ``max_imag``.

    ``block = 1`` is single-vector Arnoldi (the default). ``block > 1`` runs
    block-Arnoldi from a ``block``-vector start, which is what robustly resolves an
    *exactly* ``k``-fold degenerate soft cluster (set ``block ≥ k``) — the
    symmetry-degenerate case a single Krylov start cannot span."""
    if block <= 1:
        v0 = _rand_field_like(ref, seed)
        v_list, h, reached = arnoldi_factorization(apply, v0, krylov=krylov)
        proj = h[:reached, :reached]
    else:
        steps = max(2, -(-krylov // block))  # ceil(krylov / block)
        v_list = _block_krylov_basis(apply, ref, block=block, steps=steps, seed=seed)
        proj = _projection(apply, v_list)
    return _select_ritz(v_list, proj, apply, n_modes=n_modes, real_tol=real_tol)


# ---------------------------------------------------------------------------
# Deflated solve of the response system (1 − M) u = v̄.
#
# Near an instability an eigenvalue of M reaches +1 (and past it, > 1: the
# "gain > 1" regime), so (1 − M) is near-singular / indefinite along the soft
# mode and the plain Anderson fixed point stalls or diverges (the NiO lesson).
# Deflation removes that mode exactly (a small dense coarse solve on the soft
# subspace Q) while Anderson smooths the well-conditioned complement — which
# still spans the wide spectrum (the strong negative Hartree modes), so Anderson
# smoothing, not damped Richardson, is what handles it.
#
#   "post" — Anderson smoothing step, THEN exact coarse correction
#   "pre"  — exact coarse correction, THEN Anderson smoothing step
# (the classic multigrid pre/post-smoothing fork). Both use Anderson; the
# baseline is plain Anderson with no deflation.
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    """Outcome of a response-system solve ``(1 − M) u = v̄``."""
    u: Field
    n_iter: int
    converged: bool
    residual: float          # ‖v̄ − (1 − M)u‖ / max(1, ‖v̄‖) at return


def _sys_residual(m_apply: LinOp, vbar: Field, u: Field) -> Field:
    """r = v̄ − (1 − M)u = v̄ − u + M u (also equals g(u) − u, g = v̄ + M·)."""
    return vbar - u + m_apply(u)


def _rel(r: Field, vbar: Field) -> float:
    return _norm(r) / max(1.0, _norm(vbar))


def anderson_solve(m_apply: LinOp, vbar: Field, *, beta: float = 0.4,
                   tol: float = 1e-8, max_iter: int = 200, history: int = 8
                   ) -> SolveResult:
    """Baseline: plain Anderson fixed point of ``g(u) = v̄ + M u`` (no deflation).

    Mirrors ``scf/implicit.py::solve_adjoint`` but on the supplied operator, so it
    is the honest control for the deflated variants. Reports non-convergence
    (rather than raising) so the near-critical stall/divergence is measurable."""
    shape = vbar.shape
    u = vbar.clone()
    mixer = AndersonMixer(history, beta)
    resid = float("inf")
    for it in range(1, max_iter + 1):
        r = _sys_residual(m_apply, vbar, u)
        resid = _rel(r, vbar)
        if resid < tol:
            return SolveResult(u, it, True, resid)
        u = mixer.step(u.reshape(-1), r.reshape(-1)).reshape(shape)
        if not torch.isfinite(u).all():
            return SolveResult(u, it, False, float("inf"))
    return SolveResult(u, max_iter, False, resid)


def _coarse_matrix(m_apply: LinOp, q: list[Field]) -> torch.Tensor:
    """Galerkin coarse operator ``A_c[i,j] = ⟨Q_i, (1 − M) Q_j⟩`` (k×k dense)."""
    k = len(q)
    im_mq = [q[j] - m_apply(q[j]) for j in range(k)]
    ac = torch.zeros(k, k, dtype=torch.float64)
    for i in range(k):
        for j in range(k):
            ac[i, j] = _inner(q[i], im_mq[j])
    return ac


def _coarse_correct(m_apply: LinOp, vbar: Field, u: Field, q: list[Field],
                    ac: torch.Tensor) -> Field:
    """Exact residual removal in the soft subspace: ``u += Q A_c⁻¹ Qᵀ r``."""
    r = _sys_residual(m_apply, vbar, u)
    rc = torch.tensor([_inner(q[i], r) for i in range(len(q))], dtype=torch.float64)
    # Prefer the direct solve — it is what the deflation was tuned against, and an
    # unconditional lstsq (min-norm, less aggressive near-singular) can slow the
    # deflated solve. Only when A_c is exactly singular — a null direction of M in
    # the subspace, the near-critical regime this correction targets — fall back to
    # the min-norm least-squares solution instead of raising _LinAlgError (which
    # made the near-critical soft-mode test crash on CI).
    try:
        e = torch.linalg.solve(ac, rc)
    except torch._C._LinAlgError:
        e = torch.linalg.lstsq(ac, rc).solution
    for i in range(len(q)):
        u = u + float(e[i]) * q[i]
    return u


def deflated_solve(m_apply: LinOp, vbar: Field, subspace: list[Field], *,
                   method: str = "post", beta: float = 0.4, tol: float = 1e-8,
                   max_iter: int = 200, history: int = 8) -> SolveResult:
    """Solve ``(1 − M) u = v̄`` deflating the soft subspace ``subspace`` (columns Q).

    ``method="post"`` smooths (Anderson) then coarse-corrects; ``method="pre"``
    coarse-corrects then smooths. Correctness is method-independent — the returned
    ``residual`` is the true system residual, so ``converged`` means ``u`` actually
    solves the equation regardless of variant."""
    if method not in ("post", "pre"):
        raise ValueError(f"method must be 'post' or 'pre', got {method!r}")
    shape = vbar.shape
    q = subspace
    ac = _coarse_matrix(m_apply, q)
    mixer = AndersonMixer(history, beta)

    u = torch.zeros_like(vbar)
    u = _coarse_correct(m_apply, vbar, u, q, ac)  # exact soft-mode starting solve
    resid = float("inf")
    for it in range(1, max_iter + 1):
        r = _sys_residual(m_apply, vbar, u)
        resid = _rel(r, vbar)
        if resid < tol:
            return SolveResult(u, it, True, resid)
        if method == "pre":
            u = _coarse_correct(m_apply, vbar, u, q, ac)
            r = _sys_residual(m_apply, vbar, u)
            u = mixer.step(u.reshape(-1), r.reshape(-1)).reshape(shape)
        else:  # post
            u = mixer.step(u.reshape(-1), r.reshape(-1)).reshape(shape)
            u = _coarse_correct(m_apply, vbar, u, q, ac)
        if not torch.isfinite(u).all():
            return SolveResult(u, it, False, float("inf"))
    return SolveResult(u, max_iter, False, resid)
