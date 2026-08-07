"""Soft-mode diagnostics for the near-critical response solve (SoftModeDeflate P0).

The self-consistent response solve (``scf/implicit.py::solve_adjoint``) computes
the fixed point of

    g(u) = v̄ + M(u),      M(u) = K_Hxc[χ₀ u],

i.e. it solves the "dielectric" linear system ``(1 − M) u = v̄``. The fixed-point
map's Jacobian is exactly the screening operator ``M = K_Hxc·χ₀``, so the spectral
radius of ``M`` sets the (un-accelerated) contraction rate of the solve. Near a
spin/structural instability an eigenvalue of ``M`` approaches 1, ``(1 − M)`` goes
near-singular along the *soft mode*, and the plain fixed point stalls or diverges
(the documented NiO "gain > 1" lesson in ``solve_adjoint``/``uspp_implicit``).

This module is P0 of the SoftModeDeflate plan (docs/ideas.md): expose ``M`` and
``(1 − M)`` as explicit matrix-free linear operators over the shipped response
matvecs (``apply_chi0``, ``apply_k_hxc``), and estimate the dominant eigenvalue of
``M`` by power iteration — the "softness" indicator that later phases (symmetric
Lanczos extraction, then deflation) build on. It computes nothing new physically;
it is a lens on the operator ``solve_adjoint`` already iterates.

Validation gate (see ``tests``): on a benign insulator the dominant eigenvalue is
real, in ``(0, 1)`` (contractive), and equals the measured plain fixed-point
contraction rate — proving the operator wrappers and the estimator are correct and
that the number means what the later phases need it to mean.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from gradwave.postscf._response import fxc_hvp, fxc_hvp_spin, hartree_kernel
from gradwave.scf.implicit import apply_chi0, apply_k_hxc

__all__ = [
    "SoftModeEstimate",
    "SoftSubspace",
    "screening_apply",
    "dielectric_apply",
    "dominant_screening_eigenvalue",
    "max_real_screening_eigenvalue",
    "plain_fixed_point_rate",
    "arnoldi_factorization",
    "soft_subspace_from_operator",
    "soft_subspace",
    "SolveResult",
    "anderson_solve",
    "deflated_solve",
    "critical_coupling",
]

# A response field is a real grid tensor (nspin=1) or a stacked per-spin pair
# (2, *grid.shape) (nspin=2). The operators below are field-in, field-out.
Field = torch.Tensor
LinOp = Callable[[Field], Field]


def _k_hxc_fxc_scaled(res, xc, w_r: Field, fxc_scale: float) -> Field:
    """``(K_H + s·f_xc) w`` — the Hxc kernel with only the *xc* part scaled by ``s``.

    Mirrors ``scf/implicit.py::apply_k_hxc`` exactly (nspin, NLCC core split, the
    total-density Hartree coupling) but multiplies the ``f_xc`` HVP by ``s``, so
    the Hartree charge spectrum is untouched. ``s = 1`` is byte-identical to
    ``apply_k_hxc``."""
    grid = res.system.grid
    core = res.system.rho_core
    nspin = getattr(res, "nspin", 1)
    if nspin == 1:
        rho_xc = res.rho if core is None else res.rho + core
        return hartree_kernel(grid, w_r) + fxc_scale * fxc_hvp(xc, rho_xc, grid, w_r)
    kh = hartree_kernel(grid, w_r[0] + w_r[1])
    c2 = 0.0 if core is None else 0.5 * core
    assert res.rho_spin is not None
    fu, fd = fxc_hvp_spin(xc, res.rho_spin[0] + c2, res.rho_spin[1] + c2,
                          grid, w_r[0], w_r[1])
    return torch.stack([kh + fxc_scale * fu, kh + fxc_scale * fd])


def screening_apply(res, xc, *, coupling: float = 1.0, fxc_scale: float = 1.0,
                    chi0_tol: float = 1e-7, chi0_max_iter: int = 200) -> LinOp:
    """Matrix-free apply of the screening operator ``M = c · (K_H + s·f_xc)·χ₀``.

    Returns the Jacobian of the fixed-point map ``solve_adjoint`` iterates (at
    ``c = s = 1``). ``chi0_tol`` loosens the inner Sternheimer solve relative to
    the production adjoint default (1e-8): the eigenvalue estimates do not need a
    machine-tight χ₀ apply, and the looser tol keeps the diagnostic cheap.

    Two instability knobs, both physically-informed devices for driving a benign
    material's *real* response operator toward criticality (a mode goes soft when
    ``λ(M)`` reaches 1):

    - ``coupling`` (``c``) scales the *whole* Hxc kernel (the RPA λ-scaling). It
      reaches criticality but also amplifies the strong negative Hartree charge
      modes, so at large ``c`` the smoother, not the deflation, is the bottleneck.
    - ``fxc_scale`` (``s``) scales *only* the xc kernel — the physically correct
      knob for a Stoner / soft-phonon instability, which is xc-driven — leaving the
      Hartree charge spectrum fixed. This is the clean near-critical testbed: the
      soft mode rises toward +1 while the charge modes stay put.

    Both leave the eigenvectors of the affected block essentially in place, so a
    soft subspace extracted at one scale deflates a nearby scale unchanged.
    """
    def m_apply(u: Field) -> Field:
        chi0u = apply_chi0(res, u, tol=chi0_tol, max_iter=chi0_max_iter)
        khxc = (apply_k_hxc(res, xc, chi0u) if fxc_scale == 1.0
                else _k_hxc_fxc_scaled(res, xc, chi0u, fxc_scale))
        return coupling * khxc if coupling != 1.0 else khxc
    return m_apply


def dielectric_apply(res, xc, **kw) -> LinOp:
    """Matrix-free apply of the response operator ``L = 1 − M = 1 − K_Hxc·χ₀``.

    This is the operator ``(1 − M) u = v̄`` that the deflation in later phases
    preconditions; kept here so P1/P2 build on one definition."""
    m = screening_apply(res, xc, **kw)

    def l_apply(u: Field) -> Field:
        return u - m(u)
    return l_apply


def _rand_field_like(ref: Field, seed: int) -> Field:
    """Deterministic mean-zero random field shaped like ``ref``.

    Mean-subtracted because the kernels pin the G=0 (constant) component to zero
    — K_Hxc annihilates a uniform shift — so the constant mode is a trivial null
    vector of ``M`` that would otherwise contaminate the power start."""
    g = torch.Generator(device=ref.device).manual_seed(seed)
    v = torch.randn(ref.shape, generator=g, dtype=ref.dtype, device=ref.device)
    return v - v.mean()


@dataclass
class SoftModeEstimate:
    """An eigenpair of the screening operator ``M = K_Hxc·χ₀``."""
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


def dominant_screening_eigenvalue(res, xc, *, n_iter: int = 60,
                                  tol: float = 1e-5, seed: int = 0,
                                  **kw) -> SoftModeEstimate:
    """Dominant-*magnitude* eigenvalue of ``M = K_Hxc·χ₀`` (its spectral radius).

    ``M`` is non-symmetric (a product of two symmetric operators), so power
    iteration converges to the largest-magnitude eigenpair. For a real crystal
    this is the strong negative Hartree charge-sloshing mode, and its magnitude
    exceeds 1 even for a benign insulator — which is exactly why ``solve_adjoint``
    cannot use the plain undamped fixed point (it would diverge) and uses Anderson
    mixing instead. This is the diagnostic that explains the solver's damping, not
    the soft-mode indicator; for the latter (the eigenvalue that reaches +1 at an
    instability) see :func:`max_real_screening_eigenvalue`.

    The ``residual`` is the genuine-eigenpair check: small for a well-separated
    mode, O(1) at a defective/coalescing mode (the non-normal pathology P1 handles
    by symmetrising).
    """
    lam, v, resid, used = _power_iterate(screening_apply(res, xc, **kw), res.rho,
                                         seed=seed, n_iter=n_iter, tol=tol)
    return SoftModeEstimate(eigenvalue=lam, n_iter=used, residual=resid,
                            eigenvector=v)


def max_real_screening_eigenvalue(res, xc, *, n_iter: int = 120,
                                  tol: float = 1e-5, seed: int = 0,
                                  shift: float | None = None,
                                  **kw) -> SoftModeEstimate:
    """Maximal *real* eigenvalue of ``M = K_Hxc·χ₀`` — the soft-mode indicator.

    The response solve ``(1 − M) u = v̄`` goes singular when an eigenvalue of ``M``
    reaches +1, so the maximal real eigenvalue ``λ_max`` is the distance-to-
    instability indicator: ``1 − λ_max`` is the margin, positive and O(1) for a
    benign system, → 0 at an incipient ferroelectric / CDW / magnetic instability,
    and < 0 in the "gain > 1" regime. This is the eigenvalue the deflation in
    later phases removes.

    Found by a spectral shift: power-iterate on ``M + σI`` with ``σ`` large enough
    that every shifted eigenvalue is positive (so the largest *value* λ_max+σ is
    also the largest *magnitude*), then subtract σ. ``σ`` defaults to just above
    the spectral radius from :func:`dominant_screening_eigenvalue`.
    """
    m = screening_apply(res, xc, **kw)
    if shift is None:
        rho = dominant_screening_eigenvalue(res, xc, n_iter=n_iter, tol=tol,
                                            seed=seed, **kw).eigenvalue
        shift = abs(rho) + 0.5
    sigma = shift

    def shifted(u: Field) -> Field:
        return m(u) + sigma * u

    mu, v, _, used = _power_iterate(shifted, res.rho, seed=seed + 1,
                                    n_iter=n_iter, tol=tol)
    lam_max = mu - sigma
    # residual of the ORIGINAL operator at the recovered mode
    mv = m(v)
    resid = float(torch.linalg.norm(mv - lam_max * v))
    return SoftModeEstimate(eigenvalue=lam_max, n_iter=used, residual=resid,
                            eigenvector=v)


def plain_fixed_point_rate(res, xc, *, vbar: Field | None = None,
                           n_iter: int = 40, tail: int = 8, seed: int = 1,
                           **kw) -> float:
    """Measured asymptotic contraction rate of the *un-accelerated* fixed point.

    Runs the plain (undamped) iteration ``u ← v̄ + M u`` and returns the median
    successive-residual ratio ``‖r_{n+1}‖/‖r_n‖`` over the last ``tail`` steps.
    For a linear map the error obeys ``e_{n+1} = M e_n``, so this ratio → the
    spectral radius of ``M`` — the same magnitude ``dominant_screening_eigenvalue``
    estimates, obtained an independent way (via the actual iteration rather than the
    Rayleigh quotient). When the spectral radius exceeds 1 (the usual case, driven
    by the Hartree charge mode) the residual *grows* at this ratio: the undamped
    iteration diverges, which is precisely why ``solve_adjoint`` uses Anderson. The
    modest ``n_iter`` keeps the growing residual well inside float64 range.
    """
    m = screening_apply(res, xc, **kw)
    if vbar is None:
        vbar = _rand_field_like(res.rho, seed)
    u = torch.zeros_like(vbar)
    ratios: list[float] = []
    prev = None
    for _ in range(n_iter):
        r = vbar + m(u) - u
        rn = float(torch.linalg.norm(r))
        if prev is not None and prev > 0:
            ratios.append(rn / prev)
        prev = rn
        u = vbar + m(u)
    if not ratios:
        return float("nan")
    last = sorted(ratios[-tail:])
    return last[len(last) // 2]


# ---------------------------------------------------------------------------
# P1: soft-subspace extraction.
#
# M = K_Hxc·χ₀ is non-symmetric (a product of two symmetric operators), and the
# matrix-free χ₀ has no cheap square root, so the plan's literal "symmetric
# Lanczos on |χ₀|^½ K |χ₀|^½" is impractical. Arnoldi extracts the same near-null
# / top-of-spectrum invariant subspace directly from the non-symmetric matvec,
# and — unlike a forced symmetrisation — it *exposes* non-normality as complex
# Ritz pairs rather than hiding it (the defective-mode signal the deflation in P2
# must handle). This is the honest matrix-free route to the soft subspace.
# ---------------------------------------------------------------------------

def _inner(a: Field, b: Field) -> float:
    return float((a * b).sum())


def _norm(a: Field) -> float:
    return float(torch.linalg.vector_norm(a))


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
    """The extracted soft subspace of ``M = K_Hxc·χ₀`` (top of the real spectrum).

    Selected Ritz pairs sorted by descending real part — the modes nearest (and
    past) the +1 instability that the deflation in P2 removes. ``values`` are the
    (generally complex) Ritz values; ``residuals`` are true operator residuals
    ``‖M q − θ q‖`` at the realified Ritz vectors; ``vectors`` are those unit-norm
    real fields; ``max_imag`` is the largest ``|Im|`` among the *selected* values —
    the non-normality warning (≈ 0 for a benign, near-normal spectrum)."""
    values: list[complex]
    residuals: list[float]
    vectors: list[Field]
    n_krylov: int
    max_imag: float


def soft_subspace_from_operator(apply: LinOp, ref: Field, *, krylov: int = 24,
                                n_modes: int = 1, seed: int = 0,
                                real_tol: float = 1e-4) -> SoftSubspace:
    """Extract the top-``n_modes`` real eigenpairs of a linear operator by Arnoldi.

    Core of :func:`soft_subspace`, decoupled from the DFT operator so the
    linear-algebra can be exercised on a synthetic operator with a planted
    spectrum. "Top" is by descending real part — the soft end that reaches +1 at
    an instability. Ritz vectors for real-ish Ritz values are realified (the
    real/imag part of larger norm, which for a real eigenvalue of a real operator
    is itself a real eigenvector) and residuals are recomputed against the true
    operator, not the cheap Arnoldi bound. A benign (near-normal) spectrum returns
    modes with ``max_imag ≈ 0`` and small residuals; a defective/near-critical
    spectrum shows a growing ``max_imag`` (the P2 signal)."""
    m = apply
    v0 = _rand_field_like(ref, seed)
    v_list, h, reached = arnoldi_factorization(m, v0, krylov=krylov)
    hm = h[:reached, :reached]
    evals, evecs = torch.linalg.eig(hm)  # complex

    # rank Ritz values by descending real part
    order = sorted(range(reached), key=lambda k: float(evals[k].real),
                   reverse=True)

    values: list[complex] = []
    residuals: list[float] = []
    vectors: list[Field] = []
    max_imag = 0.0
    for k in order:
        if len(values) >= n_modes:
            break
        theta = complex(evals[k])
        if abs(theta.imag) > real_tol * (1.0 + abs(theta.real)):
            # complex (defective/rotating) mode — record its imag as the warning
            # but keep scanning for the requested number of real modes
            max_imag = max(max_imag, abs(theta.imag))
            continue
        y = evecs[:, k]
        yv = y.real if _norm(y.real) >= _norm(y.imag) else y.imag  # real e-vector
        q = sum((float(yv[i]) * v_list[i] for i in range(reached)),
                torch.zeros_like(v_list[0]))
        q = q / _norm(q)
        mq = m(q)
        theta_r = _inner(q, mq)  # Rayleigh quotient at the realified vector
        values.append(complex(theta_r, 0.0))
        residuals.append(_norm(mq - theta_r * q))
        vectors.append(q)
        max_imag = max(max_imag, abs(theta.imag))

    return SoftSubspace(values=values, residuals=residuals, vectors=vectors,
                        n_krylov=reached, max_imag=max_imag)


def soft_subspace(res, xc, *, krylov: int = 24, n_modes: int = 1, seed: int = 0,
                  real_tol: float = 1e-4, **kw) -> SoftSubspace:
    """Soft subspace of ``M = K_Hxc·χ₀`` for a converged SCF result.

    Thin convenience over :func:`soft_subspace_from_operator` bound to the DFT
    screening operator; ``**kw`` forwards to :func:`screening_apply` (e.g.
    ``chi0_tol``)."""
    return soft_subspace_from_operator(screening_apply(res, xc, **kw), res.rho,
                                       krylov=krylov, n_modes=n_modes, seed=seed,
                                       real_tol=real_tol)


# ---------------------------------------------------------------------------
# P2: deflated solve of the response system (1 − M) u = v̄.
#
# Near an instability an eigenvalue of M reaches +1 (and past it, > 1: the
# "gain > 1" regime), so (1 − M) is near-singular / indefinite along the soft
# mode and the plain Anderson fixed point stalls or diverges (the NiO lesson).
# Deflation removes that mode exactly (a small dense coarse solve on the soft
# subspace Q) while Anderson smooths the well-conditioned complement — which
# still spans the wide spectrum (the strong negative Hartree modes), so Anderson
# smoothing, not damped Richardson, is what handles it.
#
# Two variants of where the coarse correction sits in the cycle are compared:
#   "post" — Anderson smoothing step, THEN exact coarse correction
#   "pre"  — exact coarse correction, THEN Anderson smoothing step
# (the classic multigrid pre/post-smoothing fork). Both use Anderson so both
# handle the wide spectrum; the baseline is plain Anderson with no deflation.
# ---------------------------------------------------------------------------

from gradwave.core._anderson import AndersonMixer  # noqa: E402


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
    e = torch.linalg.solve(ac, rc)
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


def critical_coupling(res, xc, **kw) -> float:
    """Coupling ``c`` at which the soft mode of ``c·M`` reaches +1 (``1/λ_max``).

    Convenience over :func:`max_real_screening_eigenvalue`; a benign material is
    driven critical at this coupling and unstable ("gain > 1") above it."""
    lam = max_real_screening_eigenvalue(res, xc, **kw).eigenvalue
    return float("inf") if lam <= 0 else 1.0 / lam
