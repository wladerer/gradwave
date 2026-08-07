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

from gradwave.scf.implicit import apply_chi0, apply_k_hxc

__all__ = [
    "SoftModeEstimate",
    "screening_apply",
    "dielectric_apply",
    "dominant_screening_eigenvalue",
    "max_real_screening_eigenvalue",
    "plain_fixed_point_rate",
]

# A response field is a real grid tensor (nspin=1) or a stacked per-spin pair
# (2, *grid.shape) (nspin=2). The operators below are field-in, field-out.
Field = torch.Tensor
LinOp = Callable[[Field], Field]


def screening_apply(res, xc, *, chi0_tol: float = 1e-7,
                    chi0_max_iter: int = 200) -> LinOp:
    """Matrix-free apply of the screening operator ``M = K_Hxc·χ₀``.

    Returns ``M(u) = K_Hxc[χ₀ u]``, the Jacobian of the fixed-point map
    ``solve_adjoint`` iterates. ``chi0_tol`` loosens the inner Sternheimer solve
    relative to the production adjoint default (1e-8): the dominant eigenvalue
    does not need a machine-tight χ₀ apply, and the looser tol keeps the
    diagnostic cheap.
    """
    def m_apply(u: Field) -> Field:
        return apply_k_hxc(res, xc, apply_chi0(res, u, tol=chi0_tol,
                                               max_iter=chi0_max_iter))
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
