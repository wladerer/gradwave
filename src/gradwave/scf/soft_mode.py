"""Soft-mode diagnostics + deflated solve for the NORM-CONSERVING response.

The self-consistent response solve (``scf/implicit.py::solve_adjoint``) computes
the fixed point of

    g(u) = v̄ + M(u),      M(u) = K_Hxc[χ₀ u],

i.e. it solves the "dielectric" linear system ``(1 − M) u = v̄``. The fixed-point
map's Jacobian is exactly the screening operator ``M = K_Hxc·χ₀``, so the spectral
radius of ``M`` sets the (un-accelerated) contraction rate of the solve. Near a
spin/structural instability an eigenvalue of ``M`` approaches 1, ``(1 − M)`` goes
near-singular along the *soft mode*, and the plain fixed point stalls or diverges
(the documented NiO "gain > 1" lesson in ``solve_adjoint``/``uspp_implicit``).

This module is the norm-conserving *wiring*: it builds the concrete screening
operator ``M`` from the shipped NC response matvecs (``apply_chi0``,
``apply_k_hxc``) and hands it to the formalism-neutral deflation core in
``solvers.deflation`` (Arnoldi extraction, deflated solve). The USPP/PAW twin
lives in ``postscf.uspp_softmode`` and reuses the very same core over the
composite (ρ, becsum, +U-occupation) response operator. The core names are
re-exported here so existing imports (``from gradwave.scf.soft_mode import
soft_subspace_from_operator, deflated_solve, …``) keep working.

Validation gate (see ``tests``): on a benign insulator the dominant eigenvalue is
real, in ``(0, 1)`` (contractive) or of magnitude > 1 (the negative Hartree
charge mode), matching the measured plain fixed-point contraction rate.
"""

from __future__ import annotations

import torch

from gradwave.postscf._response import fxc_hvp, fxc_hvp_spin, hartree_kernel
from gradwave.scf.implicit import apply_chi0, apply_k_hxc
from gradwave.solvers.deflation import (
    Field,
    LinOp,
    SoftModeEstimate,
    SoftSubspace,
    SolveResult,
    _power_iterate,
    _rand_field_like,
    anderson_solve,
    arnoldi_factorization,
    deflated_solve,
    soft_subspace_from_operator,
)

__all__ = [
    # re-exported formalism-neutral core (backward-compatible import surface)
    "Field",
    "LinOp",
    "SoftModeEstimate",
    "SoftSubspace",
    "SolveResult",
    "arnoldi_factorization",
    "soft_subspace_from_operator",
    "anderson_solve",
    "deflated_solve",
    # norm-conserving screening operator + diagnostics
    "screening_apply",
    "dielectric_apply",
    "dominant_screening_eigenvalue",
    "max_real_screening_eigenvalue",
    "plain_fixed_point_rate",
    "soft_subspace",
    "critical_coupling",
    "solve_adjoint_deflated",
]


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

    This is the operator ``(1 − M) u = v̄`` that the deflation preconditions; kept
    here so callers build on one definition."""
    m = screening_apply(res, xc, **kw)

    def l_apply(u: Field) -> Field:
        return u - m(u)
    return l_apply


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
    mode, O(1) at a defective/coalescing mode (the non-normal pathology).
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
    and < 0 in the "gain > 1" regime. This is the eigenvalue deflation removes.

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


def soft_subspace(res, xc, *, krylov: int = 24, n_modes: int = 1, seed: int = 0,
                  real_tol: float = 1e-4, block: int = 1, **kw) -> SoftSubspace:
    """Soft subspace of ``M = K_Hxc·χ₀`` for a converged SCF result.

    Thin convenience over :func:`solvers.deflation.soft_subspace_from_operator`
    bound to the DFT screening operator. ``block`` selects single-vector (1) or
    block-Arnoldi (≥ the degeneracy) extraction; ``**kw`` forwards to
    :func:`screening_apply` (e.g. ``chi0_tol``, ``fxc_scale``)."""
    return soft_subspace_from_operator(screening_apply(res, xc, **kw), res.rho,
                                       krylov=krylov, n_modes=n_modes, seed=seed,
                                       real_tol=real_tol, block=block)


def critical_coupling(res, xc, **kw) -> float:
    """Coupling ``c`` at which the soft mode of ``c·M`` reaches +1 (``1/λ_max``).

    Convenience over :func:`max_real_screening_eigenvalue`; a benign material is
    driven critical at this coupling and unstable ("gain > 1") above it."""
    lam = max_real_screening_eigenvalue(res, xc, **kw).eigenvalue
    return float("inf") if lam <= 0 else 1.0 / lam


def solve_adjoint_deflated(res, xc, vbar: Field, *, subspace: list[Field] | None = None,
                           auto_threshold: float = 0.9, max_modes: int = 8,
                           krylov: int = 30, seed: int = 0, beta: float = 0.4,
                           tol: float = 1e-8, max_iter: int = 200, history: int = 8,
                           **screen_kw) -> SolveResult:
    """Deflated drop-in for ``scf/implicit.py::solve_adjoint``.

    Solves ``(1 − M) u = v̄`` for ``M = K_Hxc·χ₀`` and returns a ``SolveResult``.
    Away from criticality it is Anderson (the plain solve), and it becomes the
    deflated solve only when a soft cluster is present — so it is never worse than
    the baseline and decisively better near an instability with a degenerate soft
    cluster.

    Subspace selection, cheapest first:
    - ``subspace`` given → deflate exactly those modes (RECYCLE a cluster extracted
      once and reused across an EOS / phonon-stencil / temperature sweep, where the
      soft mode moves slowly). Skips the per-solve extraction cost.
    - otherwise AUTO → extract the top ``max_modes`` real Ritz modes of ``M`` and
      keep those above ``auto_threshold`` (near-critical). If none qualify (a benign
      system), fall back to plain Anderson.

    ``**screen_kw`` forwards to :func:`screening_apply` (``chi0_tol``, and the
    ``coupling`` / ``fxc_scale`` instability knobs for near-critical testbeds).
    """
    m = screening_apply(res, xc, **screen_kw)
    if subspace is None:
        sub = soft_subspace_from_operator(m, res.rho, krylov=krylov,
                                          n_modes=max_modes, seed=seed)
        subspace = [v for v, val in zip(sub.vectors, sub.values, strict=True)
                    if val.real > auto_threshold]
    if not subspace:
        return anderson_solve(m, vbar, beta=beta, tol=tol, max_iter=max_iter,
                              history=history)
    return deflated_solve(m, vbar, subspace, method="post", beta=beta, tol=tol,
                          max_iter=max_iter, history=history)
