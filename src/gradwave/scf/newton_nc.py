"""Norm-conserving inexact-Newton (JFNK) SCF finisher.

Every density mixer is an approximate inverse of the SCF Jacobian
``J = χ₀·K_Hxc`` built from history (Pulay/Broyden) or a model susceptibility
(Kerker). The differentiable response machinery already owns the *exact*
Jacobian matvecs — ``scf.implicit.apply_chi0`` (the metal-capable Adler-Wiser
χ₀, with the Fermi-surface ∂f/∂ε window term and the charge-conserving δμ
rank-one term) and ``scf.implicit.apply_k_hxc`` (the Hartree + f_xc kernel via
an autograd HVP). So the true Newton step on the SCF fixed-point residual

    (I − χ₀ K_Hxc) δ = r ,     r = ρ_out − ρ_in ,     ρ_new = ρ_in + δ

is computable with no model, and the outer iteration converges quadratically
near the fixed point. This is the norm-conserving analog of the USPP/PAW
``postscf.newton.newton_polish`` finisher.

It is a *finisher*: each Newton step costs one raw SCF iteration (the residual
evaluation the loop already does) plus a matrix-free inner solve of the linear
Newton system, whose every iteration is a full Sternheimer batch (``apply_chi0``)
— far more than a mixed step. Two optional devices target that inner cost, both
tunable via ``newton_kwargs`` (their benefit is system-dependent — on an easy
metal whose inner solve is already cheap they are neutral-to-negative, so both
default conservatively; the payoff is on stiff systems where the inner solve
would otherwise take many applies per Newton step):

* **Eisenstat–Walker forcing** (``ew=True``) — the inner linear tolerance is made
  adaptive (loose while the outer Newton residual is large, tightening as it
  shrinks), so early Newton steps are not oversolved. Off by default: when the
  inner solve is already cheap it undersolves and costs extra outer iterations.
* **Kerker preconditioning** (``precond=True``, default on) — the inner Anderson
  iteration is preconditioned by
  the same inverse-dielectric operator the loop's mixer already uses
  (``R̃(G)=R(G)·G²/(G²+q0²)`` on the total-charge channel, q0 reused from the
  mixer). Kerker damps the strong negative Hartree charge-sloshing mode — the
  dominant eigenvalue of ``χ₀K``, magnitude > 1 even for a benign system (see
  ``scf.soft_mode``) — which is exactly what makes the un-preconditioned inner
  fixed point stiff, so it collapses the inner iteration count. It acts on the
  *charge* channel only.
* **Stoner spin preconditioning** (``spin_precond=True``, default off, nspin=2 +
  smearing only) — the magnetization companion to Kerker. Kerker leaves the
  magnetization/Stoner inner mode un-damped, so for a magnetic system that mode
  makes the inner solve expensive (or floors the density residual above rhotol).
  This adds the mixer's rank-r Woodbury Stoner operator ``(I − χ₀^diag K_mm)⁻¹``
  (``scf.spin_precond``) on the mag channel of the inner residual, built from the
  frozen orbitals. Off by default: like its mixing-path twin it is near-identity
  outside the FM-metal regime and depends on adequate Fermi-surface sampling, so
  its benefit is system-dependent — measure before enabling on a given system.

A related tuning caveat: the inner tolerance must be tight enough to reach the
target ``rhotol``, or the *density* residual floors above it and the finisher
stalls (correct energy — the fixed point is unchanged — but ``converged=False``,
with many wasted 1-apply outer steps). This bites the stiff nspin=2 case most; a
tighter inner tol is then often *cheaper* (far fewer stalled outer iterations).

The inner linear solve is a preconditioned Anderson fixed point of
``g(x) = r + M(x)``, ``M(x) = χ₀(K_Hxc(x))`` (density space) — the same operator
``postscf.convergence_error._dyson_solve`` dresses, solved to ``(I − M) δ = r``.

Reaches the SAME fixed point as any mixer (exact at convergence: r → 0 ⇒
δ → 0). It changes the root-finding class (linear fixed point → Newton), not the
converged density. Runs under ``no_grad`` like ``newton_polish``; the implicit-
function-theorem adjoint path (``scf.implicit``) is separate and untouched.

Requires ``use_symmetry=False`` (the response matvecs need the full TR-reduced
mesh — ``apply_chi0`` raises otherwise) and nspin 1 or 2.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import torch

from gradwave.core._anderson import AndersonMixer
from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.scf.implicit import _check_no_symmetry, apply_chi0, apply_k_hxc


def _frozen_state(system, xc, nspin, smearing, width, mu, rho_s, veff_s,
                  coeffs_s, eigs_s, occ_s):
    """A minimal attribute bag the response matvecs (``apply_chi0`` /
    ``apply_k_hxc``) consume — the SCF Jacobian frozen at the current input
    density ``rho_s`` with the orbitals that diagonalize ``H[rho_s]``.

    ``coeffs_s`` / ``eigs_s`` / ``occ_s`` are the per-spin lists the loop holds
    (nspin=1 packs a length-1 list; the matvecs want the bare per-k object for
    nspin=1 and the per-spin list for nspin=2, matching ``SCFResult``)."""
    rho_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    return SimpleNamespace(
        system=system,
        nspin=nspin,
        smearing=smearing,
        width=width,
        fermi=mu,
        v_eff=(veff_s[0] if nspin == 1 else veff_s),
        coeffs=(coeffs_s[0] if nspin == 1 else coeffs_s),
        eigenvalues=(eigs_s[0] if nspin == 1 else eigs_s),
        occupations=(occ_s[0] if nspin == 1 else occ_s),
        rho=rho_tot,
        rho_spin=(None if nspin == 1 else rho_s),
    )


def kerker_precond(grid, q0: float, nspin: int,
                   stoner=None) -> Callable[[torch.Tensor], torch.Tensor]:
    """Real-space Kerker preconditioner ``P ≈ (I − χ₀K)⁻¹`` on the density
    residual: ``R̃(G) = R(G)·G²/(G²+q0²)`` on the total-charge channel.

    This is the same inverse-dielectric factor ``scf.mixing`` applies (with q0
    taken from the loop's mixer), reused here on the full FFT box: it damps the
    long-wavelength charge-sloshing mode, the dominant (|λ|>1) eigenvalue of
    ``χ₀K``. The G=0 (total-charge) component is zeroed, which both matches the
    Kerker factor's ``g2→0`` limit and keeps the update charge-neutral (the
    residual already integrates to zero). For nspin=2 the factor acts on the
    total channel only, mirroring the mixer's kerker_mask.

    ``stoner`` (optional, nspin=2 only): a :class:`scf.spin_precond.StonerSpinPrecond`
    for the magnetization channel. Without it the mag residual passes through
    un-preconditioned — the un-damped Stoner-expansive spin mode that leaves the
    magnetic inner solve stiff. With it, the mag channel gets the same rank-r
    Woodbury Newton model ``(I − χ₀^diag K_mm)⁻¹`` the mixer's ``extra_precond``
    uses; the operator lives on the density sphere, so the real-space mag field
    is FFT'd, restricted to ``grid.dens_mask``, preconditioned, and transformed
    back (components outside the sphere pass through, where the operator is the
    identity anyway). Charge and mag are preconditioned independently, exactly as
    in the mixer's packed ``[charge, mag]`` vector.
    """
    fac = (grid.g2 / (grid.g2 + q0 * q0)).to(RDTYPE)

    def _kerker_field(f: torch.Tensor) -> torch.Tensor:
        return g_to_r_box(r_to_g(f.to(CDTYPE)) * fac, real=True)

    if nspin == 1:
        return _kerker_field

    mask = grid.dens_mask.reshape(-1)

    def _apply(r: torch.Tensor) -> torch.Tensor:
        tot = _kerker_field(r[0] + r[1])
        mag = r[0] - r[1]
        if stoner is not None:
            mag_g = r_to_g(mag.to(CDTYPE)).reshape(-1)
            out = mag_g.clone()
            out[mask] = stoner.apply(mag_g[mask])
            mag = g_to_r_box(out.reshape(grid.shape), real=True)
        return torch.stack([(tot + mag) * 0.5, (tot - mag) * 0.5])

    return _apply


@torch.no_grad()
def _precond_anderson(m_apply, vbar, precond, *, beta: float, tol: float,
                      max_iter: int, history: int):
    """Preconditioned Anderson fixed point of ``g(x) = vbar + M x`` — solves
    ``(I − M) δ = vbar``. ``precond`` (or None → identity) is applied to the
    system residual before each Anderson update; the convergence test uses the
    true (un-preconditioned) residual. Returns ``(u, n_iter, converged, resid,
    n_applies)`` where ``n_applies`` counts ``m_apply`` (= χ₀+K_Hxc) calls."""
    shape = vbar.shape
    nrm = max(1.0, float(torch.linalg.norm(vbar)))
    u = vbar.clone()
    mixer = AndersonMixer(history, beta)
    resid = float("inf")
    n_applies = 0
    for it in range(1, max_iter + 1):
        mu_x = m_apply(u)
        n_applies += 1
        r = vbar - u + mu_x
        resid = float(torch.linalg.norm(r)) / nrm
        if resid < tol:
            return u, it, True, resid, n_applies
        pr = r if precond is None else precond(r)
        u = mixer.step(u.reshape(-1), pr.reshape(-1)).reshape(shape)
        if not torch.isfinite(u).all():
            return u, it, False, float("inf"), n_applies
    return u, max_iter, False, resid, n_applies


@torch.no_grad()
def newton_delta_nc(jac, xc, r, *, inner_tol: float = 1e-6, max_inner: int = 60,
                    beta: float = 0.4, history: int = 8, chi0_tol: float = 1e-8,
                    chi0_max_iter: int = 200,
                    precond: Callable | None = None) -> tuple[torch.Tensor, dict]:
    """Solve ``(I − χ₀ K_Hxc) δ = r`` for the density-space Newton update ``δ``.

    ``jac`` is the frozen-Jacobian state (see :func:`_frozen_state`); ``r`` is
    the SCF residual ``ρ_out − ρ_in`` as a real-space field (nspin=1: a grid
    tensor; nspin=2: the stacked ``(2, *grid.shape)`` per-spin pair). ``precond``
    (optional) is applied to the inner Anderson residual — pass
    :func:`kerker_precond`. Returns ``(δ, info)`` where ``info`` carries the
    inner-solve diagnostics (``inner_iters``, ``inner_applies``, ``inner_resid``,
    ``inner_converged``).

    The inner matvec ``M(x) = χ₀(K_Hxc(x))`` is the SCF map's Jacobian block in
    density space; the preconditioned Anderson finds the fixed point of
    ``g(x) = r + M(x)``. Charge is conserved automatically: ``r`` is
    charge-neutral (ρ_out and ρ_in both integrate to N) and ``apply_chi0``
    returns charge-neutral fields, so ``δ`` stays neutral and ``∫ρ_new = N``."""
    _check_no_symmetry(jac)

    def m_apply(x):
        return apply_chi0(jac, apply_k_hxc(jac, xc, x),
                          tol=chi0_tol, max_iter=chi0_max_iter)

    u, n_it, conv, resid, n_ap = _precond_anderson(
        m_apply, r, precond, beta=beta, tol=inner_tol, max_iter=max_inner,
        history=history)
    info = dict(inner_iters=n_it, inner_applies=n_ap, inner_resid=resid,
                inner_converged=conv, inner_tol=inner_tol)
    return u, info


def _ew_tol(rnorm: float, prev_rnorm: float | None, *, gamma: float, alpha: float,
            floor: float, eta_max: float) -> float:
    """Eisenstat–Walker (choice 2) forcing tolerance for the inner solve:
    ``η = clamp(γ·(‖r_k‖/‖r_{k-1}‖)^α, floor, eta_max)``. The first Newton step
    (no previous residual) starts at ``eta_max`` (loosest), and η tightens
    toward ``floor`` as the outer residual collapses — so early steps, whose
    linearization is inaccurate anyway, are not oversolved."""
    if prev_rnorm is None or prev_rnorm <= 0.0:
        return eta_max
    eta = gamma * (rnorm / prev_rnorm) ** alpha
    return float(min(eta_max, max(floor, eta)))


@torch.no_grad()
def newton_finish_step(system, xc, nspin, smearing, width, mu, rho_s,
                       rho_out_s, veff_s, coeffs_s, eigs_s, occ_s, *,
                       mixer=None, prev_rnorm: float | None = None,
                       ew: bool = False, ew_gamma: float = 0.9,
                       ew_alpha: float = 2.0, ew_floor: float = 1e-8,
                       ew_max: float = 1e-1, precond: bool = True,
                       spin_precond: bool = False,
                       inner_tol: float = 1e-6, max_inner: int = 60,
                       beta: float = 0.4, history: int = 8,
                       chi0_tol: float = 1e-8, chi0_max_iter: int = 200,
                       ) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """One norm-conserving inexact-Newton density update, drop-in for a mixer
    step inside the SCF loop.

    ``rho_s`` is the per-spin input density list, ``rho_out_s`` the raw SCF map
    output ``F(ρ_in)`` for the same iteration; the remaining positional args are
    the loop's frozen orbitals/eigenvalues/occupations at ``ρ_in``.

    ``ew`` enables Eisenstat–Walker adaptive inner tolerance (``prev_rnorm`` is
    the previous Newton step's residual norm; None on the first step). ``precond``
    enables Kerker preconditioning of the inner solve, reusing ``mixer.q0`` from
    the loop's mixer. ``spin_precond`` (nspin=2 + smearing only) additionally
    preconditions the magnetization channel with the Woodbury Stoner operator
    (built from the frozen orbitals at ρ_in); ``None`` in the insulating limit,
    in which case the mag channel passes through. Returns ``(new_rho_s, info)`` —
    the per-spin density list ``ρ_in + δ`` and a diagnostics dict (``rnorm``,
    ``inner_tol``, ``inner_iters``, ``inner_applies``, ``inner_resid``,
    ``inner_converged``, ``preconditioned``, ``spin_preconditioned``)."""
    jac = _frozen_state(system, xc, nspin, smearing, width, mu, rho_s, veff_s,
                        coeffs_s, eigs_s, occ_s)
    if nspin == 1:
        r = (rho_out_s[0] - rho_s[0]).to(RDTYPE)
    else:
        r = torch.stack([(rho_out_s[isp] - rho_s[isp]).to(RDTYPE)
                         for isp in range(nspin)])
    rnorm = float(torch.linalg.norm(r))

    tol_eff = _ew_tol(rnorm, prev_rnorm, gamma=ew_gamma, alpha=ew_alpha,
                      floor=ew_floor, eta_max=ew_max) if ew else inner_tol

    p = None
    stoner = None
    if precond and mixer is not None and getattr(mixer, "kerker", False):
        if spin_precond and nspin == 2 and smearing != "none":
            # Precondition the magnetization channel too: without it the
            # Stoner-expansive spin mode is un-damped and the nspin=2 inner
            # solve dominates. Reuses the mixer's Woodbury Stoner operator,
            # built from the frozen orbitals/eigenvalues at ρ_in.
            from gradwave.core.occupations import SCHEMES
            from gradwave.scf.spin_precond import build_stoner_precond

            stoner = build_stoner_precond(
                system, coeffs_s, eigs_s, mu, SCHEMES[smearing], width,
                rho_s[0] + rho_s[1], rho_s[0] - rho_s[1], xc)
        p = kerker_precond(system.grid, float(mixer.q0), nspin, stoner=stoner)

    delta, info = newton_delta_nc(
        jac, xc, r, inner_tol=tol_eff, max_inner=max_inner, beta=beta,
        history=history, chi0_tol=chi0_tol, chi0_max_iter=chi0_max_iter,
        precond=p)
    info["rnorm"] = rnorm
    info["preconditioned"] = p is not None
    info["spin_preconditioned"] = stoner is not None

    if nspin == 1:
        new_rho_s = [rho_s[0] + delta]
    else:
        new_rho_s = [rho_s[isp] + delta[isp] for isp in range(nspin)]
    return new_rho_s, info
