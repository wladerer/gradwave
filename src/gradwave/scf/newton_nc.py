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
— far more than a mixed step. It earns that cost only once the state is inside
the quadratic basin: switch to it below a residual threshold and it lands the
last few orders of magnitude in 1-3 steps.

The inner linear solve reuses ``solvers.deflation.anderson_solve`` on the
composition ``M(x) = χ₀(K_Hxc(x))`` (density space), exactly the operator
``postscf.convergence_error._dyson_solve`` dresses, solved to the same fixed
point ``(I − M) δ = r``.

Reaches the SAME fixed point as any mixer (it is exact at convergence: r → 0 ⇒
δ → 0). It changes the root-finding class (linear fixed point → Newton), not the
converged density. Runs under ``no_grad`` like ``newton_polish``; the implicit-
function-theorem adjoint path (``scf.implicit``) is separate and untouched.

Requires ``use_symmetry=False`` (the response matvecs need the full TR-reduced
mesh — ``apply_chi0`` raises otherwise) and nspin 1 or 2.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from gradwave.dtypes import RDTYPE
from gradwave.scf.implicit import _check_no_symmetry, apply_chi0, apply_k_hxc
from gradwave.solvers.deflation import anderson_solve


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


@torch.no_grad()
def newton_delta_nc(jac, xc, r, *, inner_tol: float = 1e-6, max_inner: int = 60,
                    beta: float = 0.4, history: int = 8,
                    chi0_tol: float = 1e-8, chi0_max_iter: int = 200):
    """Solve ``(I − χ₀ K_Hxc) δ = r`` for the density-space Newton update ``δ``.

    ``jac`` is the frozen-Jacobian state (see :func:`_frozen_state`); ``r`` is
    the SCF residual ``ρ_out − ρ_in`` as a real-space field (nspin=1: a grid
    tensor; nspin=2: the stacked ``(2, *grid.shape)`` per-spin pair). Returns
    ``(δ, SolveResult)`` in the same layout as ``r``.

    The inner matvec is ``M(x) = χ₀(K_Hxc(x))`` — the SCF map's Jacobian block
    in density space — and ``anderson_solve`` finds the fixed point of
    ``g(x) = r + M(x)`` (i.e. solves ``(I − M) δ = r``). Charge is conserved
    automatically: ``r`` is charge-neutral (both ρ_out and ρ_in integrate to
    N) and ``apply_chi0`` returns charge-neutral fields, so ``δ`` stays neutral
    and ``∫ρ_new = N`` to machine precision."""
    _check_no_symmetry(jac)

    def m_apply(x):
        return apply_chi0(jac, apply_k_hxc(jac, xc, x),
                          tol=chi0_tol, max_iter=chi0_max_iter)

    sol = anderson_solve(m_apply, r, beta=beta, tol=inner_tol,
                         max_iter=max_inner, history=history)
    return sol.u, sol


@torch.no_grad()
def newton_finish_step(system, xc, nspin, smearing, width, mu, rho_s,
                       rho_out_s, veff_s, coeffs_s, eigs_s, occ_s, **kw):
    """One norm-conserving inexact-Newton density update, drop-in for a mixer
    step inside the SCF loop.

    ``rho_s`` is the per-spin input density list, ``rho_out_s`` the raw SCF map
    output ``F(ρ_in)`` for the same iteration; the remaining arguments are the
    loop's frozen orbitals/eigenvalues/occupations at ``ρ_in``. Returns
    ``(new_rho_s, SolveResult)`` — the per-spin density list ``ρ_in + δ`` and
    the inner-solve diagnostics. ``**kw`` forwards to :func:`newton_delta_nc`."""
    jac = _frozen_state(system, xc, nspin, smearing, width, mu, rho_s, veff_s,
                        coeffs_s, eigs_s, occ_s)
    if nspin == 1:
        r = (rho_out_s[0] - rho_s[0]).to(RDTYPE)
    else:
        r = torch.stack([(rho_out_s[isp] - rho_s[isp]).to(RDTYPE)
                         for isp in range(nspin)])
    delta, sol = newton_delta_nc(jac, xc, r, **kw)
    if nspin == 1:
        return [rho_s[0] + delta], sol
    return [rho_s[isp] + delta[isp] for isp in range(nspin)], sol
