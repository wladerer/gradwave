"""Implicit differentiation through the SCF fixed point (M4) — insulators.

For a loss L(ρ) on the CONVERGED density, gradients w.r.t. functional
parameters θ require the density response (unlike energy losses, which get
dE/dθ free by stationarity). The adjoint formulation solves ONE
self-consistent linear problem regardless of the number of parameters:

    u = v̄ + K_Hxc[χ₀ u],        v̄(r) = ∂L/∂ρ(r)
    dL/dθ = ⟨χ₀ u, ∂v_xc/∂θ⟩_grid + ∂L/∂θ|_explicit

χ₀ w:  independent-particle response of the density to a local potential w,
       one conduction-projected Sternheimer solve (H − ε_n + αP_occ)|δψ_n⟩ =
       −P_c w|ψ_n⟩ per occupied band per k (CG; positive definite on the
       conduction space thanks to the gap — INSULATORS ONLY here; the
       metallic Fermi-surface term is future work).
K_Hxc: Hartree kernel 4πe²/G² + f_xc, with f_xc·w obtained as an autograd
       Hessian-vector product of E_xc — any twice-differentiable functional
       works automatically, including learnable ones.

Degeneracy note: P_c = 1 − Σ_occ |ψ⟩⟨ψ| projects out the ENTIRE occupied
subspace, so degenerate valence tops (Si Γ) are handled correctly.

This module is the mathematical core that a torch.autograd.Function wrapper
(and torch.func Hessians) will build on; the direct API here is
`density_loss_param_grads`.
"""

from __future__ import annotations

import torch

from gradwave.core.density import sigma_from_rho
from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g
from gradwave.core.hamiltonian import HamiltonianK, projectors
from gradwave.core.xc.base import xc_eager
from gradwave.dtypes import RDTYPE

# Cycle-free import direction: postscf._response depends only on core/, and
# gradwave.postscf's __init__ is empty, so scf modules may pull the shared
# response kernels from there.
from gradwave.postscf._response import (
    dyson_fixed_point,
    fxc_hvp,
    hartree_kernel,
    sternheimer_shift,
)
from gradwave.scf.loop import SCFResult
from gradwave.solvers.precond import teter


def _check_no_symmetry(res: SCFResult):
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("implicit backward for nspin=2 is future work")
    if getattr(res.system, "sym", None) is not None:
        raise NotImplementedError(
            "implicit SCF backward requires use_symmetry=False: a perturbation "
            "breaks the crystal symmetry, so the response needs the full "
            "(TR-reduced) k-mesh"
        )


def _occupied(res: SCFResult, ik: int):
    occ = res.occupations[ik]
    n_occ = int((occ > 1e-8).sum())
    if not torch.all((occ[:n_occ] - 2.0).abs() < 1e-8):
        raise NotImplementedError("implicit SCF backward supports insulators only (occ = 2)")
    return res.coeffs[ik][:n_occ], res.eigenvalues[ik][:n_occ]


def _hamiltonians(res: SCFResult) -> list[HamiltonianK]:
    system = res.system
    hs = []
    for ik, sph in enumerate(system.spheres):
        p = projectors(system.proj_data[ik], system.positions)
        hs.append(HamiltonianK(sph, system.grid.shape, res.v_eff, system.proj_data[ik], p))
    return hs


def projected_cg(a_apply, precond, x, r, tol: float, max_iter: int):
    """Projected preconditioned CG with a per-band breakdown guard.

    Solves ``A x = rhs`` in batched (per-band) form given the initial guess
    ``x`` and residual ``r = rhs − A x``. ``a_apply`` is the SPD operator on the
    conduction space; ``precond(r)`` applies the preconditioner — the USPP twin
    folds the S-projector into it, the norm-conserving path passes a plain Teter
    preconditioner. Bands whose curvature goes non-positive or non-finite are
    frozen (their search direction is zeroed); the operator is positive definite
    on the conduction space, so this only fires at the round-off floor after a
    band has already converged, where an unguarded ``pap ≤ 0`` would give a
    1e300 step → Inf → NaN.

    No autograd path — the callers are the no-grad inner solves of implicit
    differentiation. Returns ``x``; any final projection is the caller's.
    """
    z = precond(r)
    p = z
    rz = torch.einsum("bg,bg->b", r.conj(), z).real
    for _ in range(max_iter):
        ap = a_apply(p)
        pap = torch.einsum("bg,bg->b", p.conj(), ap).real
        p2 = torch.einsum("bg,bg->b", p.conj(), p).real
        ok = torch.isfinite(pap) & (pap > 1e-30 * p2.clamp_min(1e-300))
        if not bool(ok.any()):
            break
        a_cg = torch.where(ok, rz / pap.clamp_min(1e-300), torch.zeros_like(rz))
        x = x + a_cg[:, None] * p
        r = r - a_cg[:, None] * ap
        if float(torch.linalg.norm(r, dim=1).max()) < tol:
            break
        z = precond(r)
        rz_new = torch.einsum("bg,bg->b", r.conj(), z).real
        beta = torch.where(ok, rz_new / rz.clamp_min(1e-300), torch.zeros_like(rz))
        p = torch.where(ok[:, None], z + beta[:, None] * p, torch.zeros_like(p))
        rz = rz_new
    return x


def _sternheimer(h: HamiltonianK, c_occ, eps_occ, w_r, alpha: float, tol: float, max_iter: int):
    """Solve (H − ε_n + α P_occ) δψ_n = −P_c w ψ_n for all occupied n at one k.

    Returns δψ (n_occ, npw), entirely in the conduction space.
    """

    def p_c(x):
        return x - (x @ c_occ.conj().T) @ c_occ

    # RHS: −P_c (w ψ_n)
    psi_r = g_to_r(c_occ, h.sphere.flat_idx, h.shape)
    w_psi = box_to_sphere(r_to_g(psi_r * w_r), h.sphere.flat_idx)
    rhs = -p_c(w_psi)

    def a_apply(x):
        hx = h.apply(x) - eps_occ[:, None] * x
        return p_c(hx) + alpha * ((x @ c_occ.conj().T) @ c_occ)

    x = torch.zeros_like(rhs)
    r = rhs - a_apply(x)
    t_g = h.t
    t_band = torch.clamp(
        torch.einsum("bg,g,bg->b", c_occ.conj(), t_g.to(c_occ.dtype), c_occ).real, min=1e-6
    )
    x = projected_cg(a_apply, lambda rr: teter(rr, t_g, t_band), x, r, tol, max_iter)
    return p_c(x)


@torch.no_grad()
def apply_chi0(res: SCFResult, w_r: torch.Tensor, tol: float = 1e-8,
               max_iter: int = 200) -> torch.Tensor:
    """δρ(r) = χ₀ w for a real local field w(r) [insulator]."""
    _check_no_symmetry(res)
    system = res.system
    grid = system.grid
    hs = _hamiltonians(res)
    dr = torch.zeros(grid.shape, dtype=RDTYPE, device=res.v_eff.device)
    for ik, h in enumerate(hs):
        c_occ, eps_occ = _occupied(res, ik)
        gap_shift = sternheimer_shift(eps_occ)
        dpsi = _sternheimer(h, c_occ, eps_occ, w_r, alpha=gap_shift, tol=tol, max_iter=max_iter)
        psi_r = g_to_r(c_occ, h.sphere.flat_idx, grid.shape)
        dpsi_r = g_to_r(dpsi, h.sphere.flat_idx, grid.shape)
        # f = 2, factor 2 more from c.c. pair (ψ*δψ + δψ*ψ)
        contrib = 4.0 * float(system.kweights[ik]) * (psi_r.conj() * dpsi_r).real.sum(dim=0)
        dr += contrib
    return dr / grid.volume


def apply_k_hxc(res: SCFResult, xc, w_r: torch.Tensor) -> torch.Tensor:
    """(K_Hxc w)(r) = Hartree kernel + f_xc·w, via autograd HVP for the XC part.

    Both kernels are the shared response primitives in postscf._response
    (``fxc_hvp`` carries the per-grid-cell → physical n_points/Ω conversion).
    """
    grid = res.system.grid
    return hartree_kernel(grid, w_r) + fxc_hvp(xc, res.rho, grid, w_r)


@torch.no_grad()
def solve_adjoint(res: SCFResult, xc, vbar_r: torch.Tensor, beta: float = 0.4,
                  tol: float = 1e-9, max_iter: int = 100) -> torch.Tensor:
    """Solve u = v̄ + K_Hxc[χ₀ u] by damped fixed-point iteration."""

    def _fail(step):
        raise RuntimeError(
            f"adjoint fixed point not converged ({step:.2e} after {max_iter} iters)")

    # Defaults (beta 0.4, tol 1e-9, max_iter 100) and the raise-on-failure are
    # this site's historical behavior; the sibling Dyson loops differ (see
    # dyson_fixed_point's note).
    return dyson_fixed_point(
        lambda u: apply_k_hxc(res, xc, apply_chi0(res, u)), vbar_r,
        beta=beta, tol=tol, max_iter=max_iter, on_fail=_fail)


def density_loss_param_grads(res: SCFResult, xc, loss_fn) -> tuple[torch.Tensor, dict]:
    """Gradients dL/dθ of a density-dependent loss through the SCF fixed point.

    loss_fn: rho(grid tensor) -> scalar torch tensor (pure, differentiable).
    Returns (L, {param_name: grad}).
    """
    grid = res.system.grid
    rho = res.rho.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        loss = loss_fn(rho)
        (vbar,) = torch.autograd.grad(loss, rho)
    # v̄ as physical δL/δρ(r): loss_fn works on grid values directly, so vbar
    # already is ∂L/∂ρ_j — the grid-sum adjoint field
    u = solve_adjoint(res, xc, vbar)
    chi0_u = apply_chi0(res, u)

    # dL/dθ = ⟨χ₀u, ∂v_xc/∂θ⟩, differentiate ⟨χ₀u, v_xc(ρ; θ)⟩ w.r.t. θ at fixed ρ.
    # Double backward through E_xc, so force eager with xc_eager().
    rho_fixed = res.rho.detach().clone().requires_grad_(True)
    with torch.enable_grad(), xc_eager():
        sigma = sigma_from_rho(rho_fixed, grid.g_cart) if xc.needs_gradient else None
        e_xc = xc.energy(rho_fixed, grid.volume, sigma)
        (v_xc,) = torch.autograd.grad(e_xc, rho_fixed, create_graph=True)
        inner = (v_xc * chi0_u.detach()).sum() * (grid.n_points / grid.volume)
        params = list(xc.parameters())
        grads = torch.autograd.grad(inner, params, allow_unused=True)
    named = {
        name: (g if g is not None else torch.zeros_like(p))
        for (name, p), g in zip(xc.named_parameters(), grads, strict=True)
    }
    return loss.detach(), named
