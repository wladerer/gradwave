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

import math

import torch

from gradwave.constants import E2
from gradwave.core.density import sigma_from_rho
from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g
from gradwave.core.hamiltonian import HamiltonianK, projectors
from gradwave.core.xc.base import xc_eager
from gradwave.dtypes import CDTYPE, RDTYPE
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
    z = teter(r, t_g, t_band)
    p = z
    rz = torch.einsum("bg,bg->b", r.conj(), z).real
    for _ in range(max_iter):
        ap = a_apply(p)
        pap = torch.einsum("bg,bg->b", p.conj(), ap).real
        alpha_cg = rz / torch.clamp(pap, min=1e-300)
        x = x + alpha_cg[:, None] * p
        r = r - alpha_cg[:, None] * ap
        if float(torch.linalg.norm(r, dim=1).max()) < tol:
            break
        z = teter(r, t_g, t_band)
        rz_new = torch.einsum("bg,bg->b", r.conj(), z).real
        p = z + (rz_new / torch.clamp(rz, min=1e-300))[:, None] * p
        rz = rz_new
    return p_c(x)


@torch.no_grad()
def apply_chi0(res: SCFResult, w_r: torch.Tensor, tol: float = 1e-8,
               max_iter: int = 200) -> torch.Tensor:
    """δρ(r) = χ₀ w for a real local field w(r) [insulator]."""
    _check_no_symmetry(res)
    system = res.system
    grid = system.grid
    hs = _hamiltonians(res)
    dr = torch.zeros(grid.shape, dtype=RDTYPE)
    for ik, h in enumerate(hs):
        c_occ, eps_occ = _occupied(res, ik)
        gap_shift = 2.0 * float(eps_occ.max() - eps_occ.min()) + 10.0
        dpsi = _sternheimer(h, c_occ, eps_occ, w_r, alpha=gap_shift, tol=tol, max_iter=max_iter)
        psi_r = g_to_r(c_occ, h.sphere.flat_idx, grid.shape)
        dpsi_r = g_to_r(dpsi, h.sphere.flat_idx, grid.shape)
        # f = 2, factor 2 more from c.c. pair (ψ*δψ + δψ*ψ)
        contrib = 4.0 * float(system.kweights[ik]) * (psi_r.conj() * dpsi_r).real.sum(dim=0)
        dr += contrib
    return dr / grid.volume


def apply_k_hxc(res: SCFResult, xc, w_r: torch.Tensor) -> torch.Tensor:
    """(K_Hxc w)(r) = Hartree kernel + f_xc·w, via autograd HVP for the XC part."""
    grid = res.system.grid
    # Hartree: 4πe²/G² in reciprocal space
    w_g = r_to_g(w_r.to(CDTYPE))
    inv_g2 = torch.where(
        grid.g2 > 1e-12, 1.0 / torch.clamp(grid.g2, min=1e-12), torch.zeros_like(grid.g2)
    )
    kh = (torch.fft.ifftn(4.0 * math.pi * E2 * w_g * inv_g2, dim=(-3, -2, -1))
          * grid.n_points).real

    # XC: f_xc·w = d/dρ ⟨v_xc(ρ), w⟩ (double backward through E_xc). xc_eager()
    # forces eager, since compiled aot_autograd cannot double-backward.
    rho = res.rho.detach().clone().requires_grad_(True)
    with torch.enable_grad(), xc_eager():
        sigma = sigma_from_rho(rho, grid.g_cart) if xc.needs_gradient else None
        e_xc = xc.energy(rho, grid.volume, sigma)
        (v_xc,) = torch.autograd.grad(e_xc, rho, create_graph=True)
        inner = (v_xc * w_r.detach()).sum()
        (fxc_w,) = torch.autograd.grad(inner, rho)
    # v_xc here is per-grid-cell dE/dρ_j; convert both ways: kernel in physical
    # units needs (N/Ω)² · (Ω/N) = N/Ω on the product
    return kh + fxc_w * (grid.n_points / grid.volume)


@torch.no_grad()
def solve_adjoint(res: SCFResult, xc, vbar_r: torch.Tensor, beta: float = 0.4,
                  tol: float = 1e-9, max_iter: int = 100) -> torch.Tensor:
    """Solve u = v̄ + K_Hxc[χ₀ u] by damped fixed-point iteration."""
    u = vbar_r.clone()
    for _ in range(max_iter):
        u_new = vbar_r + apply_k_hxc(res, xc, apply_chi0(res, u))
        du = float(torch.linalg.norm(u_new - u)) / max(1.0, float(torch.linalg.norm(u)))
        u = u + beta * (u_new - u)
        if du < tol:
            return u
    raise RuntimeError(f"adjoint fixed point not converged ({du:.2e} after {max_iter} iters)")


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
