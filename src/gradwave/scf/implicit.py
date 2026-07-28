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

nspin=2 (collinear): the density becomes a per-channel pair (ρ↑, ρ↓) and
every structure above becomes a two-element list. χ₀ is block-diagonal over
spin — each channel has its own occupied bands, its own v_eff^σ and its own
Sternheimer solves, with the single-band degeneracy weight g = 1 instead of
2. K_Hxc keeps its cross-spin blocks: the Hartree kernel acts on the TOTAL
δρ = δρ↑ + δρ↓ and enters both channels, while f_xc^{σσ'} is the spin HVP of
the grid E_xc (fxc_hvp_spin, NLCC core split half/half). The loss stays a
functional of the total density, so v̄ = ∂L/∂ρ seeds both channels equally.
This mirrors the USPP/PAW twin in postscf/uspp_implicit.py, minus the
augmentation/one-center blocks (norm-conserving has none) and minus the
Fermi-surface channel (insulators only, both spin channels integer-filled).

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
from gradwave.postscf._anderson import AndersonMixer
from gradwave.postscf._response import (
    fxc_hvp,
    fxc_hvp_spin,
    hartree_kernel,
    spin_sigma_triple,
    sternheimer_shift,
)
from gradwave.scf.loop import SCFResult
from gradwave.solvers.precond import teter


def _check_no_symmetry(res: SCFResult):
    if getattr(res.system, "sym", None) is not None:
        raise NotImplementedError(
            "implicit SCF backward requires use_symmetry=False: a perturbation "
            "breaks the crystal symmetry, so the response needs the full "
            "(TR-reduced) k-mesh"
        )


def _occupied(res: SCFResult, isp: int, ik: int):
    """Occupied block (coeffs, ε) of spin channel ``isp`` at k-point ``ik``.

    Insulators only: every occupied band must be filled to ``f_full`` (2 for
    nspin=1, 1 per channel for nspin=2), the rest empty. Each spin channel is
    checked independently, so a collinear magnet with N↑ ≠ N↓ integer-filled
    channels is supported."""
    nspin = getattr(res, "nspin", 1)
    if nspin == 1:
        occ = res.occupations[ik]
        coeffs, eps = res.coeffs[ik], res.eigenvalues[ik]
        f_full = 2.0
    else:
        occ = res.occupations[isp][ik]
        coeffs, eps = res.coeffs[isp][ik], res.eigenvalues[isp][ik]
        f_full = 1.0
    n_occ = int((occ > 1e-8).sum())
    if not torch.all((occ[:n_occ] - f_full).abs() < 1e-8):
        raise NotImplementedError(
            f"implicit SCF backward supports insulators only (occ = {f_full})")
    return coeffs[:n_occ], eps[:n_occ]


def _hamiltonians(res: SCFResult) -> list:
    """Per-k HamiltonianK (nspin=1) or per-spin, per-k list-of-lists (nspin=2).

    For nspin=2 each channel uses its own v_eff^σ; the plane-wave sphere and
    the KB projectors are spin-independent, so they are built once per k."""
    system = res.system
    nspin = getattr(res, "nspin", 1)
    if nspin == 1:
        hs = []
        for ik, sph in enumerate(system.spheres):
            p = projectors(system.proj_data[ik], system.positions)
            hs.append(HamiltonianK(sph, system.grid.shape, res.v_eff,
                                   system.proj_data[ik], p))
        return hs
    hs = [[] for _ in range(nspin)]
    for ik, sph in enumerate(system.spheres):
        p = projectors(system.proj_data[ik], system.positions)
        for isp in range(nspin):
            hs[isp].append(HamiltonianK(sph, system.grid.shape, res.v_eff[isp],
                                        system.proj_data[ik], p))
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


def _chi0_channel(res: SCFResult, hs_k: list, isp: int, w_r: torch.Tensor,
                  f_full: float, tol: float, max_iter: int) -> torch.Tensor:
    """δρ_σ(r) = χ₀^σ w for one spin channel (block-diagonal in spin)."""
    system = res.system
    grid = system.grid
    dr = torch.zeros(grid.shape, dtype=RDTYPE, device=grid.g2.device)
    for ik, h in enumerate(hs_k):
        c_occ, eps_occ = _occupied(res, isp, ik)
        gap_shift = sternheimer_shift(eps_occ)
        dpsi = _sternheimer(h, c_occ, eps_occ, w_r, alpha=gap_shift,
                            tol=tol, max_iter=max_iter)
        psi_r = g_to_r(c_occ, h.sphere.flat_idx, grid.shape)
        dpsi_r = g_to_r(dpsi, h.sphere.flat_idx, grid.shape)
        # per-band occupation f_full, plus factor 2 from the c.c. pair
        # (ψ*δψ + δψ*ψ): nspin=1 gives 4 (f=2), nspin=2 gives 2 (f=1).
        contrib = (2.0 * f_full * float(system.kweights[ik])
                   * (psi_r.conj() * dpsi_r).real.sum(dim=0))
        dr += contrib
    return dr / grid.volume


@torch.no_grad()
def apply_chi0(res: SCFResult, w_r: torch.Tensor, tol: float = 1e-8,
               max_iter: int = 200) -> torch.Tensor:
    """δρ = χ₀ w for a real local field w [insulator].

    nspin=1: ``w_r`` is a grid field, returns a grid field. nspin=2: ``w_r`` is
    the stacked per-spin field (2, *grid.shape) and the return is the stacked
    per-spin density response — χ₀ is block-diagonal over spin."""
    _check_no_symmetry(res)
    hs = _hamiltonians(res)
    nspin = getattr(res, "nspin", 1)
    if nspin == 1:
        return _chi0_channel(res, hs, 0, w_r, 2.0, tol, max_iter)
    drs = [_chi0_channel(res, hs[isp], isp, w_r[isp], 1.0, tol, max_iter)
           for isp in range(nspin)]
    return torch.stack(drs)


def apply_k_hxc(res: SCFResult, xc, w_r: torch.Tensor) -> torch.Tensor:
    """(K_Hxc w)(r) = Hartree kernel + f_xc·w, via autograd HVP for the XC part.

    Both kernels are the shared response primitives in postscf._response
    (``fxc_hvp`` carries the per-grid-cell → physical n_points/Ω conversion).

    f_xc = ∂v_xc/∂ρ is evaluated at the SCF density INCLUDING the NLCC partial
    core, ρ + ρ_core, because the SCF builds v_xc at exactly that point
    (scf/loop.py: ``rho_for_xc = rho + rho_core``). Omitting the core would
    evaluate the response kernel at the wrong density (~1% off for NLCC
    pseudos), making the implicit-diff SCF adjoint subtly wrong. Matches the
    collinear ``_k_hxc_spin`` (which adds 0.5·ρ_core per channel) and the USPP
    ``k_hxc_grid``. For a valence-only pseudo (rho_core is None) this reduces
    exactly to the plain valence kernel — byte-identical.

    nspin=2: ``w_r`` is the stacked per-spin field (2, *grid.shape). The
    Hartree kernel acts on the TOTAL δρ = δρ↑ + δρ↓ and enters both channels;
    f_xc becomes the spin HVP f_xc^{σσ'} (fxc_hvp_spin) at the converged
    per-channel densities, NLCC core split half/half — matching the SCF's
    ``effective_potentials`` spin assembly and the USPP twin's ``k_hxc_grid``.
    """
    grid = res.system.grid
    core = res.system.rho_core
    nspin = getattr(res, "nspin", 1)
    if nspin == 1:
        rho_xc = res.rho if core is None else res.rho + core
        return hartree_kernel(grid, w_r) + fxc_hvp(xc, rho_xc, grid, w_r)
    kh = hartree_kernel(grid, w_r[0] + w_r[1])
    c2 = 0.0 if core is None else 0.5 * core
    fu, fd = fxc_hvp_spin(xc, res.rho_spin[0] + c2, res.rho_spin[1] + c2,
                          grid, w_r[0], w_r[1])
    return torch.stack([kh + fu, kh + fd])


@torch.no_grad()
def solve_adjoint(res: SCFResult, xc, vbar_r: torch.Tensor, beta: float = 0.4,
                  tol: float = 1e-9, max_iter: int = 100,
                  history: int = 8) -> torch.Tensor:
    """Solve u = v̄ + K_Hxc[χ₀ u] by Anderson-accelerated fixed-point iteration.

    For nspin=2 ``vbar_r`` (and hence ``u``) is the stacked per-spin pair
    (2, *grid.shape); the loop runs on the flattened tensor — every operation
    is per-channel or total-density-coupled inside the two kernels.

    Anderson mixing (not plain damping) because the screened response
    u = v̄ + Kχ₀u has gain > 1 modes near a spin instability where damping
    diverges — the same NiO lesson the USPP twin (uspp_implicit) and the
    dielectric/Hubbard adjoints all hit. For a nonmagnetic insulator the
    fixed point is contractive and Anderson reduces to (and converges like)
    the former damped loop, landing on the identical fixed point."""
    shape = vbar_r.shape

    def g(u):
        return vbar_r + apply_k_hxc(res, xc, apply_chi0(res, u))

    u = vbar_r.clone()
    mixer = AndersonMixer(history, beta)
    step = float("inf")
    for _ in range(max_iter):
        r = g(u) - u
        step = float(torch.linalg.norm(r)) / max(
            1.0, float(torch.linalg.norm(u)))
        if step < tol:
            return u
        u = mixer.step(u.reshape(-1), r.reshape(-1)).reshape(shape)
    raise RuntimeError(
        f"adjoint fixed point not converged ({step:.2e} after {max_iter} iters)")


def density_loss_param_grads(res: SCFResult, xc, loss_fn) -> tuple[torch.Tensor, dict]:
    """Gradients dL/dθ of a density-dependent loss through the SCF fixed point.

    loss_fn: rho(grid tensor of the TOTAL density) -> scalar torch tensor
    (pure, differentiable). For nspin=2 the loss stays a functional of ρ_tot,
    so its gradient v̄ = ∂L/∂ρ seeds both spin channels equally.
    Returns (L, {param_name: grad}).
    """
    grid = res.system.grid
    nspin = getattr(res, "nspin", 1)
    rho = res.rho.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        loss = loss_fn(rho)
        (vbar,) = torch.autograd.grad(loss, rho)
    # v̄ as physical δL/δρ(r): loss_fn works on grid values directly, so vbar
    # already is ∂L/∂ρ_j — the grid-sum adjoint field. nspin=2: the loss is a
    # functional of ρ_tot, so v̄ enters both channels equally (stacked).
    vbar_seed = vbar if nspin == 1 else torch.stack([vbar] * nspin)
    u = solve_adjoint(res, xc, vbar_seed)
    chi0_u = apply_chi0(res, u)

    # dL/dθ = Σ_σ ⟨χ₀u_σ, ∂v_xc^σ/∂θ⟩, differentiate ⟨χ₀u, v_xc(ρ; θ)⟩ w.r.t. θ
    # at fixed ρ. Double backward through E_xc, so force eager with xc_eager().
    # v_xc (hence ∂v_xc/∂θ) is evaluated at ρ + ρ_core, matching the SCF and
    # apply_k_hxc above; for valence-only pseudos (rho_core is None) this is
    # byte-identical to res.rho.
    core = res.system.rho_core
    scale = grid.n_points / grid.volume
    params = list(xc.parameters())
    with torch.enable_grad(), xc_eager():
        if nspin == 1:
            rho_xc = res.rho if core is None else res.rho + core
            rho_fixed = rho_xc.detach().clone().requires_grad_(True)
            sigma = (sigma_from_rho(rho_fixed, grid.g_cart)
                     if xc.needs_gradient else None)
            e_xc = xc.energy(rho_fixed, grid.volume, sigma)
            (v_xc,) = torch.autograd.grad(e_xc, rho_fixed, create_graph=True)
            inner = (v_xc * chi0_u.detach()).sum() * scale
        else:
            c2 = 0.0 if core is None else 0.5 * core
            ru = (res.rho_spin[0] + c2).detach().clone().requires_grad_(True)
            rd = (res.rho_spin[1] + c2).detach().clone().requires_grad_(True)
            s_uu, s_dd, s_tt = spin_sigma_triple(xc, ru, rd, grid.g_cart)
            e_xc = xc.energy(ru, rd, grid.volume, s_uu, s_dd, s_tt)
            vu, vd = torch.autograd.grad(e_xc, (ru, rd), create_graph=True)
            inner = ((vu * chi0_u[0].detach()).sum()
                     + (vd * chi0_u[1].detach()).sum()) * scale
        grads = torch.autograd.grad(inner, params, allow_unused=True)
    named = {
        name: (g if g is not None else torch.zeros_like(p))
        for (name, p), g in zip(xc.named_parameters(), grads, strict=True)
    }
    return loss.detach(), named
