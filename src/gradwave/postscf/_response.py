"""Shared linear-response primitives for the postscf modules.

One home for the machinery that the response-based estimators kept
re-implementing:

- the damped (1 − χ₀K)⁻¹ Dyson fixed point (``dyson_fixed_point``),
- the Hartree kernel and the f_xc Hessian-vector products of E_xc
  (``hartree_kernel``, ``fxc_hvp``, ``fxc_hvp_spin``),
- the batched conduction-projected Sternheimer CG and its helpers
  (``cg_sternheimer``, ``pad_coeffs``, ``insulator_window``,
  ``sternheimer_shift``),
- the (σ_uu, σ_dd, σ_tt) triple GGA spin functionals consume
  (``spin_sigma_triple``).

Import direction: this module depends only on ``gradwave.core``/
``gradwave.constants``/``gradwave.solvers``, so ``gradwave.scf`` modules
(e.g. ``scf/implicit.py``) may import it without creating a cycle —
``gradwave.postscf`` has an empty ``__init__`` and nothing here imports
``gradwave.scf``.
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import E2
from gradwave.core.density import sigma_from_rho
from gradwave.core.fftbox import r_to_g
from gradwave.core.xc.base import xc_eager
from gradwave.dtypes import CDTYPE
from gradwave.solvers.precond import teter_b


class DysonNotConverged(RuntimeError):
    """The screening Dyson fixed point did not reach ``tol`` within ``max_iter``."""


@torch.no_grad()
def dyson_fixed_point(op, rhs: torch.Tensor, *, beta: float, tol: float,
                      max_iter: int, on_fail=None, denom_new: bool = False,
                      verbose: bool = False) -> torch.Tensor:
    """Solve x = rhs + op(x) by damped fixed-point iteration.

    ``op`` applies the screening operator (χ₀K or Kχ₀ depending on which
    side of the response the caller works on). ``on_fail`` is called with
    the last relative step when the loop exhausts ``max_iter`` (raise there
    to make non-convergence fatal); None returns the unconverged iterate.
    ``denom_new`` measures the relative step against |x_new| instead of |x|.

    Historical note: the three former copies of this loop diverged in their
    defaults (max_iter 60/80/100, tol 1e-6/1e-7/1e-9), their failure behavior
    (silent return vs. raise) and the step denominator; every call site now
    passes its historical choice explicitly.
    """
    x = rhs.clone()
    step = float("inf")
    for it in range(max_iter):
        x_new = rhs + op(x)
        ref = x_new if denom_new else x
        step = float(torch.linalg.norm(x_new - x)) / max(
            1.0, float(torch.linalg.norm(ref)))
        x = x + beta * (x_new - x)
        if verbose:
            print(f"  dyson it {it}: rel step {step:.2e}", flush=True)
        if step < tol:
            return x
    if on_fail is not None:
        on_fail(step)
    return x


# --------------------------------------------------------------------------- #
#  K_Hxc pieces: Hartree kernel + f_xc Hessian-vector products                #
# --------------------------------------------------------------------------- #


def hartree_kernel(grid, w_r: torch.Tensor) -> torch.Tensor:
    """(K_H w)(r): the Hartree kernel 4πe²/G² applied to a real grid field
    (G=0 excluded)."""
    w_g = r_to_g(w_r.to(CDTYPE))
    inv_g2 = torch.where(
        grid.g2 > 1e-12, 1.0 / torch.clamp(grid.g2, min=1e-12),
        torch.zeros_like(grid.g2))
    return (torch.fft.ifftn(4.0 * math.pi * E2 * w_g * inv_g2,
                            dim=(-3, -2, -1)) * grid.n_points).real


def fxc_hvp(xc, rho0: torch.Tensor, grid, w_r: torch.Tensor) -> torch.Tensor:
    """f_xc·w at the density ``rho0`` in physical units [eV].

    d/dρ ⟨v_xc(ρ), w⟩ by double backward through E_xc; xc_eager() forces
    eager, since compiled aot_autograd cannot double-backward. The returned
    field carries the grid-cell → physical conversion n_points/Ω (v_xc from
    autograd is per-grid-cell dE/dρ_j).
    """
    rho = rho0.detach().clone().requires_grad_(True)
    with torch.enable_grad(), xc_eager():
        sigma = sigma_from_rho(rho, grid.g_cart) if xc.needs_gradient else None
        e_xc = xc.energy(rho, grid.volume, sigma)
        (v_xc,) = torch.autograd.grad(e_xc, rho, create_graph=True)
        inner = (v_xc * w_r.detach()).sum()
        (fxc_w,) = torch.autograd.grad(inner, rho)
    return fxc_w * (grid.n_points / grid.volume)


def fxc_hvp_spin(xc, ru0: torch.Tensor, rd0: torch.Tensor, grid,
                 wu: torch.Tensor, wd: torch.Tensor):
    """(f_xc^{σσ'} w^{σ'})↑, ↓ at the spin densities (ru0, rd0) [eV].

    The spin Hessian-vector product of the grid E_xc (double backward, eager
    like ``fxc_hvp``); the caller folds any NLCC core split into ru0/rd0.
    """
    ru = ru0.detach().clone().requires_grad_(True)
    rd = rd0.detach().clone().requires_grad_(True)
    with torch.enable_grad(), xc_eager():
        s_uu, s_dd, s_tt = spin_sigma_triple(xc, ru, rd, grid.g_cart)
        e_xc = xc.energy(ru, rd, grid.volume, s_uu, s_dd, s_tt)
        vu, vd = torch.autograd.grad(e_xc, (ru, rd), create_graph=True)
        inner = (vu * wu.detach()).sum() + (vd * wd.detach()).sum()
        fu, fd = torch.autograd.grad(inner, (ru, rd))
    scale = grid.n_points / grid.volume
    return fu * scale, fd * scale


def spin_sigma_triple(xc, r_u: torch.Tensor, r_d: torch.Tensor, g_cart):
    """(σ_uu, σ_dd, σ_tt) for a spin GGA call, or (None,)*3 for an LDA-type
    functional. σ_tt is the gradient invariant of the total density."""
    if not xc.needs_gradient:
        return None, None, None
    return (sigma_from_rho(r_u, g_cart), sigma_from_rho(r_d, g_cart),
            sigma_from_rho(r_u + r_d, g_cart))


# --------------------------------------------------------------------------- #
#  Sternheimer scaffolding                                                    #
# --------------------------------------------------------------------------- #


# Shift added to the occupied-window spread when projecting the Sternheimer
# operator: alpha = 2·(ε_max − ε_min) + GAP_SHIFT_EV pushes the occupied
# subspace well above the conduction spectrum so (H − ε_n + α P_occ) stays
# positive definite. The 10 eV margin is empirical headroom over the band
# gap; any value comfortably above ε_gap works, and every solver here now
# shares this one choice.
GAP_SHIFT_EV = 10.0


def sternheimer_shift(eps: torch.Tensor) -> float:
    """The occupied-projector shift 2·(ε_max − ε_min) + GAP_SHIFT_EV [eV]."""
    return 2.0 * float(eps.max() - eps.min()) + GAP_SHIFT_EV


def insulator_window(occ: torch.Tensor, f_full: float, err_msg: str) -> int:
    """Occupied-band count of an insulating (nk, nb) occupation array.

    ``f_full`` is the full filling (2 for nspin=1, 1 per spin channel).
    Raises NotImplementedError with ``err_msg`` unless every band is filled
    to f_full or empty (to 1e-6).
    """
    nocc = int((occ[0] > 0.5 * f_full).sum())
    if occ.shape[1] > nocc:
        ins = bool((occ[:, :nocc] > f_full - 1e-6).all()) \
            and bool((occ[:, nocc:] < 1e-6).all())
    else:
        ins = bool((occ > f_full - 1e-6).all())
    if not ins:
        raise NotImplementedError(err_msg)
    return nocc


def pad_coeffs(coeffs_per_k, npw_max, device=None):
    """[(nb, npw_k)] per k → padded (nk, nb, npw_max), detached. `device`
    defaults to the coeffs' own device (a no-op move in that case)."""
    nk = len(coeffs_per_k)
    nb = coeffs_per_k[0].shape[0]
    dev = device if device is not None else coeffs_per_k[0].device
    out = torch.zeros(nk, nb, npw_max, dtype=CDTYPE, device=dev)
    for ik, c in enumerate(coeffs_per_k):
        out[ik, :, : c.shape[1]] = c.detach().to(dev)
    return out


def cg_sternheimer(h, bk, c_occ, eps_occ, rhs, x0, shift, tol=1e-8,
                   max_iter=400):
    """Batched conduction-projected Sternheimer: (H − ε_n + s·P_occ)δψ = rhs,
    for all occupied bands of all k at once ((nk, nocc, npw_max), masked).
    rhs must already lie in the conduction space; positive definite there
    thanks to the gap — insulators only."""

    def p_occ(x):
        ov = torch.einsum("kng,kbg->kbn", c_occ.conj(), x)
        return torch.einsum("kbn,kng->kbg", ov, c_occ)

    def a_apply(x):
        hx = h.apply(x) - eps_occ[..., None] * x
        return hx - p_occ(hx) + shift * p_occ(x)

    t_band = torch.clamp(
        torch.einsum("kbg,kg,kbg->kb", c_occ.conj(), bk.t.to(c_occ.dtype), c_occ).real,
        min=1e-6,
    )
    x = x0.clone()
    r = rhs - a_apply(x)
    z = teter_b(r, bk.t, t_band)
    p = z
    rz = torch.einsum("kbg,kbg->kb", r.conj(), z).real
    for _ in range(max_iter):
        ap = a_apply(p)
        pap = torch.einsum("kbg,kbg->kb", p.conj(), ap).real
        a_cg = rz / torch.clamp(pap, min=1e-300)
        x = x + a_cg[..., None] * p
        r = r - a_cg[..., None] * ap
        if float(torch.linalg.norm(r, dim=-1).max()) < tol:
            break
        z = teter_b(r, bk.t, t_band)
        rz_new = torch.einsum("kbg,kbg->kb", r.conj(), z).real
        p = z + (rz_new / torch.clamp(rz, min=1e-300))[..., None] * p
        rz = rz_new
    return x - p_occ(x)
