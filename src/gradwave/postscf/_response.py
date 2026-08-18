"""Shared linear-response primitives for the postscf modules.

One home for the machinery that the response-based estimators kept
re-implementing:

- the damped (1 − χ₀K)⁻¹ Dyson fixed point (``dyson_fixed_point``),
- the Hartree kernel and the f_xc Hessian-vector products of E_xc
  (``hartree_kernel``, ``fxc_hvp``, ``fxc_hvp_spin``,
  ``fxc_hvp_noncollinear_nonmagnetic``),
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
from collections.abc import Callable
from typing import Protocol

import torch

from gradwave.constants import E2
from gradwave.core.density import sigma_from_rho
from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.core.xc.base import XCFunctional, xc_eager
from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import SpinXC
from gradwave.dtypes import CDTYPE
from gradwave.grids import FFTGrid
from gradwave.solvers.precond import teter_b


class _AppliesH(Protocol):
    """Anything ``cg_sternheimer`` can call as a batched Hamiltonian: only
    ``.apply(x)`` is read. ``BatchedHamiltonian`` satisfies this; so does
    ``scf.noncollinear.SpinorHamiltonian`` (postscf.dielectric's spinor
    Sternheimer solves reuse this same CG unchanged)."""

    # positional-only (`/`): BatchedHamiltonian/SpinorHamiltonian both name
    # this parameter `c`, not `x` -- Protocol method matching cares about
    # parameter names unless marked positional-only.
    def apply(self, x: torch.Tensor, /) -> torch.Tensor: ...


class _HasKineticTable(Protocol):
    """Anything ``cg_sternheimer`` can use for the Teter preconditioner: only
    ``.t`` (the kinetic table) is read -- not the full ``BatchedK`` contract.
    ``BatchedK`` satisfies this; so does the ``SimpleNamespace(t=...)`` shim
    postscf.dielectric's spinor path builds (a doubled-plane-wave-axis kinetic
    table is all this function needs to run unchanged on spinors)."""

    t: torch.Tensor


class DysonNotConverged(RuntimeError):
    """The screening Dyson fixed point did not reach ``tol`` within ``max_iter``."""


@torch.no_grad()
def dyson_fixed_point(op: Callable[[torch.Tensor], torch.Tensor], rhs: torch.Tensor, *,
                      beta: float, tol: float, max_iter: int,
                      on_fail: Callable[[float], None] | None = None,
                      denom_new: bool = False,
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


def hartree_kernel(grid: FFTGrid, w_r: torch.Tensor) -> torch.Tensor:
    """(K_H w)(r): the Hartree kernel 4πe²/G² applied to a real grid field
    (G=0 excluded)."""
    w_g = r_to_g(w_r.to(CDTYPE))
    inv_g2 = torch.where(
        grid.g2 > 1e-12, 1.0 / torch.clamp(grid.g2, min=1e-12),
        torch.zeros_like(grid.g2))
    return g_to_r_box(4.0 * math.pi * E2 * w_g * inv_g2, real=True)


def fxc_hvp(xc: XCFunctional, rho0: torch.Tensor, grid: FFTGrid,
            w_r: torch.Tensor) -> torch.Tensor:
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


def fxc_hvp_spin(xc: SpinXC, ru0: torch.Tensor, rd0: torch.Tensor,
                 grid: FFTGrid, wu: torch.Tensor,
                 wd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def fxc_hvp_noncollinear_nonmagnetic(xc: NoncollinearXC, rho0: torch.Tensor, grid: FFTGrid,
                                     w_r: torch.Tensor) -> torch.Tensor:
    """f_xc·w at m⃗ ≡ 0 for a ``core.xc.noncollinear.NoncollinearXC``, in
    physical units [eV]. ``rho0`` already includes any NLCC core (the same
    caller convention as ``fxc_hvp``).

    At the pinned nonmagnetic manifold the locally-collinear energy
    (``core.xc.noncollinear.energy_with_grid``) reduces exactly to the
    spin-restricted functional ``xc.collinear`` wraps (ρ± = ρ/2, up to the
    O(m_eps) regularization), so this is the (ρ,ρ)-block-only noncollinear/SOC
    counterpart of ``fxc_hvp`` — the screening kernel the fully-relativistic
    dielectric response uses at m⃗ ≡ 0 (postscf.dielectric._dielectric_born_soc,
    _k_hxc_soc). A nonzero moment needs the coupled (ρ, m⃗) Hessian-vector
    product, which this does not provide.
    """
    from gradwave.core.xc.noncollinear import energy_with_grid

    rho = rho0.detach().clone().requires_grad_(True)
    m_zero = torch.zeros(3, *rho.shape, dtype=rho.dtype, device=rho.device)
    with torch.enable_grad(), xc_eager():
        e_xc = energy_with_grid(xc, rho, m_zero, grid)
        (v_xc,) = torch.autograd.grad(e_xc, rho, create_graph=True)
        inner = (v_xc * w_r.detach()).sum()
        (fxc_w,) = torch.autograd.grad(inner, rho)
    return fxc_w * (grid.n_points / grid.volume)


# --------------------------------------------------------------------------- #
#  Energy-metric SCF convergence estimate: 1/2 <r|K_Hxc|r>                     #
# --------------------------------------------------------------------------- #


def _k_hxc_apply(grid: FFTGrid, xc, w_s: list[torch.Tensor],
                 rho_s: list[torch.Tensor], rho_core: torch.Tensor | None,
                 nspin: int) -> list[torch.Tensor]:
    """(K_Hxc w)_σ per spin channel [eV] for a per-spin field list ``w_s``.

    Mirrors ``scf.implicit.apply_k_hxc`` but takes raw grid densities (no
    ``SCFResult``), so a mid-SCF loop can call it. The Hartree kernel acts on
    the TOTAL δρ = Σ_σ w_σ and enters every channel; f_xc is the LDA/GGA spin
    HVP of E_xc at the current densities, NLCC core split half/half (nspin=2)
    or added whole (nspin=1) — the exact evaluation point the SCF's own v_xc
    uses (``scf.loop.effective_potentials``)."""
    if nspin == 1:
        rho_xc = rho_s[0] if rho_core is None else rho_s[0] + rho_core
        return [hartree_kernel(grid, w_s[0]) + fxc_hvp(xc, rho_xc, grid, w_s[0])]
    kh = hartree_kernel(grid, w_s[0] + w_s[1])
    c2 = 0.0 if rho_core is None else 0.5 * rho_core
    fu, fd = fxc_hvp_spin(xc, rho_s[0] + c2, rho_s[1] + c2, grid, w_s[0], w_s[1])
    return [kh + fu, kh + fd]


def kernel_energy_error(grid: FFTGrid, xc, r_s: list[torch.Tensor],
                        rho_s: list[torch.Tensor],
                        rho_core: torch.Tensor | None = None,
                        nspin: int = 1) -> tuple[float, float, float]:
    """Second-order SCF energy-error estimate 1/2 <r|K_Hxc|r> [eV] of a density
    residual r = ρ_out − ρ_in, decomposed into charge and magnetization
    channels. Returns ``(e_total, e_charge, e_mag)``.

    Because the energy is stationary at the SCF fixed point, the residual's
    energy error is second order, 1/2 <r|(K_Hxc + χ₀)|r> (docs/ideas.md's
    error-budget section). This computes the KERNEL-ONLY contraction
    1/2 <r|K_Hxc|r> exactly — Hartree kernel 4πe²/G² plus the f_xc HVP of the
    XC energy — which is the QE-comparable quantity (QE's "estimated scf
    accuracy" is the un-halved <r|K_Hxc|r>, so this is about half of it). It
    OMITS the χ₀ independent-particle response term, whose one application needs
    a conduction-projected Sternheimer solve per occupied band per k restricted
    to nspin=1 insulators — not cheap per SCF iteration, and inapplicable to
    the metallic magnets this gate is built for. It also omits any Hubbard (+U)
    or Fock (hybrid) second-order kernel.

    The charge/magnetization split (nspin=2) writes the residual pair as
    c = ((r↑+r↓)/2, (r↑+r↓)/2) and m = ((r↑−r↓)/2, (r↓−r↑)/2); ``e_charge`` and
    ``e_mag`` are each channel's own 1/2 <·|K_Hxc|·>. They do NOT sum to
    ``e_total`` because the f_xc kernel carries a charge<->magnetization cross
    term (``e_total − e_charge − e_mag``). The Hartree kernel contributes only
    to ``e_charge`` (it acts on the total charge, which is zero for the pure-mag
    pair). ``e_mag = 0`` for nspin=1.

    Raises ``NotImplementedError`` for a meta-GGA (``needs_tau``): the kernel
    here omits the kinetic-energy-density (τ) response.
    """
    if getattr(xc, "needs_tau", False):
        raise NotImplementedError(
            "energy-metric convergence gate does not support meta-GGA "
            "(needs_tau): the K_Hxc kernel here omits the kinetic-energy-density "
            "(tau) response")
    cell = grid.volume / grid.n_points
    kr = _k_hxc_apply(grid, xc, r_s, rho_s, rho_core, nspin)
    e_total = 0.5 * sum(float((r_s[s] * kr[s]).sum()) for s in range(nspin)) * cell
    if nspin == 1:
        return e_total, e_total, 0.0
    # Linear kernel: the swap r↑<->r↓ gives K on the (charge, mag) basis in two
    # applications (see the derivation above).
    kr_sw = _k_hxc_apply(grid, xc, [r_s[1], r_s[0]], rho_s, rho_core, nspin)
    rn2 = 0.5 * (r_s[0] + r_s[1])       # charge residual / 2 (both channels)
    rm2 = 0.5 * (r_s[0] - r_s[1])       # +mag residual / 2 (spin up channel)
    kc = [0.5 * (kr[0] + kr_sw[0]), 0.5 * (kr[1] + kr_sw[1])]
    km = [0.5 * (kr[0] - kr_sw[0]), 0.5 * (kr[1] - kr_sw[1])]
    e_charge = 0.5 * (float((rn2 * kc[0]).sum()) + float((rn2 * kc[1]).sum())) * cell
    e_mag = 0.5 * (float((rm2 * km[0]).sum()) + float((-rm2 * km[1]).sum())) * cell
    return e_total, e_charge, e_mag


def _fxc_hvp_noncollinear(
    xc: NoncollinearXC, rho0: torch.Tensor, m0: torch.Tensor, grid: FFTGrid,
    w_rho: torch.Tensor, w_m: torch.Tensor,
    rho_core: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coupled (ρ, m⃗) f_xc Hessian-vector product of the noncollinear E_xc at
    (rho0, m0), in physical units [eV]. Returns (f_ρ, f_m⃗), the ρ- and
    m⃗-components of f_xc·(w_ρ, w_m⃗).

    Double backward through ``core.xc.noncollinear.energy_with_grid``, the same
    locally-collinear energy the spinor SCF's v_xc/B⃗_xc autograd uses
    (``vxc_and_bxc``), so the kernel is evaluated at the SCF's own linearization
    point. ``xc_eager()`` forces eager mode, since compiled aot_autograd cannot
    double-backward. rho_core (NLCC) is folded into ρ inside ``energy_with_grid``,
    matching the SCF, and the returned fields carry the grid-cell → physical
    conversion n_points/Ω (see ``fxc_hvp``). Unlike
    ``fxc_hvp_noncollinear_nonmagnetic`` this keeps the full charge<->magnetization
    coupling and works at a nonzero moment.
    """
    from gradwave.core.xc.noncollinear import energy_with_grid

    rho = rho0.detach().clone().requires_grad_(True)
    m = m0.detach().clone().requires_grad_(True)
    with torch.enable_grad(), xc_eager():
        e_xc = energy_with_grid(xc, rho, m, grid, rho_core=rho_core)
        v_rho, v_m = torch.autograd.grad(e_xc, (rho, m), create_graph=True)
        inner = (v_rho * w_rho.detach()).sum() + (v_m * w_m.detach()).sum()
        f_rho, f_m = torch.autograd.grad(inner, (rho, m))
    scale = grid.n_points / grid.volume
    return f_rho * scale, f_m * scale


def kernel_energy_error_noncollinear(
    grid: FFTGrid, xc: NoncollinearXC, r_rho: torch.Tensor, r_m: torch.Tensor,
    rho0: torch.Tensor, m0: torch.Tensor,
    rho_core: torch.Tensor | None = None,
) -> tuple[float, float, float, float, float]:
    """Second-order SCF energy-error estimate 1/2 <r|K_Hxc|r> [eV] of a spinor
    density residual r = (r_ρ, r_m⃗), decomposed into charge, longitudinal, and
    transverse magnetization channels. Returns
    ``(e_total, e_charge, e_mag, e_long, e_trans)``.

    The noncollinear counterpart of ``kernel_energy_error``. K_Hxc is the Hartree
    kernel 4πe²/G² acting on the charge channel plus the EXACT coupled (ρ, m⃗)
    f_xc Hessian-vector product of the noncollinear E_xc (``_fxc_hvp_noncollinear``),
    evaluated at the iteration's input density (rho0, m0). The f_xc term carries the
    full charge<->magnetization coupling, so no channel is dropped, and at a moment
    purely along one axis this reduces exactly to the collinear ``kernel_energy_error``
    (verified to machine precision).

    Omissions, mirroring ``kernel_energy_error``. The χ₀ independent-particle
    response term needs a conduction-projected Sternheimer solve per band restricted
    to insulators, neither cheap per iteration nor applicable to the metallic magnets
    this gate targets, so it is omitted, as are the Hubbard (+U) and Fock second-order
    kernels. A meta-GGA (``needs_tau``) raises rather than return a silently-wrong
    estimate that drops the kinetic-energy-density response.

    Channel decomposition. The magnetization residual is split about the global
    integrated-moment axis m̂ = ∫m⃗/|∫m⃗| into a longitudinal part
    r_∥ = (r_m⃗·m̂) m̂ and a transverse part r_⊥ = r_m⃗ − r_∥. This is the
    campaign's floor decomposition (research/noncollinear-convergence), where the
    transverse magnon-soft modes carry the density-residual floor but little energy
    and the longitudinal near-Stoner channel carries the rest. ``e_charge``,
    ``e_long``, and ``e_trans`` are each the pure diagonal quadratic form
    1/2<·|K_Hxc|·> of that channel alone; ``e_mag`` is the full magnetization block
    1/2<(0,r_m⃗)|K_Hxc|(0,r_m⃗)> = e_long + e_trans + a longitudinal<->transverse
    cross term. The four do NOT sum to ``e_total``, which also carries the
    charge<->magnetization f_xc cross term. With no net moment (a nonmagnetic or
    fully-compensated cell) the whole magnetization residual is reported as
    transverse, there being no longitudinal axis.
    """
    if getattr(xc, "needs_tau", False):
        raise NotImplementedError(
            "energy-metric convergence gate does not support meta-GGA "
            "(needs_tau): the noncollinear K_Hxc kernel here omits the "
            "kinetic-energy-density (tau) response")
    cell = grid.volume / grid.n_points
    zero_rho = torch.zeros_like(r_rho)
    zero_m = torch.zeros_like(r_m)

    # longitudinal/transverse split about the global integrated-moment axis of
    # the state (∫ m0 dr), the near-collinear magnet's well-defined moment axis
    m_int = torch.stack([m0[i].sum() for i in range(3)]) * cell
    m_norm = float(torch.linalg.norm(m_int))
    if m_norm > 1e-8:
        m_hat = (m_int / m_norm).to(r_m.dtype)
        r_par = sum(r_m[i] * m_hat[i] for i in range(3))
        r_long = torch.stack([r_par * m_hat[i] for i in range(3)])
        r_trans = r_m - r_long
    else:
        r_long, r_trans = zero_m, r_m

    kh_r = hartree_kernel(grid, r_rho)  # charge-only Hartree kernel

    # three f_xc block applications (f_xc is linear): (r_ρ,0), (0,r_∥), (0,r_⊥)
    frr_rho, frr_m = _fxc_hvp_noncollinear(xc, rho0, m0, grid, r_rho, zero_m, rho_core)
    fl_rho, fl_m = _fxc_hvp_noncollinear(xc, rho0, m0, grid, zero_rho, r_long, rho_core)
    ft_rho, ft_m = _fxc_hvp_noncollinear(xc, rho0, m0, grid, zero_rho, r_trans, rho_core)

    def _dot_m(a, b):
        return sum(float((a[i] * b[i]).sum()) for i in range(3))

    e_charge = 0.5 * (float((r_rho * kh_r).sum()) + float((r_rho * frr_rho).sum())) * cell
    e_long = 0.5 * _dot_m(r_long, fl_m) * cell
    e_trans = 0.5 * _dot_m(r_trans, ft_m) * cell
    fm_m = [fl_m[i] + ft_m[i] for i in range(3)]  # f_xc·(0, r_m⃗), m-component
    e_mag = 0.5 * _dot_m(r_m, fm_m) * cell
    # total: K_Hxc·(r_ρ, r_m⃗) contracted with (r_ρ, r_m⃗), by linearity
    f_rho_tot = kh_r + frr_rho + fl_rho + ft_rho
    f_m_tot = [frr_m[i] + fl_m[i] + ft_m[i] for i in range(3)]
    e_total = 0.5 * (float((r_rho * f_rho_tot).sum()) + _dot_m(r_m, f_m_tot)) * cell
    return e_total, e_charge, e_mag, e_long, e_trans


def spin_sigma_triple(
    xc: SpinXC, r_u: torch.Tensor, r_d: torch.Tensor, g_cart: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
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


def is_insulating(occ: torch.Tensor, f_full: float, tol: float = 1e-6) -> bool:
    """True iff every occupation in ``occ`` is (near-)integer — 0 or ``f_full``.

    The dispatch that separates the exact insulator χ₀ path (fully-occupied
    Sternheimer, no Fermi surface) from the metallic partial-occupation path.
    A single fractional occupation anywhere flips it to the metallic branch,
    matching the wisdom-note guidance that the metallic response must be tested
    only where fractional occupations actually exist."""
    frac = (occ > tol) & (occ < f_full - tol)
    return not bool(frac.any())


def occupation_derivative(eps: torch.Tensor, mu: float, scheme, width: float,
                          degeneracy: float) -> torch.Tensor:
    """d(occ_n)/dε_n [1/eV] for a smearing ``scheme`` — the (negative) local
    density-of-states weight −(g/σ)·δ̃((ε−μ)/σ).

    ``scheme`` is a ``core.occupations.Smearing``; ``occ = g·f((ε−μ)/σ)`` so
    ``d occ/dε = (g/σ)·f'`` with ``f'`` taken by autograd through the scheme's
    own occupation function (exact for any of the four schemes, no separate
    derivative to keep in sync). Returns a tensor shaped like ``eps``, ≤ 0."""
    x = ((eps.detach() - mu) / width).clone().requires_grad_(True)
    with torch.enable_grad():
        f = scheme.occupation(x).sum()
        (df,) = torch.autograd.grad(f, x)
    return degeneracy * df / width


def divided_difference_weights(eps: torch.Tensor, occ: torch.Tensor,
                               occ_deriv: torch.Tensor,
                               deg_tol: float = 1e-6) -> torch.Tensor:
    """The Adler–Wiser divided-difference matrix β_nm = (occ_n−occ_m)/(ε_n−ε_m)
    of one k-point's window, with the analytic degenerate limit on the diagonal
    and near-degenerate pairs.

    For |ε_n−ε_m| > ``deg_tol`` this is the plain divided difference; as
    ε_m → ε_n (including n=m) it goes to the derivative, taken here as the
    average of the two endpoint slopes ½(occ'_n+occ'_m). β is real-symmetric;
    the diagonal is occ'_n (the Fermi-surface / occupation-response weight).
    Deep pairs where occ_n=occ_m (both full or both empty) give β=0 — Pauli
    blocking, no interband response. Shape (nb, nb)."""
    de = eps[:, None] - eps[None, :]
    do = occ[:, None] - occ[None, :]
    near = de.abs() < deg_tol
    safe_de = torch.where(near, torch.ones_like(de), de)
    beta = do / safe_de
    avg = 0.5 * (occ_deriv[:, None] + occ_deriv[None, :])
    return torch.where(near, avg, beta)


def pad_coeffs(coeffs_per_k: list[torch.Tensor], npw_max: int,
               device: torch.device | None=None) -> torch.Tensor:
    """[(nb, npw_k)] per k → padded (nk, nb, npw_max), detached. `device`
    defaults to the coeffs' own device (a no-op move in that case)."""
    nk = len(coeffs_per_k)
    nb = coeffs_per_k[0].shape[0]
    dev = device if device is not None else coeffs_per_k[0].device
    out = torch.zeros(nk, nb, npw_max, dtype=CDTYPE, device=dev)
    for ik, c in enumerate(coeffs_per_k):
        out[ik, :, : c.shape[1]] = c.detach().to(dev)
    return out


def cg_sternheimer(h: _AppliesH, bk: _HasKineticTable, c_occ: torch.Tensor,
                   eps_occ: torch.Tensor, rhs: torch.Tensor, x0: torch.Tensor, shift: float,
                   tol: float=1e-8, max_iter: int=400) -> torch.Tensor:
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


def resolvent_sternheimer(h, bk: _HasKineticTable, c_occ: torch.Tensor,
                          eps_occ: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Direct sum-over-states Sternheimer — a drop-in alternative to
    ``cg_sternheimer`` with the same masked (nk, nocc, npw_max) conduction-space
    output (``rhs`` already conduction-projected).

    For each k, assemble the dense ground-state Hamiltonian H_k, diagonalize it
    ONCE (H_k = U diag(Λ) Uᴴ), and back-substitute every occupied band's
    right-hand side through the conduction resolvent

        δψ_n = Σ_{c∈cond} U_c ⟨U_c|rhs_n⟩ / (Λ_c − ε_{n,k}) .

    INSULATORS and small cells only: the one eigh (O(nk·npw³)) amortizes over every
    right-hand side / perturbation / DFPT iteration that shares the fixed H, so it
    wins where many solves reuse one operator (measured ~2–4× whole-phonon-DFPT vs
    warm CG, agreeing to the CG tolerance). No shift is needed — the sum runs over
    the conduction block only, so the occupied poles never appear. ``h`` must expose
    the dense operator's pieces (a ``BatchedHamiltonian``: ``_tables``, ``shape``,
    ``v_eff_r``, ``gather_idx``). No-grad forward; the differentiable adjoint is a
    separate custom Function (it re-applies the resolvent, never differentiates the
    degenerate eigh)."""
    cdtype = rhs.dtype
    t_r, _v_eff, p, _p_conj, dij = h._tables(cdtype)
    shape = h.shape
    s1s2, s2 = shape[1] * shape[2], shape[2]
    vhat = r_to_g(h.v_eff_r.to(cdtype)).reshape(-1)
    ntorch = torch.tensor(shape, device=rhs.device)
    nk, nocc, _ = rhs.shape
    dpsi = torch.zeros_like(rhs)
    for k in range(nk):
        valid = bk.mask[k]
        flat = h.gather_idx[k][valid].to(torch.long)
        rem = flat % s1s2
        g = torch.stack([flat // s1s2, rem // s2, rem % s2], dim=-1)  # (nv, 3)
        diff = (g[:, None, :] - g[None, :, :]) % ntorch
        idx = diff[..., 0] * s1s2 + diff[..., 1] * s2 + diff[..., 2]
        hk = torch.diag(t_r[k][valid].to(cdtype)) + vhat[idx]         # kinetic + local
        if p.shape[1]:
            pk = p[k][:, valid]
            hk = hk + (pk.conj().T @ dij @ pk).T                      # KB nonlocal (col form)
        hk = 0.5 * (hk + hk.conj().T)
        w, u = torch.linalg.eigh(hk)
        uc, wc = u[:, nocc:], w[nocc:]                               # conduction block
        proj = uc.conj().T @ rhs[k][:, valid].T                      # (ncond, nocc)
        dpsi[k][:, valid] = (uc @ (proj / (wc[:, None] - eps_occ[k][None, :]))).T
    return dpsi
