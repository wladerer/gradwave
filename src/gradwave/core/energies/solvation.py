"""Implicit (continuum) solvation — SCCS generalized-Poisson solver (Layer A).

Increment 0: the electrostatic solver on a **frozen** density ρ, validated
against analytic references (Born, planar dielectric step, FD-vs-autograd). No
SCF coupling yet — that is the follow-up increment.

The Self-Consistent Continuum Solvation model (Andreussi, Dabo, Marzari,
*J. Chem. Phys.* **136**, 064102 (2012)) embeds the solute in a smooth dielectric
that is a functional of its own electron density:

    ε[ρ](r) = 1                 where ρ ≥ ρ_max   (inside the solute)
    ε[ρ](r) = ε_bulk            where ρ ≤ ρ_min   (in the bulk solvent)

with a smooth switch in between (the Andreussi 2012 form, below). The solvated
electrostatic potential solves the *generalized* Poisson equation

    ∇·(ε[ρ](r) ∇φ(r)) = −4π e² ρ(r)                                        (GPE)

(gradwave's Poisson sign/units: with ε≡1 this is −∇²φ = 4π e² ρ, exactly the
``hartree`` convention, e² = ``constants.E2`` in eV·Å). The GPE is solved by the
standard polarization-charge fixed point (Fisicaro et al., *J. Chem. Phys.*
**144**, 014103 (2016)): each iterate is one ordinary FFT-Poisson solve of an
effective charge ρ + ρ_pol, reusing the reciprocal-space Hartree kernel

    ∇²φ = −4π e² [ ρ/ε + (1/4π e²) ∇ln ε · ∇φ ]
        = −4π e² (ρ + ρ_pol),
    ρ_pol = ρ (1/ε − 1) + (1/4π e²) ∇ln ε · ∇φ  (the bound / polarization charge).

For a **uniform** ε (∇ln ε = 0) this converges in a single step to φ = φ_vac/ε,
reproducing the Born reaction field exactly.

Everything here is written out-of-place, so autograd flows through the whole
solve. The **reaction-field energy** G_sol = ½∫ρ(φ_sol − φ_vac) plus the SCCS
non-electrostatic (cavitation + dispersion–repulsion) surface/volume term is a
pure function of ρ; its autograd gradient δG/δρ is the SCF *solvation potential*,
**including the cavity term** −(1/8π e²)|∇φ|² δε/δρ that Environ/VASPsol
hand-derive. That is the differentiable-DFT novelty: the cavity potential, and
the cavity forces/stress, come from `torch.autograd` for free.

Pairs with the shipped constant-μ ESM electrode (``energies/esm.py``) toward
solvated constant-potential electrocatalysis — a combination no plane-wave code
packages. See the follow-up increments in the module's PR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gradwave.constants import E2
from gradwave.core.energies.hartree import hartree_potential_r
from gradwave.core.fftbox import g_to_r_box, r_to_g

# ρ below this positive floor is treated as ρ_floor before the log, so a slightly
# negative grid density (FFT ringing) or a true zero cannot poison ∂ε/∂ρ with a
# NaN. Physically ρ ≥ 0; the switch is flat (ε = ε_bulk) for any ρ ≤ ρ_min ≫ this.
_RHO_FLOOR = 1e-12
# Softening added under the surface-integral norm √(|∇c|² + δ²) so its gradient is
# finite where ∇c = 0 (bulk / deep interior). δ is in the field's own units.
_SURF_SOFT = 1e-12


def andreussi_epsilon(
    rho_r: torch.Tensor,
    eps_bulk: float,
    rho_min: float,
    rho_max: float,
) -> torch.Tensor:
    """Andreussi (SCCS) dielectric ε[ρ](r), differentiable in ρ.

    ε = 1 for ρ ≥ ρ_max (solute), ε = ``eps_bulk`` for ρ ≤ ρ_min (solvent), with
    the smooth Andreussi 2012 switch between the two thresholds:

        x  = clamp( (ln ρ_max − ln ρ) / (ln ρ_max − ln ρ_min), 0, 1 )
        t  = (2π x − sin 2π x) / 2π                     (0→1, C¹ at both ends)
        ε  = exp( ln(ε_bulk) · t )                      (so ε ≥ 1 everywhere)

    The exponential interpolation keeps ε ≥ 1 and t is C¹ (dt/dx = 1 − cos 2πx
    vanishes at x = 0, 1), so ε and its density derivative are continuous at the
    thresholds. ``rho_min`` < ``rho_max``, both > 0.
    """
    if not (0.0 < rho_min < rho_max):
        raise ValueError(
            f"require 0 < rho_min < rho_max, got rho_min={rho_min}, rho_max={rho_max}")
    if eps_bulk < 1.0:
        raise ValueError(f"eps_bulk must be ≥ 1, got {eps_bulk}")
    ln_rho = torch.log(rho_r.clamp(min=_RHO_FLOOR))
    denom = math.log(rho_max) - math.log(rho_min)
    x = ((math.log(rho_max) - ln_rho) / denom).clamp(0.0, 1.0)
    t = (2.0 * math.pi * x - torch.sin(2.0 * math.pi * x)) / (2.0 * math.pi)
    return torch.exp(math.log(eps_bulk) * t)


def _gradient(field_r: torch.Tensor, g_cart: torch.Tensor) -> torch.Tensor:
    """∇field(r) on the box, (3, n1, n2, n3) real. ∂_α f = ifft(i G_α f̃(G))."""
    f_g = r_to_g(field_r.to(g_cart.dtype))
    comps = [g_to_r_box(1j * g_cart[..., a] * f_g, real=True) for a in range(3)]
    return torch.stack(comps, dim=0)


@dataclass(frozen=True)
class GPESolution:
    """Result of :func:`solve_generalized_poisson`."""

    phi: torch.Tensor  # φ(r) [eV], solvated electrostatic potential (G=0 gauge = 0)
    rho_pol: torch.Tensor  # bound/polarization charge ρ_pol(r) [e/Å³]
    iterations: int  # fixed-point iterations taken
    residual: float  # final ‖φ_new − φ_old‖∞ / ‖φ_new‖∞


def solve_generalized_poisson(
    rho_r: torch.Tensor,
    eps_r: torch.Tensor,
    g2: torch.Tensor,
    g_cart: torch.Tensor,
    *,
    max_iter: int = 400,
    tol: float = 1e-9,
    mixing: float = 0.6,
) -> GPESolution:
    """Solve ∇·(ε∇φ) = −4π e² ρ on the FFT box for a **frozen** ρ and ε.

    Polarization-charge fixed point: each iterate is one FFT-Poisson solve
    (``hartree_potential_r``) of the effective charge ρ + ρ_pol, with linear
    ``mixing`` on φ. Out-of-place throughout, so autograd flows through the whole
    solve (unrolled) — that is what makes δG/δρ, including the δε/δρ cavity term,
    available by backprop.

    rho_r, eps_r: (n1,n2,n3) real. g2, g_cart: the box ``FFTGrid.g2``/``g_cart``.
    Returns a :class:`GPESolution`. For uniform ε it converges in one iterate.
    """
    inv_4pi_e2 = 1.0 / (4.0 * math.pi * E2)
    ln_eps = torch.log(eps_r)
    grad_ln_eps = _gradient(ln_eps, g_cart)  # (3, ...)
    inv_eps_m1 = 1.0 / eps_r - 1.0

    phi = hartree_potential_r(rho_r, g2)  # vacuum solve as the initial guess
    residual = float("inf")
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 — `it` is returned as the count
        grad_phi = _gradient(phi, g_cart)  # (3, ...)
        rho_pol = rho_r * inv_eps_m1 + inv_4pi_e2 * (grad_ln_eps * grad_phi).sum(0)
        phi_new = hartree_potential_r(rho_r + rho_pol, g2)
        denom = phi_new.abs().max().clamp(min=1e-30)
        residual = float(((phi_new - phi).abs().max() / denom).detach())
        phi = (1.0 - mixing) * phi + mixing * phi_new
        if residual < tol:
            break
    # Recompute the bound charge consistent with the returned φ.
    grad_phi = _gradient(phi, g_cart)
    rho_pol = rho_r * inv_eps_m1 + inv_4pi_e2 * (grad_ln_eps * grad_phi).sum(0)
    return GPESolution(phi=phi, rho_pol=rho_pol, iterations=it, residual=residual)


def reaction_field_energy(
    rho_r: torch.Tensor,
    grid,
    *,
    eps_bulk: float,
    rho_min: float,
    rho_max: float,
    max_iter: int = 400,
    tol: float = 1e-9,
    mixing: float = 0.6,
) -> torch.Tensor:
    """Electrostatic solvation (reaction-field) energy G_el [eV], a pure function
    of ρ: builds ε[ρ], solves the GPE, and returns ½∫ρ(φ_sol − φ_vac) dr.

    Differentiable in ``rho_r``; its autograd gradient is the electrostatic
    solvation potential (reaction field + the −(1/8π e²)|∇φ|² δε/δρ cavity term).
    """
    eps_r = andreussi_epsilon(rho_r, eps_bulk, rho_min, rho_max)
    sol = solve_generalized_poisson(
        rho_r, eps_r, grid.g2, grid.g_cart, max_iter=max_iter, tol=tol, mixing=mixing)
    phi_vac = hartree_potential_r(rho_r, grid.g2)
    dvol = grid.volume / rho_r.numel()
    return 0.5 * (rho_r * (sol.phi - phi_vac)).sum() * dvol


def cavity_indicator(
    rho_r: torch.Tensor,
    eps_bulk: float,
    rho_min: float,
    rho_max: float,
) -> torch.Tensor:
    """Solute-cavity indicator c(ρ) = (ε_bulk − ε[ρ])/(ε_bulk − 1) ∈ [0, 1].

    c = 1 inside the solute (ε = 1), c = 0 in bulk solvent (ε = ε_bulk). Smooth
    (inherits the Andreussi switch), so ∫c and ∫|∇c| are the SCCS cavity volume
    and quantum surface.
    """
    eps_r = andreussi_epsilon(rho_r, eps_bulk, rho_min, rho_max)
    return (eps_bulk - eps_r) / (eps_bulk - 1.0)


def cavity_volume_surface(
    rho_r: torch.Tensor,
    grid,
    *,
    eps_bulk: float,
    rho_min: float,
    rho_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SCCS cavity volume V = ∫c dr [Å³] and quantum surface S = ∫|∇c| dr [Å²],
    with c the :func:`cavity_indicator`. Both differentiable in ρ."""
    c = cavity_indicator(rho_r, eps_bulk, rho_min, rho_max)
    dvol = grid.volume / rho_r.numel()
    vol = c.sum() * dvol
    grad_c = _gradient(c, grid.g_cart)  # (3, ...)
    surf = torch.sqrt((grad_c * grad_c).sum(0) + _SURF_SOFT).sum() * dvol
    return vol, surf


def solvation_energy(
    rho_r: torch.Tensor,
    grid,
    *,
    eps_bulk: float,
    rho_min: float,
    rho_max: float,
    gamma: float = 0.0,
    pressure: float = 0.0,
    max_iter: int = 400,
    tol: float = 1e-9,
    mixing: float = 0.6,
) -> torch.Tensor:
    """Total SCCS solvation free energy G_sol [eV], a pure function of ρ:

        G_sol = G_el + γ·S + p·V

    the electrostatic reaction field (:func:`reaction_field_energy`) plus the
    non-electrostatic cavitation/dispersion–repulsion surface (``gamma``, a
    surface tension [eV/Å²]) and volume (``pressure`` [eV/Å³]) terms on the SCCS
    cavity. Differentiable in ``rho_r``: ``torch.autograd.grad(G_sol, ρ)`` is the
    full solvation potential v_solv = δG/δρ (electrostatic + cavity), and its
    gradient w.r.t. atomic positions / strain gives the cavity forces / stress.
    """
    g_el = reaction_field_energy(
        rho_r, grid, eps_bulk=eps_bulk, rho_min=rho_min, rho_max=rho_max,
        max_iter=max_iter, tol=tol, mixing=mixing)
    if gamma == 0.0 and pressure == 0.0:
        return g_el
    vol, surf = cavity_volume_surface(
        rho_r, grid, eps_bulk=eps_bulk, rho_min=rho_min, rho_max=rho_max)
    return g_el + gamma * surf + pressure * vol


def solvation_potential(
    rho_r: torch.Tensor,
    grid,
    *,
    eps_bulk: float,
    rho_min: float,
    rho_max: float,
    gamma: float = 0.0,
    pressure: float = 0.0,
    max_iter: int = 400,
    tol: float = 1e-9,
    mixing: float = 0.6,
) -> torch.Tensor:
    """Solvation SCF potential v_solv(r) [eV] = δG_sol/δρ, by autograd.

    The exact functional derivative of :func:`solvation_energy`, so it carries the
    cavity term −(1/8π e²)|∇φ|² δε/δρ automatically — no hand-derived Environ /
    VASPsol expression. For the increment-0 solver this is validated against a
    finite-difference δG/δρ (the follow-up increment feeds it into the SCF).
    """
    with torch.enable_grad():
        r = rho_r.detach().requires_grad_(True)
        g = solvation_energy(
            r, grid, eps_bulk=eps_bulk, rho_min=rho_min, rho_max=rho_max,
            gamma=gamma, pressure=pressure, max_iter=max_iter, tol=tol, mixing=mixing)
        (grad,) = torch.autograd.grad(g, r)
    dvol = grid.volume / rho_r.numel()
    return grad / dvol
