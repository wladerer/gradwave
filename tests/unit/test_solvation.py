"""SCCS generalized-Poisson solvation solver (core/energies/solvation.py).

Increment 0: the electrostatic solver on a FROZEN density, checked against
analytic references only (no SCF):

  * the Andreussi ε[ρ] switch hits 1 and ε_bulk at the density thresholds;
  * a charge in a UNIFORM dielectric reproduces the Born reaction field exactly
    (φ_sol = φ_vac/ε and G_el = (1/ε − 1)·E_H, linear in (1/ε − 1));
  * a PLANAR dielectric step reproduces the 1D generalized-Poisson (image-charge)
    solution, i.e. D = ε∇φ continuity across the step;
  * the autograd solvation potential δG/δρ (electrostatic + the δε/δρ cavity term
    + the non-electrostatic surface/volume terms) matches a finite-difference
    δG/δρ to tight tolerance — the differentiable-DFT novelty.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.core.energies.hartree import hartree_potential_r
from gradwave.core.energies.solvation import (
    andreussi_epsilon,
    cavity_indicator,
    reaction_field_energy,
    solvation_energy,
    solvation_potential,
    solve_generalized_poisson,
)
from gradwave.grids import build_fft_grid

DT = torch.float64


def _gaussian_density(grid, center, sigma, q=1.0):
    n = grid.shape
    L = [float(np.linalg.norm(np.asarray(grid.cell)[i])) for i in range(3)]
    ax = [torch.arange(n[i], dtype=DT) * (L[i] / n[i]) for i in range(3)]
    X, Y, Z = torch.meshgrid(*ax, indexing="ij")
    r2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    return q * (2 * math.pi * sigma**2) ** (-1.5) * torch.exp(-r2 / (2 * sigma**2))


# --------------------------------------------------------------------------- #
# (d) the ε[ρ] switching function
# --------------------------------------------------------------------------- #
def test_andreussi_switch_hits_thresholds():
    eps_bulk, rho_min, rho_max = 78.4, 1e-4, 5e-3
    rho = torch.tensor([1e-9, rho_min, rho_max, 1.0], dtype=DT)
    eps = andreussi_epsilon(rho, eps_bulk, rho_min, rho_max)
    # ρ ≤ ρ_min → bulk solvent; ρ ≥ ρ_max → solute (ε = 1)
    assert abs(float(eps[0]) - eps_bulk) < 1e-10  # deep solvent
    assert abs(float(eps[1]) - eps_bulk) < 1e-10  # at ρ_min
    assert abs(float(eps[2]) - 1.0) < 1e-10       # at ρ_max
    assert abs(float(eps[3]) - 1.0) < 1e-10       # deep solute
    # monotone increasing as ρ decreases through the window; C¹ midpoint
    mid = float(andreussi_epsilon(
        torch.tensor([math.sqrt(rho_min * rho_max)], dtype=DT),
        eps_bulk, rho_min, rho_max))
    assert 1.0 < mid < eps_bulk
    # geometric midpoint of ln ρ → t = 1/2 → ε = √ε_bulk
    assert abs(mid - math.sqrt(eps_bulk)) < 1e-10


def test_andreussi_switch_differentiable_and_bounded():
    eps_bulk, rho_min, rho_max = 40.0, 1e-4, 5e-3
    rho = torch.logspace(-6, 0, 50, dtype=DT).requires_grad_(True)
    eps = andreussi_epsilon(rho, eps_bulk, rho_min, rho_max)
    assert torch.all(eps >= 1.0 - 1e-12) and torch.all(eps <= eps_bulk + 1e-12)
    eps.sum().backward()
    assert torch.isfinite(rho.grad).all()  # no NaN at ρ→0 (log floor) or clamps


# --------------------------------------------------------------------------- #
# (a) Born: charge in a uniform dielectric
# --------------------------------------------------------------------------- #
def test_uniform_dielectric_is_exact_born_field():
    L = 12.0
    grid = build_fft_grid(np.eye(3) * L, ecut=300.0)
    rho = _gaussian_density(grid, (L / 2,) * 3, sigma=1.0)
    phi_vac = hartree_potential_r(rho, grid.g2)
    eps_bulk = 78.4
    eps = torch.full_like(rho, eps_bulk)
    sol = solve_generalized_poisson(rho, eps, grid.g2, grid.g_cart, tol=1e-12)
    # in a uniform dielectric the GPE solution is exactly φ_vac/ε
    scale = phi_vac.abs().max()
    assert float((sol.phi - phi_vac / eps_bulk).abs().max() / scale) < 1e-10


def test_born_energy_scaling():
    L = 12.0
    grid = build_fft_grid(np.eye(3) * L, ecut=300.0)
    rho = _gaussian_density(grid, (L / 2,) * 3, sigma=1.0)
    phi_vac = hartree_potential_r(rho, grid.g2)
    dvol = grid.volume / rho.numel()
    e_h = 0.5 * (rho * phi_vac).sum() * dvol  # vacuum self-energy on the grid

    # G_el(ε) = (1/ε − 1)·E_H exactly (Born reaction field), for every ε
    for eps_bulk in (2.0, 10.0, 40.0, 78.4):
        eps = torch.full_like(rho, eps_bulk)
        sol = solve_generalized_poisson(rho, eps, grid.g2, grid.g_cart, tol=1e-12)
        g_el = 0.5 * (rho * (sol.phi - phi_vac)).sum() * dvol
        born = (1.0 / eps_bulk - 1.0) * e_h
        assert abs(float(g_el - born)) / abs(float(born)) < 1e-9
        assert float(g_el) < 0.0  # solvation lowers the energy

    # effective Born radius from the vacuum self-energy: E_H = q²e²/(2R)
    r_eff = float(E2 / (2 * e_h))  # q = 1
    eps_bulk = 78.4
    eps = torch.full_like(rho, eps_bulk)
    sol = solve_generalized_poisson(rho, eps, grid.g2, grid.g_cart, tol=1e-12)
    g_el = float(0.5 * (rho * (sol.phi - phi_vac)).sum() * dvol)
    g_born_formula = -(1.0 - 1.0 / eps_bulk) * E2 / (2 * r_eff)  # −(1−1/ε)q²e²/2R
    assert abs(g_el - g_born_formula) / abs(g_born_formula) < 1e-9


# --------------------------------------------------------------------------- #
# (b) planar dielectric step vs the 1D generalized-Poisson (image) solution
# --------------------------------------------------------------------------- #
def test_planar_step_matches_1d_quadrature():
    L = 18.0
    grid = build_fft_grid(np.eye(3) * L, ecut=150.0)
    n = grid.shape
    nz = n[2]
    z = torch.arange(nz, dtype=DT) * (L / nz)

    def gz(z0, s):
        return torch.exp(-((z - z0) ** 2) / (2 * s * s)) / math.sqrt(2 * math.pi * s * s)

    rho_z = gz(5.5, 1.0) - gz(12.5, 1.0)  # laterally-uniform, net neutral
    rho = rho_z[None, None, :].expand(n).contiguous()
    eps_bulk = 40.0
    # dielectric slab: ε = 1 inside [4.5, 13.5], ε_bulk outside (a planar step)
    sw = 0.5 * (torch.tanh((z - 4.5) / 1.2) - torch.tanh((z - 13.5) / 1.2))
    eps_z = eps_bulk - (eps_bulk - 1.0) * sw
    eps = eps_z[None, None, :].expand(n).contiguous()

    sol = solve_generalized_poisson(rho, eps, grid.g2, grid.g_cart, tol=1e-11,
                                    mixing=0.55)
    phi_solver = sol.phi.mean(dim=(0, 1)).numpy()
    phi_solver -= phi_solver.mean()

    # 1D reference: d/dz(ε φ') = −4π e² ρ, periodic, by direct quadrature.
    dz = L / nz
    rho_np, eps_np = rho_z.numpy(), eps_z.numpy()
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rho_np[:-1] + rho_np[1:]) * dz)])[:nz]
    inv_eps = 1.0 / eps_np
    c_const = 4 * math.pi * E2 * np.sum(cum * inv_eps * dz) / np.sum(inv_eps * dz)
    phip = (c_const - 4 * math.pi * E2 * cum) * inv_eps
    phi_ref = np.concatenate([[0.0], np.cumsum(0.5 * (phip[:-1] + phip[1:]) * dz)])[:nz]
    phi_ref -= phi_ref.mean()

    err = np.max(np.abs(phi_solver - phi_ref))
    assert err / np.max(np.abs(phi_ref)) < 1e-2  # D-field continuity across the step


# --------------------------------------------------------------------------- #
# (c) autograd δG/δρ vs finite differences (the novelty)
# --------------------------------------------------------------------------- #
def _smooth_neutral_perturbation(grid, seed=0, width=0.5):
    """A smooth (low-G), charge-neutral test perturbation δρ on the box."""
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(*grid.shape, generator=gen, dtype=DT)
    smooth = torch.fft.ifftn(torch.fft.fftn(raw) * torch.exp(-grid.g2 * width)).real
    return smooth - smooth.mean()


def test_electrostatic_solvation_potential_matches_finite_difference():
    # The novelty: δG_el/δρ by autograd (reaction field + the −(1/8π e²)|∇φ|²
    # δε/δρ cavity term) vs a finite-difference directional derivative.
    L = 8.0
    grid = build_fft_grid(np.eye(3) * L, ecut=120.0)
    rho = _gaussian_density(grid, (L / 2,) * 3, sigma=0.8)
    kw = dict(eps_bulk=40.0, rho_min=1e-4, rho_max=3e-3, tol=1e-12,
              mixing=0.55, max_iter=800)
    v = solvation_potential(rho, grid, **kw)
    dvol = grid.volume / rho.numel()

    delta = _smooth_neutral_perturbation(grid)
    h = 1e-6
    g_p = float(solvation_energy(rho + h * delta, grid, **kw))
    g_m = float(solvation_energy(rho - h * delta, grid, **kw))
    fd = (g_p - g_m) / (2 * h)
    aut = float((v * delta).sum() * dvol)
    assert abs(fd - aut) / abs(aut) < 1e-6


def test_full_solvation_potential_matches_finite_difference():
    # Full G_sol = G_el + γ·S + p·V; δG/δρ (electrostatic + cavity + the
    # non-electrostatic surface/volume terms) vs finite differences.
    L = 8.0
    grid = build_fft_grid(np.eye(3) * L, ecut=120.0)
    rho = _gaussian_density(grid, (L / 2,) * 3, sigma=0.8)
    kw = dict(eps_bulk=40.0, rho_min=1e-4, rho_max=3e-3, gamma=0.05,
              pressure=0.001, tol=1e-12, mixing=0.55, max_iter=800)
    v = solvation_potential(rho, grid, **kw)
    dvol = grid.volume / rho.numel()

    delta = _smooth_neutral_perturbation(grid)
    h = 1e-6
    g_p = float(solvation_energy(rho + h * delta, grid, **kw))
    g_m = float(solvation_energy(rho - h * delta, grid, **kw))
    fd = (g_p - g_m) / (2 * h)
    aut = float((v * delta).sum() * dvol)
    assert abs(fd - aut) / abs(aut) < 1e-4


def test_reaction_field_energy_gradient_is_finite_and_signed():
    L = 8.0
    grid = build_fft_grid(np.eye(3) * L, ecut=120.0)
    rho = _gaussian_density(grid, (L / 2,) * 3, sigma=0.8).requires_grad_(True)
    g_el = reaction_field_energy(rho, grid, eps_bulk=40.0, rho_min=1e-4,
                                 rho_max=3e-3, tol=1e-10, mixing=0.55)
    assert float(g_el.detach()) < 0.0  # electrostatic solvation is stabilizing
    (grad,) = torch.autograd.grad(g_el, rho)
    assert torch.isfinite(grad).all()


def test_cavity_indicator_bounds():
    L = 8.0
    grid = build_fft_grid(np.eye(3) * L, ecut=120.0)
    rho = _gaussian_density(grid, (L / 2,) * 3, sigma=0.8)
    c = cavity_indicator(rho, eps_bulk=40.0, rho_min=1e-4, rho_max=3e-3)
    assert float(c.min()) >= -1e-12 and float(c.max()) <= 1.0 + 1e-12
    # the solute core (high density at the center) is inside the cavity (c≈1)
    ic = tuple(s // 2 for s in grid.shape)
    assert float(c[ic]) > 0.99
