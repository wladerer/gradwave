"""Constrained non-collinear magnetism: local-moment directions and torques.

Optimizing the *directions* of atomic magnetic moments needs a torque
dE/dê_I. At an unconstrained self-consistent point the local moment already
sits parallel to its own exchange field (B_xc ∥ m by construction of the
locally-collinear XC), so the naive on-site torque ∫ w_I (m × B_xc) is zero
and carries no gradient. The signal only appears when the directions are
*constrained* away from equilibrium.

This module implements the penalty constrained-DFT scheme of Ma and
Dudarev [PRB 91, 054420 (2015)]. Each atom gets a Hirshfeld weight w_I(r); the
atomic moment is M_I = ∫ w_I m⃗ dr; a penalty

    E_p = Σ_I λ |M_I - (M_I·ê_I) ê_I|²  =  Σ_I λ |M_I^⊥|²

is added to the energy, contributing a constraining field
B_c(r) = δE_p/δm⃗ = 2λ Σ_I w_I(r) M_I^⊥ to the spinor Hamiltonian. The SCF then
holds each M_I along its target ê_I. At convergence the constraining field is
minus the internal transverse field the moment feels, so the torque that would
rotate the *unconstrained* moment is

    T_I = -B_I^c^⊥ = -2λ M_I^⊥         (per-atom, transverse to ê_I)

and dE/dê_I = -T_I. A configuration is a stationary point of the true energy
when every T_I vanishes (no constraint needed). `relax_moment_directions`
descends the torque to find the ground-state configuration.

The magnitude problem
---------------------
The |M_I^⊥|² penalty ("perp" mode) constrains only the moment *direction*: it is
minimized (→ 0) at M_I = 0, so a strongly-coupled magnet forced to a large
relative angle can satisfy the constraint for free by demagnetizing. The
magnitude-robust "vector" mode pins the full moment vector,
E_p = Σ λ|M_I − m0_I ê_I|², so demagnetization costs λ m0_I² and each moment is
held at its target magnitude m0_I at any angle — enough to hold, e.g., a
metastable antiferromagnetic state that "perp" would collapse. Both penalties
(and both the SCF field and this torque) come from one differentiable definition
in `gradwave.scf.moment_penalty`, so they stay mutually consistent by
construction. All of it is validated against a finite difference of W.
"""

from __future__ import annotations

import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.scf.guess import sad_density
from gradwave.scf.moment_penalty import (
    direction_gradient,
    field_coeff,
    penalty_energy,
)
from gradwave.scf.noncollinear import scf_noncollinear


def atomic_weights(system, floor: float = 1e-6) -> torch.Tensor:
    """Hirshfeld partition weights w_I(r), shape (na, *grid). Σ_I w_I ≈ 1 where
    any atomic density exists, → 0 in vacuum. Built from the neutral-atom (SAD)
    densities, so w_I localizes on atom I.

    The per-atom SAD densities are clamped to ≥ 0 first: `sad_density` with
    n_electrons=None skips its own clamp, and the small negative FFT ringing in
    the tails would otherwise drive Σρ_at through zero and blow the weights up
    (a partition weight must stay bounded). `floor` is a small absolute density
    [e/Å³] so the denominator never vanishes and vacuum weights fall to ~0."""
    grid = system.grid
    dev = system.positions.device
    na = len(system.species_of_atom)
    rho_at = []
    for a in range(na):
        onehot = [1.0 if b == a else 0.0 for b in range(na)]
        r = sad_density(grid, system.positions, system.species_of_atom,
                        system.upfs, None, atom_scale=onehot)
        rho_at.append(r.clamp_min(0.0).to(dev))
    rho_at = torch.stack(rho_at)              # (na, *grid)
    tot = rho_at.sum(dim=0)                    # (*grid)
    return rho_at / tot.clamp_min(floor)


def _atomic_moments(m, weights, cell_factor):
    """M_I = ∫ w_I(r) m⃗(r) dr, shape (na, 3) [μB]. m is (3,*grid); weights
    (na,*grid); cell_factor = volume / n_points."""
    return torch.einsum("axyz,ixyz->ai", weights, m) * cell_factor


def _unit(v, eps: float = 1e-30):
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def reference_moment_magnitudes(system, xc: NoncollinearXC, directions, *,
                                weights=None, mag_init_scale: float = 1.5,
                                **scf_kwargs):
    """|M_I| [μB] of each atom from one *unconstrained* non-collinear SCF seeded
    along `directions`. Used as the target magnitude for the "vector" penalty so
    the constraint holds each moment at its natural self-consistent size.

    Seeded high-spin (mag_init_scale=1.5, above saturation) then relaxed down,
    because the bare non-collinear SCF is often multi-stable — a weak seed can
    collapse to a low-spin or nonmagnetic solution. This is a convenience; for a
    delicate system prefer passing `target_mag` to `constrained_moment_scf`
    explicitly (e.g. from a collinear spin-polarized reference)."""
    dirs = _unit(torch.as_tensor(directions, dtype=torch.float64,
                                 device=system.positions.device))
    if weights is None:
        weights = atomic_weights(system)
    res = scf_noncollinear(system, xc, mag_vec_init=(mag_init_scale * dirs).tolist(),
                           **scf_kwargs)
    cf = system.grid.volume / system.grid.n_points
    M = _atomic_moments(res.m, weights, cf)
    return torch.linalg.norm(M, dim=-1)                  # (na,)


def constrained_moment_scf(system, xc: NoncollinearXC, directions, *, lam: float,
                           weights=None, mode: str = "perp", target_mag=None,
                           mag_init_scale: float = 0.6, **scf_kwargs):
    """Constrained non-collinear SCF pinning each atomic moment M_I toward the
    unit direction directions[I] with penalty strength lam. Returns

        (res, info)

    `mode` selects the penalty (see gradwave.scf.moment_penalty):
      "perp"    E_p = λΣ|M_I^⊥|²          — direction only (demagnetizes when a
                                            strong pair is forced far apart).
      "vector"  E_p = λΣ|M_I − m0_I ê_I|²  — pins magnitude too, so the moment is
                                            held at full size at any angle. m0_I
                                            is `target_mag` (per-atom |M| [μB]);
                                            if None it is measured by an
                                            unconstrained reference SCF.

    info keys: atomic moments `M` (na,3), transverse residual `M_perp` (na,3),
    per-atom constraining field `B_c = ∂E_p/∂M` (na,3), the descent `torque`
    (na,3, transverse to ê), the envelope `energy_grad = dW/dê`, `energy_eV` (the
    true KS free energy, penalty excluded), `W_eV` (constrained functional),
    `target_mag` (na, or None), and `converged`.
    """
    dirs = _unit(torch.as_tensor(directions, dtype=torch.float64,
                                 device=system.positions.device))
    if weights is None:
        weights = atomic_weights(system)
    if mode == "vector" and target_mag is None:
        target_mag = reference_moment_magnitudes(
            system, xc, dirs, weights=weights, mag_init_scale=mag_init_scale,
            **scf_kwargs)
    res = scf_noncollinear(
        system, xc, mag_vec_init=(mag_init_scale * dirs).tolist(),
        constrain_dirs=dirs, constrain_lambda=lam, atom_weights=weights,
        constrain_mode=mode, constrain_target_mag=target_mag, **scf_kwargs)
    cf = system.grid.volume / system.grid.n_points
    M = _atomic_moments(res.m, weights, cf)
    m_dot_e = (M * dirs).sum(-1, keepdim=True)          # (M_I·ê_I), (na,1)
    Mperp = M - m_dot_e * dirs                           # transverse moment M_I^⊥
    # Field and envelope gradient both come from autograd on the same penalty,
    # so "perp" and "vector" stay consistent with the SCF field with no
    # hand-derived formula to keep in sync.
    Bc = field_coeff(M, dirs, lam, mode, target_mag)     # ∂E_p/∂M_I
    grad = direction_gradient(M, dirs, lam, mode, target_mag)  # dW/dê_I ⟂ ê
    ep = float(penalty_energy(M, dirs, lam, mode, target_mag))
    info = {
        "directions": dirs, "M": M, "M_perp": Mperp, "B_c": Bc,
        "energy_grad": grad, "torque": -grad,
        "target_mag": target_mag,
        "energy_eV": float(res.energies.free_energy),   # physical KS energy
        "W_eV": float(res.energies.free_energy) + ep,   # constrained functional
        "converged": bool(res.converged),
    }
    return res, info


def relax_moment_directions(system, xc: NoncollinearXC, directions0, *,
                            lam: float, step: float = 0.5, tol: float = 1e-2,
                            max_sweeps: int = 40, weights=None, mode: str = "perp",
                            target_mag=None, verbose: bool = True, **scf_kwargs):
    """Gradient-descend the moment directions to the ground-state configuration.

    Each sweep runs a constrained SCF at the current targets, reads the descent
    torque T_I = -dW/dê_I, and rotates every ê_I downhill in energy
    (ê ← unit(ê + step·T)). Convergence is measured by the moment misalignment
    max_I |M_I^⊥| [μB]: it → 0 when each moment already sits along its target, so
    no constraint is needed and the configuration is a self-consistent stationary
    point of the true energy. `mode`/`target_mag` pick the penalty (see
    `constrained_moment_scf`); for mode="vector" the reference magnitude is
    measured once here and reused across sweeps. Returns (directions (na,3),
    history)."""
    dirs = _unit(torch.as_tensor(directions0, dtype=torch.float64,
                                 device=system.positions.device))
    if weights is None:
        weights = atomic_weights(system)
    if mode == "vector" and target_mag is None:
        target_mag = reference_moment_magnitudes(system, xc, dirs, weights=weights,
                                                 **scf_kwargs)
    history = []
    for sweep in range(1, max_sweeps + 1):
        _, info = constrained_moment_scf(system, xc, dirs, lam=lam, weights=weights,
                                         mode=mode, target_mag=target_mag,
                                         verbose=False, **scf_kwargs)
        misalign = float(torch.linalg.norm(info["M_perp"], dim=-1).max())
        history.append({"sweep": sweep, "energy_eV": info["energy_eV"],
                        "misalign_muB": misalign, "directions": dirs.tolist()})
        if verbose:
            print(f"  moment sweep {sweep:2d}  E = {info['energy_eV']:+.8f} eV  "
                  f"max|M⊥| = {misalign:.3e} μB", flush=True)
        if misalign < tol:
            break
        dirs = _unit(dirs + step * info["torque"])
    return dirs, history
