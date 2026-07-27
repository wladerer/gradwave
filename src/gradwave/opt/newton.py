"""Steihaug trust-region Newton-CG over the joint (strain, positions, orbitals).

The first-order joint engine (``opt/joint.py``) descends on the KS total energy
with L-BFGS. This module replaces the inner optimizer with a second-order one:
a Steihaug truncated-CG trust-region method that needs the Hessian only through
products ``Hv``, which autograd supplies exactly by double-backward on
``joint_energy`` (ideas.md, "exact Hvp Newton-CG", step 4). No Sternheimer
solve: the joint functional is an explicit function of all its variables, so
``Hv`` is plain double-backward at ~2× gradient cost.

Structure mirrors ``joint_relax``: an outer basis-rebuild loop (the plane-wave
count rides the cell) wraps the inner solver, orbitals cross rebuilds by
Miller-index transfer, and cycle 0 seeds the orbitals with a few loose SCF
iterations (charged exactly to ``h_seed``). The inner solver is the trust-region
Newton loop instead of L-BFGS.

All optimization variables are REAL tensors (strain 3×3, fractional positions
na×3, and the real-view ``view_as_real`` of the per-k orbital variables), so the
CG runs in a plain real inner-product space; the complex orbital structure lives
entirely inside ``joint_energy``. A block-vector is just a list of tensors
aligned with the leaves.

Preconditioner hook. The electronic block is already Teter-preconditioned by the
parametrization (the leaves are ``C/p(G)``, so L-BFGS's / CG's identity metric on
Z is the Teter–Payne–Allan metric on C); the ionic block is identity (acceptable
v1, ideas.md step 4). ``preconditioner`` lets a caller slot a different
block-diagonal ``M⁻¹`` without touching the solver.

H-application accounting (comparable to ``JointResult.h_equiv``, one unit = one
band·k H|ψ⟩ ≈ 2 FFTs): a gradient eval (fwd+bwd) is 1 unit, an Hvp (a second
backward through the graph) is charged 2 units — the "2–3× gradient cost" of
double-backward, taken at the low end — and a trial energy (fwd only) is 0.5.
``h_equiv = h_seed + round((n_grad + 2·n_hvp + 0.5·n_trial)·nk·n_occ)``. Wall
time is reported alongside as the ultimate arbiter.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from gradwave.core.xc.base import XCFunctional
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.opt.joint import (
    _coeffs_from_z,
    _transfer_coeffs,
    count_h_applies,
    joint_energy,
    kinetic_band,
    lowdin,
    teter_precond,
)
from gradwave.pseudo.radial_torch import RadialTables
from gradwave.scf.loop import System, scf, setup_system

logger = logging.getLogger(__name__)


# ------------------------------------------------------------- block-vector
# A "block vector" is a list of real tensors aligned with the optimizer leaves.
def _dot(a, b) -> torch.Tensor:
    return sum((x * y).sum() for x, y in zip(a, b, strict=True))


def _axpy(alpha: float, x, y):
    """y + alpha·x (new list)."""
    return [yi + alpha * xi for xi, yi in zip(x, y, strict=True)]


def _norm(x) -> float:
    return float(torch.sqrt(_dot(x, x).clamp_min(0.0)))


def _to_boundary(z, d, delta: float) -> float:
    """τ ≥ 0 solving ‖z + τ d‖ = Δ (the positive root)."""
    a = float(_dot(d, d))
    b = 2.0 * float(_dot(z, d))
    c = float(_dot(z, z)) - delta * delta
    disc = max(b * b - 4.0 * a * c, 0.0)
    return (-b + math.sqrt(disc)) / (2.0 * a) if a > 0 else 0.0


def _identity_precond(r):
    """Default M⁻¹ = I. The Teter-like electronic preconditioning is already
    folded into the leaf parametrization (C = Löwdin(Z·p)); the ionic block is
    identity (ideas.md step 4, v1)."""
    return r


# ------------------------------------------------------------- Steihaug CG
def steihaug_cg(grad, hvp, delta: float, precond, max_iter: int,
                tol_rel: float):
    """Truncated-CG solve of the trust-region subproblem (Steihaug 1983).

    min_p  gᵀp + ½ pᵀHp   s.t.  ‖p‖ ≤ Δ,   using only H through ``hvp``.
    Exits to the trust boundary on negative curvature (dᵀHd ≤ 0) or when an
    iterate leaves the trust region; otherwise on a relative-residual drop.
    Returns ``(p, hit_boundary, n_hvp)``.
    """
    z = [torch.zeros_like(g) for g in grad]
    # detach: the CG state is a numeric vector; the Hessian action re-differentiates
    # the retained energy graph (captured in ``hvp``), not these vectors.
    r = [g.detach().clone() for g in grad]  # residual Hz + g, with z = 0 → g
    y = precond(r)
    d = [-yi for yi in y]
    r_dot_y = _dot(r, y)
    r0 = _norm(r)
    n_hvp = 0
    if r0 == 0.0:
        return z, False, n_hvp
    for _ in range(max_iter):
        hd = hvp(d)
        n_hvp += 1
        d_hd = float(_dot(d, hd))
        if d_hd <= 0.0:                       # negative curvature
            tau = _to_boundary(z, d, delta)
            return _axpy(tau, d, z), True, n_hvp
        alpha = float(r_dot_y) / d_hd
        z_next = _axpy(alpha, d, z)
        if _norm(z_next) >= delta:            # crossed the trust boundary
            tau = _to_boundary(z, d, delta)
            return _axpy(tau, d, z), True, n_hvp
        z = z_next
        r = _axpy(alpha, hd, r)
        if _norm(r) < tol_rel * r0:
            return z, False, n_hvp
        y = precond(r)
        r_dot_y_new = _dot(r, y)
        beta = float(r_dot_y_new) / float(r_dot_y)
        d = _axpy(beta, d, [-yi for yi in y])
        r_dot_y = r_dot_y_new
    return z, False, n_hvp


def _newton_inner(system, xc, tabs, precond_cols, npws, occ, a0_np, omega0,
                  eps_p, frac_p, leaves, *, fix_cell, checkpoint, device,
                  fmax, smax, ctol, max_newton, cg_max_iter, cg_tol,
                  trust_radius, trust_max, eta_accept, verbose, cycle):
    """One frozen-basis trust-region Newton-CG solve; mutates the leaf tensors
    in place. A top-level function (not a closure) so the energy / Hvp / metric
    closures bind their state as parameters, not late-bound loop variables.
    Returns a dict of counts, residuals, and the energy trace."""
    dev = torch.device(device)
    eye3 = torch.eye(3, dtype=RDTYPE, device=dev)
    eps_zero = torch.zeros(3, 3, dtype=RDTYPE, device=dev)
    n_grad = n_hvp = n_trial = 0
    history: list = []

    def energy(_leaves):
        if fix_cell:
            eps, frac, zs = eps_zero, _leaves[0], _leaves[1:]
        else:
            eps, frac, zs = _leaves[0], _leaves[1], _leaves[2:]
        eps_s = 0.5 * (eps + eps.transpose(-2, -1))
        detj = torch.linalg.det(eye3 + eps_s)
        if float(detj.detach()) < 0.2 or float(detj.detach()) > 5.0:
            return 1e3 * ((detj - 1.0) ** 2 + (eps_s ** 2).sum() + 1.0)
        coeffs = _coeffs_from_z(zs, precond_cols, npws)
        return joint_energy(system, xc, tabs, eps, frac, coeffs, occ,
                            checkpoint=checkpoint)

    def energy_and_grad(_leaves):
        e = energy(_leaves)
        return e, list(torch.autograd.grad(e, _leaves, create_graph=True))

    def metrics(grads):
        a_e = a0_np @ (np.eye(3) + eps_p.detach().cpu().numpy()).T
        if fix_cell:
            g_frac, g_z, smax_now = grads[0], grads[1:], 0.0
        else:
            g_eps, g_frac, g_z = grads[0], grads[1], grads[2:]
            gg = g_eps.detach()
            smax_now = float((0.5 * (gg + gg.T) / omega0).abs().max())
        de_dpos = g_frac.detach().cpu().numpy() @ np.linalg.inv(a_e).T
        de_dpos = de_dpos - de_dpos.mean(axis=0, keepdims=True)
        return (float(np.abs(de_dpos).max()), smax_now,
                max(float(g.detach().abs().max()) for g in g_z))

    delta = trust_radius
    converged = False
    newton_steps = 0
    e_val, grads = energy_and_grad(leaves)
    n_grad += 1
    f_now = s_now = c_now = float("inf")
    for _ in range(max_newton):
        f_now, s_now, c_now = metrics(grads)
        history.append(float(e_val.detach()))
        if verbose:
            print(f"  newton cycle {cycle} step {newton_steps:3d}  "
                  f"E = {float(e_val.detach()):+.8f} eV  fmax {f_now:.2e}  "
                  f"smax {s_now:.2e}  cmax {c_now:.2e}  Δ {delta:.2e}", flush=True)
        if f_now < fmax and c_now < ctol and (fix_cell or s_now < smax):
            converged = True
            break

        def hvp(v, _g=grads):
            return list(torch.autograd.grad(_dot(_g, v), leaves,
                                            retain_graph=True))

        g_det = [g.detach() for g in grads]
        accepted = False
        for _trial in range(8):                # shrink Δ, reuse the gradient
            p, hit_bnd, nh = steihaug_cg(
                grads, hvp, delta, _identity_precond, cg_max_iter, cg_tol)
            hp = hvp(p)                        # predicted decrease (one Hvp)
            n_hvp += nh + 1
            pred = -(float(_dot(g_det, p))
                     + 0.5 * float(_dot(p, [h.detach() for h in hp])))
            with torch.no_grad():
                for t, pi in zip(leaves, p, strict=True):
                    t.add_(pi)
            n_trial += 1
            try:                               # a trial can drive Z rank-deficient
                with torch.no_grad():          # (Löwdin Cholesky) — reject it
                    e_new = float(energy(leaves).detach())
                ok = math.isfinite(e_new)
            except RuntimeError:               # torch LinAlgError ⊂ RuntimeError
                e_new, ok = float("inf"), False
            rho = ((float(e_val.detach()) - e_new) / pred
                   if ok and abs(pred) > 1e-300 else -1.0)
            if rho < 0.25:
                delta *= 0.25
            elif rho > 0.75 and hit_bnd:
                delta = min(2.0 * delta, trust_max)
            if rho > eta_accept:
                accepted = True
                newton_steps += 1
                break
            with torch.no_grad():              # reject: undo, retry smaller Δ
                for t, pi in zip(leaves, p, strict=True):
                    t.sub_(pi)
            if delta < 1e-9:
                break
        if not accepted:                       # stalled → fall back to nested
            break
        del grads
        e_val, grads = energy_and_grad(leaves)
        n_grad += 1

    return {
        "converged": converged, "newton_steps": newton_steps,
        "n_grad": n_grad, "n_hvp": n_hvp, "n_trial": n_trial,
        "fmax": f_now, "smax": s_now, "energy": float(e_val.detach()),
        "history": history,
    }


# --------------------------------------------------------------- result
@dataclass
class NewtonResult:
    converged: bool
    energy: float
    cell: np.ndarray
    positions: np.ndarray
    coeffs: list
    system: System
    n_newton: int = 0        # accepted trust-region steps
    n_grad: int = 0          # energy+gradient evaluations
    n_hvp: int = 0           # Hessian-vector products (incl. predicted-decrease)
    n_trial: int = 0         # trial energy-only evaluations
    h_seed: int = 0
    h_equiv: int = 0
    n_cycles: int = 0
    fmax: float = 0.0
    smax: float = 0.0
    history: list = field(default_factory=list)


# --------------------------------------------------------------- driver
def newton_cg_relax(
    cell: np.ndarray,
    positions: np.ndarray,
    species_of_atom: list[int],
    upfs: list,
    xc: XCFunctional,
    *,
    ecut: float,
    kmesh=(1, 1, 1),
    n_occ: int | None = None,
    fmax: float = 0.005,
    smax: float = 5e-4,
    ctol: float = 5e-4,
    max_newton: int = 40,
    cg_max_iter: int = 40,
    cg_tol: float = 0.1,          # relative-residual (Steihaug forcing) target
    trust_radius: float = 0.3,
    trust_max: float = 2.0,
    eta_accept: float = 0.1,      # accept step if actual/predicted ≥ η
    seed_scf_iters: int = 3,
    max_rebuilds: int = 3,
    rebuild_tol: float = 2e-4,
    fix_cell: bool = False,
    checkpoint: bool | None = None,
    device: str = "cpu",
    verbose: bool = False,
) -> NewtonResult:
    """Trust-region Newton-CG relaxation of an NC insulator's (cell, positions,
    orbitals). Same contract as ``joint_relax`` (NC, nspin=1, fixed integer
    occupations, insulators only); ``converged=False`` signals the caller to
    fall back to the nested engine, exactly as the first-order path does."""
    cell = np.asarray(cell, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    coeffs_init = None
    prev_spheres = None
    h_seed = n_grad = n_hvp = n_trial = 0
    history: list = []

    for cycle in range(max_rebuilds + 1):
        system = setup_system(
            cell=cell, positions=positions, species_of_atom=species_of_atom,
            upfs=upfs, ecut=ecut, kmesh=kmesh, use_symmetry=False,
        ).to(device)
        nk = len(system.spheres)
        if n_occ is None:
            if abs(system.n_electrons / 2 - round(system.n_electrons / 2)) > 1e-9:
                raise ValueError("odd electron count — not an insulator at "
                                 "fixed occupations; pass n_occ explicitly")
            n_occ = int(round(system.n_electrons / 2))
        occ = torch.full((nk, n_occ), 2.0, dtype=RDTYPE, device=device)
        tabs = [RadialTables(u, device=device) for u in system.upfs]

        # ---- orbital seed (loose SCF on cycle 0, Miller transfer afterwards)
        if coeffs_init is None:
            with count_h_applies() as counter:
                res = scf(system, xc, max_iter=seed_scf_iters, verbose=False,
                          etol=0.0, rhotol=0.0, diago_tol=1e-4)
            h_seed += counter.count
            coeffs_init = [lowdin(c[:n_occ].to(CDTYPE)) for c in res.coeffs]
        else:
            coeffs_init = [lowdin(c) for c in
                           _transfer_coeffs(prev_spheres, system.spheres,
                                            coeffs_init)]

        ekin_ref = float(np.mean([
            kinetic_band(coeffs_init[ik], system.spheres[ik].kpg2)
            .mean().item() for ik in range(nk)]))
        precond_cols = [teter_precond(sph.kpg2, ekin_ref).to(RDTYPE)
                        for sph in system.spheres]

        a0_np = np.asarray(system.grid.cell, dtype=np.float64)
        frac0 = positions @ np.linalg.inv(a0_np)
        eps_p = torch.zeros(3, 3, dtype=RDTYPE, device=device,
                            requires_grad=not fix_cell)
        frac_p = torch.tensor(frac0, dtype=RDTYPE, device=device,
                              requires_grad=True)
        z_params = []
        for ik in range(nk):
            z0 = coeffs_init[ik] / precond_cols[ik][None, :].to(CDTYPE)
            z_params.append(torch.view_as_real(z0.contiguous())
                            .clone().requires_grad_(True))
        npws = [sph.npw for sph in system.spheres]
        omega0 = system.grid.volume

        leaves = ([frac_p] if fix_cell else [eps_p, frac_p]) + z_params
        inner = _newton_inner(
            system, xc, tabs, precond_cols, npws, occ, a0_np, omega0,
            eps_p, frac_p, leaves, fix_cell=fix_cell, checkpoint=checkpoint,
            device=device, fmax=fmax, smax=smax, ctol=ctol,
            max_newton=max_newton, cg_max_iter=cg_max_iter, cg_tol=cg_tol,
            trust_radius=trust_radius, trust_max=trust_max,
            eta_accept=eta_accept, verbose=verbose, cycle=cycle)
        n_grad += inner["n_grad"]
        n_hvp += inner["n_hvp"]
        n_trial += inner["n_trial"]
        base = len(history)
        history.extend((base + i, e) for i, e in enumerate(inner["history"]))
        converged_inner = inner["converged"]
        newton_steps = inner["newton_steps"]
        f_final, s_final, energy_final = (
            inner["fmax"], inner["smax"], inner["energy"])

        # ---- apply relaxed strain/positions; decide on a rebuild
        eps_np = 0.5 * (eps_p.detach().cpu().numpy()
                        + eps_p.detach().cpu().numpy().T)
        cell = a0_np @ (np.eye(3) + eps_np).T
        frac_np = frac_p.detach().cpu().numpy()
        positions = frac_np @ cell
        coeffs_init = [c.detach() for c in
                       _coeffs_from_z(z_params, precond_cols, npws)]
        prev_spheres = system.spheres
        strain_step = float(np.abs(eps_np).max())
        logger.info("newton cycle %d: %d steps, max|eps|=%.2e, converged=%s",
                    cycle, newton_steps, strain_step, converged_inner)
        if (fix_cell or strain_step < rebuild_tol) and converged_inner:
            break

    h_equiv = h_seed + round(
        (n_grad + 2 * n_hvp + 0.5 * n_trial) * nk * n_occ)
    return NewtonResult(
        converged=converged_inner,
        energy=energy_final,
        cell=cell, positions=positions, coeffs=coeffs_init, system=system,
        n_newton=newton_steps, n_grad=n_grad, n_hvp=n_hvp, n_trial=n_trial,
        h_seed=h_seed, h_equiv=int(h_equiv), n_cycles=cycle + 1,
        fmax=f_final, smax=s_final, history=history,
    )
