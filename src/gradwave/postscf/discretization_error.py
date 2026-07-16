"""Plane-wave discretization (Ecut) error estimation + AD propagation.

Estimates the plane-wave basis-set error of a converged SCF and propagates it to
quantities of interest (forces, energy, stress) by forward-mode AD. Following:

  E. Cancès, G. Dusson, Y. Maday, B. Stamm, M. Vohralík, "A perturbation-method-
  based post-processing for the planewave discretization of Kohn-Sham models,"
  J. Comput. Phys. 307, 446 (2016);
  M. F. Herbst, A. Levitt, E. Cancès, "Practical error bounds for properties in
  plane-wave electronic structure calculations," arXiv:2111.01470;
  and the differentiable-DFT coupling of arXiv:2509.07785.

Idea. The exact (infinite-basis) solution is a perturbation of the converged
Ecut solution. At high kinetic energy the Hamiltonian is dominated by the
diagonal Laplacian, so a first-order orbital correction in the high-G
"complement" annulus (Ecut < T_G <= Ecut_large) is a cheap diagonal solve,

    R_i = P_annulus (H - eps_i) psi_i = P_annulus H psi_i     (psi_i has no
                                                               annulus support)
    dpsi_i = -R_i / (T_G - eps_i)          on the annulus, 0 elsewhere,

and the density-matrix perturbation follows as dP <-> {dpsi_i}. Optionally the
coarse-space response is dressed by the SCF dielectric operator (an approximate
Dyson equation) to capture the self-consistent part of the error.

This is a post-processing step on a converged SCF. It never enters the SCF hot
path, so it does not affect solve performance; the cost is a handful of extra
FFTs per occupied band plus (if requested) one dielectric solve.

Coverage here: norm-conserving, nspin=1, use_symmetry=False (a perturbation
breaks the crystal symmetry, so the response needs the full k-mesh; the same
restriction the implicit-diff response carries). USPP/PAW and metals are staged
separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g, sphere_to_box
from gradwave.core.hamiltonian import (
    HamiltonianK,
    becp,
    build_projector_data,
    projectors,
)
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_gsphere, gmax_from_ecut
from gradwave.pseudo.kb import beta_form_factors
from gradwave.scf.implicit import _check_no_symmetry, apply_chi0, apply_k_hxc
from gradwave.scf.loop import SCFResult


@dataclass
class DiscretizationError:
    """Estimated Ecut discretization error of a converged SCF.

    The sign convention is that the exact-basis quantities are approximated by
    adding the estimate, e.g. rho_exact ~= res.rho + drho.
    """

    drho: torch.Tensor           # (n1,n2,n3) real, estimated density error
    drho_first_order: torch.Tensor   # drho before any Dyson dressing
    denergy: float               # estimated total-energy error [eV] (2nd order, < 0)
    dpsi: list                   # per-k (n_occ, npw_large) complex, orbital correction
    psi_large: list              # per-k (n_occ, npw_large) complex, occ orbitals on large sphere
    occ: list                    # per-k (n_occ,) occupations
    spheres_large: list          # per-k enlarged GSphere
    ecut: float                  # eV
    ecut_large: float            # eV
    dyson: bool


def _occupied(res: SCFResult, ik: int):
    """Occupied coefficients, eigenvalues, occupations at one k (nspin=1)."""
    occ = res.occupations[ik]
    n_occ = int((occ > 1e-8).sum())
    return res.coeffs[ik][:n_occ], res.eigenvalues[ik][:n_occ], occ[:n_occ]


def _enlarged_hamiltonian(res: SCFResult, k_frac, ecut_large: float, device):
    """Build a HamiltonianK on the enlarged G-sphere at ecut_large for one k."""
    system = res.system
    grid = system.grid
    sph = build_gsphere(grid, ecut_large, k_frac, device=device)
    beta_ls = [[b.l for b in upf.betas] for upf in system.upfs]
    dij_species = [torch.as_tensor(upf.dij, dtype=RDTYPE, device=device) for upf in system.upfs]
    q = np.sqrt(sph.kpg2.cpu().numpy())
    beta_tables = [
        torch.as_tensor(beta_form_factors(upf, q), dtype=RDTYPE, device=device)
        for upf in system.upfs
    ]
    pd = build_projector_data(
        sph, system.species_of_atom, beta_tables, beta_ls, dij_species, grid.volume
    )
    p = projectors(pd, system.positions)
    return sph, HamiltonianK(sph, grid.shape, res.v_eff, pd, p)


@torch.no_grad()
def _dyson_dress(res, xc, drho0, *, beta, tol, max_iter, verbose):
    """Dress the first-order density error by the SCF dielectric operator.

    Solves the fixed point  drho = drho0 + chi0[ K_Hxc[ drho ] ], i.e. applies
    (1 - chi0 K)^-1 to the complement estimate, capturing the coarse-space
    self-consistent part of the error. Damped iteration reusing the response
    primitives from scf/implicit.py.
    """
    drho = drho0.clone()
    for it in range(max_iter):
        induced = apply_chi0(res, apply_k_hxc(res, xc, drho))
        drho_new = drho0 + induced
        denom_n = max(1.0, float(torch.linalg.norm(drho_new)))
        step = float(torch.linalg.norm(drho_new - drho)) / denom_n
        drho = drho + beta * (drho_new - drho)
        if verbose:
            print(f"  dyson it {it}: rel step {step:.2e}", flush=True)
        if step < tol:
            break
    return drho


@torch.no_grad()
def estimate_density_error(
    res: SCFResult,
    *,
    ecut_large: float | None = None,
    factor: float = 2.5,
    dyson: bool = False,
    xc=None,
    dyson_beta: float = 0.4,
    dyson_tol: float = 1e-6,
    dyson_max_iter: int = 60,
    verbose: bool = False,
) -> DiscretizationError:
    """Estimate the plane-wave discretization error in the converged density.

    Parameters
    ----------
    res : SCFResult
        A converged norm-conserving SCF (nspin=1, use_symmetry=False).
    ecut_large : float, optional
        Cutoff [eV] of the enlarged basis defining the complement annulus. If
        None, uses ``factor * res.system.ecut``. Must satisfy
        ecut_large <= 4*ecut so the enlarged sphere fits the density FFT box.
    dyson : bool
        If True, dress the first-order estimate with the coarse-space dielectric
        response (needs ``xc``).
    """
    _check_no_symmetry(res)
    system = res.system
    grid = system.grid
    device = res.v_eff.device
    ecut = float(system.ecut)
    if ecut_large is None:
        ecut_large = factor * ecut
    if gmax_from_ecut(ecut_large) > 2.0 * gmax_from_ecut(ecut) * (1.0 + 1e-9):
        raise ValueError(
            f"ecut_large={ecut_large:.1f} eV exceeds 4*ecut={4 * ecut:.1f} eV; the "
            "enlarged sphere would not fit the density FFT box. Lower ecut_large."
        )

    drho = torch.zeros(grid.shape, dtype=RDTYPE, device=device)
    denergy = 0.0
    dpsi_all, psi_large_all, occ_all, sph_all = [], [], [], []

    for ik, sph0 in enumerate(system.spheres):
        c_occ, eps_occ, occ = _occupied(res, ik)
        sph1, h1 = _enlarged_hamiltonian(res, sph0.k_frac, ecut_large, device)

        # zero-pad occupied orbitals from the Ecut sphere onto the enlarged one
        box = sphere_to_box(c_occ, sph0.flat_idx, grid.shape)
        c_occ_1 = box_to_sphere(box, sph1.flat_idx)  # (n_occ, npw1)

        # complement residual R = P_annulus (H - eps) psi
        resid = h1.apply(c_occ_1) - eps_occ[:, None] * c_occ_1
        annulus = h1.t > ecut * (1.0 + 1e-9)  # (npw1,) bool
        denom = torch.clamp(h1.t[None, :] - eps_occ[:, None], min=1e-3)
        corr = -resid / denom.to(resid.dtype)
        dpsi = torch.where(annulus[None, :], corr, torch.zeros_like(resid))

        # density error contribution: drho = sum_i f_i * 2 Re(psi_i* dpsi_i)
        psi_r = g_to_r(c_occ, sph0.flat_idx, grid.shape)
        dpsi_r = g_to_r(dpsi, sph1.flat_idx, grid.shape)
        w = 2.0 * float(system.kweights[ik])  # 2 = c.c. pair (psi* dpsi + dpsi* psi)
        drho += w * (occ.view(-1, 1, 1, 1) * (psi_r.conj() * dpsi_r).real).sum(dim=0)

        # energy error contribution: dE = sum_i f_i <dpsi_i | R_i> (2nd order, < 0);
        # dpsi is annulus-only, so this picks the complement part of the residual
        de_band = (dpsi.conj() * resid).real.sum(dim=1)  # (n_occ,)
        denergy += float(system.kweights[ik]) * float((occ * de_band).sum())

        dpsi_all.append(dpsi)
        psi_large_all.append(c_occ_1)
        occ_all.append(occ)
        sph_all.append(sph1)

    drho = drho / grid.volume
    drho_fo = drho.clone()

    if dyson:
        if xc is None:
            raise ValueError("dyson=True requires the xc functional")
        drho = _dyson_dress(
            res, xc, drho_fo, beta=dyson_beta, tol=dyson_tol,
            max_iter=dyson_max_iter, verbose=verbose,
        )

    return DiscretizationError(
        drho=drho, drho_first_order=drho_fo, denergy=denergy, dpsi=dpsi_all,
        psi_large=psi_large_all, occ=occ_all, spheres_large=sph_all, ecut=ecut,
        ecut_large=ecut_large, dyson=dyson,
    )


def _enlarged_projector_data(res: SCFResult, sph, device):
    """ProjectorData on a given (enlarged) sphere for the nonlocal energy."""
    system = res.system
    grid = system.grid
    beta_ls = [[b.l for b in upf.betas] for upf in system.upfs]
    dij_species = [torch.as_tensor(upf.dij, dtype=RDTYPE, device=device) for upf in system.upfs]
    q = np.sqrt(sph.kpg2.cpu().numpy())
    beta_tables = [
        torch.as_tensor(beta_form_factors(upf, q), dtype=RDTYPE, device=device)
        for upf in system.upfs
    ]
    return build_projector_data(
        sph, system.species_of_atom, beta_tables, beta_ls, dij_species, grid.volume
    )


def estimate_force_error(
    res: SCFResult,
    err: DiscretizationError,
    *,
    remove_net: bool = True,
) -> torch.Tensor:
    """Discretization error in the Hellmann-Feynman forces, δF ≈ (∂F/∂P) δP.

    Propagates the density-matrix perturbation δP (the density change δρ into the
    local term, the orbital corrections δφ into the nonlocal term) through the
    force F(P) by a single forward-mode directional derivative along δP, then a
    reverse pass over the positions:

        δF = -∂²E / ∂τ ∂ε   with   P(ε) = P + ε δP.

    No extra response solve is taken here; δP is the estimator's output. Returns
    (na, 3) [eV/Å], the estimated F_exact - F(res). The sign matches
    ``postscf.forces.forces`` (add δF to move toward the large-basis force).

    The first-order δP is used (drho_first_order together with dpsi) so the two
    channels stay consistent; a Dyson-dressed δρ would need the matching dressed
    δφ, which is future work.
    """
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("force error estimate is nspin=1 only for now")
    if getattr(res.system, "rho_core", None) is not None:
        raise NotImplementedError("NLCC force term not supported in the error estimate")
    system = res.system
    grid = system.grid
    device = res.v_eff.device

    drho = err.drho_first_order  # consistent with err.dpsi
    rho0 = res.rho.detach()

    pos = system.positions.detach().clone().requires_grad_(True)
    eps = torch.zeros((), dtype=RDTYPE, device=device, requires_grad=True)

    # local channel: density enters through rho_g
    rho_e_g = r_to_g((rho0 + eps * drho).to(CDTYPE))
    vloc_g = local_potential_g(
        pos, system.species_index, system.vloc_tables, grid.g_cart, grid.volume
    )
    energy = local_energy(rho_e_g, vloc_g, grid.volume)

    # nonlocal channel: orbital corrections enter through becp on the enlarged sphere
    nk = len(err.spheres_large)
    nocc_max = max(o.shape[0] for o in err.occ)
    occ_t = torch.zeros(nk, nocc_max, dtype=RDTYPE, device=device)
    becps = []
    dij_enl = None
    for ik, sph1 in enumerate(err.spheres_large):
        pd1 = _enlarged_projector_data(res, sph1, device)
        if dij_enl is None:
            dij_enl = pd1.dij_full
        p1 = projectors(pd1, pos)  # differentiable in positions
        c1_e = err.psi_large[ik] + eps * err.dpsi[ik]  # (n_occ, npw1), differentiable in eps
        becps.append(becp(p1, c1_e))
        occ_t[ik, : err.occ[ik].shape[0]] = err.occ[ik]
    energy = energy + nonlocal_energy(becps, dij_enl, occ_t, system.kweights)
    # Ewald has no δP dependence, so it drops out of ∂/∂ε.

    (de_deps,) = torch.autograd.grad(energy, eps, create_graph=True)
    (dgrad,) = torch.autograd.grad(de_deps, pos)
    dF = -dgrad
    if remove_net:
        dF = dF - dF.mean(dim=0, keepdim=True)
    return dF


# NOTE on stress. The naive fixed-δP linearization that works for forces,
# δσ ≈ (∂σ/∂P) δP by one forward-mode pass, does NOT estimate the stress
# discretization error: it comes out cleanly anti-correlated (corr ≈ -1 on
# diamond, capturing ≈ -0.5× the true error). The reason is that σ = ∂E/∂ε and
# its δP-response is dominated by the STRAIN-response of the orbital correction,
# the ⟨∂δφ/∂ε | R⟩ term, which a fixed-δφ forward pass omits. Forces avoid this
# because ⟨δφ | ∂R/∂τ⟩ (the ion moving) dominates there. A correct stress
# estimate needs a strain-parameterized residual (δφ differentiated through
# strain), reusing the ``rho``/``coeffs``/``spheres`` overrides now on
# ``postscf.stress._energy_strained``. Deferred; do not ship the naive form.
