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

Coverage. Density and energy error: norm-conserving and ultrasoft/PAW, nspin=1
and nspin=2, use_symmetry=False (a perturbation breaks the crystal symmetry, so
the response needs the full k-mesh; the same restriction the implicit-diff
response carries). Force error: norm-conserving nspin=1. The USPP/PAW path uses
the generalized residual R = P_annulus(H - eps S) psi and adds the
augmentation-charge response, dpsi perturbs the on-site occupations (becsum),
which feed the augmentation density through the Q functions. For nspin=2 the
complement correction runs per spin channel (each with its own v_eff and
eigenvalues) and the density and energy errors sum over the two channels. When
nspin=2 is used with smearing (required), the estimate on partially occupied
bands is the leading first-order term only.
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
from gradwave.scf.implicit import apply_chi0, apply_k_hxc
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
    uspp: bool = False           # USPP/PAW generalized-metric path
    dbecsum: list | None = None  # USPP: per-atom (nproj_a, nproj_a) on-site becsum change
    drho_smooth: torch.Tensor | None = None  # USPP: smooth part of drho (aug excluded)


def _occupied(res: SCFResult, ik: int, sp: int | None = None):
    """Occupied coefficients, eigenvalues, occupations at one k.

    ``sp`` selects the spin channel for nspin=2 (occupations then in [0,1]);
    None is the nspin=1 path (occupations in [0,2]).
    """
    if sp is None:
        occ, coeffs, eig = res.occupations[ik], res.coeffs[ik], res.eigenvalues[ik]
    else:
        occ = res.occupations[sp][ik]
        coeffs = res.coeffs[sp][ik]
        eig = res.eigenvalues[sp][ik]
    n_occ = int((occ > 1e-8).sum())
    return coeffs[:n_occ], eig[:n_occ], occ[:n_occ]


def _enlarged_hamiltonian(res: SCFResult, k_frac, ecut_large: float, device,
                          v_eff=None):
    """Build a HamiltonianK on the enlarged G-sphere at ecut_large for one k.

    ``v_eff`` overrides ``res.v_eff`` (the per-spin potential for nspin=2).
    """
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
    v = res.v_eff if v_eff is None else v_eff
    return sph, HamiltonianK(sph, grid.shape, v, pd, p)


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
    res : SCFResult or dict
        A converged SCF, use_symmetry=False. An ``SCFResult`` (norm-conserving,
        nspin=1 or 2) or the dict returned by ``scf_uspp`` (USPP/PAW, nspin=1).
    ecut_large : float, optional
        Cutoff [eV] of the enlarged basis defining the complement annulus. If
        None, uses ``factor * res.system.ecut``. Must satisfy
        ecut_large <= 4*ecut so the enlarged sphere fits the density FFT box.
    dyson : bool
        If True, dress the first-order estimate with the coarse-space dielectric
        response (needs ``xc``).
    """
    if isinstance(res, dict):
        return _estimate_density_error_uspp(
            res, ecut_large=ecut_large, factor=factor, xc=xc, verbose=verbose,
        )
    if getattr(res.system, "sym", None) is not None:
        raise NotImplementedError(
            "discretization error requires use_symmetry=False: a perturbation "
            "breaks the crystal symmetry, so the response needs the full k-mesh")
    system = res.system
    grid = system.grid
    device = res.v_eff.device
    ecut = float(system.ecut)
    nspin = int(getattr(res, "nspin", 1))
    if ecut_large is None:
        ecut_large = factor * ecut
    if gmax_from_ecut(ecut_large) > 2.0 * gmax_from_ecut(ecut) * (1.0 + 1e-9):
        raise ValueError(
            f"ecut_large={ecut_large:.1f} eV exceeds 4*ecut={4 * ecut:.1f} eV; the "
            "enlarged sphere would not fit the density FFT box. Lower ecut_large."
        )

    drho = torch.zeros(grid.shape, dtype=RDTYPE, device=device)
    denergy = 0.0
    # per-spin lists (single spin channel when nspin=1)
    spins = [None] if nspin == 1 else list(range(nspin))
    dpsi_s, psi_large_s, occ_s, sph_s = [], [], [], []

    for sp in spins:
        v_eff_sp = res.v_eff if sp is None else res.v_eff[sp]
        dpsi_k, psi_large_k, occ_k, sph_k = [], [], [], []
        for ik, sph0 in enumerate(system.spheres):
            c_occ, eps_occ, occ = _occupied(res, ik, sp)
            sph1, h1 = _enlarged_hamiltonian(
                res, sph0.k_frac, ecut_large, device, v_eff=v_eff_sp)

            # zero-pad occupied orbitals from the Ecut sphere onto the enlarged one
            box = sphere_to_box(c_occ, sph0.flat_idx, grid.shape)
            c_occ_1 = box_to_sphere(box, sph1.flat_idx)  # (n_occ, npw1)

            # complement residual R = P_annulus (H - eps) psi
            resid = h1.apply(c_occ_1) - eps_occ[:, None] * c_occ_1
            annulus = h1.t > ecut * (1.0 + 1e-9)  # (npw1,) bool
            denom = torch.clamp(h1.t[None, :] - eps_occ[:, None], min=1e-3)
            corr = -resid / denom.to(resid.dtype)
            dpsi = torch.where(annulus[None, :], corr, torch.zeros_like(resid))

            # density error contribution: drho = sum_i f_i * 2 Re(psi_i* dpsi_i).
            # occ is [0,2] for nspin=1 and [0,1] per spin channel for nspin=2;
            # the factor 2 below is the c.c. pair, not spin degeneracy.
            psi_r = g_to_r(c_occ, sph0.flat_idx, grid.shape)
            dpsi_r = g_to_r(dpsi, sph1.flat_idx, grid.shape)
            w = 2.0 * float(system.kweights[ik])
            drho += w * (occ.view(-1, 1, 1, 1) * (psi_r.conj() * dpsi_r).real).sum(dim=0)

            # energy error contribution: dE = sum_i f_i <dpsi_i | R_i> (2nd order,
            # < 0); dpsi is annulus-only, so this picks the complement residual
            de_band = (dpsi.conj() * resid).real.sum(dim=1)  # (n_occ,)
            denergy += float(system.kweights[ik]) * float((occ * de_band).sum())

            dpsi_k.append(dpsi)
            psi_large_k.append(c_occ_1)
            occ_k.append(occ)
            sph_k.append(sph1)
        dpsi_s.append(dpsi_k)
        psi_large_s.append(psi_large_k)
        occ_s.append(occ_k)
        sph_s.append(sph_k)

    # nspin=1 keeps the flat per-k layout the force path consumes
    if nspin == 1:
        dpsi_all, psi_large_all, occ_all, sph_all = (
            dpsi_s[0], psi_large_s[0], occ_s[0], sph_s[0])
    else:
        dpsi_all, psi_large_all, occ_all, sph_all = (
            dpsi_s, psi_large_s, occ_s, sph_s)

    drho = drho / grid.volume
    drho_fo = drho.clone()

    if dyson:
        if nspin != 1:
            raise NotImplementedError("Dyson dressing is nspin=1 only")
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


# --------------------------------------------------------------------------- #
#  USPP / PAW path                                                            #
# --------------------------------------------------------------------------- #


def _uspp_frozen_operators(res: dict, xc):
    """Frozen v_eff and screened D (dscr_full) of a converged USPP/PAW SCF.

    Rebuilt from the converged density and becsum exactly as
    ``postscf.uspp_bands.bands_uspp``. Together with ``system.q_full`` these
    define the band Hamiltonian H(k) c = eps S(k) c that the complement
    correction perturbs.
    """
    from gradwave.core.energies.hartree import hartree_potential_g
    from gradwave.scf.loop import vxc_potential

    system = res["system"]
    grid = system.grid
    vol = grid.volume
    dev = system.positions.device
    mask_flat = grid.dens_mask.reshape(-1)

    rho = res["rho"].detach()
    rho_g_box = r_to_g(rho.to(CDTYPE))
    v_h = (torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2),
                           dim=(-3, -2, -1)) * grid.n_points).real
    rho_xc = rho if system.rho_core is None else rho + system.rho_core
    v_xc, _ = vxc_potential(xc, rho_xc, grid)
    vloc_g = local_potential_g(
        system.positions, torch.as_tensor(system.species_of_atom, device=dev),
        system.vloc_tables, grid.g_cart, vol)
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real
    v_eff = v_h + v_xc + vloc_r

    v_eff_g = r_to_g(v_eff.to(CDTYPE)).reshape(-1)[mask_flat]
    phase_arg = system.g_sphere @ system.positions.T
    phase_pos = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
    dscr = torch.zeros_like(system.q_full)
    for a, sp in enumerate(system.species_of_atom):
        s0, s1 = system.atom_slices[a]
        contr = torch.einsum("ijg,g->ij", system.aug[sp].q_g.conj(),
                             v_eff_g * phase_pos[:, a])
        dscr[s0:s1, s0:s1] = (0.5 * (contr + contr.conj().T)).real
    dscr_full = dscr + system.proj_data[0].dij_full
    if any(p.is_paw for p in system.paws):
        from gradwave.scf.paw_onsite import OneCenter

        onec = {sp: OneCenter(system.paws[sp], xc)
                for sp in set(system.species_of_atom)}
        dscr_full = dscr_full.clone()
        for a, sp in enumerate(system.species_of_atom):
            _, ddd = onec[sp].energy_and_ddd(res["rho_ij_atoms"][a])
            s0, s1 = system.atom_slices[a]
            dscr_full[s0:s1, s0:s1] += ddd.to(dev)
    return v_eff, dscr_full


def _uspp_enlarged_hks(system, k_frac, ecut_large, v_eff, dscr_full, device):
    """Enlarged-sphere generalized band Hamiltonian _HkS at one k for USPP."""
    from gradwave.scf.uspp import _HkS

    grid = system.grid
    sph = build_gsphere(grid, ecut_large, np.asarray(k_frac, dtype=float),
                        device=device)
    q = np.sqrt(sph.kpg2.cpu().numpy())
    beta_ls = [[b.l for b in p.betas] for p in system.paws]
    dij_species = [torch.as_tensor(p.dij, dtype=RDTYPE, device=device)
                   for p in system.paws]
    beta_tables = [torch.as_tensor(beta_form_factors(p, q), dtype=RDTYPE,
                                   device=device) for p in system.paws]
    pd = build_projector_data(sph, system.species_of_atom, beta_tables,
                              beta_ls, dij_species, grid.volume)
    p = projectors(pd, system.positions)
    hs = _HkS(sph, grid.shape, v_eff, pd, p, dscr_full, system.q_full)
    return sph, hs, pd, p


def _aug_density_from_becsum(system, becsum, device):
    """Augmentation density on the real grid from a per-atom becsum list."""
    grid = system.grid
    phase_arg = system.g_sphere @ system.positions.T
    phase_pos = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
    aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE, device=device)
    for a, sp in enumerate(system.species_of_atom):
        aug_sph = aug_sph + phase_pos[:, a].conj() * torch.einsum(
            "ij,ijg->g", becsum[a], system.aug[sp].q_g)
    aug_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=device)
    aug_box[system.sphere_idx] = aug_sph / grid.volume
    return (torch.fft.ifftn(aug_box.reshape(grid.shape) * grid.n_points,
                            dim=(-3, -2, -1))).real


@torch.no_grad()
def _estimate_density_error_uspp(res: dict, *, ecut_large, factor, xc, verbose):
    """USPP/PAW discretization density error (nspin=1, use_symmetry=False).

    Adds two things to the NC recipe. The complement residual uses the
    generalized metric, R = P_annulus (H - eps S) psi, built from the enlarged
    _HkS. And the density change has two channels, the smooth part from dpsi
    and an augmentation part from the on-site occupation (becsum) change

        dbecsum^a_ij = sum_k w_k sum_b f_b [<psi_b|beta_i><beta_j|dpsi_b> + c.c.]

    fed through the Q functions exactly as the SCF builds rho_aug.
    """
    if res.get("nspin", 1) != 1:
        raise NotImplementedError("USPP density error is nspin=1 only for now")
    if res.get("hub_sites") is not None:
        raise NotImplementedError("USPP density error with DFT+U not implemented")
    if getattr(res["system"], "sym", None) is not None:
        raise NotImplementedError(
            "USPP density error requires use_symmetry=False: a perturbation "
            "breaks the crystal symmetry, so the response needs the full k-mesh")
    if xc is None:
        raise ValueError("USPP density error requires the xc functional (to "
                         "rebuild v_eff and the one-center ddd)")

    system = res["system"]
    grid = system.grid
    vol = grid.volume
    dev = system.positions.device
    ecut = float(system.ecut)
    if ecut_large is None:
        ecut_large = factor * ecut
    if gmax_from_ecut(ecut_large) > 2.0 * gmax_from_ecut(ecut) * (1.0 + 1e-9):
        raise ValueError(
            f"ecut_large={ecut_large:.1f} eV exceeds 4*ecut={4 * ecut:.1f} eV; the "
            "enlarged sphere would not fit the density FFT box. Lower ecut_large."
        )

    v_eff, dscr_full = _uspp_frozen_operators(res, xc)
    coeffs = res["coeffs"]
    eigs = res["eigenvalues"]
    occs = res["occupations"]

    drho_smooth = torch.zeros(grid.shape, dtype=RDTYPE, device=dev)
    denergy = 0.0
    dpsi_all, psi_large_all, occ_all, sph_all = [], [], [], []
    dbecsum = [torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=dev)
               for (s0, s1) in system.atom_slices]

    for ik, sph0 in enumerate(system.spheres):
        occ_k = occs[ik]
        n_occ = int((occ_k > 1e-8).sum())
        c_occ = coeffs[ik][:n_occ]
        eps_occ = eigs[ik][:n_occ]
        occ = occ_k[:n_occ]
        sph1, hs, pd1, p1 = _uspp_enlarged_hks(
            system, sph0.k_frac, ecut_large, v_eff, dscr_full, dev)

        # zero-pad occupied orbitals onto the enlarged sphere
        box = sphere_to_box(c_occ, sph0.flat_idx, grid.shape)
        c_occ_1 = box_to_sphere(box, sph1.flat_idx)

        # generalized complement residual R = P_annulus (H - eps S) psi
        resid = hs.h(c_occ_1) - eps_occ[:, None].to(CDTYPE) * hs.s(c_occ_1)
        annulus = hs.t > ecut * (1.0 + 1e-9)
        denom = torch.clamp(hs.t[None, :] - eps_occ[:, None], min=1e-3)
        corr = -resid / denom.to(resid.dtype)
        dpsi = torch.where(annulus[None, :], corr, torch.zeros_like(resid))

        # smooth-density change
        psi_r = g_to_r(c_occ, sph0.flat_idx, grid.shape)
        dpsi_r = g_to_r(dpsi, sph1.flat_idx, grid.shape)
        w = float(system.kweights[ik])
        drho_smooth += (2.0 * w) * (occ.view(-1, 1, 1, 1)
                                    * (psi_r.conj() * dpsi_r).real).sum(dim=0)

        # on-site occupation (becsum) change from the enlarged projectors
        bpsi = becp(p1, c_occ_1)
        bdps = becp(p1, dpsi)
        wk = (w * occ).to(CDTYPE)
        for a, (s0, s1) in enumerate(system.atom_slices):
            m = torch.einsum("b,bi,bj->ij", wk, bpsi[:, s0:s1].conj(),
                             bdps[:, s0:s1])
            dbecsum[a] = dbecsum[a] + m + m.conj().T

        # energy change (2nd order): sum_i f_i <dpsi_i | R_i>
        de_band = (dpsi.conj() * resid).real.sum(dim=1)
        denergy += w * float((occ * de_band).sum())

        dpsi_all.append(dpsi)
        psi_large_all.append(c_occ_1)
        occ_all.append(occ)
        sph_all.append(sph1)

    drho_smooth = drho_smooth / vol
    dbecsum = [0.5 * (m + m.conj().T) for m in dbecsum]
    if system.becsum_sym is not None:
        dbecsum = system.becsum_sym.apply(dbecsum)
    drho_aug = _aug_density_from_becsum(system, dbecsum, dev)
    drho = drho_smooth + drho_aug

    if verbose:
        dq = float(drho.sum()) * vol / grid.n_points
        print(f"  USPP drho: int(drho)={dq:.3e}, denergy={denergy:.4e} eV",
              flush=True)

    return DiscretizationError(
        drho=drho, drho_first_order=drho, denergy=denergy, dpsi=dpsi_all,
        psi_large=psi_large_all, occ=occ_all, spheres_large=sph_all, ecut=ecut,
        ecut_large=ecut_large, dyson=False, uspp=True, dbecsum=dbecsum,
        drho_smooth=drho_smooth,
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
    if err.uspp:
        raise NotImplementedError(
            "force error estimate is norm-conserving only; the USPP/PAW force "
            "needs the augmentation and one-center force terms in P(eps)")
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
