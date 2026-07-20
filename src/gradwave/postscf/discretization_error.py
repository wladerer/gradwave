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

Coverage.
  Density + energy error: norm-conserving, ultrasoft/PAW, and non-collinear/SOC
    (spinor); nspin=1 and nspin=2 throughout.
  Eigenvalue + band-gap error: the same three formalisms, nspin=1 and 2 (the
    per-band second-order shift the energy error already sums over occupations).
  Force error: norm-conserving collinear (nspin=1 and 2, no NLCC) and USPP/PAW
    (nspin=1 and 2, including NLCC). The USPP/PAW path propagates delta-P through
    the augmentation density, the S-orthogonality constraint, and the PAW
    one-center ddd response (following postscf.uspp_position.hessian_column). Not
    assembled: the norm-conserving NLCC force term (blocked on the ground-state
    NLCC force in postscf.forces, itself unimplemented) and the spinor force.

The USPP/PAW path uses the generalized residual R = P_annulus(H - eps S) psi and
adds the augmentation-charge response: dpsi perturbs the on-site occupations
(becsum), which feed the augmentation density through the Q functions. The
non-collinear path runs the correction on the doubled (up, down) plane-wave
axis with the spinor Hamiltonian rebuilt from the converged (rho, m); the
density error sums both spin blocks and occupations use degeneracy 1 (one
electron per spinor band). For nspin=2 the correction runs per spin channel
(each with its own v_eff and eigenvalues) and the errors sum over channels.

Symmetry: for norm-conserving nspin=1 the estimate runs on the IBZ k-points and
folds the density error over the star (the same operator the SCF applies to
rho), while the force error symmetrizes the output dF vector like ground-state
forces. USPP/PAW, nspin=2, the non-collinear path, and the Dyson dressing
require use_symmetry=False (a perturbation breaks the crystal symmetry, so a
genuine response needs the full k-mesh; folding the IBZ output only works for a
fully symmetric perturbation like the complement correction). When smearing is
used (required for nspin=2 and the spinor path), the estimate on partially
occupied bands is the leading first-order term only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from gradwave.core.batch import build_batched, g_to_r_b, projectors_b
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g, sphere_to_box
from gradwave.core.occupations import SCHEMES, find_fermi, occupations_and_entropy
from gradwave.core.hamiltonian import (
    HamiltonianK,
    becp,
    build_projector_data,
    projectors,
)
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_gsphere, gmax_from_ecut
from gradwave.postscf.uspp_frozen import (
    aug_density_from_becsum,
    frozen_veff,
    screen_phase,
    screened_dscr,
)
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
    drho_first_order: torch.Tensor   # pre-Dyson drho (== drho on USPP/spinor, no dressing there)
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
    drho_smooth: torch.Tensor | None = None  # USPP: smooth part of drho (aug excluded, spin-summed)
    deig: list | None = None     # USPP: per-k (n_occ,) occupied eigenvalue error (S-term response)
    drho_smooth_spin: list | None = None  # USPP: per-spin smooth drho (force error XC channel)


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


def _bands_at(res: SCFResult, ik: int, sp: int | None = None):
    """All computed coefficients and eigenvalues at one k (and spin channel).

    Accepts an ``SCFResult`` (norm-conserving) or the ``scf_uspp`` result dict;
    both carry ``coeffs``/``eigenvalues`` with the same per-k (nspin=1) or
    [spin][k] (nspin=2) layout.
    """
    coeffs = res["coeffs"] if isinstance(res, dict) else res.coeffs
    eigs = res["eigenvalues"] if isinstance(res, dict) else res.eigenvalues
    if sp is None:
        return coeffs[ik], eigs[ik]
    return coeffs[sp][ik], eigs[sp][ik]


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
    smearing: str | None = None,
    width: float | None = None,
    verbose: bool = False,
) -> DiscretizationError:
    """Estimate the plane-wave discretization error in the converged density.

    Parameters
    ----------
    res : SCFResult, dict, or NCResult
        A converged SCF. An ``SCFResult`` (norm-conserving, nspin=1 or 2;
        use_symmetry allowed for nspin=1), the dict returned by ``scf_uspp``
        (USPP/PAW, nspin=1 or 2, use_symmetry=False), or an ``NCResult``
        (non-collinear/SOC spinor SCF, use_symmetry=False -- pass ``xc`` as the
        NoncollinearXC and the run's ``smearing``/``width``).
    ecut_large : float, optional
        Cutoff [eV] of the enlarged basis defining the complement annulus. If
        None, uses ``factor * res.system.ecut``. Must satisfy
        ecut_large <= 4*ecut so the enlarged sphere fits the density FFT box.
    dyson : bool
        If True, dress the first-order estimate with the coarse-space dielectric
        response (needs ``xc``). Only implemented on the norm-conserving
        (``SCFResult``) path; requesting it on the USPP/PAW or spinor paths
        raises ``NotImplementedError`` (the ``dyson_*`` tuning kwargs are inert
        there).
    """
    if isinstance(res, dict):
        if dyson:
            raise NotImplementedError(
                "Dyson dressing not implemented for the USPP/spinor path")
        return _estimate_density_error_uspp(
            res, ecut_large=ecut_large, factor=factor, xc=xc, verbose=verbose,
        )
    if _is_ncresult(res):
        if dyson:
            raise NotImplementedError(
                "Dyson dressing not implemented for the USPP/spinor path")
        if xc is None:
            raise ValueError("non-collinear density error requires xc (the "
                             "NoncollinearXC used for the run)")
        scheme, w = _spinor_smearing(smearing, width)
        return _estimate_density_error_noncollinear(
            res, ecut_large=ecut_large, factor=factor, xc=xc,
            smearing=scheme, width=w)
    system = res.system
    grid = system.grid
    device = res.v_eff.device
    ecut = float(system.ecut)
    nspin = int(getattr(res, "nspin", 1))
    sym = getattr(system, "sym", None)
    # With symmetry on, the complement runs on the IBZ k-points and the density
    # error is symmetrized (folded over the star) exactly as the SCF symmetrizes
    # rho. The energy error is a scalar BZ integral, so the IBZ-weighted sum is
    # already correct. Only the density channel needs the fold, and only for the
    # collinear-nonmagnetic case (nspin=1); AFM/ferrimagnetic symmetry is out of
    # scope. The force propagation then symmetrizes the output dF vector.
    if sym is not None and nspin != 1:
        raise NotImplementedError(
            "symmetric discretization error is nspin=1 only; use "
            "use_symmetry=False for nspin=2")
    if sym is not None and dyson:
        raise NotImplementedError("Dyson dressing requires use_symmetry=False")
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
    if sym is not None:
        # fold the IBZ complement over the star, same operator the SCF uses on rho
        sym_g = system.rho_symmetrizer.apply(r_to_g(drho.to(CDTYPE)))
        drho = (torch.fft.ifftn(sym_g * grid.n_points, dim=(-3, -2, -1))).real
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
    """Frozen per-spin v_eff and screened D of a converged USPP/PAW SCF.

    Rebuilt from the converged density and becsum exactly as the USPP SCF map
    (postscf.uspp_bands.bands_uspp for nspin=1; the per-spin v_xc and one-center
    ddd for nspin=2). Together with ``system.q_full`` these define the band
    Hamiltonian H_s(k) c = eps S(k) c that the complement correction perturbs.
    Returns ``(veff_s, dscr_s)`` as lists of length nspin.
    """
    veff_s = frozen_veff(res, xc)
    dscr_s = screened_dscr(res, xc, veff_s)
    return veff_s, dscr_s


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


def _aug_density_from_becsum(system, becsum):
    """Augmentation density on the real grid from a per-atom becsum list."""
    return aug_density_from_becsum(system, becsum, screen_phase(system))


@torch.no_grad()
def _estimate_density_error_uspp(res: dict, *, ecut_large, factor, xc, verbose):
    """USPP/PAW discretization density error (nspin=1 or 2, use_symmetry=False).

    Adds two things to the NC recipe. The complement residual uses the
    generalized metric, R = P_annulus (H - eps S) psi, built from the enlarged
    _HkS. And the density change has two channels, the smooth part from dpsi
    and an augmentation part from the on-site occupation (becsum) change

        dbecsum^a_ij = sum_k w_k sum_b f_b [<psi_b|beta_i><beta_j|dpsi_b> + c.c.]

    fed through the Q functions exactly as the SCF builds rho_aug.
    """
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
    nspin = int(res.get("nspin", 1))
    if ecut_large is None:
        ecut_large = factor * ecut
    if gmax_from_ecut(ecut_large) > 2.0 * gmax_from_ecut(ecut) * (1.0 + 1e-9):
        raise ValueError(
            f"ecut_large={ecut_large:.1f} eV exceeds 4*ecut={4 * ecut:.1f} eV; the "
            "enlarged sphere would not fit the density FFT box. Lower ecut_large."
        )

    veff_s, dscr_s = _uspp_frozen_operators(res, xc)
    coeffs = res["coeffs"]
    eigs = res["eigenvalues"]
    occs = res["occupations"]

    denergy = 0.0
    spins = [None] if nspin == 1 else list(range(nspin))
    drho_smooth_sp = [torch.zeros(grid.shape, dtype=RDTYPE, device=dev)
                      for _ in spins]
    dpsi_s, psi_large_s, occ_s_out, sph_s, deig_s = [], [], [], [], []
    # per-spin becsum change (a single channel when nspin=1)
    dbecsum_s = [[torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=dev)
                  for (s0, s1) in system.atom_slices] for _ in spins]

    for isp_i, sp in enumerate(spins):
        c_sp = coeffs if sp is None else coeffs[sp]
        e_sp = eigs if sp is None else eigs[sp]
        o_sp = occs if sp is None else occs[sp]
        v_eff, dscr_full = veff_s[isp_i], dscr_s[isp_i]
        dbecsum = dbecsum_s[isp_i]
        dpsi_k, psi_large_k, occ_k_out, sph_k, deig_k = [], [], [], [], []
        for ik, sph0 in enumerate(system.spheres):
            occ_k = o_sp[ik]
            n_occ = int((occ_k > 1e-8).sum())
            c_occ = c_sp[ik][:n_occ]
            eps_occ = e_sp[ik][:n_occ]
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

            # smooth-density change (occ is [0,2] for nspin=1, [0,1] per spin)
            psi_r = g_to_r(c_occ, sph0.flat_idx, grid.shape)
            dpsi_r = g_to_r(dpsi, sph1.flat_idx, grid.shape)
            w = float(system.kweights[ik])
            drho_smooth_sp[isp_i] += (2.0 * w) * (occ.view(-1, 1, 1, 1)
                                        * (psi_r.conj() * dpsi_r).real).sum(dim=0)

            # on-site occupation (becsum) change from the enlarged projectors
            bpsi = becp(p1, c_occ_1)
            bdps = becp(p1, dpsi)
            wk = (w * occ).to(CDTYPE)
            for a, (s0, s1) in enumerate(system.atom_slices):
                m = torch.einsum("b,bi,bj->ij", wk, bpsi[:, s0:s1].conj(),
                                 bdps[:, s0:s1])
                dbecsum[a] = dbecsum[a] + m + m.conj().T

            # energy change (2nd order): sum_i f_i <dpsi_i | R_i>. de_band is also
            # the per-band eigenvalue error, the S-orthogonality response the
            # force-error estimate pairs against eps.
            de_band = (dpsi.conj() * resid).real.sum(dim=1)
            denergy += w * float((occ * de_band).sum())

            dpsi_k.append(dpsi)
            psi_large_k.append(c_occ_1)
            occ_k_out.append(occ)
            sph_k.append(sph1)
            deig_k.append(de_band)
        dpsi_s.append(dpsi_k)
        psi_large_s.append(psi_large_k)
        occ_s_out.append(occ_k_out)
        sph_s.append(sph_k)
        deig_s.append(deig_k)

    drho_smooth_sp = [d / vol for d in drho_smooth_sp]
    drho_smooth = sum(drho_smooth_sp)
    # augmentation change, summed over spin channels
    drho_aug = torch.zeros(grid.shape, dtype=RDTYPE, device=dev)
    dbecsum_out = []
    for isp_i in range(len(spins)):
        dbec = [0.5 * (m + m.conj().T) for m in dbecsum_s[isp_i]]
        # becsum symmetrization is gated above (sym is None here, so becsum_sym is too)
        drho_aug = drho_aug + _aug_density_from_becsum(system, dbec)
        dbecsum_out.append(dbec)
    drho = drho_smooth + drho_aug

    if verbose:
        dq = float(drho.sum()) * vol / grid.n_points
        print(f"  USPP drho: int(drho)={dq:.3e}, denergy={denergy:.4e} eV",
              flush=True)

    if nspin == 1:
        dpsi_all, psi_large_all = dpsi_s[0], psi_large_s[0]
        occ_all, sph_all, dbecsum_ret = occ_s_out[0], sph_s[0], dbecsum_out[0]
        deig_all = deig_s[0]
    else:
        dpsi_all, psi_large_all = dpsi_s, psi_large_s
        occ_all, sph_all, dbecsum_ret = occ_s_out, sph_s, dbecsum_out
        deig_all = deig_s

    return DiscretizationError(
        drho=drho, drho_first_order=drho, denergy=denergy, dpsi=dpsi_all,
        psi_large=psi_large_all, occ=occ_all, spheres_large=sph_all, ecut=ecut,
        ecut_large=ecut_large, dyson=False, uspp=True, dbecsum=dbecsum_ret,
        drho_smooth=drho_smooth, deig=deig_all, drho_smooth_spin=drho_smooth_sp,
    )


@torch.no_grad()
def _estimate_eigenvalue_error_uspp(res: dict, *, ecut_large, factor, xc, bands):
    """USPP/PAW eigenvalue error via the generalized complement correction.

    The same per-band second-order shift the USPP density error already sums over
    occupations, run on every requested band with the generalized metric:

        R = P_annulus (H - eps S) psi,  dpsi = -R / (T_G - eps),
        deps = <dpsi | R> = -sum_annulus |R|^2 / (T_G - eps) <= 0.

    Reuses the frozen per-spin (v_eff, screened D) operators and the enlarged
    _HkS the density path builds, so the occupation-weighted sum of the occupied
    shifts reproduces ``estimate_density_error(res, ...).denergy`` exactly.
    """
    if getattr(res["system"], "sym", None) is not None:
        raise NotImplementedError(
            "USPP eigenvalue error requires use_symmetry=False: a perturbation "
            "breaks the crystal symmetry, so the response needs the full k-mesh")
    if xc is None:
        raise ValueError("USPP eigenvalue error requires the xc functional (to "
                         "rebuild v_eff and the one-center ddd)")
    system = res["system"]
    grid = system.grid
    dev = system.positions.device
    ecut = float(system.ecut)
    nspin = int(res.get("nspin", 1))
    if ecut_large is None:
        ecut_large = factor * ecut
    if gmax_from_ecut(ecut_large) > 2.0 * gmax_from_ecut(ecut) * (1.0 + 1e-9):
        raise ValueError(
            f"ecut_large={ecut_large:.1f} eV exceeds 4*ecut={4 * ecut:.1f} eV; the "
            "enlarged sphere would not fit the density FFT box. Lower ecut_large."
        )

    veff_s, dscr_s = _uspp_frozen_operators(res, xc)
    coeffs, eigs = res["coeffs"], res["eigenvalues"]
    spins = [None] if nspin == 1 else list(range(nspin))
    deig_s, eig_s = [], []
    for isp, sp in enumerate(spins):
        c_sp = coeffs if sp is None else coeffs[sp]
        e_sp = eigs if sp is None else eigs[sp]
        v_eff, dscr_full = veff_s[isp], dscr_s[isp]
        deig_k, eig_k = [], []
        for ik, sph0 in enumerate(system.spheres):
            sel = slice(None) if bands is None else bands
            c_sel = c_sp[ik][sel]
            eps_sel = e_sp[ik][sel]
            sph1, hs, _pd1, _p1 = _uspp_enlarged_hks(
                system, sph0.k_frac, ecut_large, v_eff, dscr_full, dev)
            box = sphere_to_box(c_sel, sph0.flat_idx, grid.shape)
            c1 = box_to_sphere(box, sph1.flat_idx)
            resid = hs.h(c1) - eps_sel[:, None].to(CDTYPE) * hs.s(c1)
            annulus = hs.t > ecut * (1.0 + 1e-9)
            denom = torch.clamp(hs.t[None, :] - eps_sel[:, None], min=1e-3)
            dpsi = torch.where(annulus[None, :], -resid / denom.to(resid.dtype),
                               torch.zeros_like(resid))
            deig_k.append((dpsi.conj() * resid).real.sum(dim=1))
            eig_k.append(eps_sel.clone())
        deig_s.append(deig_k)
        eig_s.append(eig_k)

    if nspin == 1:
        deig, eig = deig_s[0], eig_s[0]
    else:
        deig, eig = deig_s, eig_s
    return EigenvalueError(deig=deig, eig=eig, ecut=ecut,
                           ecut_large=ecut_large, nspin=nspin)


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


def _uspp_enlarged_projdata(system, sph1, device):
    """USPP/PAW ProjectorData on a given enlarged sphere (differentiable becp)."""
    q = np.sqrt(sph1.kpg2.cpu().numpy())
    beta_ls = [[b.l for b in p.betas] for p in system.paws]
    dij_species = [torch.as_tensor(p.dij, dtype=RDTYPE, device=device)
                   for p in system.paws]
    beta_tables = [torch.as_tensor(beta_form_factors(p, q), dtype=RDTYPE,
                                   device=device) for p in system.paws]
    return build_projector_data(sph1, system.species_of_atom, beta_tables,
                                beta_ls, dij_species, system.grid.volume)


def _estimate_force_error_uspp(res: dict, err: DiscretizationError, xc, *,
                               remove_net: bool = True) -> torch.Tensor:
    """USPP/PAW discretization force error, δF ≈ ∂²E/∂τ∂ε along the estimator's δP.

    Rebuilds the τ-differentiable USPP/PAW force energy (postscf.paw_forces) with
    the state carried by leaves -- the smooth density, the occupied orbitals on
    the enlarged sphere, the eigenvalues (in the S-orthogonality constraint), and
    the PAW one-center ddd -- and takes the mixed derivative along the complement
    perturbation δP. The δP pieces are the estimator's own outputs: drho_smooth
    (per spin), dpsi (orbital corrections), deig (eigenvalue error, the response
    the S-constraint pairs against), and dbecsum (feeding the augmentation density
    and, through the one-center Hessian-vector product, the ddd response). Follows
    postscf.uspp_position.hessian_column but with δP in place of the SCF's
    position response and no explicit ∂²E/∂τ∂τ' term. nspin=1 or 2; the crystal
    symmetry must be off (as for the USPP density error).
    """
    from gradwave.core.density import sigma_from_rho
    from gradwave.core.energies.hartree import hartree_energy
    from gradwave.core.xc.base import xc_eager
    from gradwave.postscf.paw_forces import _aug_at_fixed, _aug_from_becsum
    from gradwave.postscf.uspp_implicit import _ConvergedUSPP

    system = res["system"]
    if getattr(system, "sym", None) is not None:
        raise NotImplementedError(
            "USPP force error requires use_symmetry=False (a perturbation breaks "
            "the crystal symmetry, so the response needs the full k-mesh)")
    if res.get("hub_sites") is not None:
        raise NotImplementedError("USPP force error with DFT+U not implemented")
    if xc is None:
        raise ValueError("USPP force error requires the xc functional (to rebuild "
                         "v_eff, the augmentation, and the one-center ddd)")

    grid = system.grid
    vol, shape = grid.volume, grid.shape
    kw = system.kweights
    dev = system.positions.device
    nspin = int(res.get("nspin", 1))

    def _s(x):
        return [x] if nspin == 1 else list(x)

    psi_s, dpsi_s, occ_s = _s(err.psi_large), _s(err.dpsi), _s(err.occ)
    deig_s, dbec_s, sph_s = _s(err.deig), _s(err.dbecsum), _s(err.spheres_large)
    drho_sm_s = err.drho_smooth_spin
    eigs_src = res["eigenvalues"]
    nk = len(sph_s[0])
    species = system.species_of_atom
    is_paw = any(p.is_paw for p in system.paws)

    pos = system.positions.detach().clone().requires_grad_(True)
    # enlarged projectors (the enlarged spheres are spin-independent)
    pd_list = [_uspp_enlarged_projdata(system, sph_s[0][ik], dev)
               for ik in range(nk)]
    projs = [projectors(pd, pos) for pd in pd_list]
    phase_arg = system.g_sphere @ pos.T
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
    q_full = system.q_full.to(CDTYPE)
    dij_full = system.proj_data[0].dij_full.to(CDTYPE)

    # ddd response to the becsum perturbation (PAW one-center Hessian-vector
    # product); ddd leaves frozen at the converged becsum.
    dddd_s = None
    ddd_leaves = [[None] * len(system.atom_slices) for _ in range(nspin)]
    if is_paw:
        with torch.no_grad():
            cs = _ConvergedUSPP(res, xc)
            dddd_s = cs.hvp_onecenter(
                [[m.to(CDTYPE) for m in dbec_s[isp]] for isp in range(nspin)])
        from gradwave.scf.paw_onsite import OneCenter
        onec = {sp: OneCenter(system.paws[sp], xc) for sp in set(species)}
        becsum = res["rho_ij_atoms"]
        for at, sp in enumerate(species):
            bec = becsum[at] if nspin == 1 else [becsum[0][at], becsum[1][at]]
            _, ddd = onec[sp].energy_and_ddd(bec)
            ddd_ch = [ddd] if nspin == 1 else list(ddd)
            for isp in range(nspin):
                ddd_leaves[isp][at] = (ddd_ch[isp].detach().clone().to(dev)
                                       .requires_grad_(True))

    leaves, responses = [], []      # matched (leaf, δstate) pairs for the s scalar

    # smooth-density leaves (per spin), response = per-spin smooth drho
    rho_s_leaves = []
    for isp in range(nspin):
        rho_sp_isp = res["rho"] if nspin == 1 else res["rho_spin"][isp]
        fixed = (rho_sp_isp.detach() - _aug_at_fixed(res, system, isp)).detach()
        leaf = fixed.clone().requires_grad_(True)
        rho_s_leaves.append(leaf)
        leaves.append(leaf)
        responses.append(drho_sm_s[isp])

    # orbital + eigenvalue leaves and the augmentation/one-center energy per spin
    e = pos.sum() * 0.0             # scalar seed on the pos graph
    rho_chans = []
    for isp in range(nspin):
        e_full = eigs_src if nspin == 1 else eigs_src[isp]
        rho_ij = [torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=dev)
                  for (s0, s1) in system.atom_slices]
        for ik in range(nk):
            c_leaf = psi_s[isp][ik].detach().clone().requires_grad_(True)
            leaves.append(c_leaf)
            responses.append(dpsi_s[isp][ik])
            n_occ = occ_s[isp][ik].shape[0]
            eps_leaf = e_full[ik][:n_occ].detach().clone().requires_grad_(True)
            leaves.append(eps_leaf)
            responses.append(deig_s[isp][ik])

            b = becp(projs[ik], c_leaf)
            w = (kw[ik] * occ_s[isp][ik]).to(CDTYPE)
            for a, (s0, s1) in enumerate(system.atom_slices):
                ba = b[:, s0:s1]
                rho_ij[a] = rho_ij[a] + torch.einsum("b,bi,bj->ij", w,
                                                     ba.conj(), ba)
            # nonlocal (bare D) + S-orthogonality constraint (−Σ w f ε ⟨ψ|Q|ψ⟩)
            nl = torch.einsum("bi,ij,bj->b", b.conj(), dij_full, b).real
            e = e + (kw[ik] * occ_s[isp][ik] * nl).sum()
            so = torch.einsum("bi,ij,bj->b", b.conj(), q_full, b).real
            e = e - (kw[ik] * occ_s[isp][ik] * eps_leaf * so).sum()
        rho_ij = [0.5 * (m + m.conj().T) for m in rho_ij]
        if is_paw:
            for a in range(len(system.atom_slices)):
                e = e + (ddd_leaves[isp][a].to(CDTYPE) * rho_ij[a]).sum().real
                leaves.append(ddd_leaves[isp][a])
                responses.append(dddd_s[isp][a].real)
        rho_chans.append(rho_s_leaves[isp] + _aug_from_becsum(system, rho_ij, phases))

    # local + Hartree see the total density; XC sees the spin channels (+ NLCC)
    rho_tot = sum(rho_chans)
    rho_core = _uspp_rho_core_on_graph(system, phases, pos)
    if nspin == 1:
        rho_xc = rho_tot if rho_core is None else rho_tot + rho_core
        sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
        with xc_eager():
            e = e + xc.energy(rho_xc, vol, sigma)
    else:
        c2 = 0.0 if rho_core is None else 0.5 * rho_core
        r_u, r_d = rho_chans[0] + c2, rho_chans[1] + c2
        if xc.needs_gradient:
            s_uu = sigma_from_rho(r_u, grid.g_cart)
            s_dd = sigma_from_rho(r_d, grid.g_cart)
            s_tt = sigma_from_rho(r_u + r_d, grid.g_cart)
        else:
            s_uu = s_dd = s_tt = None
        with xc_eager():
            e = e + xc.energy(r_u, r_d, vol, s_uu, s_dd, s_tt)
    rho_g = r_to_g(rho_tot.to(CDTYPE))
    species_index = torch.tensor(species, dtype=torch.int64, device=dev)
    vloc_g = local_potential_g(pos, species_index, system.vloc_tables,
                               grid.g_cart, vol)
    e = e + hartree_energy(rho_g, grid.g2, vol) + local_energy(rho_g, vloc_g, vol)

    # s = ∂E/∂ε along δP (no explicit ∂E/∂τ term: ε perturbs only the state);
    # δF = -∂/∂τ s. Complex leaves pair via Re⟨g, δz⟩ (conjugate-Wirtinger).
    grads = torch.autograd.grad(e, leaves, create_graph=True)
    s = pos.sum() * 0.0
    for g, d in zip(grads, responses, strict=True):
        if g.is_complex():
            s = s + (g.conj() * d).real.sum()
        else:
            s = s + (g * d.real if d.is_complex() else g * d).sum()
    (dgrad,) = torch.autograd.grad(s, pos)
    dF = -dgrad
    if remove_net:
        dF = dF - dF.mean(dim=0, keepdim=True)
    return dF.detach()


def _uspp_rho_core_on_graph(system, phases, pos):
    """NLCC core density on the graph (differentiable in positions), or None.

    Copied from postscf.paw_forces: the core rides the same e^{+iGτ} phases as
    the augmentation, so its τ-derivative is the NLCC core force and it enters
    the XC argument here.
    """
    if system.rho_core is None:
        return None
    from gradwave.pseudo.radial_torch import RadialTables

    grid = system.grid
    vol = grid.volume
    q_sph = torch.linalg.norm(system.g_sphere, dim=1)
    core = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE, device=pos.device)
    for sp in set(system.species_of_atom):
        paw = system.paws[sp]
        if paw.core_rho is None:
            continue
        tab = RadialTables(paw, device=pos.device)
        with torch.no_grad():
            f_core = tab.core_of_g(q_sph)
        atoms = [a for a, sa in enumerate(system.species_of_atom) if sa == sp]
        core = core + phases[:, atoms].conj().sum(dim=1) * f_core.to(CDTYPE) / vol
    core_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=pos.device)
    core_box[system.sphere_idx] = core
    return torch.fft.ifftn(core_box.reshape(grid.shape) * grid.n_points,
                           dim=(-3, -2, -1)).real


def estimate_force_error(
    res: SCFResult,
    err: DiscretizationError,
    *,
    xc=None,
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

    Norm-conserving (nspin=1 or 2, no NLCC) or USPP/PAW (nspin=1 or 2, needs
    ``xc``). For nspin=2 the nonlocal channel sums over the two spin channels --
    each with its own orbital corrections and occupations (in [0,1]) -- while the
    local channel already sees the spin-summed density error. The USPP/PAW path
    additionally propagates δP through the augmentation density (δbecsum), the
    S-orthogonality constraint (via the eigenvalue error), and the PAW one-center
    ddd response, following ``postscf.uspp_position.hessian_column``.
    """
    if _is_ncresult(res):
        raise NotImplementedError(
            "force error estimate is norm-conserving collinear only; the spinor "
            "force terms in P(eps) are not assembled")
    if err.uspp:
        return _estimate_force_error_uspp(res, err, xc, remove_net=remove_net)
    if getattr(res.system, "rho_core", None) is not None:
        raise NotImplementedError(
            "NLCC force term not supported in the error estimate: it is blocked "
            "on the ground-state NLCC force (postscf.forces), itself unimplemented")
    nspin = int(getattr(res, "nspin", 1))
    system = res.system
    grid = system.grid
    device = res.v_eff.device

    drho = err.drho_first_order  # consistent with err.dpsi; total (spin-summed) drho
    rho0 = res.rho.detach()      # total density for nspin=1 and 2

    pos = system.positions.detach().clone().requires_grad_(True)
    eps = torch.zeros((), dtype=RDTYPE, device=device, requires_grad=True)

    # local channel: total density enters through rho_g
    rho_e_g = r_to_g((rho0 + eps * drho).to(CDTYPE))
    vloc_g = local_potential_g(
        pos, system.species_index, system.vloc_tables, grid.g_cart, grid.volume
    )
    energy = local_energy(rho_e_g, vloc_g, grid.volume)

    # nonlocal channel: orbital corrections enter through becp on the enlarged
    # sphere, summed over spin channels. For nspin=1 the estimator returns flat
    # per-k lists; for nspin=2 they are nested [spin][k], so normalize to a
    # per-spin layout here. Each spin channel carries its own occupations
    # (in [0,1]) and its own δφ.
    if nspin == 1:
        dpsi_s, psi_large_s, occ_s, sph_s = (
            [err.dpsi], [err.psi_large], [err.occ], [err.spheres_large])
    else:
        dpsi_s, psi_large_s, occ_s, sph_s = (
            err.dpsi, err.psi_large, err.occ, err.spheres_large)

    nk = len(sph_s[0])
    nocc_max = max(o.shape[0] for occ_k in occ_s for o in occ_k)
    dij_enl = None
    for isp in range(nspin):
        occ_t = torch.zeros(nk, nocc_max, dtype=RDTYPE, device=device)
        becps = []
        for ik, sph1 in enumerate(sph_s[isp]):
            pd1 = _enlarged_projector_data(res, sph1, device)
            if dij_enl is None:
                dij_enl = pd1.dij_full
            p1 = projectors(pd1, pos)  # differentiable in positions
            # (n_occ, npw1), differentiable in eps
            c1_e = psi_large_s[isp][ik] + eps * dpsi_s[isp][ik]
            becps.append(becp(p1, c1_e))
            occ_t[ik, : occ_s[isp][ik].shape[0]] = occ_s[isp][ik]
        energy = energy + nonlocal_energy(becps, dij_enl, occ_t, system.kweights)
    # Ewald has no δP dependence, so it drops out of ∂/∂ε.

    (de_deps,) = torch.autograd.grad(energy, eps, create_graph=True)
    (dgrad,) = torch.autograd.grad(de_deps, pos)
    dF = -dgrad
    if remove_net:
        dF = dF - dF.mean(dim=0, keepdim=True)
    if system.sym is not None:
        # the IBZ estimate gives the force error on the irreducible set; project
        # onto the symmetry-invariant subspace exactly as ground-state forces do
        from gradwave.symmetry import symmetrize_forces

        dF = symmetrize_forces(dF, system.sym, grid.cell)
    return dF


# --------------------------------------------------------------------------- #
#  Non-collinear (spinor) path                                                #
# --------------------------------------------------------------------------- #


def _is_ncresult(res) -> bool:
    """True for a non-collinear ``NCResult`` (spinor SCF output).

    Duck-typed to avoid importing scf.noncollinear at module load (it would
    close an import cycle). An NCResult carries the magnetization field ``m``
    and the integrated moment ``mag_vec`` but no ``v_eff``/``occupations``.
    """
    return (not isinstance(res, dict)
            and hasattr(res, "mag_vec") and hasattr(res, "m")
            and not hasattr(res, "v_eff"))


def _spinor_smearing(smearing, width):
    """Resolve the (scheme, width) for recomputing spinor occupations.

    NCResult stores neither the occupations nor the smearing it was run with, so
    the caller supplies them (the CLI passes ``inp.smearing``). A non-collinear
    SCF always uses a real scheme -- ``none`` maps to gaussian, exactly as
    ``api._run_scf_noncollinear`` does when seeding the run.
    """
    scheme = "gaussian" if (smearing is None or smearing == "none") else smearing
    if scheme not in SCHEMES:
        raise ValueError(f"unknown smearing scheme {scheme!r}")
    return scheme, (0.1 if width is None else float(width))


def _spinor_enlarged_batch(system, ecut_large, device):
    """Enlarged-sphere BatchedK + spinor projectors at ``ecut_large``.

    Rebuilds the k-batched plane-wave machinery on the complement basis exactly
    as ``scf.loop.setup_system`` does at the run cutoff: per-k GSphere, scalar
    KB projector data (empty for fully-relativistic pseudos, whose projectors
    are the spinor SO set), and -- when ``system.is_fr`` -- the per-k SO radial
    tables the spinor projector builder consumes. Returns
    ``(bk_large, spheres_large, so_tables_large)``.
    """
    grid = system.grid
    spheres1 = [build_gsphere(grid, ecut_large, sph0.k_frac, device=device)
                for sph0 in system.spheres]
    is_fr = bool(getattr(system, "is_fr", False))
    npw_max1 = max(s.npw for s in spheres1)
    dij_species = [torch.as_tensor(u.dij, dtype=RDTYPE, device=device)
                   for u in system.upfs]
    proj_data1 = []
    so_tabs1 = ([torch.zeros(len(spheres1), u.n_proj, npw_max1, dtype=RDTYPE,
                             device=device) for u in system.upfs]
                if is_fr else None)
    for ik, sph in enumerate(spheres1):
        q = np.sqrt(sph.kpg2.cpu().numpy())
        beta_tables = [torch.as_tensor(beta_form_factors(u, q), dtype=RDTYPE,
                                       device=device) for u in system.upfs]
        if is_fr:
            for sp_i in range(len(system.upfs)):
                so_tabs1[sp_i][ik, :, :sph.npw] = beta_tables[sp_i]
            beta_ls = [[] for _ in system.upfs]
            beta_tables = [t[:0] for t in beta_tables]
        else:
            beta_ls = [[b.l for b in u.betas] for u in system.upfs]
        proj_data1.append(build_projector_data(
            sph, system.species_of_atom, beta_tables, beta_ls, dij_species,
            grid.volume))
    bk1 = build_batched(spheres1, proj_data1, device)
    return bk1, spheres1, so_tabs1


@torch.no_grad()
def _spinor_complement(res, *, ecut_large, factor, xc, smearing, width):
    """Shared spinor complement correction for the density/energy/eigenvalue error.

    Runs the first-order correction  R = P_annulus (H - eps) psi,
    dpsi = -R/(T_G - eps)  on the enlarged spinor basis for every band, exactly
    as the collinear path does per band, but with the 2-component (up/down)
    block structure and the enlarged spinor Hamiltonian rebuilt post-hoc from
    the converged (rho, m). Returns a dict of the tensors both the density and
    the eigenvalue estimators consume.
    """
    from gradwave.core.energies.hartree import hartree_potential_g
    from gradwave.core.xc.noncollinear import vxc_and_bxc
    from gradwave.scf.noncollinear import SpinorHamiltonian

    system = res.system
    if getattr(system, "sym", None) is not None \
            or getattr(system, "rho_symmetrizer", None) is not None:
        raise NotImplementedError(
            "non-collinear discretization error requires use_symmetry=False: the "
            "spinor complement fold over the (magnetic) star is not implemented")
    grid = system.grid
    device = res.rho.device
    ecut = float(system.ecut)
    vol = grid.volume
    if ecut_large is None:
        ecut_large = factor * ecut
    if gmax_from_ecut(ecut_large) > 2.0 * gmax_from_ecut(ecut) * (1.0 + 1e-9):
        raise ValueError(
            f"ecut_large={ecut_large:.1f} eV exceeds 4*ecut={4 * ecut:.1f} eV; the "
            "enlarged sphere would not fit the density FFT box. Lower ecut_large.")

    # frozen spinor potential rebuilt from the converged (rho, m), the same
    # v_r = v_H + v_xc + v_loc and exchange field b_xc the SCF forms each step.
    rho_g = r_to_g(res.rho.to(CDTYPE))
    v_h = (torch.fft.ifftn(hartree_potential_g(rho_g, grid.g2), dim=(-3, -2, -1))
           * grid.n_points).real
    v_xc, b_xc, _ = vxc_and_bxc(xc, res.rho, res.m, grid, rho_core=system.rho_core)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, vol)
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real
    v_r = v_h + v_xc + vloc_r

    bk0 = system.batch
    m0 = bk0.npw_max
    bk1, spheres1, so_tabs1 = _spinor_enlarged_batch(system, ecut_large, device)
    m1 = bk1.npw_max
    projs_b1 = projectors_b(bk1, system.positions)
    q_so1 = dij_so1 = None
    if bool(getattr(system, "is_fr", False)):
        from gradwave.core.spinor_proj import build_so_projectors
        q_so1, dij_so1 = build_so_projectors(bk1, system, so_tables=so_tabs1)
    h1 = SpinorHamiltonian(bk1, grid.shape, v_r, b_xc, projs_b1,
                           q=q_so1, dij_so=dij_so1)

    # recompute occupations (NCResult stores none): degeneracy 1.0 -- each
    # spinor band holds one electron.
    scheme_obj = SCHEMES[smearing]
    eps = res.eigenvalues.to(device)          # (nk, nb)
    mu = find_fermi(eps, system.kweights, scheme_obj, width,
                    system.n_electrons, degeneracy=1.0)
    occ, _ = occupations_and_entropy(eps, mu.to(device), scheme_obj, width,
                                     degeneracy=1.0)   # (nk, nb) in [0,1]

    # zero-pad the spinor coefficients from the run sphere onto the enlarged one,
    # per spin block (both blocks share their k's sphere ordering).
    coeffs = res.coeffs.to(device)            # (nk, nb, 2*m0)
    nk, nb, _ = coeffs.shape
    c1 = torch.zeros(nk, nb, 2 * m1, dtype=CDTYPE, device=device)
    for ik, (sph0, sph1) in enumerate(zip(system.spheres, spheres1)):
        n0 = sph0.npw
        for blk, off in ((0, 0), (1, m1)):
            src = coeffs[ik, :, blk * m0: blk * m0 + n0]      # (nb, n0)
            box = sphere_to_box(src, sph0.flat_idx.to(device), grid.shape)
            c1[ik, :, off: off + sph1.npw] = box_to_sphere(box, sph1.flat_idx.to(device))

    # complement residual and correction on the doubled (2*m1) axis
    t2 = torch.cat([bk1.t, bk1.t], dim=-1)               # (nk, 2*m1)
    mask2 = torch.cat([bk1.mask, bk1.mask], dim=-1)      # (nk, 2*m1)
    hc1 = h1.apply(c1)
    resid = hc1 - eps[:, :, None] * c1
    annulus = t2 > ecut * (1.0 + 1e-9)                   # (nk, 2*m1)
    denom = torch.clamp(t2[:, None, :] - eps[:, :, None], min=1e-3)
    dpsi = torch.where(annulus[:, None, :], -resid / denom.to(resid.dtype),
                       torch.zeros_like(resid))
    dpsi = dpsi * mask2[:, None, :]

    return {
        "system": system, "grid": grid, "vol": vol, "ecut": ecut,
        "ecut_large": ecut_large, "bk1": bk1, "m1": m1, "spheres1": spheres1,
        "c1": c1, "dpsi": dpsi, "resid": resid, "eps": eps, "occ": occ,
    }


@torch.no_grad()
def _estimate_density_error_noncollinear(res, *, ecut_large, factor, xc,
                                         smearing, width):
    """Non-collinear (spinor) plane-wave discretization density/energy error.

    The complement correction of the collinear path, generalized to a 2-component
    spinor: the residual and dpsi live on the doubled (up, down) plane-wave axis,
    the density error is the spin-summed  drho = sum_i f_i 2 Re(psi_i^dag dpsi_i)
    (both blocks), and the energy error is the second-order occupation-weighted
    sum of <dpsi_i | R_i>. Occupations use degeneracy 1.0 (one electron per
    spinor band). Norm-conserving spinor pseudopotentials, with or without SOC;
    use_symmetry=False only.
    """
    d = _spinor_complement(res, ecut_large=ecut_large, factor=factor, xc=xc,
                           smearing=smearing, width=width)
    system, grid, vol = d["system"], d["grid"], d["vol"]
    bk1, m1, c1, dpsi = d["bk1"], d["m1"], d["c1"], d["dpsi"]
    eps, occ, resid = d["eps"], d["occ"], d["resid"]
    device = c1.device
    nk, nb, _ = c1.shape

    w_kb = (system.kweights[:, None].to(device) * occ)     # (nk, nb)

    # energy error: second-order occupation-weighted <dpsi|R>, spin summed over
    # the doubled axis (resid/dpsi already carry both blocks).
    de_band = (dpsi.conj() * resid).real.sum(dim=-1)       # (nk, nb)
    denergy = float((w_kb * de_band).sum())

    # density error: 2 Re(psi^dag dpsi) summed over both spin blocks, band
    # chunked to bound the dense-grid temporaries (mirrors the SCF density).
    drho = torch.zeros(grid.shape, dtype=RDTYPE, device=device)
    chunk = 64 if device.type == "cuda" else nb
    for lo in range(0, nb, chunk):
        hi = min(lo + chunk, nb)
        cu, cd = c1[:, lo:hi, :m1], c1[:, lo:hi, m1:]
        du, dd = dpsi[:, lo:hi, :m1], dpsi[:, lo:hi, m1:]
        psi_u = g_to_r_b(cu, bk1, grid.shape)
        psi_d = g_to_r_b(cd, bk1, grid.shape)
        dps_u = g_to_r_b(du, bk1, grid.shape)
        dps_d = g_to_r_b(dd, bk1, grid.shape)
        cross = (psi_u.conj() * dps_u + psi_d.conj() * dps_d).real
        w = (2.0 * w_kb[:, lo:hi]).to(cross.dtype)
        drho += torch.einsum("kb,kbxyz->xyz", w, cross)
    drho = drho / vol

    return DiscretizationError(
        drho=drho, drho_first_order=drho, denergy=denergy, dpsi=d["dpsi"],
        psi_large=c1, occ=occ, spheres_large=d["spheres1"], ecut=d["ecut"],
        ecut_large=d["ecut_large"], dyson=False,
    )


def _estimate_eigenvalue_error_noncollinear(res, *, ecut_large, factor, xc,
                                            bands, smearing, width):
    """Non-collinear eigenvalue error: per-band second-order spinor shift.

    Same complement correction as the density path, reporting the per-band
    deps = <dpsi|R> <= 0 for every spinor band. The occupation-weighted sum over
    the occupied bands reproduces the density-path ``denergy`` exactly. Returns a
    per-k list of (nb,) tensors (nspin field set to 1 -- a spinor run is a single
    combined channel, not two collinear ones). ``bands`` selects a subset.
    """
    d = _spinor_complement(res, ecut_large=ecut_large, factor=factor, xc=xc,
                           smearing=smearing, width=width)
    dpsi, resid, eps = d["dpsi"], d["resid"], d["eps"]
    de_all = (dpsi.conj() * resid).real.sum(dim=-1)        # (nk, nb)
    sel = slice(None) if bands is None else bands
    deig = [de_all[ik][sel].clone() for ik in range(de_all.shape[0])]
    eig = [eps[ik][sel].clone() for ik in range(eps.shape[0])]
    return EigenvalueError(deig=deig, eig=eig, ecut=d["ecut"],
                           ecut_large=d["ecut_large"], nspin=1)


# --------------------------------------------------------------------------- #
#  Eigenvalue / band-gap error                                                #
# --------------------------------------------------------------------------- #


@dataclass
class EigenvalueError:
    """Estimated Ecut discretization error of the Kohn-Sham eigenvalues.

    ``deig`` is the second-order eigenvalue shift toward the infinite-basis
    limit (<= 0, a definite lowering), same sign convention as the energy
    error: eps_exact ~= eps + deig. Layout mirrors the SCF eigenvalues -- a
    per-k list of (nband_sel,) tensors for nspin=1, nested [spin][k] for
    nspin=2. ``eig`` carries the matching base eigenvalues [eV].
    """

    deig: list
    eig: list
    ecut: float
    ecut_large: float
    nspin: int = 1


def _band_eig_error(h1, sph0, sph1, coeffs, eig, grid_shape, ecut):
    """Per-band δε on one k/spin from the enlarged Hamiltonian ``h1``.

    The same complement correction as the density estimate, run on every band:
    R = P_annulus (H - eps) psi, δψ = -R/(T_G - eps), and the second-order
    eigenvalue shift is δε = <δψ | R> = -sum_annulus |R|^2/(T_G - eps) <= 0.
    This is the per-band term the energy error sums over occupations.
    """
    box = sphere_to_box(coeffs, sph0.flat_idx, grid_shape)
    c1 = box_to_sphere(box, sph1.flat_idx)
    resid = h1.apply(c1) - eig[:, None] * c1
    annulus = h1.t > ecut * (1.0 + 1e-9)
    denom = torch.clamp(h1.t[None, :] - eig[:, None], min=1e-3)
    dpsi = torch.where(annulus[None, :], -resid / denom.to(resid.dtype),
                       torch.zeros_like(resid))
    return (dpsi.conj() * resid).real.sum(dim=1)  # (nband,) <= 0


@torch.no_grad()
def estimate_eigenvalue_error(
    res: SCFResult,
    *,
    ecut_large: float | None = None,
    factor: float = 2.5,
    bands=None,
    xc=None,
    smearing: str | None = None,
    width: float | None = None,
) -> EigenvalueError:
    """Estimate the plane-wave discretization error of the KS eigenvalues.

    Reuses the complement correction of the density/energy estimate. The term
    the energy error already sums over occupations, δε_i = <δψ_i | R_i>, is
    exactly the second-order eigenvalue shift; running it on the empty bands as
    well turns the estimator into a band-structure and band-gap error tool. The
    shift is a definite lowering (δε <= 0), so the occupation-weighted sum of
    the occupied-band shifts reproduces ``estimate_density_error(...).denergy``.

    Norm-conserving, USPP/PAW, or non-collinear/SOC; nspin=1 or 2. USPP/PAW and
    the spinor path need ``xc`` (to rebuild the band operator); the spinor path
    also takes the run's ``smearing``/``width``. ``bands`` selects a subset (an
    index list or slice); None uses every band the SCF computed -- keep the
    default for gap analysis, which needs the frontier bands.
    """
    if isinstance(res, dict):
        return _estimate_eigenvalue_error_uspp(
            res, ecut_large=ecut_large, factor=factor, xc=xc, bands=bands)
    if _is_ncresult(res):
        if xc is None:
            raise ValueError("non-collinear eigenvalue error requires xc (the "
                             "NoncollinearXC used for the run)")
        scheme, w = _spinor_smearing(smearing, width)
        return _estimate_eigenvalue_error_noncollinear(
            res, ecut_large=ecut_large, factor=factor, xc=xc, bands=bands,
            smearing=scheme, width=w)
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

    spins = [None] if nspin == 1 else list(range(nspin))
    deig_s, eig_s = [], []
    for sp in spins:
        v_eff_sp = res.v_eff if sp is None else res.v_eff[sp]
        deig_k, eig_k = [], []
        for ik, sph0 in enumerate(system.spheres):
            coeffs, eig = _bands_at(res, ik, sp)
            sel = slice(None) if bands is None else bands
            c_sel, eps_sel = coeffs[sel], eig[sel]
            sph1, h1 = _enlarged_hamiltonian(
                res, sph0.k_frac, ecut_large, device, v_eff=v_eff_sp)
            deig_k.append(
                _band_eig_error(h1, sph0, sph1, c_sel, eps_sel, grid.shape, ecut))
            eig_k.append(eps_sel.clone())
        deig_s.append(deig_k)
        eig_s.append(eig_k)

    if nspin == 1:
        deig, eig = deig_s[0], eig_s[0]
    else:
        deig, eig = deig_s, eig_s
    return EigenvalueError(deig=deig, eig=eig, ecut=ecut,
                           ecut_large=ecut_large, nspin=nspin)


def estimate_gap_error(res: SCFResult, eigerr: EigenvalueError, *,
                       occ_threshold: float | None = None,
                       occupations=None) -> dict:
    """Band-gap discretization error from a full-band ``EigenvalueError``.

    Locates the valence-band maximum and conduction-band minimum over the BZ
    (and both spin channels), then reports the base gap, the extrapolated gap
    (eps + δε at each edge), and their difference. The eigenvalue error must
    have been computed with the default ``bands=None`` so the band index lines
    up with the occupations. Accepts an ``SCFResult`` or the ``scf_uspp`` result
    dict. Raises ValueError for a metal/semimetal (VBM >= CBM) or when no empty
    band is available to resolve the CBM.
    """
    is_dict = isinstance(res, dict)
    system = res["system"] if is_dict else res.system
    if occupations is not None:
        occs = occupations                       # NCResult stores no occupations
    else:
        occs = res["occupations"] if is_dict else res.occupations
    nspin = int(res.get("nspin", 1)) if is_dict else int(getattr(res, "nspin", 1))
    full = 2.0 if nspin == 1 else 1.0
    thr = 0.5 * full if occ_threshold is None else occ_threshold
    deig_s = [eigerr.deig] if nspin == 1 else eigerr.deig
    spins = [None] if nspin == 1 else list(range(nspin))

    vbm, cbm = -math.inf, math.inf
    dvbm = dcbm = 0.0
    vk = ck = vsp = csp = -1
    for isp, sp in enumerate(spins):
        for ik in range(len(system.spheres)):
            occ = occs[ik] if sp is None else occs[sp][ik]
            _, eig = _bands_at(res, ik, sp)
            de = deig_s[isp][ik]
            n = de.shape[0]
            occ, eig = occ[:n], eig[:n]
            filled = occ > thr
            if bool(filled.any()):
                iv = int(torch.where(filled)[0].max())
                if float(eig[iv]) > vbm:
                    vbm, dvbm, vk, vsp = float(eig[iv]), float(de[iv]), ik, isp
            empty = ~filled
            if bool(empty.any()):
                ic = int(torch.where(empty)[0].min())
                if float(eig[ic]) < cbm:
                    cbm, dcbm, ck, csp = float(eig[ic]), float(de[ic]), ik, isp

    if not (math.isfinite(vbm) and math.isfinite(cbm)):
        raise ValueError(
            "cannot resolve a gap: no empty band available -- increase nbands")
    if vbm >= cbm:
        raise ValueError(
            f"system is metallic/semimetallic (VBM {vbm:.3f} >= CBM {cbm:.3f} "
            "eV); band-gap error is undefined")
    gap, dgap = cbm - vbm, dcbm - dvbm
    return {
        "gap_eV": gap,
        "gap_extrapolated_eV": gap + dgap,
        "dgap_eV": dgap,
        "vbm_eV": vbm, "cbm_eV": cbm,
        "dvbm_eV": dvbm, "dcbm_eV": dcbm,
        "direct": (vk == ck and vsp == csp),
    }


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
