"""Hydrostatic (pressure) component of the plane-wave stress error.

The stress discretization error is dominated by its trace: the incomplete basis
produces a spurious isotropic "Pulay pressure" (on sheared silicon the shear
part of the basis-set stress error is ~1% of the hydrostatic part). This module
estimates that pressure error; the full anisotropic tensor is deferred (see the
NOTE in ``discretization_error``).

The naive recipes fail. The fixed-δP forward pass that works for forces comes
out ANTI-correlated for stress (it omits the strain-response of the orbital
correction), and so does the volume-derivative of the reported energy error at
fixed ``ecut`` -- both land near -0.3x the true value, because differentiating
through a basis whose plane-wave count JUMPS as G-vectors cross ``ecut`` adds a
spurious discrete term.

The fix is the Nielsen-Martin fixed-basis convention: hold the integer Miller
indices and strain only the metric. A homogeneous scale by ``s`` (cell -> s*cell)
maps ``ecut -> ecut/s**2`` at fixed Miller set, so evaluating the (frozen-state)
energy error at ``ecut/s**2`` on the ``s``-scaled cell differentiates the SAME
basis. The pressure error is then the volume-derivative of that energy error,

    P_error = -d(dE_error)/dV = -(1/3) tr(sigma_exact - sigma_coarse),

by a central finite difference in ``s``. This reuses ``estimate_density_error``
on a frozen electronic state (fixed coefficients, density scaled to conserve N,
potential rebuilt from it) at the two scaled cells; no new SCF is taken.

Accuracy. A first-order indicator, not a bound. It is correctly signed
(Pulay pressure) and captures ~0.45-0.75x of the true pressure error over
ecut ~ 10-18 Ry on silicon, the ratio rising toward 1 as the cutoff converges
(a consistent under-estimate -- it does not give false confidence). It inherits
the ~0.75x absolute accuracy of the underlying energy-error estimate. The opt-in
iterative annulus solver (``solver="cg"``) recovers more of it, ~0.6-0.8x on
silicon, by including the potential coupling the diagonal resolvent drops (see
``estimate_pressure_error`` and benchmarks/pulay_accuracy/RESULTS.md).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.dtypes import RDTYPE
from gradwave.grids import FFTGrid, GSphere, reciprocal_cell
from gradwave.postscf.discretization_error import estimate_density_error
from gradwave.scf.loop import (
    SCFResult,
    effective_potentials,
    local_potential_r,
    setup_system,
)

EV_A3_TO_KBAR = 1602.176634  # 1 eV/Å³ = 160.2176634 GPa

__all__ = ["estimate_pressure_error"]


def _sphere_on_cell(sph: GSphere, grid: FFTGrid) -> GSphere:
    """``sph``'s exact Miller set, in its exact order, with ``grid``'s metric.

    The strained rebuild must reproduce the run's k-set and per-k basis
    exactly, and regenerating them does not do that. ``symmetry: false`` still
    applies time-reversal reduction, so the run k-list is not a full MP mesh
    and cannot be recovered from an inferred mesh (a Γ-centered 3×3×3 on
    hexagonal quartz keeps only 2 unique fractions on one axis after TR
    reduction, so the old per-axis-unique-count inference rebuilt a (2,3,3)
    mesh with misaligned spheres — the #217-fix crash). And even at the right
    k, ``build_gsphere``'s cutoff re-selection can tie-break a boundary
    G-shell differently under the scaled metric. Taking the run sphere's
    integer Millers verbatim implements the fixed-Miller-set convention by
    construction; ``flat_idx`` carries over because the rebuild is pinned to
    the run's FFT shape."""
    b = reciprocal_cell(np.asarray(grid.cell, dtype=np.float64))
    mm = sph.miller.detach().cpu().numpy()
    kpg = (mm + sph.k_frac) @ b
    kpg2 = np.einsum("ij,ij->i", kpg, kpg)
    return GSphere(
        k_frac=sph.k_frac,
        k_cart=torch.as_tensor(sph.k_frac @ b, dtype=RDTYPE),
        miller=sph.miller.detach().cpu(),
        kpg=torch.as_tensor(kpg, dtype=RDTYPE),
        kpg2=torch.as_tensor(kpg2, dtype=RDTYPE),
        flat_idx=sph.flat_idx.detach().cpu(),
    )


def _fit_dE_inf(ecls: list[float], des: list[float]) -> float:
    """Extrapolate the frozen-state energy error to the complete-annulus limit.

    Fits ``dE(ecl) = dE_inf + c * ecl**(-p)`` to the (ecut_large, denergy)
    samples and returns ``dE_inf``. ``denergy`` is negative and grows more
    negative as the annulus widens, so the missing tail decays as a power law in
    the annulus cutoff. The exponent ``p`` is chosen by a coarse grid search
    (linear least squares for ``dE_inf`` and ``c`` at each ``p``), which avoids a
    scipy dependency and is stable for the 3 to 4 samples used here. Falls back
    to the widest-annulus sample if the fit does not extend past it (a
    non-monotone or ill-conditioned tail), so the extrapolation never returns a
    value less converged than the samples themselves.
    """
    x = np.asarray(ecls, dtype=np.float64)
    y = np.asarray(des, dtype=np.float64)
    widest = float(y[int(np.argmax(x))])
    best_resid, best_inf = np.inf, widest
    for p in np.linspace(0.5, 4.0, 36):
        a_mat = np.stack([np.ones_like(x), x ** (-p)], axis=1)
        coef, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
        resid = float(np.sum((a_mat @ coef - y) ** 2))
        if resid < best_resid:
            best_resid, best_inf = resid, float(coef[0])
    # dE_inf must be at least as converged (as negative) as the widest sample;
    # otherwise the tail model misbehaved and we keep the widest annulus.
    return best_inf if best_inf <= widest else widest


@torch.no_grad()
def estimate_pressure_error(res: SCFResult, xc: XCFunctional | SpinXC, *,
                            ecut_large: float | None = None, factor: float = 2.5,
                            strain: float = 0.01, extrapolate: bool = False,
                            extrap_factors: tuple[float, ...] = (1.8, 2.2, 2.6, 3.0),
                            solver: str = "diagonal", cg_tol: float = 1e-6,
                            cg_max_iter: int = 20,
                            ) -> dict[str, float | str]:
    """Estimate the hydrostatic (pressure) plane-wave stress error of a run.

    Returns a dict with ``pressure_error_kbar`` and ``pressure_error_eV_A3``:
    the estimated ``P_exact - P_coarse`` with ``P = -(1/3) tr(sigma)``. Add it to
    the reported pressure to approach the large-basis value (a positive value is
    the usual Pulay under-pressure of a too-small basis). Also returns the two
    ``denergy`` samples and the cell volume for transparency.

    Norm-conserving, nspin=1 or 2, scalar-relativistic, ``use_symmetry=False``
    (the frozen strained rebuild must reproduce the run's k-points). For nspin=2
    the per-spin frozen densities rebuild the per-spin v_eff and the energy error
    sums both spin channels. ``ecut_large``
    defaults to ``factor*ecut`` and sets the complement annulus, exactly as in
    ``estimate_density_error``. ``strain`` is the finite-difference half-step in
    the linear scale ``s`` (the estimate is flat in it from ~0.005 to ~0.02).

    ``extrapolate`` replaces the single-annulus energy error with a power-law
    extrapolation to the complete-annulus limit, evaluating the frozen-state
    ``denergy`` at ``extrap_factors`` and fitting the tail (see ``_fit_dE_inf``).
    It is off by default because the annulus tail is nearly volume-independent on
    silicon, so it moves the pressure estimate only marginally (the diagonal
    resolvent, not annulus truncation, is the dominant source of the
    under-estimate; see benchmarks/pulay_accuracy/RESULTS.md). ``extrap_factors``
    must all be at most 4 so each enlarged sphere fits the density FFT box.

    ``solver="cg"`` replaces the diagonal complement resolvent with a
    preconditioned iterative solve of the annulus-projected P(H-eps)P operator
    (``discretization_error._annulus_cg``), which captures the potential coupling
    the diagonal drops and roughly doubles the recovered fraction of the true
    Pulay pressure on silicon (see benchmarks/pulay_accuracy/RESULTS.md). It
    costs extra H-applies, reported as ``n_h_apply`` in the return dict;
    ``cg_tol``/``cg_max_iter`` tune it.
    """
    system = res.system
    if getattr(system, "sym", None) is not None:
        raise NotImplementedError(
            "pressure error requires use_symmetry=False: the symmetrized "
            "density-error path is untested under the strained rebuild")
    nspin = int(getattr(res, "nspin", 1))
    if getattr(system, "is_fr", False):
        raise NotImplementedError(
            "pressure error not implemented for fully-relativistic pseudos")
    if getattr(res, "hub_occ", None) is not None:
        raise NotImplementedError("pressure error with DFT+U not implemented")

    grid = system.grid
    cell0 = np.asarray(grid.cell, dtype=np.float64)
    pos0 = system.positions.detach().cpu().numpy()
    ecut = float(system.ecut)
    ecl = float(ecut_large) if ecut_large is not None else factor * ecut
    vol0 = float(grid.volume)

    def _scaled_result(s: float):
        # fixed Miller set: ecut/s**2 on the s-scaled cell strains only the
        # metric. The Γ-only kmesh is a placeholder — the run's own k-set
        # (spheres + weights) is grafted in below (see _sphere_on_cell), so
        # the mesh passed here only sizes unused per-k tables.
        ss = setup_system(s * cell0, s * pos0, system.species_of_atom, system.upfs,
                          ecut=ecut / s ** 2, kmesh=(1, 1, 1), fft_shape=grid.shape)
        ss = dataclasses.replace(
            ss,
            spheres=[_sphere_on_cell(sph, ss.grid) for sph in system.spheres],
            kweights=system.kweights.detach().cpu(),
        )
        # follow the run's device (per-tensor .to is a no-op when already there)
        ss = ss.to(str(res.rho.device))
        scale = vol0 / float(ss.grid.volume)               # conserve electron count
        vloc_g = local_potential_g(ss.positions, ss.species_index, ss.vloc_tables,
                                   ss.grid.g_cart, ss.grid.volume)
        vloc_r = local_potential_r(ss, vloc_g)
        # per-spin frozen density -> per-spin v_eff (nspin=2 sums both channels'
        # errors in estimate_density_error, exactly as the fixed-basis stress does)
        if nspin == 1:
            rho_list = [res.rho * scale]
        else:
            assert res.rho_spin is not None  # guaranteed by nspin == 2
            rho_list = [rsp * scale for rsp in res.rho_spin]
        veff = effective_potentials(ss, xc, rho_list, vloc_r)
        if nspin == 1:
            res_s = dataclasses.replace(res, system=ss, v_eff=veff[0])
        else:
            res_s = dataclasses.replace(res, system=ss, v_eff=torch.stack(veff),
                                        rho_spin=rho_list)
        return res_s, float(ss.grid.volume)

    n_h_apply = 0

    def _de(res_s, ecut_large_s: float) -> float:
        nonlocal n_h_apply
        err = estimate_density_error(res_s, ecut_large=ecut_large_s,
                                     solver=solver, cg_tol=cg_tol,
                                     cg_max_iter=cg_max_iter)
        n_h_apply += err.n_h_apply
        return float(err.denergy)

    def _denergy_at(s: float) -> tuple[float, float]:
        res_s, vol_s = _scaled_result(s)
        if not extrapolate:
            return _de(res_s, ecl / s ** 2), vol_s
        # sample the frozen-state energy error over a range of annulus cutoffs
        # (at fixed metric) and extrapolate to the complete-annulus limit.
        ecut_s = float(res_s.system.ecut)
        ecls = [f * ecut_s for f in extrap_factors]
        des = [_de(res_s, e) for e in ecls]
        return _fit_dE_inf(ecls, des), vol_s

    d_minus, v_minus = _denergy_at(1.0 - strain)
    d_plus, v_plus = _denergy_at(1.0 + strain)
    dden_dvol = (d_plus - d_minus) / (v_plus - v_minus)   # eV/Å³
    p_err = -dden_dvol                                    # P_exact - P_coarse [eV/Å³]
    return {
        "pressure_error_eV_A3": p_err,
        "pressure_error_kbar": p_err * EV_A3_TO_KBAR,
        "denergy_minus_eV": d_minus,
        "denergy_plus_eV": d_plus,
        "volume_A3": vol0,
        "n_h_apply": n_h_apply,
        "note": "first-order indicator (under-estimates ~0.5-0.75x, correctly "
                "signed); hydrostatic component only",
    }
