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
the ~0.75x absolute accuracy of the underlying energy-error estimate.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from gradwave.core.energies.local_pp import local_potential_g
from gradwave.postscf.discretization_error import estimate_density_error
from gradwave.scf.loop import (
    effective_potentials,
    local_potential_r,
    setup_system,
)

EV_A3_TO_KBAR = 1602.176634  # 1 eV/Å³ = 160.2176634 GPa

__all__ = ["estimate_pressure_error"]


def _infer_kmesh(system) -> tuple[int, int, int]:
    """Monkhorst-Pack mesh dimensions from a full (unreduced) k-point set.

    Each axis carries ``N`` distinct fractional values for an ``N``-fold mesh,
    shift or not; count them. Only valid when the run kept the full BZ
    (``use_symmetry=False``), which the estimator requires so the rebuilt
    strained system reproduces the run's k-point ordering.
    """
    kf = np.array([np.asarray(sph.k_frac, dtype=float) for sph in system.spheres])
    return tuple(len(np.unique(np.round(kf[:, i] % 1.0, 6))) for i in range(3))


@torch.no_grad()
def estimate_pressure_error(res, xc, *, ecut_large: float | None = None,
                            factor: float = 2.5, strain: float = 0.01) -> dict:
    """Estimate the hydrostatic (pressure) plane-wave stress error of a run.

    Returns a dict with ``pressure_error_kbar`` and ``pressure_error_eV_A3``:
    the estimated ``P_exact - P_coarse`` with ``P = -(1/3) tr(sigma)``. Add it to
    the reported pressure to approach the large-basis value (a positive value is
    the usual Pulay under-pressure of a too-small basis). Also returns the two
    ``denergy`` samples and the cell volume for transparency.

    Norm-conserving, nspin=1, scalar-relativistic, ``use_symmetry=False`` (the
    frozen strained rebuild must reproduce the run's k-points). ``ecut_large``
    defaults to ``factor*ecut`` and sets the complement annulus, exactly as in
    ``estimate_density_error``. ``strain`` is the finite-difference half-step in
    the linear scale ``s`` (the estimate is flat in it from ~0.005 to ~0.02).
    """
    system = res.system
    if getattr(system, "sym", None) is not None:
        raise NotImplementedError(
            "pressure error requires use_symmetry=False: the frozen strained "
            "rebuild reproduces the run's full k-point set")
    if int(getattr(res, "nspin", 1)) != 1:
        raise NotImplementedError("pressure error is nspin=1 only")
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
    kmesh = _infer_kmesh(system)
    vol0 = float(grid.volume)

    def _denergy_at(s: float):
        # fixed Miller set: ecut/s**2 on the s-scaled cell strains only the metric
        ss = setup_system(s * cell0, s * pos0, system.species_of_atom, system.upfs,
                          ecut=ecut / s ** 2, kmesh=kmesh, fft_shape=grid.shape)
        rho_s = res.rho * (vol0 / float(ss.grid.volume))   # conserve electron count
        vloc_g = local_potential_g(ss.positions, ss.species_index, ss.vloc_tables,
                                   ss.grid.g_cart, ss.grid.volume)
        veff = effective_potentials(ss, xc, [rho_s], local_potential_r(ss, vloc_g))
        res_s = dataclasses.replace(res, system=ss, v_eff=veff[0])
        err = estimate_density_error(res_s, ecut_large=ecl / s ** 2)
        return float(err.denergy), float(ss.grid.volume)

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
        "note": "first-order indicator (under-estimates ~0.5-0.75x, correctly "
                "signed); hydrostatic component only",
    }
