"""Prototype: the whole error-vs-cutoff CURVE from one converged SCF.

The shipped estimator (``postscf.discretization_error.estimate_density_error``)
returns a single scalar ``denergy`` = the second-order energy lowering summed
over the complement annulus ecut < T_G <= ecut_large:

    dE = sum_k w_k sum_i f_i sum_G  Re[ dpsi_i(G)* R_i(G) ],
    dpsi_i(G) = -R_i(G) / (T_G - eps_i)   on the annulus, 0 elsewhere.

Because that is a sum over annulus G-vectors and each summand is TAGGED with its
own kinetic energy T_G (and its denominator depends only on T_G, not on where the
basis was truncated), we can bin the summand by T_G and cumulatively sum from the
top. Summing only the part with T_G > ecut' gives the second-order energy still
sitting above the cutoff ecut', i.e. the predicted remaining basis-set error at
EVERY virtual cutoff ecut' in [ecut, ecut_large], from one calculation.

This module does NOT modify the shipped estimator. It reuses its private helpers
to reproduce the exact same per-G summand, then bins it. The full sum over the
annulus reproduces ``estimate_density_error(...).denergy`` to round-off.

Norm-conserving nspin=1 only (the probe scope). Works with use_symmetry on or off
(the energy error is a scalar BZ integral, so the IBZ kweighted sum is correct,
exactly as the shipped estimator relies on).
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.fftbox import g_to_r
from gradwave.dtypes import RDTYPE
from gradwave.postscf.discretization_error import (
    DiscretizationError,
    _complement_correction,
    _enlarged_hamiltonian,
    _occupied,
    _pad_to_sphere,
    _resolve_ecut_large,
)
from gradwave.scf.loop import SCFResult


@torch.no_grad()
def energy_error_curve(
    res: SCFResult,
    ecut_grid: np.ndarray,
    *,
    ecut_large: float | None = None,
    factor: float = 2.5,
) -> np.ndarray:
    """Predicted remaining energy error dE(ecut') at each virtual cutoff.

    Returns an array parallel to ``ecut_grid`` [eV], each entry the second-order
    energy still above that virtual cutoff (<= 0, a lowering). ``ecut_grid`` is
    interpreted in eV and must lie in [res.system.ecut, ecut_large]; entries at
    or below the probe cutoff return the full denergy (the annulus does not reach
    below the probe cutoff, so there is no information there) and entries at/above
    ecut_large return 0.
    """
    system = res.system
    grid = system.grid
    device = res.v_eff.device
    ecut_large = _resolve_ecut_large(system, ecut_large, factor)

    t_all: list[torch.Tensor] = []
    s_all: list[torch.Tensor] = []  # per-G weighted summand, summed over bands
    for _ik, sph0 in enumerate(system.spheres):
        c_occ, eps_occ, occ = _occupied(res, _ik, None)
        sph1, h1 = _enlarged_hamiltonian(res, sph0.k_frac, ecut_large, device)
        c_occ_1 = _pad_to_sphere(c_occ, sph0, sph1, grid.shape)
        resid = h1.apply(c_occ_1) - eps_occ[:, None] * c_occ_1
        dpsi = _complement_correction(resid, h1.t, eps_occ, float(system.ecut))
        summand = (dpsi.conj() * resid).real  # (n_occ, npw1)
        w = float(system.kweights[_ik])
        weighted = summand * (w * occ)[:, None]
        t_all.append(h1.t)
        s_all.append(weighted.sum(dim=0))  # (npw1,)

    T = torch.cat(t_all)
    S = torch.cat(s_all)
    out = np.empty(len(ecut_grid), dtype=float)
    for j, ec in enumerate(ecut_grid):
        mask = T > float(ec) * (1.0 + 1e-9)
        out[j] = float(S[mask].sum())
    return out


@torch.no_grad()
def density_error_at_virtual_cutoff(
    res: SCFResult,
    ecut_virtual_ev: float,
    *,
    ecut_large: float | None = None,
    factor: float = 2.5,
) -> DiscretizationError:
    """Complement density/orbital error using a RAISED annulus floor ecut_virtual.

    Identical to the shipped ``estimate_density_error`` NC nspin=1 path, except
    the annulus lower bound is ``ecut_virtual_ev`` (>= the probe cutoff) instead
    of the probe cutoff. That truncates dpsi and drho to the part of the
    complement above ecut_virtual, so feeding the result into
    ``estimate_force_error`` yields the predicted force error STILL REMAINING at
    the virtual cutoff -- the force analogue of the energy curve. nspin=1,
    use_symmetry=False (the force curve is probed on a displaced, low-symmetry
    cell).
    """
    system = res.system
    grid = system.grid
    device = res.v_eff.device
    ecut_large = _resolve_ecut_large(system, ecut_large, factor)
    if getattr(system, "sym", None) is not None:
        raise NotImplementedError("virtual-cutoff force curve: use_symmetry=False")

    drho = torch.zeros(grid.shape, dtype=RDTYPE, device=device)
    dpsi_k, psi_large_k, occ_k, sph_k = [], [], [], []
    for ik, sph0 in enumerate(system.spheres):
        c_occ, eps_occ, occ = _occupied(res, ik, None)
        sph1, h1 = _enlarged_hamiltonian(res, sph0.k_frac, ecut_large, device)
        c_occ_1 = _pad_to_sphere(c_occ, sph0, sph1, grid.shape)
        resid = h1.apply(c_occ_1) - eps_occ[:, None] * c_occ_1
        dpsi = _complement_correction(resid, h1.t, eps_occ, float(ecut_virtual_ev))
        psi_r = g_to_r(c_occ, sph0.flat_idx, grid.shape)
        dpsi_r = g_to_r(dpsi, sph1.flat_idx, grid.shape)
        w = 2.0 * float(system.kweights[ik])
        drho += w * (occ.view(-1, 1, 1, 1) * (psi_r.conj() * dpsi_r).real).sum(dim=0)
        dpsi_k.append(dpsi)
        psi_large_k.append(c_occ_1)
        occ_k.append(occ)
        sph_k.append(sph1)
    drho = drho / grid.volume

    return DiscretizationError(
        drho=drho, drho_first_order=drho.clone(), denergy=0.0, dpsi=dpsi_k,
        psi_large=psi_large_k, occ=occ_k, spheres_large=sph_k,
        ecut=float(system.ecut), ecut_large=ecut_large, dyson=False,
    )


def recommend_ecut(
    ecut_grid: np.ndarray,
    curve_ev: np.ndarray,
    target_ev: float,
) -> float:
    """Smallest ecut' on the grid whose |predicted error| <= target_ev [eV].

    Returns np.nan if no grid point meets the target (target already met at the
    probe cutoff, or never met within the annulus).
    """
    ok = np.abs(curve_ev) <= target_ev
    if not ok.any():
        return float("nan")
    return float(ecut_grid[np.argmax(ok)])
