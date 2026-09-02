"""Torch-differentiable EFG evaluation w.r.t. atomic position (FLAPW-DFPT, Phase 1).

A ``torch`` shadow of the numpy EFG evaluation in :mod:`gradwave.flapw.efg`. At a
FIXED (detached) converged density it recomputes the nuclear ``V_zz`` as a
differentiable function of the atom's Cartesian position ``tau``. This delivers the
EXPLICIT frozen-density partial ``dV_zz/dtau|_rho`` (autograd == finite difference at
frozen rho/psi) — the first, self-contained half of the differentiable-EFG chain

    dV_zz/du = dV_zz/dtau|_rho (this module, explicit)  +  (dV_zz/drho)(drho/du)  (Phase 2, chi_0).

The whole point of the AutoAPW thesis is the *moving muffin-tin* term. When the atom
moves, its muffin-tin sphere moves with it, so the two-region partition (interstitial
Theta_I + sphere) moves too. On a fixed FFT grid that boundary motion is INVISIBLE to
naive real-space autograd (the region membership is a Heaviside the grid never
resolves — see the ``grad_mask_autograd == 0`` control in
``experiments/autoapw/surface_term_toy.py``), but it is recovered EXACTLY when the
boundary is expressed through its analytic Fourier transform: the atom enters the
interstitial surface projection only through the smooth phase ``exp(iG·tau)``. That
phase — the gate-A Theta(G) surface/Pulay term — is the ONLY tau-dependence at frozen
density here (the on-site valence multipole term ``_valence_v`` and the own-sphere
``q^MT`` subtraction have no explicit position dependence), so it is exactly what this
module makes autograd-clean.

The numpy path in :mod:`gradwave.flapw.efg` stays the reference; this torch path is
asserted equal to it in value (``tests/unit/test_flapw_efg_torch.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.flapw.coulomb import cell_matrix
from gradwave.flapw.efg import _angular_grid, _surface_phases

# Cartesian-tensor prefactors of the complex-harmonic r^2 potential coefficients
# (identical to gradwave.flapw.efg._tensor_from_v).
_C0 = math.sqrt(5.0 / math.pi)
_C2 = math.sqrt(15.0 / (2.0 * math.pi))
_VAL_PREF = 4.0 * math.pi * E2 / 5.0            # the l=2 valence Poisson prefactor


def _as_c128(x, device) -> torch.Tensor:
    return torch.tensor(np.array(x), dtype=torch.complex128, device=device)


def _as_f64(x, device) -> torch.Tensor:
    return torch.tensor(np.array(x, dtype=float), dtype=torch.float64, device=device)


def _tensor_from_v_torch(v0, v1, v2):
    """Cartesian EFG tensor + ``(V_zz, eta)`` from the l=2 r^2 coefficients (torch, autograd).

    Mirrors :func:`gradwave.flapw.efg._tensor_from_v` exactly: ``v0=v_{m=0}`` (real part used),
    ``v1=v_{m=1}``, ``v2=v_{m=2}`` (complex). ``V_zz`` is the eigenvalue of largest magnitude,
    ``eta = |V_xx - V_yy| / |V_zz|`` with ``|V_xx| <= |V_yy| <= |V_zz|``."""
    v0r = v0.real
    vxx = -0.5 * _C0 * v0r + _C2 * v2.real
    vyy = -0.5 * _C0 * v0r - _C2 * v2.real
    vzz = _C0 * v0r
    vxy = -_C2 * v2.imag
    vxz = -_C2 * v1.real
    vyz = _C2 * v1.imag
    row0 = torch.stack([vxx, vxy, vxz])
    row1 = torch.stack([vxy, vyy, vyz])
    row2 = torch.stack([vxz, vyz, vzz])
    tensor = torch.stack([row0, row1, row2])
    w = torch.linalg.eigvalsh(tensor)
    order = torch.argsort(w.abs())                     # |V_xx| <= |V_yy| <= |V_zz|
    v_zz = w[order[2]]
    v_yy = w[order[1]]
    v_xx = w[order[0]]
    eta = (v_xx - v_yy).abs() / v_zz.abs().clamp_min(1e-30)
    return tensor, v_zz, eta


def _valence_v_torch(multipoles, rr, drw):
    """The frozen-density valence r^2 coefficients ``v_m = (4 pi E2/5) sum(rho_2m / r) dr`` (torch).

    ``multipoles[(2, m)]`` are the (detached) l=2 sphere-density multipoles; ``rr`` and ``drw``
    the radial mesh and its ``dr`` weights. No explicit position dependence — a constant of the
    autograd graph — so this reproduces the dominant on-site term while carrying no ``tau`` grad."""
    return {m: _VAL_PREF * (multipoles[(2, m)] / rr * drw).sum() for m in range(-2, 3)}


@dataclass
class BoundaryContext:
    """Geometry- and density-frozen constants of the interstitial surface projection.

    Everything here is detached (independent of ``tau``); the atom position enters only through
    the ``exp(iG·tau)`` phase applied in :func:`interstitial_boundary_torch`. Build once per
    (frozen ``v_grid``, cell, radius) with :func:`build_boundary_ctx`."""

    vg: torch.Tensor           # (nG,)   complex   V_int(G) = fftn(v_grid)/nfft^3, flattened
    e0: torch.Tensor           # (npts, nG) complex  center-free surface phases exp(iR G·Omega)
    gvec: torch.Tensor         # (nG, 3) real     reciprocal vectors (matches e0 column order)
    wgt: torch.Tensor          # (nx, nphi) real   angular quadrature weights
    ylm_conj: dict[tuple[int, int], torch.Tensor]   # conj(Y_LM) on the (nx, nphi) angular grid
    shape: tuple[int, int]     # (nx, nphi)


def build_boundary_ctx(v_grid, R, cell, lset, nx: int = 16, nphi: int = 24,
                       device=None) -> BoundaryContext:
    """Precompute the detached constants of :func:`interstitial_boundary_torch`.

    Mirrors the setup inside :func:`gradwave.flapw.efg.interstitial_boundary_multi` (same FFT
    normalization, the same memoized ``_surface_phases`` geometry table, the same angular grid),
    so :func:`interstitial_boundary_torch` reproduces it to round-off. ``v_grid`` is the FROZEN
    interstitial Coulomb-only grid; ``R`` the muffin-tin radius; ``cell`` the lattice."""
    from scipy.special import sph_harm_y

    device = torch.device("cpu") if device is None else device
    a = cell_matrix(cell)
    nfft = v_grid.shape[0]
    vg = (np.fft.fftn(v_grid) / nfft**3).reshape(-1)
    e0, gvec = _surface_phases(a, nfft, float(R), nx, nphi)
    th, ph, wgt = _angular_grid(nx, nphi)
    ylm_conj = {(lang, m): np.conj(sph_harm_y(lang, m, th, ph)) for (lang, m) in lset}
    return BoundaryContext(
        vg=_as_c128(vg, device),
        e0=_as_c128(e0, device),
        gvec=_as_f64(gvec, device),
        wgt=_as_f64(wgt, device),
        ylm_conj={lm: _as_c128(y, device) for lm, y in ylm_conj.items()},
        shape=th.shape,
    )


def interstitial_boundary_torch(ctx: BoundaryContext, center_cart, lset):
    """The (L,M) surface projections ``v_bc_LM = oint V_int(center + R Omega) Y*_LM dOmega``.

    ``center_cart`` is the atom position (Angstrom) and the ONLY differentiable input: it enters
    solely through the phase ``exp(iG·center)``. This is the moving muffin-tin Theta(G) term — the
    gate-A surface/Pulay primitive wired into the crystal EFG-eval path. Autograd through this phase
    recovers the moving-boundary surface integral exactly (invisible to a naive real-space mask)."""
    cphase = torch.exp(1j * (ctx.gvec @ center_cart.to(torch.float64)))   # (nG,)
    vals = (ctx.e0 @ (cphase * ctx.vg)).reshape(ctx.shape).real           # (nx, nphi)
    return {lm: (vals * ctx.wgt * ctx.ylm_conj[lm]).sum() for lm in lset}


def efg_vzz_torch(multipoles, rr, drw, ctx: BoundaryContext, center_cart, R, qmt=None,
                  include_pulay: bool = True):
    """Full nuclear ``V_zz`` at a FROZEN density, differentiable in the position ``center_cart``.

    A torch mirror of the per-atom body of :func:`gradwave.flapw.scf._efg_from_multipoles`:
    ``V_zz`` = valence on-site l=2 sphere Poisson (frozen ``multipoles``) + the interstitial
    lattice term (surface projection of the frozen Coulomb grid, minus the analytic own-sphere
    ``q^MT`` field). At frozen density the sole ``tau``-dependence is the moving-boundary phase in
    :func:`interstitial_boundary_torch`.

    ``include_pulay=False`` DETACHES ``center_cart`` from that phase (the boundary is held at its
    current location while the atom nominally moves) — i.e. it OMITS the moving muffin-tin Pulay
    term. Then ``V_zz`` is constant in ``tau`` and autograd returns 0, which no longer matches the
    frozen finite difference: the switch is the load-bearing demonstration that the Theta(G) term is
    real and correctly captured. Returns the scalar ``V_zz`` (torch, autograd)."""
    lset2 = [(2, m) for m in range(-2, 3)]
    v_val = _valence_v_torch(multipoles, rr, drw)
    center = center_cart if include_pulay else center_cart.detach()
    v_bc_raw = interstitial_boundary_torch(ctx, center, lset2)
    qmt = qmt or {}
    r2 = float(R) ** 2
    r3 = float(R) ** 3
    v0 = v_val[0] + (v_bc_raw[(2, 0)] - _VAL_PREF * complex(qmt.get((2, 0), 0.0)) / r3) / r2
    v1 = v_val[1] + (v_bc_raw[(2, 1)] - _VAL_PREF * complex(qmt.get((2, 1), 0.0)) / r3) / r2
    v2 = v_val[2] + (v_bc_raw[(2, 2)] - _VAL_PREF * complex(qmt.get((2, 2), 0.0)) / r3) / r2
    _, v_zz, _eta = _tensor_from_v_torch(v0, v1, v2)
    return v_zz


