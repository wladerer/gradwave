"""Open-boundary (ESM) Hartree potential — periodic in-plane, open in z.

Phase 1 of the differentiable open-boundary (DtN) surface engine
(``docs/design/dtn-3d-engine.md``). The Otani–Sugino *Effective Screening
Medium* electrostatics (PRB 73, 115407 (2006)), made differentiable.

In-plane periodicity makes ``G∥`` a good quantum number, so the Poisson equation
decouples per in-plane reciprocal vector:

    (∂²_z − |G∥|²) v_H(G∥, z) = −4π e² ρ(G∥, z).

Solved with the *open* 1D Green's function (v_H → 0 as z → ±∞) instead of the
periodic ``1/(|G∥|² + G_z²)`` — so a slab feels no z-images and asymmetric
surfaces need no dipole correction:

    |G∥| > 0 :  v_H(G∥, z) = (2π e² / |G∥|) ∫ e^{−|G∥||z−z'|} ρ(G∥, z') dz'
    |G∥| = 0 :  v_H(0, z)  = −2π e² ∫ |z − z'| ρ(0, z') dz'      (neutral channel)

The screened ``|G∥|>0`` solve uses an O(Nz) forward/backward recursion for the
exponential kernel (the dense Nz×Nz kernel over every in-plane mode would be
hundreds of GB); the recursion is written without in-place writes so autograd
flows through it — the differentiable-ESM novelty. The ``|G∥|=0`` channel is the
1D prototype's open Poisson (``experiments/dtn_1d``), here as the closed-form 1D
Coulomb kernel |z−z'| (one channel, O(Nz²), cheap).

Constraint: the open axis (``open_axis``, default the c/2 axis) must be
orthogonal to the two periodic axes, so "in-plane" and "z" separate cleanly.
This is the ESM slab geometry; a general triclinic cell is not supported.

The ``|G∥|=Gz=0`` divergence is handled here by neutralizing the ``G∥=0``
channel against a uniform background (``neutralize_g0``), matching the periodic
``v_H(0)≡0`` convention. In the SCF this is superseded by feeding the neutral
*total* charge (electrons + ion background); that bookkeeping lives in
``energies/total.py`` (a later Phase-1 increment).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.grids import reciprocal_cell

# |G∥|² [Å⁻²] at/below this is the G∥=0 channel (open 1D Coulomb, handled apart).
_GPAR2_ZERO_TOL = 1e-12


def _decay_conv(rho: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Σ_j q^{|i−j|} ρ[j] along the last axis, in O(Nz) per mode.

    rho: (..., nz) complex; q = e^{−|G∥|·dz}: (...) real in [0, 1). Two linear
    recurrences (a decaying prefix sum + suffix sum) replace the Nz×Nz kernel
    multiply; built with list/stack (no in-place writes) so autograd flows.
    """
    nz = rho.shape[-1]
    q = q.unsqueeze(-1) if q.ndim == rho.ndim - 1 else q
    fwd = [rho[..., 0]]
    for i in range(1, nz):
        fwd.append(q[..., 0] * fwd[-1] + rho[..., i])
    bwd = [rho[..., nz - 1]]
    for i in range(nz - 2, -1, -1):
        bwd.append(q[..., 0] * bwd[-1] + rho[..., i])
    bwd.reverse()
    return torch.stack([fwd[i] + bwd[i] for i in range(nz)], dim=-1) - rho


def _inplane_gpar(cell: np.ndarray, shape: tuple[int, int],
                  in_axes: tuple[int, ...], device, dtype) -> torch.Tensor:
    """|G∥| [Å⁻¹] on the in-plane FFT grid for the two periodic axes."""
    b = reciprocal_cell(cell)  # rows b_i, b_i·a_j = 2π δ_ij
    b0, b1 = b[in_axes[0]], b[in_axes[1]]
    k0 = np.fft.fftfreq(shape[0], d=1.0 / shape[0]).astype(np.int64)
    k1 = np.fft.fftfreq(shape[1], d=1.0 / shape[1]).astype(np.int64)
    gvec = k0[:, None, None] * b0[None, None, :] + k1[None, :, None] * b1[None, None, :]
    gpar = np.sqrt(np.einsum("...i,...i->...", gvec, gvec))
    return torch.as_tensor(gpar, device=device, dtype=dtype)


def hartree_potential_esm(rho_r: torch.Tensor, cell: np.ndarray,
                          open_axis: int = 2, *,
                          neutralize_g0: bool = True) -> torch.Tensor:
    """Open-boundary Hartree potential v_H(r) [eV] for a slab.

    rho_r: real electron density ρ(r) [e/Å³] on the FFT box (fftbox layout).
    cell:  (3,3) rows a_i [Å]; ``cell[open_axis]`` must be ⊥ the other two rows.
    open_axis: the non-periodic (vacuum) direction; default 2 (the c axis).
    neutralize_g0: subtract the mean sheet charge from the G∥=0 channel (a
        uniform neutralizing background, matching periodic v_H(0)≡0). Leave True
        unless the caller already passes a z-neutral in-plane average.

    Returns v_H(r) real, same shape as rho_r. Differentiable in rho_r.
    """
    cell = np.asarray(cell, dtype=np.float64)
    in_axes = tuple(a for a in (0, 1, 2) if a != open_axis)
    c = cell[open_axis]
    # ESM geometry: open axis orthogonal to both periodic axes.
    for a in in_axes:
        cos = abs(float(np.dot(c, cell[a]))) / (
            np.linalg.norm(c) * np.linalg.norm(cell[a]))
        if cos > 1e-8:
            raise ValueError(
                f"ESM open_axis={open_axis} must be orthogonal to the periodic "
                f"axes; axis {a} has |cos|={cos:.2e}. ESM needs slab geometry.")

    # Work with the open axis last: (n_a, n_b, nz).
    rho_m = torch.movedim(rho_r, open_axis, -1)
    n_a, n_b, nz = rho_m.shape
    dz = float(np.linalg.norm(c)) / nz
    dev, rdt = rho_r.device, rho_r.dtype

    gpar = _inplane_gpar(cell, (n_a, n_b), in_axes, dev, rdt)  # (n_a, n_b)
    rho_hat = torch.fft.fft2(rho_m, dim=(0, 1))               # (n_a, n_b, nz) complex

    # Screened channels |G∥|>0: v = (2π e²/|G∥|) · dz · Σ_j e^{−|G∥|dz|i−j|} ρ[j].
    gpar_safe = torch.clamp(gpar, min=math.sqrt(_GPAR2_ZERO_TOL))
    q = torch.exp(-gpar_safe * dz)
    conv = _decay_conv(rho_hat, q)
    pref = torch.where(gpar * gpar > _GPAR2_ZERO_TOL,
                       2.0 * math.pi * E2 * dz / gpar_safe,
                       torch.zeros_like(gpar))
    v_hat = pref.unsqueeze(-1) * conv

    # G∥=0 channel: open 1D Coulomb v(0,z) = −2π e² ∫|z−z'| ρ(0,z') dz'.
    s = rho_hat[0, 0, :]
    if neutralize_g0:
        s = s - s.sum() / nz  # uniform neutralizing background (Σ s dz = 0)
    zc = torch.arange(nz, device=dev, dtype=rdt) * dz
    zmat = (zc[:, None] - zc[None, :]).abs().to(v_hat.dtype)
    v00 = -2.0 * math.pi * E2 * dz * (zmat @ s)
    # replace the (0,0) mode without an in-place scatter (autograd-safe)
    mask = torch.zeros((n_a, n_b), device=dev, dtype=v_hat.dtype)
    mask[0, 0] = 1.0
    v_hat = v_hat * (1.0 - mask).unsqueeze(-1) + mask.unsqueeze(-1) * v00

    v_r = torch.fft.ifft2(v_hat, dim=(0, 1)).real
    return torch.movedim(v_r, -1, open_axis)
