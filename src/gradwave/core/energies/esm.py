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
from gradwave.core.fftbox import g_to_r_box
from gradwave.core.structure import structure_factors
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


# --- SCF wiring: the ESM electrostatic correction (Gaussian counter-charge) ---
#
# The physically-correct ESM restructures the whole electrostatics: the electron
# Hartree, the ion-Coulomb tail (baked into vloc), and the Ewald ion-ion sum all
# merge into ONE open Poisson of the neutral total charge ρ_tot = ρ_elec − ρ_ion.
# Rather than split vloc / rewrite Ewald, we add ESM as a correction to the
# periodic KS energy (Otani–Sugino's add-on structure):
#
#     ΔE = E_es^open(ρ_tot) − E_es^periodic(ρ_tot) = ½ ∫ ρ_tot (v^open − v^periodic) dr
#
# so E_total^ESM = E_total^periodic + ΔE. The ions enter as narrow Gaussians
# (ρ_ion); the Gaussian-vs-true-ion error is short-range and IDENTICAL in the
# open and periodic terms, so it cancels in the difference — only the long-range
# open-vs-periodic part (the image/dipole artifact) survives, which is the point.
#
# The SCF potential is δΔE/δρ_elec by AUTOGRAD, NOT the raw field v^open−v^periodic.
# The two agree in the continuum (symmetric Coulomb operators), but the discrete
# open solver goes through a complex-FFT `.real` whose autograd adjoint differs
# from its forward by ~1% (a Wirtinger subtlety, not the periodic Nyquist one).
# Committing to ΔE as the energy functional and taking its exact gradient keeps
# the discrete SCF variational: v_eff is δE/δρ, so SCF stationarity and the
# Hellmann–Feynman force δΔE/δpositions stay mutually consistent (verified by FD).


def gaussian_ion_density(positions: torch.Tensor, charges: torch.Tensor, grid,
                         beta: float) -> torch.Tensor:
    """Neutralizing nuclear number density Σ_a Z_a G_β(r−R_a) [e/Å³] on the box.

    Built in G-space (exact structure-factor phases, no real-space aliasing of a
    sharp Gaussian): ρ_ion(G) = (1/Ω) Σ_a Z_a e^{−iG·R_a} e^{−G²β²/2}. Integrates
    to Σ_a Z_a = N_e, so ρ_elec − ρ_ion is net neutral. β cancels in ΔE (it is the
    short-range width) — keep it a few grid spacings: representable, narrow.
    """
    s = structure_factors(positions.to(grid.g_cart.dtype), grid.g_cart)  # e^{−iG·R}
    form = torch.exp(-0.5 * grid.g2 * beta * beta).to(s.dtype)
    z = charges.to(s.dtype)
    rho_ion_g = torch.einsum("a,axyz,xyz->xyz", z, s, form) / grid.volume
    return g_to_r_box(rho_ion_g, real=True)


def _default_beta(grid, open_axis: int) -> float:
    n_open = grid.shape[open_axis]
    dz = float(np.linalg.norm(np.asarray(grid.cell)[open_axis])) / n_open
    return 2.0 * dz


def esm_delta_potential(rho_r: torch.Tensor, cell: np.ndarray,
                        open_axis: int = 2) -> torch.Tensor:
    """v_H^open(r) − v_H^periodic(r) [eV] — the open-minus-periodic field.

    Both potentials are built with the SAME in-plane FFT and the SAME real-space
    z quadrature, differing ONLY in the z boundary: a *linear* convolution with
    the 1D Green's function (open, no z-images) vs a *circular* one (periodic,
    z-images summed). Because the kernel is sampled identically, the short-range
    (self-)field cancels exactly in the difference — only the long-range image
    part survives, so the correction is smooth and independent of any sharp
    features in ρ (unlike differencing against the spectral ``hartree_potential_r``,
    whose mismatched z-discretization leaves grid-scale residue that a sharp ion
    charge amplifies without bound).
    """
    cell = np.asarray(cell, dtype=np.float64)
    in_axes = tuple(a for a in (0, 1, 2) if a != open_axis)
    rho_m = torch.movedim(rho_r, open_axis, -1)
    n_a, n_b, nz = rho_m.shape
    dz = float(np.linalg.norm(cell[open_axis])) / nz
    dev, rdt = rho_r.device, rho_r.dtype

    gpar = _inplane_gpar(cell, (n_a, n_b), in_axes, dev, rdt)
    rho_hat = torch.fft.fft2(rho_m, dim=(0, 1))
    gpar_safe = torch.clamp(gpar, min=math.sqrt(_GPAR2_ZERO_TOL))
    q = torch.exp(-gpar_safe * dz)

    # screened channels |G∥|>0: linear (open) minus circular (periodic) conv of
    # the SAME kernel q^{|i−j|}, scaled by 2π e² dz / |G∥|.
    conv_open = _decay_conv(rho_hat, q)                     # Σ_j q^{|i−j|} ρ[j]
    m = torch.arange(nz, device=dev, dtype=rdt)
    qm = q.unsqueeze(-1) ** m                               # (n_a,n_b,nz)
    kring = (qm + q.unsqueeze(-1) ** (nz - m)) / (1.0 - q.unsqueeze(-1) ** nz + 0.0)
    conv_per = torch.fft.ifft(
        torch.fft.fft(rho_hat, dim=-1) * torch.fft.fft(kring.to(rho_hat.dtype), dim=-1),
        dim=-1)
    pref = torch.where(gpar * gpar > _GPAR2_ZERO_TOL,
                       2.0 * math.pi * E2 * dz / gpar_safe,
                       torch.zeros_like(gpar))
    dv_hat = pref.unsqueeze(-1) * (conv_open - conv_per)

    # G∥=0 channel: open 1D Coulomb (−|u|/2 kernel) minus the periodic one
    # (background-neutralized, kernel −|u_wrap|/2 + u_wrap²/(2L)). Shared −|u|/2
    # part cancels for a localized neutral ρ; the parabola/wrap is the image term.
    s = rho_hat[0, 0, :]
    zc = torch.arange(nz, device=dev, dtype=rdt) * dz
    u = zc[:, None] - zc[None, :]
    lz = nz * dz
    u_wrap = u - lz * torch.round(u / lz)
    k0 = (-0.5 * u.abs()) - (-0.5 * u_wrap.abs() + u_wrap * u_wrap / (2.0 * lz))
    dv00 = 4.0 * math.pi * E2 * dz * (k0.to(rho_hat.dtype) @ s)
    mask = torch.zeros((n_a, n_b), device=dev, dtype=dv_hat.dtype)
    mask[0, 0] = 1.0
    dv_hat = dv_hat * (1.0 - mask).unsqueeze(-1) + mask.unsqueeze(-1) * dv00

    dv_r = torch.fft.ifft2(dv_hat, dim=(0, 1)).real
    return torch.movedim(dv_r, -1, open_axis)


def esm_energy(rho_elec: torch.Tensor, positions: torch.Tensor,
               charges: torch.Tensor, grid, *, beta: float | None = None,
               open_axis: int = 2) -> torch.Tensor:
    """ΔE [eV] — the open-minus-periodic electrostatic energy correction.

    Add to the periodic KS total energy for an open-boundary (ESM) calculation.
    A pure function, differentiable in ``rho_elec`` (→ the SCF potential) and in
    ``positions`` (→ the ESM contribution to the force). ``beta`` (the Gaussian
    ion width) defaults to ~2 grid spacings; the discretization-matched
    difference makes ΔE independent of it in the point-ion limit.
    """
    if beta is None:
        beta = _default_beta(grid, open_axis)
    rho_tot = rho_elec - gaussian_ion_density(positions, charges, grid, beta)
    dv = esm_delta_potential(rho_tot, grid.cell, open_axis)
    dvol = grid.volume / rho_elec.numel()
    return 0.5 * (rho_tot * dv).sum() * dvol


def esm_potential(rho_elec: torch.Tensor, positions: torch.Tensor,
                  charges: torch.Tensor, grid, *, beta: float | None = None,
                  open_axis: int = 2) -> torch.Tensor:
    """ΔV(r) [eV] = δΔE/δρ_elec — the consistent open-boundary SCF potential.

    Add to the electron effective potential each SCF step. Computed as the exact
    autograd gradient of :func:`esm_energy` (rho detached; one cheap backward
    through a few FFTs), so it is variationally consistent with the energy.
    """
    # Force grad on — the SCF applies this under torch.no_grad(); the outer SCF
    # gradient goes through the IFT adjoint, not this local backward.
    with torch.enable_grad():
        r = rho_elec.detach().requires_grad_(True)
        de = esm_energy(r, positions.detach(), charges, grid, beta=beta, open_axis=open_axis)
        (grad,) = torch.autograd.grad(de, r)
    dvol = grid.volume / rho_elec.numel()
    return grad / dvol
