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
from gradwave.core.energies.hartree import hartree_potential_r
from gradwave.core.fftbox import g_to_r_box
from gradwave.core.structure import structure_factors

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


def _esm_geom(cell, shape: tuple[int, ...], open_axis: int, device, dtype):
    """In-plane |G∥| [Å⁻¹] (n_a, n_b), z spacing dz, and z extent lz [Å].

    Computed in torch so a differentiable (strained) torch ``cell`` flows to the
    stress; an ``np.ndarray`` cell takes the same path with no grad. n_a/n_b are
    the two periodic-axis sizes, nz the open-axis size (matching the movedim'd
    density layout in :func:`esm_delta_potential`).
    """
    in_axes = tuple(a for a in (0, 1, 2) if a != open_axis)
    if not torch.is_tensor(cell):
        cell = torch.as_tensor(np.asarray(cell, dtype=np.float64), device=device, dtype=dtype)
    b = 2.0 * math.pi * torch.linalg.inv(cell).transpose(-2, -1)  # rows b_i
    lz = torch.linalg.norm(cell[open_axis])
    dz = lz / shape[open_axis]
    b0, b1 = b[in_axes[0]], b[in_axes[1]]
    na, nb = shape[in_axes[0]], shape[in_axes[1]]
    k0 = torch.as_tensor(np.fft.fftfreq(na, d=1.0 / na).astype(np.int64), device=device).to(dtype)
    k1 = torch.as_tensor(np.fft.fftfreq(nb, d=1.0 / nb).astype(np.int64), device=device).to(dtype)
    gvec = k0[:, None, None] * b0[None, None, :] + k1[None, :, None] * b1[None, None, :]
    return torch.linalg.norm(gvec, dim=-1), dz, lz


def hartree_potential_esm(rho_r: torch.Tensor, cell: "np.ndarray | torch.Tensor",
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
    # Orthogonality check on a detached numpy copy — keep the original ``cell``
    # (which may be a grad-requiring strained tensor for the stress) untouched so
    # _esm_geom below stays differentiable in it.
    cell_np = np.asarray(cell.detach().cpu() if torch.is_tensor(cell) else cell,
                         dtype=np.float64)
    in_axes = tuple(a for a in (0, 1, 2) if a != open_axis)
    c = cell_np[open_axis]
    # ESM geometry: open axis orthogonal to both periodic axes.
    for a in in_axes:
        cos = abs(float(np.dot(c, cell_np[a]))) / (
            np.linalg.norm(c) * np.linalg.norm(cell_np[a]))
        if cos > 1e-8:
            raise ValueError(
                f"ESM open_axis={open_axis} must be orthogonal to the periodic "
                f"axes; axis {a} has |cos|={cos:.2e}. ESM needs slab geometry.")

    # Work with the open axis last: (n_a, n_b, nz).
    rho_m = torch.movedim(rho_r, open_axis, -1)
    n_a, n_b, nz = rho_m.shape
    dev, rdt = rho_r.device, rho_r.dtype

    gpar, dz, _lz = _esm_geom(cell, tuple(rho_r.shape), open_axis, dev, rdt)
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


def hartree_potential_capacitor(rho_r: torch.Tensor, cell: "np.ndarray | torch.Tensor",
                                open_axis: int = 2, *, bias: float = 0.0) -> torch.Tensor:
    """Hartree v_H(r) [eV] with METAL (Dirichlet) planes at both z-box edges — the
    ESM capacitor / metal-metal mode.

    The metal planes are grounded equipotentials at the first and last grid slabs
    (v=0 there for every in-plane channel), with an optional applied ``bias`` [V]
    lifting the far plane (a uniform vacuum field bias/L — field-effect / biased
    electrochemistry). Built from the validated open (vacuum) potential plus the
    homogeneous image solution that cancels its boundary values:

        v_cap(G∥, z) = v_open(G∥, z) + v_image(G∥, z),
        v_image(G∥, z) = −[v_open(G∥,0)·sinh(g(L−z)) + v_open(G∥,L)·sinh(g z)]/sinh(gL)

    (the linear analogue for G∥=0), so v_cap satisfies the same Poisson equation
    but with v_cap(0)=v_cap(L)=0. Differentiable in rho_r.
    """
    v_open = hartree_potential_esm(rho_r, cell, open_axis)
    v_m = torch.movedim(v_open, open_axis, -1)
    n_a, n_b, nz = v_m.shape
    dev, rdt = rho_r.device, rho_r.dtype
    gpar, dz, _lz = _esm_geom(cell, tuple(rho_r.shape), open_axis, dev, rdt)
    zc = torch.arange(nz, device=dev, dtype=rdt) * dz
    lm = zc[-1]  # plane separation: first grid point to last

    v_hat = torch.fft.fft2(v_m, dim=(0, 1))
    v0 = v_hat[:, :, 0].unsqueeze(-1)   # (n_a,n_b,1) boundary values per G∥
    vl = v_hat[:, :, -1].unsqueeze(-1)
    g = gpar.unsqueeze(-1)              # (n_a,n_b,1)
    glm = (gpar * lm).unsqueeze(-1)
    gz = g * zc                         # (n_a,n_b,nz)

    # stable sinh(g(L−z))/sinh(gL) and sinh(gz)/sinh(gL): divide by e^{gL} so all
    # exponents are ≤ 0 (no overflow for large g·L).
    denom = 1.0 - torch.exp(-2.0 * glm)
    ra = (torch.exp(-gz) - torch.exp(-(2.0 * glm - gz)))          # · /denom = sinh ratio a
    rb = (torch.exp(-(glm - gz)) - torch.exp(-(glm + gz)))        # · /denom = sinh ratio b
    # v0/vL are complex Fourier coefficients; keep the full complex value (the real
    # ratios ra/rb broadcast against them) — do NOT cast to real.
    v_img = -(v0 * ra + vl * rb) / denom.clamp(min=1e-30)

    # G∥=0 (g=0) mode: solve the Dirichlet BVP DIRECTLY with the parabolic Green's
    # function G_D0(z,z')=z_<(L−z_>)/L. This handles a NET-CHARGED channel (the
    # plates carry the induced counter-charge) rather than neutralizing it — the
    # prerequisite for constant-potential / charged-cell electrochemistry. For a
    # neutral channel it equals the open+linear-image result.
    rho00 = torch.fft.fft2(torch.movedim(rho_r, open_axis, -1), dim=(0, 1))[0, 0, :]
    zlo = torch.minimum(zc[:, None], zc[None, :])
    zhi = torch.maximum(zc[:, None], zc[None, :])
    gd0 = (zlo * (lm - zhi) / lm).to(rho00.dtype)
    v00 = 4.0 * math.pi * E2 * dz * (gd0 @ rho00)
    mask = torch.zeros((n_a, n_b), device=dev, dtype=v_img.dtype)
    mask[0, 0] = 1.0
    combined = (v_hat + v_img) * (1.0 - mask).unsqueeze(-1) + mask.unsqueeze(-1) * v00

    v_cap_m = torch.fft.ifft2(combined, dim=(0, 1)).real
    if bias != 0.0:  # applied ramp: v(0)=0, v(L)=bias (uniform in-plane)
        v_cap_m = v_cap_m + bias * (zc / lm)
    return torch.movedim(v_cap_m, -1, open_axis)


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
    return _gaussian_ion(positions, charges, grid.g_cart, grid.g2, grid.volume, beta)


def _gaussian_ion(positions: torch.Tensor, charges: torch.Tensor,
                  g_cart: torch.Tensor, g2: torch.Tensor, volume, beta: float) -> torch.Tensor:
    """gaussian_ion_density from explicit (possibly strained) G-vectors/volume."""
    s = structure_factors(positions.to(g_cart.dtype), g_cart)  # e^{−iG·R}
    form = torch.exp(-0.5 * g2 * beta * beta).to(s.dtype)
    z = charges.to(s.dtype)
    rho_ion_g = torch.einsum("a,axyz,xyz->xyz", z, s, form) / volume
    return g_to_r_box(rho_ion_g, real=True)


def _default_beta(grid, open_axis: int) -> float:
    n_open = grid.shape[open_axis]
    dz = float(np.linalg.norm(np.asarray(grid.cell)[open_axis])) / n_open
    return 2.0 * dz


def esm_delta_potential(rho_r: torch.Tensor, cell: "np.ndarray | torch.Tensor",
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
    rho_m = torch.movedim(rho_r, open_axis, -1)
    n_a, n_b, nz = rho_m.shape
    dev, rdt = rho_r.device, rho_r.dtype
    gpar, dz, lz = _esm_geom(cell, tuple(rho_r.shape), open_axis, dev, rdt)
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
    u_wrap = u - lz * torch.round(u / lz)
    k0 = (-0.5 * u.abs()) - (-0.5 * u_wrap.abs() + u_wrap * u_wrap / (2.0 * lz))
    dv00 = 4.0 * math.pi * E2 * dz * (k0.to(rho_hat.dtype) @ s)
    mask = torch.zeros((n_a, n_b), device=dev, dtype=dv_hat.dtype)
    mask[0, 0] = 1.0
    dv_hat = dv_hat * (1.0 - mask).unsqueeze(-1) + mask.unsqueeze(-1) * dv00

    dv_r = torch.fft.ifft2(dv_hat, dim=(0, 1)).real
    return torch.movedim(dv_r, -1, open_axis)


def esm_mode_of(boundary: str) -> str | None:
    """ESM electrostatics mode for an SCF ``boundary`` string, or None for a
    periodic (non-ESM) run. ``open_z`` → vacuum, ``open_z_metal`` → capacitor."""
    return {"open_z": "vacuum", "open_z_metal": "capacitor"}.get(boundary)


def _bias_ramp(grid, open_axis: int, bias: float, ref: torch.Tensor) -> torch.Tensor:
    """Applied capacitor bias as a uniform-in-plane ramp v(z)=bias·z/L on the box
    (v=0 at the near plane, v=bias at the far plane)."""
    nz = grid.shape[open_axis]
    ramp = bias * torch.arange(nz, device=ref.device, dtype=ref.dtype) / (nz - 1)
    shp = [1, 1, 1]
    shp[open_axis] = nz
    return ramp.reshape(shp).expand(grid.shape)


def _matched_cap_dv(rho: torch.Tensor, cell, open_axis: int) -> torch.Tensor:
    """β-safe matched capacitor correction ΔV = ΔV_vacuum + image (v_cap − v_open)."""
    return (esm_delta_potential(rho, cell, open_axis)
            + hartree_potential_capacitor(rho, cell, open_axis, bias=0.0)
            - hartree_potential_esm(rho, cell, open_axis))


def _smooth_charge_carrier(q, cell, volume, shape, open_axis: int,
                           ref: torch.Tensor) -> torch.Tensor:
    """The net charge q spread as an in-plane-uniform broad Gaussian in z (∫=q).
    Smooth, so its spectral-referenced electrostatics stays grid-safe.

    Takes explicit geometry (``cell``, ``volume``, ``shape``) rather than a grid
    so the SAME carrier serves the value/force path and the strained stress path.
    The Gaussian *shape* is strain-independent (the dz factor cancels between the
    z-coordinates and the width), so only ``q`` and ``volume`` carry the strain
    dependence — both flow through here when ``volume``/``q`` are strained tensors.
    """
    shape = tuple(int(s) for s in shape)
    nz = shape[open_axis]
    cell_np = np.asarray(cell.detach().cpu() if torch.is_tensor(cell) else cell,
                         dtype=np.float64)
    dz = float(np.linalg.norm(cell_np[open_axis])) / nz
    zc = torch.arange(nz, device=ref.device, dtype=ref.dtype) * dz
    lz = (nz - 1) * dz
    w = lz / 6.0
    g1 = torch.exp(-((zc - 0.5 * lz) ** 2) / (2.0 * w * w))
    shp = [1, 1, 1]
    shp[open_axis] = nz
    rho_q = g1.reshape(shp).expand(shape)
    dvol = volume / rho_q.numel()
    return rho_q * (q / (rho_q.sum() * dvol))


def _capacitor_grounded_de(rho_tot: torch.Tensor, cell, g2, volume, shape,
                           open_axis: int) -> torch.Tensor:
    """Grounded capacitor charge-induced energy ½∫ρ_tot ΔV, with the NET charge's
    reference calibrated to the spectral periodic (matching the KS energy). Split
    ρ_tot = ρ_n (neutral) + ρ_q (smooth net charge): the neutral part uses the
    β-safe matched correction; the net-charge terms use the spectral reference (v_cap
    − v_periodic) directly, valid for a charged cell and grid-safe since ρ_q is
    smooth. Reduces to the plain matched energy for a neutral cell (ρ_q → 0).

    Takes explicit (possibly strained) geometry — ``cell``, the full-box ``g2``,
    the ``volume`` and box ``shape`` — so this ONE function is the capacitor
    grounded energy for BOTH the value/force path (:func:`esm_energy`) and the
    stress path (:func:`esm_energy_strained`). Because it is a single functional,
    its strain derivative is exactly the stress of the energy it reports, for a
    net-charged cell as well as a neutral one (the derivative-identity fix)."""
    dvol = volume / rho_tot.numel()
    q = rho_tot.sum() * dvol
    rho_q = _smooth_charge_carrier(q, cell, volume, shape, open_axis, rho_tot)
    rho_n = rho_tot - rho_q
    s_q = hartree_potential_capacitor(rho_q, cell, open_axis) - hartree_potential_r(rho_q, g2)
    de_n = 0.5 * (rho_n * _matched_cap_dv(rho_n, cell, open_axis)).sum()
    return (de_n + (rho_n * s_q).sum() + 0.5 * (rho_q * s_q).sum()) * dvol


def esm_energy(rho_elec: torch.Tensor, positions: torch.Tensor,
               charges: torch.Tensor, grid, *, beta: float | None = None,
               open_axis: int = 2, mode: str = "vacuum",
               bias: float = 0.0) -> torch.Tensor:
    """ΔE [eV] — the ESM electrostatic energy correction to add to the periodic KS
    total energy for an open-boundary calculation.

    ``mode``: ``vacuum`` (open both sides) or ``capacitor`` (metal Dirichlet planes
    at both z-box edges, with an optional applied ``bias`` [V] — field-effect /
    biased electrochemistry). A pure function, differentiable in ``rho_elec``
    (→ the SCF potential) and in ``positions`` (→ the force). ``beta`` (the Gaussian
    ion width) defaults to ~2 grid spacings; the matched difference makes the
    charge-induced part independent of it in the point-ion limit.
    """
    if beta is None:
        beta = _default_beta(grid, open_axis)
    rho_tot = rho_elec - gaussian_ion_density(positions, charges, grid, beta)
    dvol = grid.volume / rho_elec.numel()
    if mode == "vacuum":
        dv = esm_delta_potential(rho_tot, grid.cell, open_axis)
        return 0.5 * (rho_tot * dv).sum() * dvol
    if mode != "capacitor":
        raise ValueError(f"esm mode must be 'vacuum' or 'capacitor', got {mode!r}")
    # capacitor: grounded charge-induced correction (quadratic, net-charge
    # reference calibrated to the spectral periodic) + applied bias (linear).
    de = _capacitor_grounded_de(rho_tot, grid.cell, grid.g2, grid.volume,
                                grid.shape, open_axis)
    if bias != 0.0:
        de = de + (rho_tot * _bias_ramp(grid, open_axis, bias, rho_elec)).sum() * dvol
    return de


def esm_energy_strained(rho_elec: torch.Tensor, positions: torch.Tensor,
                        charges: torch.Tensor, cell: torch.Tensor,
                        shape: tuple[int, int, int], beta: float,
                        open_axis: int = 2, mode: str = "vacuum",
                        bias: float = 0.0) -> torch.Tensor:
    """ΔE [eV] from a differentiable (strained) ``cell`` — the stress entry point.

    Builds the strained full-box G-vectors and volume from ``cell`` (a torch
    tensor of the strained lattice rows, requires_grad through the strain), so
    autograd of the returned energy w.r.t. the strain gives the ESM stress
    contribution. Supports ``mode="vacuum"|"capacitor"`` (+ ``bias``); the
    capacitor sub-terms flow through the same differentiable cell. ``beta`` is a
    fixed physical width (strain-independent).
    """
    b = 2.0 * math.pi * torch.linalg.inv(cell).transpose(-2, -1)  # rows b_i
    dev = cell.device

    def _miller(n):
        k = np.fft.fftfreq(n, d=1.0 / n).astype(np.int64)
        return torch.as_tensor(k, device=dev).to(cell.dtype)

    axes = [_miller(n) for n in shape]
    miller = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)  # (n1,n2,n3,3)
    g_cart = miller @ b                                                  # strained Cartesian G
    g2 = (g_cart * g_cart).sum(-1)
    volume = torch.abs(torch.linalg.det(cell))

    rho_ion = _gaussian_ion(positions, charges, g_cart, g2, volume, beta)
    rho_tot = rho_elec - rho_ion
    dvol = volume / rho_elec.numel()
    if mode == "vacuum":
        dv = esm_delta_potential(rho_tot, cell, open_axis)
        return 0.5 * (rho_tot * dv).sum() * dvol
    if mode != "capacitor":
        raise ValueError(f"esm mode must be 'vacuum' or 'capacitor', got {mode!r}")
    # Same grounded functional as the value/force path (single source of the
    # capacitor energy), through the strained geometry — so this energy's strain
    # derivative is the stress even for a net-charged cell (was the plain matched
    # functional on full rho_tot, which only equals this at q=0).
    de = _capacitor_grounded_de(rho_tot, cell, g2, volume, shape, open_axis)
    if bias != 0.0:
        nz = shape[open_axis]
        ramp = bias * torch.arange(nz, device=dev, dtype=rho_elec.dtype) / (nz - 1)
        shp = [1, 1, 1]
        shp[open_axis] = nz
        de = de + (rho_tot * ramp.reshape(shp)).sum() * dvol
    return de


def esm_potential(rho_elec: torch.Tensor, positions: torch.Tensor,
                  charges: torch.Tensor, grid, *, beta: float | None = None,
                  open_axis: int = 2, mode: str = "vacuum",
                  bias: float = 0.0) -> torch.Tensor:
    """ΔV(r) [eV] = δΔE/δρ_elec — the consistent open-boundary SCF potential.

    Add to the electron effective potential each SCF step. Computed as the exact
    autograd gradient of :func:`esm_energy` (rho detached; one cheap backward
    through a few FFTs), so it is variationally consistent with the energy — for
    both ``mode="vacuum"`` and ``mode="capacitor"`` (incl. the applied ``bias``).
    """
    # Force grad on — the SCF applies this under torch.no_grad(); the outer SCF
    # gradient goes through the IFT adjoint, not this local backward.
    with torch.enable_grad():
        r = rho_elec.detach().requires_grad_(True)
        de = esm_energy(r, positions.detach(), charges, grid, beta=beta,
                        open_axis=open_axis, mode=mode, bias=bias)
        (grad,) = torch.autograd.grad(de, r)
    dvol = grid.volume / rho_elec.numel()
    return grad / dvol
