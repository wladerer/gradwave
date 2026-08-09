"""Scanning-tunneling-microscopy (STM) images in the Tersoff-Hamann approximation,
resolved into charge and spin.

Tersoff & Hamann (PRB 31, 805 (1985)): for an s-wave tip the tunneling current is
proportional to the sample local density of states at the tip position. Resolving
the KS states into charge and spin gives the local spectral density as a scalar and
a 3-vector,

    ρ(r; E)   = Σ_{n,k} w_k · g_σ(E_{nk} − E) · ψ†_{nk} ψ_{nk},
    m_α(r; E) = Σ_{n,k} w_k · g_σ(E_{nk} − E) · ψ†_{nk} σ_α ψ_{nk},

windowed by a Gaussian g_σ about E (E = E_F at low bias). A magnetized tip with
polarization P measures I ∝ ρ + P·m (spin-polarized STM); an unpolarized tip
measures ρ. `spin_ldos_grid` returns (ρ, m); `ldos_grid` returns a single scalar
channel; `stm_constant_height` / `stm_constant_current` form the tip planes.

The three formalisms are special cases of the same (ρ, m):
  * nspin=1 — ρ = Σ|ψ|², m = 0.
  * nspin=2 collinear — ρ = ρ↑+ρ↓, m = (0, 0, ρ↑−ρ↓).
  * non-collinear / SOC — full ρ and m from the two-component spinors.

A symmetry-reduced SCF sums over the irreducible zone, so the map is symmetrized
over the space group with the matching G-space symmetrizer (charge via
`RhoSymmetrizer.apply`, the collinear pair via `apply_pair`, the vector m via
`MagneticSymmetrizer.apply_m`). This recovers the full-BZ result from the reduced
one. `apply_pair` symmetrizes each spin channel over the magnetic group and, where
a sublattice-swap anti-unitary op is present, couples ρ↑ and ρ↓. For a
body-centered antiferromagnet that swap op shares its rotation with a unitary op
and is removed by the per-rotation translation dedup in `find_spacegroup`, so the
fold there rests on the per-channel symmetrization plus the plain k → −k mesh fold
rather than the channel swap. The collinear paths are verified against full-BZ to
~1e-6; the non-collinear path reuses the validated `apply_m` but has not been
benchmarked on an SOC surface.
"""

from __future__ import annotations

import math

import torch

from gradwave.core.fftbox import g_to_r, g_to_r_box, r_to_g
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.scf.loop import SCFResult


def _formalism(result: SCFResult) -> str:
    """'nspin1' | 'collinear' | 'spinor' from the result's storage."""
    if isinstance(result.coeffs, torch.Tensor):          # (nk, nb, 2*npw) spinors
        return "spinor"
    return "collinear" if getattr(result, "nspin", 1) == 2 else "nspin1"


def _coeffs(result: SCFResult, ispin: int, ik: int) -> torch.Tensor:
    return result.coeffs[ispin][ik] if result.nspin == 2 else result.coeffs[ik]  # type: ignore[index]


def _eigs(result: SCFResult, ispin: int, ik: int) -> torch.Tensor:
    return result.eigenvalues[ispin, ik] if result.nspin == 2 else result.eigenvalues[ik]


def _sym_scalar(field: torch.Tensor, symr) -> torch.Tensor:
    """Symmetrize a real scalar grid over the space group (G-space RhoSymmetrizer)."""
    return g_to_r_box(symr.apply(r_to_g(field.to(CDTYPE))), real=True)


def _sym_pair(up: torch.Tensor, dn: torch.Tensor, symr) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrize a collinear (ρ↑, ρ↓) pair; anti-unitary ops swap the channels."""
    up_g, dn_g = symr.apply_pair(r_to_g(up.to(CDTYPE)), r_to_g(dn.to(CDTYPE)))
    return g_to_r_box(up_g, real=True), g_to_r_box(dn_g, real=True)


def _sym_vector(m: torch.Tensor, symr) -> torch.Tensor:
    """Symmetrize the axial vector m (3, n1, n2, n3) via apply_m, if available."""
    if not hasattr(symr, "apply_m"):
        return m
    return g_to_r_box(symr.apply_m(r_to_g(m.to(CDTYPE))), real=True)


def _gauss(eigs: torch.Tensor, e_ref: float, sigma: float, window: float):
    """Mask of states within `window`·σ of e_ref and their Gaussian weights."""
    mask = (eigs - e_ref).abs() < window * sigma
    norm = 1.0 / (sigma * math.sqrt(2 * math.pi))
    wts = norm * torch.exp(-((eigs[mask] - e_ref) ** 2) / (2 * sigma**2))
    return mask, wts


def _channel_ldos(result, e_ref, sigma, window, ispin, shape, kw):
    """Σ_k w_k g_σ |ψ_{k,ispin}|² on the grid (one collinear/charge channel)."""
    out = torch.zeros(shape, dtype=RDTYPE, device=kw.device)
    for ik, sphere in enumerate(result.system.spheres):
        mask, wts = _gauss(_eigs(result, ispin, ik), e_ref, sigma, window)
        if not bool(mask.any()):
            continue
        psi_r = g_to_r(_coeffs(result, ispin, ik)[mask], sphere.flat_idx, shape)
        out = out + float(kw[ik]) * (wts[:, None, None, None]
                                     * (psi_r.real**2 + psi_r.imag**2)).sum(0)
    return out


def _spinor_ldos(result, e_ref, sigma, window, shape, kw):
    """Charge ρ and spin m = ψ†σψ on the grid from two-component spinors."""
    rho = torch.zeros(shape, dtype=RDTYPE, device=kw.device)
    m = torch.zeros((3, *shape), dtype=RDTYPE, device=kw.device)
    npw_max = result.coeffs.shape[-1] // 2
    for ik, sphere in enumerate(result.system.spheres):
        npw = sphere.npw
        mask, wts = _gauss(result.eigenvalues[ik], e_ref, sigma, window)
        if not bool(mask.any()):
            continue
        c = result.coeffs[ik][mask]                                   # (nb, 2*npw_max)
        up = g_to_r(c[:, :npw], sphere.flat_idx, shape)               # ψ↑(r)
        dn = g_to_r(c[:, npw_max:npw_max + npw], sphere.flat_idx, shape)  # ψ↓(r)
        w = wts[:, None, None, None]
        f = float(kw[ik])
        nu, nd = up.real**2 + up.imag**2, dn.real**2 + dn.imag**2
        ud = up.conj() * dn
        rho = rho + f * (w * (nu + nd)).sum(0)
        m[0] = m[0] + f * (w * 2 * ud.real).sum(0)
        m[1] = m[1] + f * (w * 2 * ud.imag).sum(0)
        m[2] = m[2] + f * (w * (nu - nd)).sum(0)
    return rho, m


def spin_ldos_grid(
    result: SCFResult,
    energy: float | None = None,
    sigma: float = 0.1,
    window: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Charge and spin local density of states on the real-space grid at `energy`
    (default the Fermi level), symmetrized over the space group.

    Returns:
        rho: (n1, n2, n3) charge LDOS.
        m:   (3, n1, n2, n3) spin LDOS (m_x, m_y, m_z); zero for a non-magnetic run.
    """
    system = result.system
    shape = system.grid.shape
    kw = system.kweights
    e_ref = float(result.fermi if energy is None else energy)
    symr = getattr(system, "rho_symmetrizer", None)
    form = _formalism(result)

    if form == "spinor":
        rho, m = _spinor_ldos(result, e_ref, sigma, window, shape, kw)
        if symr is not None:
            rho = _sym_scalar(rho, symr)
            m = _sym_vector(m, symr)
        return rho, m

    m = torch.zeros((3, *shape), dtype=RDTYPE, device=kw.device)
    if form == "nspin1":
        rho = _channel_ldos(result, e_ref, sigma, window, 0, shape, kw)
        if symr is not None:
            rho = _sym_scalar(rho, symr)
        return rho, m

    up = _channel_ldos(result, e_ref, sigma, window, 0, shape, kw)
    dn = _channel_ldos(result, e_ref, sigma, window, 1, shape, kw)
    if symr is not None:
        up, dn = (_sym_pair(up, dn, symr) if hasattr(symr, "apply_pair")
                  else (_sym_scalar(up, symr), _sym_scalar(dn, symr)))
    m[2] = up - dn
    return up + dn, m


def ldos_grid(
    result: SCFResult,
    energy: float | None = None,
    sigma: float = 0.1,
    window: float = 8.0,
    spin: int | None = None,
) -> torch.Tensor:
    """A single scalar LDOS channel [states/eV per grid cell]. `spin=None` returns
    the charge ρ; `spin=0`/`spin=1` return the collinear channels ρ↑=(ρ+m_z)/2 /
    ρ↓=(ρ−m_z)/2."""
    rho, m = spin_ldos_grid(result, energy, sigma, window)
    if spin is None:
        return rho
    return 0.5 * (rho + (m[2] if spin == 0 else -m[2]))


def _tip_field(result, energy, sigma, spin, tip_polarization):
    """The imaged scalar: I ∝ ρ + P·m for a magnetized tip, else the ρ/channel."""
    if tip_polarization is None:
        return ldos_grid(result, energy, sigma, spin=spin)
    rho, m = spin_ldos_grid(result, energy, sigma)
    p = torch.as_tensor(tip_polarization, dtype=RDTYPE, device=rho.device)
    return rho + torch.einsum("a,axyz->xyz", p, m)


def _surface_top_z(result: SCFResult, axis: int = 2) -> float:
    """Highest atomic coordinate along `axis` [Å] — the surface the tip scans over."""
    return float(result.system.positions[:, axis].max())


def stm_constant_height(
    result: SCFResult,
    height: float,
    energy: float | None = None,
    sigma: float = 0.1,
    axis: int = 2,
    surface_z: float | None = None,
    spin: int | None = None,
    tip_polarization: tuple[float, float, float] | None = None,
) -> tuple[torch.Tensor, float]:
    """Constant-height STM map on the plane `height` [Å] above the topmost atom (or
    `surface_z`), normal along `axis` (default z=2). Images the charge ρ, a collinear
    spin channel (`spin=`), or the spin-polarized current ρ + P·m (`tip_polarization`
    a 3-vector tip magnetization). Returns (image, z_tip [Å])."""
    field = _tip_field(result, energy, sigma, spin, tip_polarization)
    cell = result.system.grid.cell
    length = float(cell[axis, axis])
    n = field.shape[axis]
    z_surf = _surface_top_z(result, axis) if surface_z is None else surface_z
    z_tip = z_surf + height
    iz = int(round((z_tip % length) / length * n)) % n
    return field.index_select(axis, torch.tensor([iz], device=field.device)).squeeze(axis), z_tip


def stm_constant_current(
    result: SCFResult,
    target: float,
    z_lo: float,
    z_hi: float,
    energy: float | None = None,
    sigma: float = 0.1,
    axis: int = 2,
    surface_z: float | None = None,
    spin: int | None = None,
    tip_polarization: tuple[float, float, float] | None = None,
    n_planes: int = 48,
) -> torch.Tensor:
    """Constant-current STM topograph: the tip height z(x, y) [Å above the surface]
    where the imaged signal (see `stm_constant_height`) equals `target`, over a stack
    of `n_planes` heights from `z_lo` to `z_hi`. Returns the 2D height map [Å]."""
    field = _tip_field(result, energy, sigma, spin, tip_polarization)
    cell = result.system.grid.cell
    length = float(cell[axis, axis])
    n = field.shape[axis]
    z_surf = _surface_top_z(result, axis) if surface_z is None else surface_z
    heights = torch.linspace(z_lo, z_hi, n_planes)

    def plane(h: float) -> torch.Tensor:
        iz = int(round(((z_surf + h) % length) / length * n)) % n
        return field.index_select(axis, torch.tensor([iz], device=field.device)).squeeze(axis)

    stack = torch.stack([plane(float(h)) for h in heights])
    dh = (z_hi - z_lo) / (n_planes - 1)
    count = (stack >= target).sum(0)
    return z_lo + (count.clamp(min=1) - 1).to(RDTYPE) * dh
