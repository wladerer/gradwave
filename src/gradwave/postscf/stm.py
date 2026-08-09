"""Scanning-tunneling-microscopy (STM) images in the Tersoff-Hamann approximation.

Tersoff & Hamann (PRB 31, 805 (1985)): for an s-wave tip the tunneling current is
proportional to the sample's LOCAL DENSITY OF STATES at the tip position, at the
Fermi level (low bias) or integrated over the bias window,

    I(x, y, z_tip) ∝ ρ(x, y, z_tip; E_F) = Σ_{n,k} w_k · g_σ(E_{nk} − E) · |ψ_{nk}(r)|²,

with the states energy-windowed by a Gaussian g_σ around E (E = E_F by default).
Evaluated at a tip plane in the vacuum above the surface. Two imaging modes:

  * constant-height — ρ(x, y) on a fixed plane z_tip = z_surface + height.
  * constant-current — the height z(x, y) where ρ reaches a target isovalue (a
    bisection between two tip planes); this is what an experimental STM records.

The LDOS is built from the converged KS orbitals (`SCFResult.coeffs`), so it is a
pure post-SCF analysis — no tip, no transport — and it is differentiable in the tip
energy/height. For the full tunneling current at finite bias or strong coupling use
the NEGF transport module (`postscf.transport`) with a tip lead.

nspin=2 sums both spin channels (spin-resolved maps: pass one spin via `spin=`).
"""

from __future__ import annotations

import math

import torch

from gradwave.core.fftbox import g_to_r
from gradwave.dtypes import RDTYPE
from gradwave.scf.loop import SCFResult


def _coeffs(result: SCFResult, ispin: int, ik: int) -> torch.Tensor:
    return result.coeffs[ispin][ik] if result.nspin == 2 else result.coeffs[ik]  # type: ignore[index]


def _eigs(result: SCFResult, ispin: int, ik: int) -> torch.Tensor:
    return result.eigenvalues[ispin, ik] if result.nspin == 2 else result.eigenvalues[ik]


def ldos_grid(
    result: SCFResult,
    energy: float | None = None,
    sigma: float = 0.1,
    window: float = 8.0,
    spin: int | None = None,
) -> torch.Tensor:
    """Tersoff-Hamann LDOS on the real-space FFT grid [states/eV per grid cell]:

        ρ(r) = Σ_{n,k} w_k · g_σ(E_{nk} − energy) · |ψ_{nk}(r)|²

    Args:
        result: converged SCFResult.
        energy: reference energy [eV]; default the Fermi level.
        sigma: Gaussian energy window width [eV]. Only states within `window`·σ of
            `energy` are transformed to real space (the rest are negligible).
        spin: if set (0 or 1), only that spin channel (nspin=2 only).
    Returns:
        ρ: (n1, n2, n3) real LDOS on the grid.
    """
    system = result.system
    shape = system.grid.shape
    e_ref = float(result.fermi if energy is None else energy)
    kw = system.kweights
    norm = 1.0 / (sigma * math.sqrt(2 * math.pi))
    spins = [spin] if spin is not None else range(result.nspin)
    ldos = torch.zeros(shape, dtype=RDTYPE, device=kw.device)
    for ispin in spins:
        for ik, sphere in enumerate(system.spheres):
            eigs = _eigs(result, ispin, ik)
            mask = (eigs - e_ref).abs() < window * sigma
            if not bool(mask.any()):
                continue
            wts = norm * torch.exp(-((eigs[mask] - e_ref) ** 2) / (2 * sigma**2))
            psi_r = g_to_r(_coeffs(result, ispin, ik)[mask], sphere.flat_idx, shape)
            ldos = ldos + float(kw[ik]) * (
                wts[:, None, None, None] * (psi_r.real**2 + psi_r.imag**2)
            ).sum(0)
    return ldos


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
) -> tuple[torch.Tensor, float]:
    """Constant-height Tersoff-Hamann STM map: LDOS on the plane `height` [Å] above
    the topmost atom (or `surface_z`), normal along `axis` (default z=2).

    Returns (image, z_tip) where image is the 2D LDOS in the plane [the other two
    grid axes] and z_tip is the tip height [Å].
    """
    ldos = ldos_grid(result, energy, sigma, spin=spin)
    cell = result.system.grid.cell
    length = float(cell[axis, axis])
    n = ldos.shape[axis]
    z_surf = _surface_top_z(result, axis) if surface_z is None else surface_z
    z_tip = z_surf + height
    iz = int(round((z_tip % length) / length * n)) % n
    return ldos.index_select(axis, torch.tensor([iz])).squeeze(axis), z_tip


def stm_constant_current(
    result: SCFResult,
    target: float,
    z_lo: float,
    z_hi: float,
    energy: float | None = None,
    sigma: float = 0.1,
    axis: int = 2,
    surface_z: float | None = None,
    n_planes: int = 48,
) -> torch.Tensor:
    """Constant-current STM topograph: the tip height z(x, y) [Å above the surface]
    where the LDOS equals `target` — what an experimental STM records. LDOS decays
    monotonically into the vacuum, so scanning a stack of `n_planes` heights from
    `z_lo` to `z_hi`, the topograph is the highest plane still at/above `target`.
    Returns the 2D height map [Å].
    """
    ldos = ldos_grid(result, energy, sigma)
    cell = result.system.grid.cell
    length = float(cell[axis, axis])
    n = ldos.shape[axis]
    z_surf = _surface_top_z(result, axis) if surface_z is None else surface_z
    heights = torch.linspace(z_lo, z_hi, n_planes)

    def plane(h: float) -> torch.Tensor:
        iz = int(round(((z_surf + h) % length) / length * n)) % n
        return ldos.index_select(axis, torch.tensor([iz], device=ldos.device)).squeeze(axis)

    stack = torch.stack([plane(float(h)) for h in heights])       # (n_planes, ., .)
    dh = (z_hi - z_lo) / (n_planes - 1)
    count = (stack >= target).sum(0)                              # planes above target
    return z_lo + (count.clamp(min=1) - 1).to(RDTYPE) * dh
