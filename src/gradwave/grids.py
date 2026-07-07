"""Plane-wave basis geometry: G-vector spheres, FFT grid sizing, index maps.

Setup layer — numpy for construction, frozen torch tensors as output.

Cutoff logic (eV/Å units, T(G) = HBAR2_2M·|k+G|²):

- wavefunction sphere at k:   HBAR2_2M |k+G|² ≤ ecut
- density/potential sphere:   |G| ≤ 2·G_max  (products of two wavefunctions)
- FFT box: per axis i, the largest Miller index of any density-sphere vector
  is m_i ≤ 2·G_max·|a_i|/(2π) (projection m_i = G·a_i/2π), so n_i ≥ 2m_i + 1,
  rounded up to a 2^a·3^b·5^c·7^d size. Undersizing this box aliases ρ(G)
  silently — test_grids checks product representability explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.dtypes import RDTYPE


def reciprocal_cell(cell: np.ndarray) -> np.ndarray:
    """Rows b_i [Å⁻¹] with a_i·b_j = 2π δ_ij; cell rows a_i in Å."""
    return 2.0 * np.pi * np.linalg.inv(cell).T


def good_fft_size(n: int) -> int:
    """Smallest size ≥ n whose prime factors are all ≤ 7."""
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def gmax_from_ecut(ecut: float) -> float:
    return float(np.sqrt(ecut / HBAR2_2M))


@dataclass(frozen=True)
class FFTGrid:
    """The dense real-space/G-space grid shared by density and potentials."""

    cell: np.ndarray  # (3,3) rows a_i [Å]
    shape: tuple[int, int, int]
    g_cart: torch.Tensor  # (n1,n2,n3,3) Cartesian G of every box point [Å⁻¹]
    g2: torch.Tensor  # (n1,n2,n3) |G|² [Å⁻²]
    dens_mask: torch.Tensor  # (n1,n2,n3) bool, |G| ≤ 2·G_max(ecut) — the density sphere

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    @property
    def n_points(self) -> int:
        s = self.shape
        return s[0] * s[1] * s[2]


def build_fft_grid(
    cell: np.ndarray,
    ecut: float,
    device=None,
    equal_dims: bool = False,
    shape_override=None,
) -> FFTGrid:
    """shape_override pins the FFT box (e.g. to QE's dims for µeV-level
    comparisons: XC grid integration differs at the meV level for sharp
    semicore densities when box sizes differ, even though both boxes hold
    the density sphere exactly)."""
    cell = np.asarray(cell, dtype=np.float64)
    b = reciprocal_cell(cell)
    gmax_dens = 2.0 * gmax_from_ecut(ecut)

    if shape_override is not None:
        shape = tuple(int(n) for n in shape_override)
    else:
        shape = []
        for i in range(3):
            # minimal box: integer Miller extent of the density sphere
            m_i = int(np.floor(gmax_dens * np.linalg.norm(cell[i]) / (2.0 * np.pi)))
            shape.append(good_fft_size(2 * m_i + 1))
        if equal_dims is True:
            # symmetry operations permute axes; a cubic box is always closed
            # under m → Wᵀm mod n
            shape = [max(shape)] * 3
        elif equal_dims:  # iterable of axis groups, e.g. [(0, 1)] for a slab
            for group in equal_dims:
                n = max(shape[i] for i in group)
                for i in group:
                    shape[i] = n
        shape = tuple(shape)

    millers = np.meshgrid(
        *[np.fft.fftfreq(n, d=1.0 / n).astype(np.int64) for n in shape], indexing="ij"
    )
    m = np.stack(millers, axis=-1)  # (n1,n2,n3,3) integer Miller indices
    g = m @ b  # Cartesian G
    g2 = np.einsum("...i,...i->...", g, g)

    return FFTGrid(
        cell=cell,
        shape=shape,
        g_cart=torch.as_tensor(g, dtype=RDTYPE, device=device),
        g2=torch.as_tensor(g2, dtype=RDTYPE, device=device),
        dens_mask=torch.as_tensor(g2 <= gmax_dens**2 * (1 + 1e-12), device=device),
    )


@dataclass(frozen=True)
class GSphere:
    """Wavefunction plane-wave sphere at one k-point."""

    k_frac: np.ndarray  # (3,) fractional k
    k_cart: torch.Tensor  # (3,) [Å⁻¹]
    miller: torch.Tensor  # (npw, 3) int64
    kpg: torch.Tensor  # (npw, 3) Cartesian k+G [Å⁻¹]
    kpg2: torch.Tensor  # (npw,) |k+G|² [Å⁻²]
    flat_idx: torch.Tensor  # (npw,) int64 indices into the flattened FFT box

    @property
    def npw(self) -> int:
        return int(self.miller.shape[0])


def build_gsphere(grid: FFTGrid, ecut: float, k_frac, device=None) -> GSphere:
    """All G with HBAR2_2M|k+G|² ≤ ecut, as indices into `grid`'s FFT box."""
    cell = grid.cell
    b = reciprocal_cell(cell)
    k_frac = np.asarray(k_frac, dtype=np.float64)
    k_cart = k_frac @ b

    n1, n2, n3 = grid.shape
    # candidate Miller range: |m_i + k_frac_i| ≤ G_max|a_i|/2π
    gmax = gmax_from_ecut(ecut)
    bound = [int(np.floor(gmax * np.linalg.norm(cell[i]) / (2 * np.pi) + 1)) + 1 for i in range(3)]
    axes = [np.arange(-bound[i], bound[i] + 1) for i in range(3)]
    mm = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    kpg = (mm + k_frac) @ b
    kpg2 = np.einsum("ij,ij->i", kpg, kpg)
    keep = HBAR2_2M * kpg2 <= ecut * (1 + 1e-12)
    mm, kpg, kpg2 = mm[keep], kpg[keep], kpg2[keep]

    for i, n in enumerate((n1, n2, n3)):
        if np.any(np.abs(mm[:, i]) > n // 2):
            raise RuntimeError("wavefunction sphere exceeds FFT box — grid sizing bug")
    flat = (mm[:, 0] % n1) * (n2 * n3) + (mm[:, 1] % n2) * n3 + (mm[:, 2] % n3)

    # deterministic ordering: by |k+G|², then Miller lexicographic
    order = np.lexsort((mm[:, 2], mm[:, 1], mm[:, 0], np.round(kpg2, 10)))
    mm, kpg, kpg2, flat = mm[order], kpg[order], kpg2[order], flat[order]

    return GSphere(
        k_frac=k_frac,
        k_cart=torch.as_tensor(k_cart, dtype=RDTYPE, device=device),
        miller=torch.as_tensor(mm, dtype=torch.int64, device=device),
        kpg=torch.as_tensor(kpg, dtype=RDTYPE, device=device),
        kpg2=torch.as_tensor(kpg2, dtype=RDTYPE, device=device),
        flat_idx=torch.as_tensor(flat, dtype=torch.int64, device=device),
    )
