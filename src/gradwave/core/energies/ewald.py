"""Ion–ion Ewald sum — differentiable in positions (Layer A).

E_ewald = E_real + E_recip + E_self + E_background

  E_real = (e²/2) Σ'_{a,b,R} Z_a Z_b erfc(√η r)/r,   r = |τ_a − τ_b + R|
  E_recip = (2π e²/Ω) Σ_{G≠0} e^{−G²/4η}/G² |Σ_a Z_a e^{−iG·τ_a}|²
  E_self = −e² √(η/π) Σ_a Z_a²
  E_bg   = −(π e²/(2ηΩ)) (Σ_a Z_a)²    (neutralizing background, the ion-side
                                        share of the global G=0 cancellation —
                                        see energies/total.py)

η and the real/reciprocal image lists are chosen at setup (non-differentiable,
plain numbers); the summed expression is differentiable in τ. η-independence
is a unit test.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.dtypes import RDTYPE
from gradwave.grids import reciprocal_cell

# Cutoff parameter: erfc(_ACC) and e^{-_ACC²} bound the LAST term, but the
# error is the sum over all excluded shells — 4.8 leaves ~1e-3 eV errors in
# ionic crystals. 8.0 is machine-precision converged (verified vs Madelung
# constants for all η in [0.3, 1.5]) and still cheap for unit-cell systems.
_ACC = 8.0


def _image_vectors(cell: np.ndarray, rcut: float) -> np.ndarray:
    """All lattice vectors R with |R| ≤ rcut (including R = 0)."""
    binv = reciprocal_cell(cell) / (2.0 * math.pi)  # rows: b_i/2π, R·(b_i/2π) = n_i
    bounds = [int(np.ceil(rcut * np.linalg.norm(binv[i]))) + 1 for i in range(3)]
    axes = [np.arange(-n, n + 1) for n in bounds]
    n = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    r = n @ cell
    return r[np.linalg.norm(r, axis=1) <= rcut + 1e-9]


def _g_vectors(cell: np.ndarray, gcut: float) -> np.ndarray:
    """All reciprocal vectors 0 < |G| ≤ gcut."""
    b = reciprocal_cell(cell)
    bounds = [int(np.ceil(gcut * np.linalg.norm(cell[i]) / (2 * math.pi))) + 1 for i in range(3)]
    axes = [np.arange(-n, n + 1) for n in bounds]
    m = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    g = m @ b
    g2 = np.einsum("ij,ij->i", g, g)
    keep = (g2 > 1e-12) & (g2 <= gcut**2 + 1e-9)
    return g[keep]


def ewald_energy(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: np.ndarray,
    eta: float | None = None,
) -> torch.Tensor:
    """Ewald energy [eV]. positions (na,3) Å (may require grad); charges (na,)."""
    cell = np.asarray(cell, dtype=np.float64)
    omega = abs(np.linalg.det(cell))
    if eta is None:
        # balance: real-space cutoff ~ ACC/√η vs recip cutoff 2√η·ACC; pick η so
        # the real-space sum needs only a few shells of a typical cell
        eta = (math.pi / omega ** (1.0 / 3.0)) ** 2

    sqrt_eta = math.sqrt(eta)
    rcut = _ACC / sqrt_eta
    gcut = 2.0 * sqrt_eta * _ACC

    dev = positions.device
    images = torch.as_tensor(_image_vectors(cell, rcut), dtype=RDTYPE, device=dev)
    gvecs = torch.as_tensor(_g_vectors(cell, gcut), dtype=RDTYPE, device=dev)
    z = charges.to(RDTYPE)

    # real space: pair separations (na, na, nR, 3)
    d = positions[:, None, None, :] - positions[None, :, None, :] + images[None, None, :, :]
    na = positions.shape[0]
    self_pair = (
        (torch.eye(na, dtype=torch.bool, device=dev))[:, :, None]
        & (torch.linalg.norm(images, dim=-1) < 1e-12)[None, None, :]
    )
    # shift the (masked-out) self pairs to |d| = 1 BEFORE the norm: the
    # norm at d = 0 has NaN second derivatives even when the entries are
    # masked afterward (the dead branch poisons double backward — the
    # Hessian path needs create_graph through this sum; forces only ever
    # saw the guarded first derivative)
    offset = torch.zeros(3, dtype=RDTYPE, device=dev)
    offset[0] = 1.0
    d = d + self_pair[..., None].to(RDTYPE) * offset
    r = torch.linalg.norm(d, dim=-1)
    pair = torch.erfc(sqrt_eta * r) / r
    pair = torch.where(self_pair, torch.zeros_like(pair), pair)
    e_real = 0.5 * E2 * torch.einsum("a,b,abr->", z, z, pair)

    # reciprocal space
    g2 = (gvecs**2).sum(-1)
    phase = positions @ gvecs.T  # (na, ng)
    s_re = (z[:, None] * torch.cos(phase)).sum(0)
    s_im = (z[:, None] * torch.sin(phase)).sum(0)
    e_recip = (
        (2.0 * math.pi * E2 / omega)
        * ((s_re**2 + s_im**2) * torch.exp(-g2 / (4.0 * eta)) / g2).sum()
    )

    e_self = -E2 * sqrt_eta / math.sqrt(math.pi) * (z**2).sum()
    e_bg = -math.pi * E2 / (2.0 * eta * omega) * z.sum() ** 2
    return e_real + e_recip + e_self + e_bg
