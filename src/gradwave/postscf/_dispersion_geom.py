"""Shared periodic-image geometry for the dispersion corrections.

The D3 (``dispersion``) and D4 (``dispersion_d4``) modules build the same
real-space image lattice and pairwise-distance tensor before applying their
(different) damping and C6 models. Both helpers live here so the geometry — and
the double-backward guard on the masked self-pair — is defined once; the callers
import them as their module-private ``_image_labels`` / ``_pair_distances``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.constants import BOHR_ANG
from gradwave.grids import reciprocal_cell


def image_labels(cell_ang: np.ndarray | None, rcut_bohr: float) -> np.ndarray:
    """Integer lattice labels n with |n·cell| ≤ rcut (incl. n=0). (nR, 3).

    Returns integer labels (not Cartesian vectors) so the images can be rebuilt
    on the autograd cell in the stress path; a molecule (cell is None) has only
    the n=0 image.
    """
    if cell_ang is None:
        return np.zeros((1, 3), dtype=np.int64)
    cell = np.asarray(cell_ang, dtype=np.float64) / BOHR_ANG  # Bohr
    binv = reciprocal_cell(cell) / (2.0 * math.pi)  # rows b_i/2π
    bounds = [int(np.ceil(rcut_bohr * np.linalg.norm(binv[i]))) + 1 for i in range(3)]
    axes = [np.arange(-n, n + 1) for n in bounds]
    n = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    r = n @ cell
    return n[np.linalg.norm(r, axis=1) <= rcut_bohr + 1e-9].astype(np.int64)


def pair_distances(
    pos_bohr: torch.Tensor, images_bohr: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(na,na,nR) distances |τ_a − τ_b + L| and the self-pair (A=B, L=0) mask.

    The masked self separation is shifted off zero *before* the norm so the dead
    branch keeps finite second derivatives (the Ewald double-backward guard);
    callers must still exclude/replace self pairs via the returned mask.
    """
    na = pos_bohr.shape[0]
    d = (
        pos_bohr[:, None, None, :]
        - pos_bohr[None, :, None, :]
        + images_bohr[None, None, :, :]
    )
    dev, dt = pos_bohr.device, pos_bohr.dtype
    img0 = torch.linalg.norm(images_bohr, dim=-1) < 1e-12
    self_mask = torch.eye(na, dtype=torch.bool, device=dev)[:, :, None] & img0[None, None, :]
    offset = torch.zeros(3, dtype=dt, device=dev)
    offset[0] = 1.0
    d = d + self_mask[..., None].to(dt) * offset
    r = torch.linalg.norm(d, dim=-1)
    return r, self_mask
