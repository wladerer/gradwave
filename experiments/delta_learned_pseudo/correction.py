"""Differentiable local-potential correction dv_loc(G; theta) for a single species.

The correction is a smooth few-parameter form in |G| added to a species' local
form-factor table [eV.Ang^3], forced to vanish at G=0 (so the alpha-Z / charge
normalization the base table carries is untouched) and at G->inf (so it is a
genuine short-range correction, not a shifted potential).

Basis: dv(G) = sum_k theta_k * (G/mu_k)^2 * exp(-(G/mu_k)^2)
  - the (G/mu_k)^2 prefactor kills G=0 exactly
  - the Gaussian kills G->inf
  - each bump peaks at |G| = mu_k, so the theta_k are near-orthogonal knobs on
    the low-G shells where a local pseudopotential form factor actually lives

theta is [eV.Ang^3]. dv is gathered onto the dense box exactly the way
setup_common.build_vloc_tables gathers the base table (via the |G| shell
inverse index), so a corrected per-atom table drops straight into
System.vloc_atom, which local_potential_g already consumes.
"""
from __future__ import annotations

import numpy as np
import torch

from gradwave.dtypes import RDTYPE
from gradwave.scf.setup_common import _unique_shells


def default_centers(n: int = 4) -> torch.Tensor:
    """mu_k in Ang^-1, peaks of the correction bumps. Low-G is where the local
    form factor has structure, so span 1..n Ang^-1."""
    return torch.arange(1, n + 1, dtype=RDTYPE)


def dv_on_shells(g: torch.Tensor, theta: torch.Tensor,
                 centers: torch.Tensor) -> torch.Tensor:
    """dv(|G|; theta) [eV.Ang^3] on a 1-D tensor of |G| values (Ang^-1)."""
    g = g.to(RDTYPE)
    centers = centers.to(RDTYPE)
    x = g[:, None] / centers[None, :]  # (nshell, K)
    basis = x**2 * torch.exp(-(x**2))  # vanishes at G=0 and G->inf
    return basis @ theta.to(RDTYPE)


def dv_table(grid, theta: torch.Tensor,
             centers: torch.Tensor) -> torch.Tensor:
    """dv correction gathered onto the dense box (n1,n2,n3) [eV.Ang^3],
    differentiable in theta. Uses the same unique-|G|-shell path as
    build_vloc_tables so shells align with the base table exactly."""
    g_flat = np.sqrt(grid.g2.reshape(-1).detach().cpu().numpy())
    uniq, inverse = _unique_shells(g_flat)
    uniq_t = torch.as_tensor(uniq, dtype=RDTYPE)
    dv_shells = dv_on_shells(uniq_t, theta, centers)  # (nshell,)
    inv_t = torch.as_tensor(inverse, dtype=torch.long)
    return dv_shells[inv_t].reshape(grid.shape)


def corrected_vloc_atom(system, theta: torch.Tensor, centers: torch.Tensor,
                        species: int = 0) -> torch.Tensor:
    """Per-atom local table (na, n1,n2,n3) = base species table + dv(theta) on
    atoms of `species`, differentiable in theta. Drops into System.vloc_atom.

    theta=0 returns the base per-atom table bit-for-bit (dv is identically 0),
    which is the endpoint-exactness guarantee."""
    base = system.vloc_tables[system.species_index]  # (na, n1,n2,n3)
    dv = dv_table(system.grid, theta, centers)  # (n1,n2,n3)
    sp = system.species_index.detach().cpu().numpy()
    rows = []
    for a in range(base.shape[0]):
        rows.append(base[a] + dv if sp[a] == species else base[a])
    return torch.stack(rows, dim=0)
