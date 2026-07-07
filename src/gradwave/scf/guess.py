"""Superposition-of-atomic-densities (SAD) initial guess (Layer B)."""

from __future__ import annotations

import numpy as np
import torch

from gradwave.dtypes import CDTYPE
from gradwave.pseudo.atomic import rhoatom_of_q


def sad_density(
    grid,
    positions: torch.Tensor,  # (na, 3) Å (detached — guess is not differentiated)
    species_of_atom: list[int],
    upfs: list,  # [UPFData] per species
    n_electrons: float,
    species_scale=None,  # per-species factor (spin-channel splits), default 1
) -> torch.Tensor:
    """ρ₀(r) on the dense grid [e/Å³], rescaled to exactly N_e electrons."""
    device = grid.g2.device
    g = torch.sqrt(grid.g2).reshape(-1).cpu().numpy()  # radial tables are numpy-side
    uniq, inverse = np.unique(np.round(g, 9), return_inverse=True)
    vol = grid.volume

    rho_g = torch.zeros(grid.n_points, dtype=CDTYPE, device=device)
    pos = positions.detach()
    for s, upf in enumerate(upfs):
        scale = 1.0 if species_scale is None else float(species_scale[s])
        table = scale * torch.as_tensor(
            rhoatom_of_q(upf, uniq), dtype=torch.float64, device=device)
        shell = table[torch.as_tensor(inverse, device=device)]
        atoms = [a for a, sa in enumerate(species_of_atom) if sa == s]
        if not atoms:
            continue
        gvec = grid.g_cart.reshape(-1, 3)
        phase = gvec @ pos[atoms].T  # (nG, natoms_s)
        sfac = torch.exp(torch.complex(torch.zeros_like(phase), -phase)).sum(dim=1)
        rho_g += sfac * shell.to(CDTYPE) / vol

    # rescale so that Ω·ρ(G=0) = N_e exactly (mesh-truncation fix)
    scale = n_electrons / (vol * rho_g[0].real)
    rho_g = rho_g * scale
    rho_g = torch.where(grid.dens_mask.reshape(-1), rho_g, torch.zeros_like(rho_g))

    rho_r = torch.fft.ifftn(rho_g.reshape(grid.shape) * grid.n_points, dim=(-3, -2, -1)).real
    # SAD can dip slightly negative between atoms; floor tiny negatives, then
    # restore exact normalization (the mixer pins the G=0 channel)
    rho_r = torch.clamp(rho_r, min=1e-12)
    return rho_r * (n_electrons / (rho_r.mean() * vol))
