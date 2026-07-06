"""Hartree energy and potential in reciprocal space (Layer A).

With ρ(G) the Fourier-series coefficients of the electron density [e/Å³]
(fftbox convention), for the periodic neutralized system:

    E_H = (Ω/2) Σ_{G≠0} 4π e² |ρ(G)|² / G²
    v_H(G) = 4π e² ρ(G)/G²,  v_H(G=0) ≡ 0

The divergent G=0 term is EXCLUDED here — its cancellation against the
local-pseudopotential tail and the Ewald background is documented in
energies/total.py.
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import E2


def hartree_energy(rho_g: torch.Tensor, g2: torch.Tensor, volume: float) -> torch.Tensor:
    """E_H [eV]. rho_g, g2: dense-box tensors (fftbox layout)."""
    inv_g2 = torch.where(g2 > 1e-12, 1.0 / torch.clamp(g2, min=1e-12), torch.zeros_like(g2))
    return 0.5 * volume * 4.0 * math.pi * E2 * ((rho_g.abs() ** 2) * inv_g2).sum()


def hartree_potential_g(rho_g: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """v_H(G) [eV] on the dense box, v_H(0) = 0."""
    inv_g2 = torch.where(g2 > 1e-12, 1.0 / torch.clamp(g2, min=1e-12), torch.zeros_like(g2))
    return 4.0 * math.pi * E2 * rho_g * inv_g2
