"""XC functional interface (Layer A).

An XCFunctional maps grid densities to an XC energy density; the potential
v_xc = δE_xc/δρ is obtained by autograd — one differentiable implementation
serves as (a) the potential generator inside SCF, (b) the twice-differentiable
energy term for forces/Hessians, and (c) the trainable object for functional
learning (parameters are ordinary nn.Module parameters).

Internally functionals convert ρ [e/Å³] to atomic units, evaluate the
standard Hartree-a.u. expressions, and return e_xc [eV/Å³]:

    E_xc = (Ω/N) Σ_j e_xc(r_j)

GGA functionals receive σ = |∇ρ|² computed spectrally by the caller
(density.py) INSIDE the autograd graph, so autograd's v_xc automatically
contains the −∇·(∂e/∂∇ρ) term with spectral accuracy.

NaN discipline: no torch.where over expressions that are NaN in the dead
branch (NaN·0 = NaN in backward). Densities are floored via clamp before
fractional powers/logs; test densities sit far above the floor.
"""

from __future__ import annotations

import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV

RHO_FLOOR_AU = 1e-14  # a.u.; well below any physical grid density


class XCFunctional(torch.nn.Module):
    """Base class. Subclasses implement energy_density()."""

    needs_gradient: bool = False  # True for GGAs

    def energy_density(self, rho: torch.Tensor, sigma: torch.Tensor | None = None) -> torch.Tensor:
        """e_xc [eV/Å³] pointwise. rho [e/Å³]; sigma = |∇ρ|² [e²/Å⁸] for GGAs."""
        raise NotImplementedError

    def energy(
        self, rho: torch.Tensor, volume: float, sigma: torch.Tensor | None = None
    ) -> torch.Tensor:
        """E_xc [eV] = (Ω/N)Σ e_xc."""
        e = self.energy_density(rho, sigma)
        return e.sum() * (volume / e.numel())


def to_au(rho: torch.Tensor) -> torch.Tensor:
    """ρ [e/Å³] → ρ [e/bohr³], floored for NaN-safe powers/logs."""
    return torch.clamp(rho * BOHR_ANG**3, min=RHO_FLOOR_AU)


def eps_to_ev_density(rho_ang: torch.Tensor, eps_au: torch.Tensor) -> torch.Tensor:
    """ε [Ha/electron] → e_xc [eV/Å³] = ρ[e/Å³]·ε[eV]."""
    return rho_ang * eps_au * HARTREE_EV
