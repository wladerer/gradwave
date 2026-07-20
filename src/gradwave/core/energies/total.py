"""Total energy assembly — THE pure function autograd differentiates (Layer A).

E_total = E_kin + E_H + E_xc + E_loc + E_NL + E_ewald            (KS energy)
F       = E_total − σS                                            (free energy)
E₀      = (E_total + F)/2                                         (σ→0 extrap.)

QE's printed "!    total energy" corresponds to F when smearing is on.

## G = 0 ownership table (the three divergences and their survivors)

Every Coulomb piece diverges alone at G=0 for a charged sublattice; for a
neutral cell (Σ Z_val = N_e) the divergences cancel exactly. Which module
keeps which piece:

| term                       | G=0 handling                       | module            |
|----------------------------|------------------------------------|-------------------|
| Hartree e-e                | EXCLUDED (v_H(0) ≡ 0)              | energies/hartree  |
| local PSP Coulomb tail     | EXCLUDED from vloc_of_g            | pseudo/local      |
| local PSP short-range part | KEPT as alpha-Z (finite moment)    | pseudo/local +    |
|                            |   via the G=0 table entry          | energies/local_pp |
| ion-ion background         | KEPT inside Ewald (−π(ΣZ)²/2ηΩ)    | energies/ewald    |

Consistency test: total energy invariant (1e-10) under rigid translation of
all atoms, and the per-term breakdown regroups to QE's printout.

Purity contract: no in-place ops, no .detach() inside, no data-dependent
Python branching on tensors — this function is differentiated once for
forces and twice (via torch.func) for Hessians. Callers decide what is
detached BEFORE passing arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gradwave.core.density import sigma_from_rho
from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.hartree import hartree_energy
from gradwave.core.energies.kinetic import kinetic_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.fftbox import r_to_g
from gradwave.core.xc.base import XCFunctional


@dataclass
class EnergyBreakdown:
    kinetic: torch.Tensor
    hartree: torch.Tensor
    xc: torch.Tensor
    local: torch.Tensor
    nonlocal_: torch.Tensor
    ewald: torch.Tensor
    smearing: torch.Tensor  # −σS (zero for fixed occupations)
    hubbard: torch.Tensor | float = 0.0  # Dudarev E_U (zero without +U)
    onecenter: torch.Tensor | float = 0.0  # PAW one-center Σ±(E_H+E_xc) (zero for NC/USPP)

    @property
    def total(self) -> torch.Tensor:
        """Kohn–Sham energy E (without the smearing term)."""
        return (self.kinetic + self.hartree + self.xc + self.local + self.nonlocal_
                + self.ewald + self.hubbard + self.onecenter)

    @property
    def free_energy(self) -> torch.Tensor:
        """F = E − σS — compare against QE's '! total energy' when smeared."""
        return self.total + self.smearing

    @property
    def e0(self) -> torch.Tensor:
        """(E + F)/2, the σ→0 extrapolation for Gaussian smearing."""
        return self.total + 0.5 * self.smearing


def total_energy(
    *,
    coeffs_per_k: list[torch.Tensor],
    occ: torch.Tensor,
    kweights: torch.Tensor,
    spheres: list,
    grid,  # grids.FFTGrid
    rho: torch.Tensor,  # (n1,n2,n3) [e/Å³] — pass the SCF density (detached or not)
    positions: torch.Tensor,  # (na,3) Å
    charges: torch.Tensor,  # (na,) ionic Z_val
    species_index: torch.Tensor,
    vloc_tables: torch.Tensor,
    becp_per_k: list[torch.Tensor],
    dij_full: torch.Tensor,
    xc: XCFunctional,
    entropy_term: torch.Tensor | None = None,  # −σS, precomputed by SCF layer
    rho_core: torch.Tensor | None = None,  # NLCC: shifts the XC argument ONLY
) -> EnergyBreakdown:
    volume = grid.volume
    rho_g = r_to_g(rho.to(torch.complex128))

    e_kin = kinetic_energy(coeffs_per_k, occ, kweights, spheres)
    e_h = hartree_energy(rho_g, grid.g2, volume)
    rho_xc = rho if rho_core is None else rho + rho_core
    sigma = sigma_from_rho(rho_xc, grid.g_cart) if xc.needs_gradient else None
    e_xc = xc.energy(rho_xc, volume, sigma)
    vloc_g = local_potential_g(positions, species_index, vloc_tables, grid.g_cart, volume)
    e_loc = local_energy(rho_g, vloc_g, volume)
    e_nl = nonlocal_energy(becp_per_k, dij_full, occ, kweights)
    e_ew = ewald_energy(positions, charges, grid.cell)

    zero = torch.zeros((), dtype=e_kin.dtype, device=e_kin.device)
    return EnergyBreakdown(
        kinetic=e_kin,
        hartree=e_h,
        xc=e_xc,
        local=e_loc,
        nonlocal_=e_nl,
        ewald=e_ew,
        smearing=entropy_term if entropy_term is not None else zero,
    )
