"""Charged-cell finite-size corrections (Makov-Payne).

A periodic supercell with a net charge q is neutralized by an implicit uniform
jellium background (gradwave's ``setup_system(..., tot_charge=q)`` sets
``n_electrons = ΣZ − q``; the G=0 electrostatics carry the −q background). The
computed total energy is then the jellium-neutralized periodic energy, which
differs from the physical isolated (dilute-limit) energy by the spurious
interaction of the charge with its periodic images and the background:

    E_periodic(L) = E_isolated + q² E_M / ε + O(L⁻³)

where E_M = −α_M e²/(2L) is the Madelung energy of a unit point charge plus
neutralizing background in the cell (α_M the lattice Madelung constant, L the
supercell size). ``makov_payne_correction`` returns the leading monopole term to
add back, and computes E_M from gradwave's own Ewald sum so it is exact for any
(possibly non-cubic) lattice, not just the simple-cubic α_M = 2.837297.

This is a post-SCF correction, the standard practice for charged defects and
ionization energies (Makov & Payne 1995; the FNV / Lany-Zunger schemes refine it
for localized charge and the dielectric screening). It is not part of the SCF
total energy — the SCF gives the periodic jellium cell, this estimates the
isolated limit.
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.dtypes import RDTYPE


def madelung_energy(cell) -> float:
    """Madelung energy [eV] of a single unit point charge plus a neutralizing
    uniform background in ``cell`` (rows a_i in Å): E_M = −α_M e²/(2L).

    Evaluated with the code's own Ewald sum (a single q=1 charge at the origin),
    so it is the exact lattice Madelung energy for any cell shape — for a simple
    cubic cell of side L it reproduces −2.837297·e²/(2L)."""
    from gradwave.core.energies.ewald import ewald_energy

    cell_t = torch.as_tensor(np.asarray(cell), dtype=RDTYPE)
    pos = torch.zeros((1, 3), dtype=RDTYPE)
    q = torch.ones(1, dtype=RDTYPE)
    return float(ewald_energy(pos, q, cell_t))


def makov_payne_correction(q: float, cell, epsilon: float = 1.0) -> float:
    """Leading Makov-Payne finite-size correction [eV] to ADD to a periodic
    charged-cell total energy to estimate the isolated (dilute-limit) energy.

    ``q`` is the net cell charge [e] (the ``tot_charge`` the cell was run at),
    ``cell`` the lattice (rows a_i in Å), ``epsilon`` the static dielectric
    constant screening the correction (1 for a molecule/atom in vacuum; the bulk
    ε for a defect in a solid). Returns −q²·E_M/ε = +q²α_M e²/(2Lε) > 0: the
    periodic images and jellium spuriously lower the charged cell by that amount,
    and this adds it back. Only the monopole term; the O(L⁻³) quadrupole term
    needs the charge distribution's second moment and is left to the caller.
    """
    e_m = madelung_energy(cell)
    return -(q ** 2) * e_m / float(epsilon)
