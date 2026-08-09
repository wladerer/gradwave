"""Atomic valence density form factors — the superposition-of-atomic-densities
(SAD) initial guess.

UPF's PP_RHOATOM is 4πr²ρ_atom(r); its l=0 transform

    ρ̂(q) = ∫ 4πr²ρ_atom(r) j₀(q r) dr

satisfies ρ̂(0) = Z_val once the FULL radial mesh is integrated. Unlike the
local potential, the atomic density carries no long-range −Z/r tail, so there
is no reason to apply QE's 10-bohr (_msh) local-channel truncation here — doing
so clips the density tail and loses ~0.1–0.3% of the valence charge (worst for
diffuse pseudos, e.g. Al: −2.4e-3), which the crystal-guess G=0 rescale then
smears back over every shell, distorting the guess shape. Integrating the whole
mesh recovers Z_val to ~1e-6.
"""

from __future__ import annotations

import numpy as np

from gradwave.pseudo.local import _msh
from gradwave.pseudo.radial import sbt
from gradwave.pseudo.upf import UPFData


def rhoatom_of_q(upf: UPFData, q: np.ndarray) -> np.ndarray:
    """ρ̂(q) for q (nq,) in Å⁻¹ (q=0 allowed → Z_val). Returns (nq,).

    Integrates the full radial mesh (not the 10-bohr _msh clip): the atomic
    density decays on its own, so truncating it only sheds charge from the tail.
    """
    return sbt(0, upf.rhoatom, upf.r, upf.rab, np.asarray(q, dtype=np.float64))


def core_density_of_q(upf: UPFData, q: np.ndarray) -> np.ndarray:
    """NLCC core density transform ∫4πr²ρc j₀(qr) dr; zeros if no NLCC."""
    if upf.core_rho is None:
        return np.zeros_like(np.asarray(q, dtype=np.float64))
    n = _msh(upf)
    g = 4.0 * np.pi * upf.r[:n] ** 2 * upf.core_rho[:n]
    return sbt(0, g, upf.r[:n], upf.rab[:n], np.asarray(q, dtype=np.float64))
