"""Atomic valence density form factors — the superposition-of-atomic-densities
(SAD) initial guess.

UPF's PP_RHOATOM is 4πr²ρ_atom(r); its l=0 transform

    ρ̂(q) = ∫ 4πr²ρ_atom(r) j₀(q r) dr

satisfies ρ̂(0) ≈ Z_val (approximately, after mesh truncation — callers
rescale to the exact electron count when assembling the crystal guess).
"""

from __future__ import annotations

import numpy as np

from gradwave.pseudo.local import _msh
from gradwave.pseudo.radial import sbt
from gradwave.pseudo.upf import UPFData


def rhoatom_of_q(upf: UPFData, q: np.ndarray) -> np.ndarray:
    """ρ̂(q) for q (nq,) in Å⁻¹ (q=0 allowed → ≈ Z_val). Returns (nq,)."""
    n = _msh(upf)
    return sbt(0, upf.rhoatom[:n], upf.r[:n], upf.rab[:n], np.asarray(q, dtype=np.float64))


def core_density_of_q(upf: UPFData, q: np.ndarray) -> np.ndarray:
    """NLCC core density transform ∫4πr²ρc j₀(qr) dr; zeros if no NLCC."""
    if upf.core_rho is None:
        return np.zeros_like(np.asarray(q, dtype=np.float64))
    n = _msh(upf)
    g = 4.0 * np.pi * upf.r[:n] ** 2 * upf.core_rho[:n]
    return sbt(0, g, upf.r[:n], upf.rab[:n], np.asarray(q, dtype=np.float64))
