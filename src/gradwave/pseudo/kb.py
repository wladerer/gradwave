"""Kleinman–Bylander projector form factors.

For each projector i (angular momentum l_i), the radial form factor

    F_i(q) = ∫ (r·β_i)(r) · j_{l_i}(q r) · r dr

evaluated directly at every requested |k+G| (no spline interpolation).
With the parse-time scalings in upf.py, the plane-wave projector

    p_i(k+G) = (4π/√Ω) · (−i)^{l_i} · Y_{l_i m}(k+G^) · F_i(|k+G|) · e^{−i(k+G)·τ}

and the contraction E_NL = Σ_ij D_ij ⟨ψ|p_i⟩⟨p_j|ψ⟩ come out directly in eV
with Ω in Å³ — this reproduces Quantum ESPRESSO's Ry-unit formula exactly.

Note the UPF stores one β per (l, radial channel); the m-degeneracy is
expanded downstream (core/hamiltonian.py) where Ylm and phases live.
"""

from __future__ import annotations

import numpy as np

from gradwave.pseudo.radial import sbt
from gradwave.pseudo.upf import UPFData


def beta_form_factors(upf: UPFData, q: np.ndarray) -> np.ndarray:
    """F_i(q) for all projectors. q: (nq,) in Å⁻¹ (q=0 allowed). Returns (nproj, nq)."""
    q = np.asarray(q, dtype=np.float64)
    out = np.empty((upf.n_proj, q.shape[0]))
    for i, beta in enumerate(upf.betas):
        n = beta.cutoff_idx
        r = upf.r[:n]
        out[i] = sbt(beta.l, beta.rbeta * r, r, upf.rab[:n], q)
    return out
