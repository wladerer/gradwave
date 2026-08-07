"""Kleinman–Bylander projector form factors.

For each projector i (angular momentum l_i), the radial form factor

    F_i(q) = ∫ (r·β_i)(r) · j_{l_i}(q r) · r dr

evaluated at every requested |k+G|. With the parse-time scalings in upf.py, the
plane-wave projector

    p_i(k+G) = (4π/√Ω) · (−i)^{l_i} · Y_{l_i m}(k+G^) · F_i(|k+G|) · e^{−i(k+G)·τ}

and the contraction E_NL = Σ_ij D_ij ⟨ψ|p_i⟩⟨p_j|ψ⟩ come out directly in eV
with Ω in Å³ — this reproduces Quantum ESPRESSO's Ry-unit formula exactly.

F_i(q) is the spherical Bessel transform of a *fixed* radial function, so it
depends only on the pseudopotential and on q — NOT on the cell. In a variable-cell
relaxation the |k+G| set changes every ionic step, so evaluating the (expensive)
transform at those magnitudes re-does the same radial integrals for a slightly
shifted q-grid each time. Instead we build F_i(q) ONCE per pseudopotential on a
fine fixed q-grid and interpolate it (cubic spline) at each cell's |k+G|. The
interpolation error is ~1e-9 vs the direct transform (below every SCF tolerance;
guarded by tests/unit/test_form_factor_cache.py), so this is a speedup, not an
approximation the user can see. QE does the same (`tab_beta` / `init_us_1`).

Note the UPF stores one β per (l, radial channel); the m-degeneracy is
expanded downstream (core/hamiltonian.py) where Ylm and phases live — the
structure factor e^{-i(k+G)·τ} and positions never enter this radial table.
"""

from __future__ import annotations

import weakref

import numpy as np
from scipy.interpolate import CubicSpline

from gradwave.pseudo.radial import sbt
from gradwave.pseudo.upf import UPFData

# id(upf) -> (q_max_covered, [CubicSpline per projector]). Keyed on the pseudo
# object (stable across cells: the parsed UPF is path/symbol-cached upstream),
# with a weakref finalizer so a dropped pseudo evicts its splines — the same
# pattern as scf.uspp_setup._AUG_CACHE.
_BETA_FF_CACHE: dict[int, tuple[float, list[CubicSpline]]] = {}

# q-grid step [Å⁻¹]. dq = 0.02 puts the cubic-spline error ~1e-9 below the direct
# transform across the ecut range we run; test_form_factor_cache.py asserts it.
_FF_DQ = 0.02


def _beta_form_factors_exact(upf: UPFData, q: np.ndarray) -> np.ndarray:
    """F_i(q) for all projectors, evaluated DIRECTLY at q via the radial SBT — no
    interpolation. The normative path; ``beta_form_factors`` splines this across
    cells. q: (nq,) Å⁻¹ (q=0 allowed). Returns (nproj, nq)."""
    q = np.asarray(q, dtype=np.float64)
    out = np.empty((upf.n_proj, q.shape[0]))
    for i, beta in enumerate(upf.betas):
        n = beta.cutoff_idx
        r = upf.r[:n]
        out[i] = sbt(beta.l, beta.rbeta * r, r, upf.rab[:n], q)
    return out


def _beta_splines(upf: UPFData, q_max: float) -> list[CubicSpline]:
    """Per-projector CubicSpline of F_i(q) on [0, q_max], cached per pseudo. Built
    once and reused across cells; rebuilt only if a later cell needs a larger
    q_max (ecut is fixed, so at most once after the first build in a relax)."""
    hit = _BETA_FF_CACHE.get(id(upf))
    if hit is not None and hit[0] >= q_max:
        return hit[1]
    # margin so nearby cells reuse the table without a rebuild
    q_hi = q_max * 1.10 + _FF_DQ
    n_tab = max(2, int(np.ceil(q_hi / _FF_DQ)) + 1)
    q_tab = np.linspace(0.0, q_hi, n_tab)
    tab = _beta_form_factors_exact(upf, q_tab)  # (nproj, n_tab)
    splines = [CubicSpline(q_tab, tab[i]) for i in range(upf.n_proj)]
    if id(upf) not in _BETA_FF_CACHE:
        weakref.finalize(upf, _BETA_FF_CACHE.pop, id(upf), None)
    _BETA_FF_CACHE[id(upf)] = (q_hi, splines)
    return splines


def beta_form_factors(upf: UPFData, q: np.ndarray) -> np.ndarray:
    """F_i(q) for all projectors. q: (nq,) in Å⁻¹ (q=0 allowed). Returns (nproj, nq).

    Interpolates a cell-independent cubic-spline table (built once per pseudo,
    reused across cells) instead of re-running the radial transform every cell —
    the variable-cell-relax speedup. Matches ``_beta_form_factors_exact`` to
    ~1e-9, below every SCF tolerance."""
    q = np.asarray(q, dtype=np.float64)
    if q.size == 0:
        return np.empty((upf.n_proj, 0))
    splines = _beta_splines(upf, float(q.max()))
    out = np.empty((upf.n_proj, q.shape[0]))
    for i in range(upf.n_proj):
        out[i] = splines[i](q)
    return out
