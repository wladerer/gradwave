"""Γ-point force constants, dynamical matrix, and phonon frequencies (M4).

The finite-difference route: central differences of ANALYTIC Hellmann–Feynman
forces — each Hessian column costs two SCF runs but inherits force-level (not
energy-level) accuracy. It needs no response machinery, so it is the reference
against which the analytic Hessian is checked. The analytic route now lives in
postscf/phonons.py (symmetry-reduced columns from postscf/uspp_position, built
on the implicit-diff SCF and Sternheimer solves in scf/implicit.py).

Units: force constants eV/Å²; frequencies via
ω[cm⁻¹] = SQRT_EV_AMU_ANG2_TO_CM1 · sqrt(λ[eV/(amu·Å²)]).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np

from gradwave.postscf.forces import forces

# Same cm⁻¹ conversion the analytic route uses: reuse phonons.py's single
# derived value (from gradwave.constants) so the two Γ-frequency paths cannot
# drift — an earlier hand-typed literal here disagreed at the 8th digit.
from gradwave.postscf.phonons import _SQRT_EV_AMU_ANG2_TO_CM1 as SQRT_EV_AMU_ANG2_TO_CM1
from gradwave.scf.loop import SCFResult

logger = logging.getLogger(__name__)


def force_constants_gamma(
    make_scf: Callable[[np.ndarray], SCFResult],  # positions(na,3)->SCFResult (converged)
    positions: np.ndarray,
    h: float = 5e-3,
    acoustic_sum_rule: bool = True,
    active: Sequence[int] | None = None,
) -> np.ndarray:
    """Φ_(ai),(bj) = ∂²E/∂τ_ai∂τ_bj by central FD of analytic forces.

    ``active`` restricts the *displaced* atoms to a subset (e.g. an adsorbate +
    the top slab layer): only those atoms are perturbed, and only their rows and
    columns are returned — a ``(3·n_active, 3·n_active)`` partial Hessian built
    from ``6·n_active`` SCFs instead of the full ``6·na``. This is the
    partial-Hessian vibrational analysis (PHVA) block used for TS prefactors and
    adsorbate frequencies, where the substrate is treated as rigid. ``None`` (or
    a list naming every atom, in order) reproduces the full ``(3na, 3na)``
    Hessian bit-for-bit.

    Acoustic sum rule. The translational invariance Σ_b Φ_(ai),(bj) = 0 sums a
    row over *all* atoms, so it is only meaningful for the full Hessian. On a
    strict active subset the row is truncated and the correction is ill-defined;
    ``acoustic_sum_rule=True`` is therefore ignored for a partial Hessian (with a
    warning) — a PHVA block is used as-is (the rigid substrate breaks
    translational invariance by construction). Pass ``acoustic_sum_rule=False``
    to silence the warning.
    """
    na = positions.shape[0]
    if active is None:
        idx = np.arange(na, dtype=int)
    else:
        idx = np.asarray(list(active), dtype=int)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("active must be a non-empty 1-D sequence of atom indices")
        if idx.min() < 0 or idx.max() >= na:
            raise ValueError(
                f"active atom index out of range for {na} atoms: {list(idx)}")
        if len(set(idx.tolist())) != idx.size:
            raise ValueError(f"active has duplicate indices: {list(idx)}")
    # "full" iff every atom is displaced in canonical order — then the returned
    # block is the whole Hessian and the full ASR applies unchanged.
    is_full = np.array_equal(idx, np.arange(na, dtype=int))
    nact = idx.size

    phi = np.zeros((3 * nact, 3 * nact))
    for local, a in enumerate(idx):
        for i in range(3):
            col = 3 * local + i
            fplus, fminus = [], []
            for sign, store in ((+1, fplus), (-1, fminus)):
                pos = positions.copy()
                pos[a, i] += sign * h
                res = make_scf(pos)
                # forces on ALL atoms; keep only the active rows (the block's
                # row DOFs) — for is_full this is the whole force vector, so the
                # column matches the original all-atom computation exactly.
                f_full = forces(res).cpu().numpy()
                store.append(f_full[idx].reshape(-1))
            phi[:, col] = -(fplus[0] - fminus[0]) / (2.0 * h)
    phi = 0.5 * (phi + phi.T)
    if acoustic_sum_rule and not is_full:
        logger.warning(
            "force_constants_gamma: acoustic_sum_rule ignored for a partial "
            "Hessian (active names %d of %d atoms) — the translational sum rule "
            "needs every atom's column; the rigid-substrate PHVA block is "
            "returned uncorrected. Pass acoustic_sum_rule=False to silence.",
            nact, na)
    elif acoustic_sum_rule:
        # enforce Σ_b Φ_(ai),(bj) = 0 by correcting the self blocks
        for i in range(3 * na):
            a = i // 3
            for j in range(3):
                block_sum = sum(phi[i, 3 * b + j] for b in range(na))
                phi[i, 3 * a + j] -= block_sum
        phi = 0.5 * (phi + phi.T)
    return phi


def gamma_phonons(phi: np.ndarray, masses_amu: np.ndarray) -> np.ndarray:
    """Frequencies [cm⁻¹] from Γ force constants; negative = imaginary mode."""
    msqrt = np.repeat(np.sqrt(masses_amu), 3)
    dyn = phi / np.outer(msqrt, msqrt)
    dyn = 0.5 * (dyn + dyn.T)
    evals = np.linalg.eigvalsh(dyn)
    return np.sign(evals) * SQRT_EV_AMU_ANG2_TO_CM1 * np.sqrt(np.abs(evals))
