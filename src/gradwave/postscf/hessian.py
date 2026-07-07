"""Γ-point force constants, dynamical matrix, and phonon frequencies (M4).

Current method: central finite differences of ANALYTIC Hellmann–Feynman
forces — each Hessian column costs two SCF runs but inherits force-level
(not energy-level) accuracy. The fully-automatic route (torch.func over the
implicit-diff SCF, one Sternheimer solve per column) plugs in behind the
same interface once scf/implicit.py lands.

Units: force constants eV/Å²; frequencies via
ω[cm⁻¹] = 521.47091 · sqrt(λ[eV/(amu·Å²)]).
"""

from __future__ import annotations

import numpy as np

from gradwave.postscf.forces import forces

SQRT_EV_AMU_ANG2_TO_CM1 = 521.4709116794098


def force_constants_gamma(
    make_scf,  # callable positions(na,3)->SCFResult (converged)
    positions: np.ndarray,
    h: float = 5e-3,
    acoustic_sum_rule: bool = True,
) -> np.ndarray:
    """Φ_(ai),(bj) = ∂²E/∂τ_ai∂τ_bj (3na, 3na) by central FD of analytic forces."""
    na = positions.shape[0]
    phi = np.zeros((3 * na, 3 * na))
    for a in range(na):
        for i in range(3):
            col = 3 * a + i
            fplus, fminus = [], []
            for sign, store in ((+1, fplus), (-1, fminus)):
                pos = positions.copy()
                pos[a, i] += sign * h
                res = make_scf(pos)
                store.append(forces(res).numpy().reshape(-1))
            phi[:, col] = -(fplus[0] - fminus[0]) / (2.0 * h)
    phi = 0.5 * (phi + phi.T)
    if acoustic_sum_rule:
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


def hessian_wrt_positions_fd_energy(
    make_scf, positions: np.ndarray, h: float = 5e-3
) -> np.ndarray:
    """Reference: d²E/dτ² by second differences of the ENERGY (validation only)."""
    na = positions.shape[0]
    n = 3 * na

    def e_at(pos):
        return float(make_scf(pos).energies.total)

    e0 = e_at(positions)
    hess = np.zeros((n, n))
    for p in range(n):
        for q in range(p, n):
            dp = np.zeros(n)
            dq = np.zeros(n)
            dp[p] = h
            dq[q] = h
            dpm, dqm = dp.reshape(na, 3), dq.reshape(na, 3)
            if p == q:
                hess[p, p] = (
                    e_at(positions + dpm) - 2 * e0 + e_at(positions - dpm)
                ) / h**2
            else:
                hess[p, q] = hess[q, p] = (
                    e_at(positions + dpm + dqm)
                    - e_at(positions + dpm - dqm)
                    - e_at(positions - dpm + dqm)
                    + e_at(positions - dpm - dqm)
                ) / (4 * h**2)
    return hess
