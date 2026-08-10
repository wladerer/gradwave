"""Lindh model Hessian — a cheap, curvature-aware initial Hessian for BFGS.

BFGS starts from a scaled-identity Hessian and only learns curvature by
accumulating ``(Δx, Δg)`` pairs, so its first several steps are effectively
scaled steepest descent. On a system whose force constants span two orders of
magnitude — a stiff covalent stretch next to a soft intermolecular or lateral
mode — the identity mis-scales the step and overshoots the stiff directions
(the transient force *increase* seen early in a molecular-crystal relax).

Seeding BFGS with a model Hessian that already knows which directions are stiff
removes that overshoot. This module builds the Lindh (1995) pairwise-stretch
model: every atom pair contributes a stretch force constant
``k·ρ_ij`` along its bond axis, with

    ρ_ij = exp(α_AB · (r_ref,AB² − r_ij²)),

so bonded pairs (``r ≈ r_ref``) are stiff and distant pairs decay away. The
``α_AB`` and ``r_ref,AB`` depend only on the periods of the two atoms (Lindh
et al., Chem. Phys. Lett. 241, 423). This is the dominant (stretch) term of the
full Lindh force field — bends and torsions are omitted — which is enough to
separate stiff from soft directions and kill the early overshoot; the C=O
stretch it produces (~114 eV/Å²) matches experiment (~116 eV/Å²).

The assembled Hessian has zero modes (rigid translation/rotation) and soft
modes; its eigenvalues are floored to keep it positive-definite and to bound the
first-step size (ASE's ``maxstep`` caps it further).
"""
from __future__ import annotations

import numpy as np

from gradwave.constants import BOHR_ANG, HARTREE_EV

# Lindh parameters, indexed by period row (0: H–He, 1: Li–Ne, 2: Na and beyond).
_ALPHA = np.array([          # Bohr^-2
    [1.0000, 0.3949, 0.3949],
    [0.3949, 0.2800, 0.2800],
    [0.3949, 0.2800, 0.2800],
])
_RREF = np.array([           # Bohr
    [1.35, 2.10, 2.53],
    [2.10, 2.87, 3.40],
    [2.53, 3.40, 3.40],
])
_K_R = 0.45                  # Hartree / Bohr^2, reference stretch force constant


def _row(z: int) -> int:
    if z <= 2:
        return 0
    if z <= 10:
        return 1
    return 2


def lindh_hessian(
    positions: np.ndarray,
    numbers: list[int] | np.ndarray,
    *,
    floor: float = 30.0,
    cutoff_ang: float = 15.0,
) -> np.ndarray:
    """Cartesian Lindh stretch-model Hessian, shape ``(3N, 3N)`` in eV/Å².

    ``positions`` are Cartesian [Å], ``numbers`` the atomic numbers. ``floor`` is
    the minimum eigenvalue [eV/Å²] imposed to keep the matrix positive-definite
    (so rigid/soft modes get a finite, bounded initial curvature rather than a
    near-zero one that would blow the first step up). ``cutoff_ang`` drops pairs
    whose ``ρ`` is negligible, keeping the build affordable on larger cells.
    """
    pos_bohr = np.asarray(positions, dtype=float) / BOHR_ANG
    z = [int(n) for n in numbers]
    n = len(z)
    cut_bohr = cutoff_ang / BOHR_ANG
    h = np.zeros((3 * n, 3 * n))
    for i in range(n):
        ri = _row(z[i])
        for j in range(i + 1, n):
            d = pos_bohr[i] - pos_bohr[j]
            r = float(np.linalg.norm(d))
            if r < 1e-6 or r > cut_bohr:
                continue
            u = d / r
            rj = _row(z[j])
            rho = np.exp(_ALPHA[ri, rj] * (_RREF[ri, rj] ** 2 - r * r))
            k = _K_R * rho
            b = np.zeros(3 * n)
            b[3 * i:3 * i + 3] = u
            b[3 * j:3 * j + 3] = -u
            h += k * np.outer(b, b)
    h *= HARTREE_EV / BOHR_ANG ** 2  # Ha/Bohr² → eV/Å²
    # symmetric eigenvalue floor: positive-definite, bounded first step
    w, v = np.linalg.eigh(0.5 * (h + h.T))
    w = np.maximum(w, float(floor))
    return (v * w) @ v.T


def seed_bfgs_hessian(optimizer: object, h0: np.ndarray) -> bool:
    """Install ``h0`` (eV/Å², shape ``(ndof, ndof)``) as an ASE BFGS optimizer's
    starting Hessian, replacing the scaled-identity default.

    Returns True if applied, False if the optimizer's degree-of-freedom count does
    not match ``h0`` (a cell relax adds cell DOFs the atomic model Hessian does not
    cover) — the caller then keeps the ASE default. Tolerant of ASE-version
    differences in how the Hessian state is stored.
    """
    ndof = None
    try:
        ndof = optimizer.optimizable.ndofs()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - older ASE without .optimizable
        ndof = None
    if ndof is not None and int(h0.shape[0]) != int(ndof):
        return False
    if hasattr(optimizer, "H0"):
        # ASE (3.29) builds its BFGSMethod from ``H0`` LAZILY on the first step,
        # guarded by ``state is None``. Set only ``H0`` and leave ``state`` unset so
        # that guard still fires — installing a live state here defeats it and
        # crashes updating from the (None) previous forces.
        optimizer.H0 = h0                 # type: ignore[attr-defined]
        if hasattr(optimizer, "state"):
            optimizer.state = None        # type: ignore[attr-defined]
        return True
    if hasattr(optimizer, "H"):  # pragma: no cover - very old ASE stored H directly
        optimizer.H = h0                  # type: ignore[attr-defined]
        return True
    return False
