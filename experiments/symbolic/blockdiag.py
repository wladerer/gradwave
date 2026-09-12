"""Dependency-free symmetry block-diagonalizer (numpy only).

Generalizes track3b to 3D crystals and COMPLEX phased representations (needed for
nonsymmorphic space groups, where D(g) carries an e^{-iG·τ} fractional-translation
phase). The construction is unchanged: build the class sums C_i = Σ_{g∈class} D(g)
— which span the center of the group algebra and commute with any H that has the
symmetry — then take a generic HERMITIAN central element

    T = Σ_i [ a_i (C_i + C_i†)/2 + b_i (C_i − C_i†)/(2i) ],   a_i, b_i random,

whose eigenspaces are the isotypic components (real+imag parts sample the full
central character, so complex irreps separate too). eigh(T) → the block basis.

No character tables, no irrep matrices, no Sage. Only the group operations (as
integer rotation matrices) and their representation D(g) on the basis.
"""

from __future__ import annotations

import numpy as np


def _key(m: np.ndarray) -> tuple:
    return tuple(np.rint(m).astype(int).flatten().tolist())


def conjugacy_classes(rots: list[np.ndarray]) -> list[list[int]]:
    """Partition a closed group of integer rotation matrices into conjugacy
    classes { h g h⁻¹ }, from the group multiplication alone."""
    by_key = {_key(R): i for i, R in enumerate(rots)}
    inv = [np.rint(np.linalg.inv(R)).astype(int) for R in rots]
    classes: list[list[int]] = []
    seen: set[int] = set()
    for gi in range(len(rots)):
        if gi in seen:
            continue
        orbit = set()
        for hi in range(len(rots)):
            orbit.add(by_key[_key(rots[hi] @ rots[gi] @ inv[hi])])
        cls = sorted(orbit)
        classes.append(cls)
        seen |= orbit
    return classes


def cluster_sorted(w: np.ndarray, tol: float) -> list[int]:
    """Sizes of contiguous groups of a SORTED real array within `tol`."""
    bnds = [0]
    for i in range(1, len(w)):
        if w[i] - w[i - 1] > tol:
            bnds.append(i)
    bnds.append(len(w))
    return [bnds[i + 1] - bnds[i] for i in range(len(bnds) - 1)]


def block_diagonalize(H: np.ndarray, Dmats: list[np.ndarray],
                      classes: list[list[int]], seed: int = 0,
                      tol: float | None = None):
    """Block-diagonalize Hermitian H using the symmetry rep {Dmats} grouped into
    `classes`. Returns (U, sizes, B, diag) with B = U†HU block-diagonal by irrep
    type, sizes the block dimensions, and diag a dict of validation numbers."""
    N = H.shape[0]
    rng = np.random.default_rng(seed)
    Csum = [sum(Dmats[g] for g in cls) for cls in classes]

    # generic Hermitian central element
    a = rng.standard_normal(len(classes))
    b = rng.standard_normal(len(classes))
    T = np.zeros((N, N), dtype=complex)
    for Ci, ai, bi in zip(Csum, a, b):
        T += ai * (Ci + Ci.conj().T) / 2 + bi * (Ci - Ci.conj().T) / (2j)

    commHT = float(np.abs(T @ H - H @ T).max())          # T must commute with H
    commHD = max(float(np.abs(D @ H - H @ D).max()) for D in Dmats)  # H must be symmetric

    w, U = np.linalg.eigh(T)
    scale = max(1.0, float(np.abs(w).max()))
    tol = tol if tol is not None else 1e-6 * scale
    sizes = cluster_sorted(w, tol)

    B = U.conj().T @ H @ U
    # inter-block leakage
    off = B.copy()
    s = 0
    for r in sizes:
        off[s:s + r, s:s + r] = 0.0
        s += r
    leak = float(np.abs(off).max())

    # spectrum: per-block eig vs full eig
    ev_full = np.sort(np.linalg.eigvalsh(H).real)
    ev_blk = []
    s = 0
    for r in sizes:
        ev_blk += list(np.linalg.eigvalsh(B[s:s + r, s:s + r]).real)
        s += r
    ev_err = float(np.abs(ev_full - np.sort(ev_blk)).max())

    diag = {"commute_TH": commHT, "commute_HD": commHD,
            "leak": leak, "ev_err": ev_err, "n_classes": len(classes)}
    return U, sizes, B, diag


def flop_reduction(N: int, sizes: list[int]) -> float:
    return N ** 3 / sum(s ** 3 for s in sizes)


if __name__ == "__main__":
    # self-test on the C4v toy (real rep) — must match track3b
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from track3_symmetry import build_hamiltonian
    from track3b_commutant import R as C4V_R, perm_matrix_R

    H, grid, idx = build_hamiltonian(3)
    rots = list(C4V_R.values())
    Dmats = [perm_matrix_R(Rm, grid, idx).astype(complex) for Rm in rots]
    classes = conjugacy_classes(rots)
    U, sizes, B, diag = block_diagonalize(H.astype(complex), Dmats, classes)
    print("C4v self-test:", sorted(sizes), diag)
    assert sorted(sizes) == [3, 6, 6, 10, 24], sizes
    assert diag["leak"] < 1e-9 and diag["ev_err"] < 1e-9
    print(f"PASS — reduction {flop_reduction(len(grid), sizes):.1f}×")
