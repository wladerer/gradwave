"""Track 3b — Dependency-FREE symmetry block-diagonalization (no Sage, no tables).

Track 3 used a hardcoded C4v character table to build the irrep projectors. That
table (and its space-group generalization) was the ONLY reason Track 3 named Sage.
This shows the table is unnecessary: the symmetry-adapted basis can be built from
the group OPERATIONS alone, via linear algebra.

Key fact (Schur / group-algebra center): the CLASS SUMS  C_i = Σ_{g∈class_i} D(g)
span the center of the representation. They commute with the group AND with H, and
a generic real combination  T = Σ_i r_i C_i  is a scalar ω_λ on each irrep type λ,
with the ω_λ generically distinct. So eigendecomposing ONE matrix T groups the
basis by irrep type — exactly the isotypic blocks — using no characters at all.

Everything here is numpy: conjugacy classes are computed from the operations
(matrix products), class sums from permutation matrices, blocks from one eigh.

Real crystals: the operations come from spglib (already a gradwave dep, in
symmetry.py) — NOT Sage. Fractional translations (nonsymmorphic groups) enter only
as phases in D(g); the class sums still commute with H and still block H, because
this construction NEVER classifies irreps, so it needs no ray/projective-rep
machinery. Sage buys convenience (irrep LABELS), never the block-diagonalization.

Run:  uv run python experiments/symbolic/track3b_commutant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track3_symmetry import build_hamiltonian  # noqa: E402  (sibling module)

# C4v operations as 2×2 integer matrices acting on the G-index (n1,n2).
# (Same 8 ops as track3, expressed as matrices so we can multiply/invert them.)
R = {
    "E":    np.array([[1, 0], [0, 1]]),
    "C4":   np.array([[0, -1], [1, 0]]),
    "C4^3": np.array([[0, 1], [-1, 0]]),
    "C2":   np.array([[-1, 0], [0, -1]]),
    "sx":   np.array([[1, 0], [0, -1]]),
    "sy":   np.array([[-1, 0], [0, 1]]),
    "sd":   np.array([[0, 1], [1, 0]]),
    "sd'":  np.array([[0, -1], [-1, 0]]),
}


def _key(m):
    return tuple(m.flatten().tolist())


def conjugacy_classes():
    """Group the 8 ops into conjugacy classes { h g h⁻¹ } — from the group
    multiplication alone, no character table."""
    by_key = {_key(m): name for name, m in R.items()}
    classes, seen = [], set()
    for g in R:
        if g in seen:
            continue
        orbit = set()
        for h in R:
            Hm = R[h]
            Hinv = np.rint(np.linalg.inv(Hm)).astype(int)  # orthogonal → integer inverse
            orbit.add(by_key[_key(Hm @ R[g] @ Hinv)])
        classes.append(sorted(orbit))
        seen |= orbit
    return classes


def perm_matrix_R(Rmat, grid, idx):
    """Permutation rep D(g) on plane waves from the 2×2 op matrix."""
    N = len(grid)
    D = np.zeros((N, N))
    for i, g in enumerate(grid):
        img = (int(Rmat[0, 0] * g[0] + Rmat[0, 1] * g[1]),
               int(Rmat[1, 0] * g[0] + Rmat[1, 1] * g[1]))
        D[idx[img], i] = 1.0
    return D


def cluster_by_value(w, tol=1e-6):
    """Split sorted eigenvalues into groups (contiguous within tol) → block sizes."""
    order = np.argsort(w)
    ws = w[order]
    boundaries = [0]
    for i in range(1, len(ws)):
        if ws[i] - ws[i - 1] > tol:
            boundaries.append(i)
    boundaries.append(len(ws))
    sizes = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    return order, sizes


def main() -> None:
    M = 3
    H, grid, idx = build_hamiltonian(M)
    N = len(grid)
    print("=== Track 3b: dependency-free block-diagonalization (no character table) ===\n")
    print(f"square lattice, C4v, {N} plane waves at Γ\n")

    classes = conjugacy_classes()
    print(f"conjugacy classes computed from the ops (no tables): {len(classes)}")
    for c in classes:
        print(f"  {{{', '.join(c)}}}")

    D = {name: perm_matrix_R(R[name], grid, idx) for name in R}

    # class sums C_i = Σ_{g∈class} D(g) — these span the center, commute with H
    Csums = [sum(D[g] for g in cls) for cls in classes]
    max_commute = max(np.abs(C @ H - H @ C).max() for C in Csums)
    print(f"\nmax |[class-sum, H]| = {max_commute:.2e}  "
          f"(class sums commute with H ⇒ they block it)\n")

    # ONE generic real combination of class sums → isotypic blocks
    rng = np.random.default_rng(0)
    T = sum(float(r) * C for r, C in zip(rng.standard_normal(len(Csums)), Csums))
    T = 0.5 * (T + T.T)  # symmetric (C4v classes are ambivalent); real spectrum
    wT, U = np.linalg.eigh(T)
    order, sizes = cluster_by_value(wT)
    U = U[:, order]

    # transform H; measure inter-block leakage
    B = U.T @ H @ U
    off = B.copy()
    s = 0
    for r in sizes:
        off[s:s + r, s:s + r] = 0.0
        s += r
    print(f"blocks from eigen-clustering of T (Σr_i·C_i): sizes {sorted(sizes)}")
    print(f"  (Track 3 character-table blocks were [3, 6, 6, 10, 24] — same multiset)")
    print(f"max inter-block element of UᵀHU = {np.abs(off).max():.2e}")

    # correctness: block spectra == full spectrum
    ev_full = np.sort(np.linalg.eigvalsh(H))
    ev_block, s = [], 0
    for r in sizes:
        ev_block += list(np.linalg.eigvalsh(B[s:s + r, s:s + r]))
        s += r
    ev_err = np.abs(ev_full - np.sort(ev_block)).max()
    print(f"eigenvalues: max |full − per-block| = {ev_err:.2e}")

    ok = (max_commute < 1e-10 and np.abs(off).max() < 1e-9 and ev_err < 1e-9
          and sorted(sizes) == [3, 6, 6, 10, 24])
    print(f"\n{'PASS' if ok else 'FAIL'}: identical blocks to Track 3, built from the "
          f"group ops + one eigh. No character table, no Sage.")
    print("\nnonsymmorphic note: fractional translations ride along as phases in "
          "D(g); the class sums still commute with H and still block it — no "
          "ray-rep machinery, because we never label the irreps.")


if __name__ == "__main__":
    main()
