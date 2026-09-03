"""Cached, density-INDEPENDENT symmetry-adapted basis + compiled sparse assembly.

The adapted basis U(k) depends only on (crystal, k) — NOT on the density / v_eff —
so it is built ONCE and reused for every diagonalization of H(k) (every SCF
iteration, every parametric/DFPT solve, every property at fixed geometry).

U is stored as a scipy CSC (each column is a star-local combination of ≤|G_k|
plane waves → sparse). The reduced blocks are assembled as B = Uᴴ H U with two
C-backed sparse·dense products (cost O(N²·s̄), s̄ = mean star size), instead of the
O(N³) dense matmul the Python prototype used. Then eigh each diagonal block.

This turns the flop win (Σnλ³ ≪ N³) into a walltime win once the basis is amortized.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from blockdiag import cluster_sorted, conjugacy_classes
from blockdiag_matfree import orbits, sparse_reps


class SymBasis:
    """Density-independent symmetry-adapted basis for H(k); build once, reuse."""

    def __init__(self, miller, ops, Ws, k_frac, seed: int = 0):
        self.npw = len(miller)
        perms, phases = sparse_reps(miller, ops, np.asarray(k_frac, float))
        stars = orbits(perms, self.npw)
        classes = conjugacy_classes(Ws)
        rng = np.random.default_rng(seed)
        a = rng.standard_normal(len(classes))
        b = rng.standard_normal(len(classes))

        rows, cols, vals, tags = [], [], [], []
        col = 0
        sum_s3, max_s = 0, 0
        for S in stars:
            s = len(S)
            sum_s3 += s ** 3
            max_s = max(max_s, s)
            Sarr = np.asarray(S)
            posmap = {g: k for k, g in enumerate(S)}
            # local generic Hermitian central element on this star
            T = np.zeros((s, s), dtype=complex)
            for ci, cls in enumerate(classes):
                C = np.zeros((s, s), dtype=complex)
                for gi in cls:
                    perm, phase = perms[gi], phases[gi]
                    lr = np.array([posmap[int(perm[j])] for j in S])
                    lc = np.arange(s)
                    np.add.at(C, (lr, lc), phase[Sarr])
                T += a[ci] * (C + C.conj().T) / 2 + b[ci] * (C - C.conj().T) / (2j)
            w, V = np.linalg.eigh(T)
            for c in range(s):
                rows.extend(Sarr.tolist())
                cols.extend([col] * s)
                vals.extend(V[:, c].tolist())
                tags.append(w[c])
                col += 1

        tags = np.array(tags)
        order = np.argsort(tags)
        # remap columns into tag-sorted (block) order
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        cols = inv[np.asarray(cols)]
        self.U = sp.csc_matrix((np.asarray(vals), (np.asarray(rows), cols)),
                               shape=(self.npw, self.npw))
        self.Uh = self.U.getH().tocsr()
        self.sizes = cluster_sorted(tags[order],
                                    tol=1e-6 * max(1.0, float(np.abs(tags).max())))
        self.max_star = max_s
        self.sum_s3 = sum_s3
        self.n_stars = len(stars)

    def block_diagonalize(self, H: np.ndarray):
        """Full spectrum of H via the cached basis. Sparse·dense assembly + eigh."""
        # B = Uᴴ H U, both products are sparse·dense (C-backed), O(N²·s̄).
        # H @ U via sparse-left transpose identity: H@U = (Uᵀ @ Hᵀ)ᵀ = (Uᵀ @ conj(H))ᵀ
        # for Hermitian H (Hᵀ = conj(H)).  Uᵀ is a plain (non-conjugate) transpose.
        HU = (self.U.T @ H.conj()).T               # = H @ U
        B = self.Uh @ HU                           # (N×N) dense, block-diagonal
        evals = []
        s = 0
        for r in self.sizes:
            blk = B[s:s + r, s:s + r]
            blk = 0.5 * (blk + blk.conj().T)
            evals.extend(np.linalg.eigvalsh(blk).tolist())
            s += r
        return np.sort(np.array(evals))

    def block_eig(self, H: np.ndarray):
        """Full spectrum AND eigenvectors (in the plane-wave basis) of H via the
        cached basis. For consumers that need |ψ⟩ (e.g. optical matrix elements)."""
        HU = (self.U.T @ H.conj()).T
        B = self.Uh @ HU
        evals = np.empty(self.npw)
        vecs = np.empty((self.npw, self.npw), dtype=complex)
        s = 0
        for r in self.sizes:
            blk = 0.5 * (B[s:s + r, s:s + r] + B[s:s + r, s:s + r].conj().T)
            w, y = np.linalg.eigh(blk)                     # block eigenpairs
            evals[s:s + r] = w
            # lift block eigenvectors back to the plane-wave basis: ψ = U_block y
            vecs[:, s:s + r] = (self.U[:, s:s + r] @ y).toarray() \
                if hasattr(self.U[:, s:s + r] @ y, "toarray") else self.U[:, s:s + r] @ y
            s += r
        order = np.argsort(evals)
        return evals[order], vecs[:, order]


if __name__ == "__main__":
    # validate against dense baseline on a random Hermitian with C4v symmetry
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from track3_symmetry import build_hamiltonian
    from track3b_commutant import R as C4V_R

    H, grid, idx = build_hamiltonian(4)  # bigger toy
    miller3 = np.c_[np.array(grid), np.zeros(len(grid), int)]
    Ws = [np.rint(np.pad(np.asarray(R), ((0, 1), (0, 1))) + np.diag([0, 0, 1])).astype(int)
          for R in C4V_R.values()]
    ops = [{"Winv_t": np.rint(np.linalg.inv(W).T).astype(int),
            "g0": np.zeros(3, int), "w": np.zeros(3)} for W in Ws]
    sb = SymBasis(miller3, ops, Ws, np.zeros(3))
    Hc = H.astype(complex)
    ev = sb.block_diagonalize(Hc)
    ev0 = np.sort(np.linalg.eigvalsh(H))
    print(f"C4v: N={len(grid)}, blocks={sb.sizes}, err={np.abs(ev-ev0).max():.1e}")
    assert np.abs(ev - ev0).max() < 1e-9
    print("PASS")
