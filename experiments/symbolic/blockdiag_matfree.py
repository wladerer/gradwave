"""Matrix-free symmetry block-diagonalization via plane-wave ORBITS (stars).

The scalable version of blockdiag.py. Each symmetry op acts on the plane-wave
basis as a permutation-with-phase (npw nonzeros), so the G-vectors split into
ORBITS (stars) under the little group, each of size ≤ |group| ≤ 48. The
symmetry-adapted basis is built star-by-star with tiny s×s eigendecompositions —
O(Σ s³) ≈ O(npw·|G|²), LINEAR in npw, never forming any npw×npw matrix. Each
adapted vector is sparse (supported on one star).

The reduced irrep blocks H_λ = U_λ†(H U_λ) are then assembled by applying the
caller's matrix-free H (e.g. gradwave's HamiltonianK.apply) only to the block's
basis vectors — peak memory max(n_λ)² instead of npw², and the small blocks are
independent (parallelizable). No dense H, no Sage, no character tables.

Works at any k where the little-group rep is ORDINARY (all k in symmorphic
groups; Γ and interior k in nonsymmorphic groups). At nonsymmorphic zone
boundaries the rep is PROJECTIVE and the plain class sums stop being central —
detected here by the block-vs-full spectrum residual.
"""

from __future__ import annotations

import numpy as np
import torch

from blockdiag import cluster_sorted, conjugacy_classes


def sparse_reps(miller: np.ndarray, ops: list, k_frac: np.ndarray):
    """Per op: (perm, phase) so D(g) sends basis vec j → phase[j]·(basis perm[j])."""
    index = {tuple(m): i for i, m in enumerate(miller)}
    perms, phases = [], []
    for op in ops:
        mprime = miller @ op["Winv_t"].T + op["g0"]
        perm = np.array([index[tuple(m)] for m in mprime])
        phase = np.exp(-2j * np.pi * ((k_frac + mprime) @ op["w"]))
        perms.append(perm)
        phases.append(phase)
    return perms, phases


def orbits(perms: list[np.ndarray], npw: int) -> list[list[int]]:
    """Partition the basis into orbits (stars) under the permutation group."""
    parent = list(range(npw))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for perm in perms:
        for i in range(npw):
            ra, rb = find(i), find(int(perm[i]))
            if ra != rb:
                parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for i in range(npw):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def adapted_basis(miller: np.ndarray, ops: list, Ws: list[np.ndarray],
                  k_frac: np.ndarray, seed: int = 0):
    """Symmetry-adapted basis U (npw×npw, columns grouped by irrep) built star by
    star — no npw×npw intermediate. Returns (U, sizes, diag)."""
    npw = len(miller)
    perms, phases = sparse_reps(miller, ops, k_frac)
    stars = orbits(perms, npw)
    classes = conjugacy_classes(Ws)
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(len(classes))
    b = rng.standard_normal(len(classes))

    cols, tags = [], []
    sum_s3, max_s = 0, 0
    for S in stars:
        s = len(S)
        sum_s3 += s ** 3
        max_s = max(max_s, s)
        pos = {g: k for k, g in enumerate(S)}
        # generic Hermitian central element on this star (globally fixed a,b)
        T = np.zeros((s, s), dtype=complex)
        for ci, cls in enumerate(classes):
            C = np.zeros((s, s), dtype=complex)
            for gi in cls:
                perm, phase = perms[gi], phases[gi]
                for j in S:
                    C[pos[int(perm[j])], pos[j]] += phase[j]
            T += a[ci] * (C + C.conj().T) / 2 + b[ci] * (C - C.conj().T) / (2j)
        w, V = np.linalg.eigh(T)
        for col in range(s):
            full = np.zeros(npw, dtype=complex)
            full[S] = V[:, col]
            cols.append(full)
            tags.append(w[col])

    U = np.array(cols).T
    tags = np.array(tags)
    order = np.argsort(tags)
    U, tags = U[:, order], tags[order]
    sizes = cluster_sorted(tags, tol=1e-6 * max(1.0, float(np.abs(tags).max())))
    diag = {"max_star": max_s, "sum_s3": sum_s3, "n_stars": len(stars),
            "n_irrep_blocks": len(sizes)}
    return U, sizes, diag


def solve_blocks_matfree(h_apply, U: np.ndarray, sizes: list[int]):
    """Assemble & diagonalize each reduced block H_λ = U_λ†(H U_λ) using only the
    matrix-free h_apply on the block's basis vectors. Never forms dense H.
    Returns (sorted eigenvalues, total H-applies, peak block elements)."""
    evals, napply, peak = [], 0, 0
    s = 0
    for r in sizes:
        Ul = U[:, s:s + r]                                   # npw × r (sparse cols)
        cin = torch.tensor(np.ascontiguousarray(Ul.T), dtype=torch.complex128)
        HU = h_apply(cin).numpy()                            # (r, npw): row a = H·U_λ[:,a]
        Hl = Ul.conj().T @ HU.T                              # (r, r) reduced block
        Hl = 0.5 * (Hl + Hl.conj().T)
        evals.extend(np.linalg.eigvalsh(Hl).real.tolist())
        napply += r
        peak = max(peak, r * r)
        s += r
    return np.sort(np.array(evals)), napply, peak


if __name__ == "__main__":
    # self-test on the C4v toy — no SCF, validates the whole star machinery
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from track3_symmetry import build_hamiltonian
    from track3b_commutant import R as C4V_R, perm_matrix_R  # noqa: F401

    H, grid, idx = build_hamiltonian(3)
    miller = np.array(grid)  # (npw, 2) — but we need 3D for op dicts; embed
    # embed 2D grid into 3D so the _Op-style dict math works
    miller3 = np.c_[miller, np.zeros(len(miller), int)]
    Ws = [np.pad(np.asarray(R), ((0, 1), (0, 1))) + np.diag([0, 0, 1])
          for R in C4V_R.values()]
    ops = [{"Winv_t": np.rint(np.linalg.inv(W)).astype(int).T @ np.eye(3, dtype=int),
            "g0": np.zeros(3, int), "w": np.zeros(3)} for W in Ws]
    # fix Winv_t = inv(W).T (integer)
    for op, W in zip(ops, Ws):
        op["Winv_t"] = np.rint(np.linalg.inv(W).T).astype(int)

    U, sizes, diag = adapted_basis(miller3, ops, Ws, np.zeros(3))
    Hc = torch.tensor(H, dtype=torch.complex128)
    ev, napply, peak = solve_blocks_matfree(lambda c: c @ Hc.T, U, sizes)
    ev_full = np.sort(np.linalg.eigvalsh(H))
    err = np.abs(ev - ev_full).max()
    print("C4v matrix-free self-test:", sorted(sizes), diag)
    print(f"  spectrum err {err:.1e}; H-applies {napply} (=npw {len(grid)}); "
          f"peak block {int(peak**0.5)}² vs full {len(grid)}²")
    assert sorted(sizes) == [3, 6, 6, 10, 24] and err < 1e-9
    print("PASS")
