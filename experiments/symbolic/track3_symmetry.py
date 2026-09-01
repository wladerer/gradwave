"""Track 3 — Symmetry-adapted plane waves block-diagonalize H (proof-of-concept).

The eigensolver is the dominant memory/time cost in PW-DFT, and the ONE lever that
shrinks it is symmetry. At a high-symmetry k-point, H(k) commutes with the little
group; group-theory projection operators

    P^Γ = (d_Γ/|G|) Σ_g χ^Γ(g)* D(g)

split the plane-wave basis into irrep subspaces that block-diagonalize H →
several small independent eigenproblems instead of one big one (Σ n_Γ³ ≪ N³).

This demonstrates the claim end-to-end on a concrete C4v square-lattice PW
Hamiltonian at Γ, using pure numpy + the hardcoded C4v character table. FULL SAGE
IS NEEDED for the real thing: space-group irreps and PROJECTIVE (ray) reps for
nonsymmorphic groups (fractional-translation phases) — spglib gives the operations
but not the irreps. This POC uses a symmorphic point group so no Sage is required.

Run:  uv run python experiments/symbolic/track3_symmetry.py   (no sympy needed)
"""

from __future__ import annotations

import numpy as np

# --- C4v: operations on integer G-index (n1,n2), grouped by class -------------
OPS = {
    "E":    lambda n: (n[0], n[1]),      # class E
    "C4":   lambda n: (-n[1], n[0]),     # class 2C4
    "C4^3": lambda n: (n[1], -n[0]),     #   "
    "C2":   lambda n: (-n[0], -n[1]),    # class C2
    "sx":   lambda n: (n[0], -n[1]),     # class 2σv
    "sy":   lambda n: (-n[0], n[1]),     #   "
    "sd":   lambda n: (n[1], n[0]),      # class 2σd
    "sd'":  lambda n: (-n[1], -n[0]),    #   "
}
OP_CLASS = {"E": "E", "C4": "2C4", "C4^3": "2C4", "C2": "C2",
            "sx": "2sv", "sy": "2sv", "sd": "2sd", "sd'": "2sd"}
# C4v character table: rows = irreps, cols = classes (E, 2C4, C2, 2σv, 2σd)
CHAR = {  # (dim, {class: character})
    "A1": (1, {"E": 1, "2C4": 1, "C2": 1, "2sv": 1, "2sd": 1}),
    "A2": (1, {"E": 1, "2C4": 1, "C2": 1, "2sv": -1, "2sd": -1}),
    "B1": (1, {"E": 1, "2C4": -1, "C2": 1, "2sv": 1, "2sd": -1}),
    "B2": (1, {"E": 1, "2C4": -1, "C2": 1, "2sv": -1, "2sd": 1}),
    "E":  (2, {"E": 2, "2C4": 0, "C2": -2, "2sv": 0, "2sd": 0}),
}


def build_hamiltonian(M: int):
    """H(Γ) for a square lattice: |G|² kinetic + a C4v-symmetric local potential."""
    grid = [(n1, n2) for n1 in range(-M, M + 1) for n2 in range(-M, M + 1)]
    idx = {g: i for i, g in enumerate(grid)}
    N = len(grid)
    H = np.zeros((N, N))
    # V(ΔG): closed C4v shells → manifestly symmetric (from V(r)=cos x+cos y + …)
    Vshell = {1: -1.0, 2: -0.3}  # |Δn|² = 1 : (±1,0),(0,±1);  = 2 : (±1,±1)
    for a, ga in enumerate(grid):
        H[a, a] = ga[0] ** 2 + ga[1] ** 2  # kinetic (units of (2π/a)²)
        for b, gb in enumerate(grid):
            d2 = (ga[0] - gb[0]) ** 2 + (ga[1] - gb[1]) ** 2
            if d2 in Vshell:
                H[a, b] += Vshell[d2]
    return H, grid, idx


def perm_matrix(op, grid, idx):
    """Permutation rep D(g) of a point op on the plane-wave basis."""
    N = len(grid)
    D = np.zeros((N, N))
    for i, g in enumerate(grid):
        D[idx[op(g)], i] = 1.0
    return D


def main() -> None:
    M = 3
    H, grid, idx = build_hamiltonian(M)
    N = len(grid)
    print("=== Track 3: symmetry-adapted plane waves block-diagonalize H ===\n")
    print(f"square lattice, C4v, {2*M+1}×{2*M+1} = {N} plane waves at Γ\n")

    Dm = {name: perm_matrix(op, grid, idx) for name, op in OPS.items()}

    # sanity: H must commute with every group operation
    commute = max(np.abs(H @ Dm[n] - Dm[n] @ H).max() for n in OPS)
    print(f"max |[H, D(g)]| over the 8 ops = {commute:.2e}  "
          f"({'H is C4v-symmetric' if commute < 1e-12 else 'NOT symmetric!'})\n")

    # projection operators → orthonormal symmetry-adapted basis Q, irrep by irrep
    Q_cols, blocks = [], []
    for irrep, (dim, chi) in CHAR.items():
        P = np.zeros((N, N))
        for name, D in Dm.items():
            P += chi[OP_CLASS[name]] * D
        P *= dim / len(OPS)
        rank = int(round(np.trace(P)))
        if rank == 0:
            blocks.append((irrep, 0))
            continue
        U, S, _ = np.linalg.svd(P)          # P symmetric idempotent → range = top singulars
        Q_cols.append(U[:, :rank])
        blocks.append((irrep, rank))
    Q = np.hstack(Q_cols)

    # transform: B = Qᵀ H Q should be block-diagonal by irrep
    B = Q.T @ H @ Q
    # zero out within-irrep blocks to measure the OFF-diagonal (inter-irrep) leakage
    off = B.copy()
    s = 0
    for _, r in blocks:
        off[s:s + r, s:s + r] = 0.0
        s += r
    print("irrep blocks (block-diagonal structure of QᵀHQ):")
    for irrep, r in blocks:
        print(f"  {irrep:3s}: size {r}")
    sizes = [r for _, r in blocks if r]
    print(f"\nmax inter-irrep (off-block) element of QᵀHQ = {np.abs(off).max():.2e}\n")

    # correctness: eigenvalues from the blocks == eigenvalues of full H
    ev_full = np.sort(np.linalg.eigvalsh(H))
    ev_block = []
    s = 0
    for _, r in blocks:
        if r:
            ev_block += list(np.linalg.eigvalsh(B[s:s + r, s:s + r]))
        s += r
    ev_block = np.sort(ev_block)
    ev_err = np.abs(ev_full - ev_block).max()
    print(f"eigenvalues: max |full − per-block| = {ev_err:.2e}")

    # the payoff, quantified
    cost_full = N ** 3
    cost_block = sum(r ** 3 for r in sizes)
    print(f"\ndiagonalization cost proxy (Σ n³):")
    print(f"  full block : {N}³           = {cost_full:,}")
    print(f"  irrep blocks: {'+'.join(f'{r}³' for r in sizes)} = {cost_block:,}")
    print(f"  → {cost_full / cost_block:.1f}× fewer flops; "
          f"subspace memory set by max block {max(sizes)} not {N}")
    ok = commute < 1e-12 and np.abs(off).max() < 1e-10 and ev_err < 1e-10
    print(f"\n{'PASS' if ok else 'FAIL'}: symmetry projection block-diagonalizes H "
          f"exactly; blocks reproduce the full spectrum.")
    print("\n(E-irrep block further halves: its 2 partners decouple — an extra 2× "
          "not counted here. Real gain needs Sage for space-group irreps.)")


if __name__ == "__main__":
    main()
