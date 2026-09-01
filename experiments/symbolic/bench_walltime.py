"""Walltime benchmark: does symmetry block-diagonalization actually pay off?

The symmetry win is in the DIAGONALIZATION: one N×N eigenproblem → Σ small blocks
(Σn³ ≪ N³). This is a real walltime win for DENSE / near-full diagonalization of
H(k) (exact diag, spectral methods, many-band DOS) — NOT for gradwave's iterative
Davidson, whose per-apply FFT cost symmetry does not reduce. There is a crossover:
small N is dominated by Python/basis overhead, large N by the Σn³ saving.

We sweep N by building H(Γ) on smaller G-spheres from ONE high-cutoff converged
v_eff, and time three paths that all yield the FULL spectrum from the same
symmetrized H:

  A. baseline      : numpy eigvalsh(H)                       (full N×N)
  B. sym (dense H) : adapted_basis + U_λ†HU_λ + eigh blocks  (reuse dense H)
  C. sym (mat-free): adapted_basis + solve_blocks_matfree    (never forms dense H)

Run:  uv run python experiments/symbolic/bench_walltime.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.api import build_system, run_scf
from gradwave.core.hamiltonian import HamiltonianK, projectors
from gradwave.grids import build_gsphere
from gradwave.inputs import load_input
from gradwave.postscf._kb import projector_data_at_k, species_projector_tables
from gradwave.postscf.irreps import little_group
from gradwave.symmetry import find_spacegroup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blockdiag_matfree import adapted_basis, solve_blocks_matfree, sparse_reps  # noqa: E402

HERE = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def median_time(fn, repeats=3):
    ts, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def spacegroup(system):
    grid = system.grid
    frac = system.positions.cpu().numpy() @ np.linalg.inv(grid.cell)
    return find_spacegroup(grid.cell, frac, system.species_of_atom)


def symmetrize(H, miller, ops, k_frac):
    perms, phases = sparse_reps(miller, ops, np.asarray(k_frac, float))
    Hs = np.zeros_like(H)
    for P, ph in zip(perms, phases):
        Hs[np.ix_(P, P)] += (ph[:, None] * H) * np.conj(ph)[None, :]
    return Hs / len(ops)


def sym_dense(H, U, sizes):
    """Path B: reduced blocks from dense H, eigh each."""
    evals, s = [], 0
    for r in sizes:
        Ul = U[:, s:s + r]
        Hl = Ul.conj().T @ (H @ Ul)
        Hl = 0.5 * (Hl + Hl.conj().T)
        evals.extend(np.linalg.eigvalsh(Hl).tolist())
        s += r
    return np.sort(np.array(evals))


def main() -> None:
    print("Walltime benchmark — symmetry block-diagonalization of H(Γ)")
    print("(times are ms; A=full eigh, B=sym reuse dense H, C=sym matrix-free)\n")

    for name, yaml, sweep in [
        ("diamond", "diamond_bench.yaml", [300, 500, 700, 900]),
        ("bcc Fe", "fe_bench.yaml", [500, 700, 900]),
    ]:
        inp = load_input(str(HERE / yaml))
        system = build_system(inp)
        t0 = time.perf_counter()
        res = run_scf(inp, system=system, verbose=False)
        v_eff = res.v_eff if res.v_eff.ndim == 3 else res.v_eff[0]
        sg = spacegroup(system)
        grid = system.grid
        print(f"=== {name} (SCF {time.perf_counter()-t0:.0f}s, grid {tuple(grid.shape)}, "
              f"ecut {system.ecut:.0f} eV, {sg.international}) ===")
        print(f"{'N':>5} {'blks':>4} {'maxblk':>6} {'1/R':>6} "
              f"{'A:eigh':>9} {'B:sym':>8} {'C:matfree':>9} "
              f"{'A/B':>6} {'A/C':>6} {'specerr':>8}")

        for se in sweep:
            if se > system.ecut:
                continue
            sph = build_gsphere(grid, se, np.zeros(3))
            npw = sph.npw
            miller = sph.miller.cpu().numpy()
            ops = little_group(np.zeros(3), sg, grid.cell)
            Ws = [np.asarray(op["W"], int) for op in ops]
            beta_ls, dij_sp = species_projector_tables(system.upfs, None)
            pd = projector_data_at_k(sph, system.species_of_atom, system.upfs,
                                     beta_ls, dij_sp, grid.volume, None)
            p = projectors(pd, system.positions)
            h = HamiltonianK(sph, grid.shape, v_eff, pd, p)
            H = h.apply(torch.eye(npw, dtype=torch.complex128)).transpose(0, 1).numpy()
            H = 0.5 * (H + H.conj().T)
            H = symmetrize(H, miller, ops, np.zeros(3))

            U, sizes, _ = adapted_basis(miller, ops, Ws, np.zeros(3))
            R = npw ** 3 / sum(s ** 3 for s in sizes)

            tA, evA = median_time(lambda: np.sort(np.linalg.eigvalsh(H)))
            # Path B includes the basis construction (fair: it's part of the method)
            tB, evB = median_time(
                lambda: sym_dense(H, *adapted_basis(miller, ops, Ws, np.zeros(3))[:2]))
            tC, evC = median_time(
                lambda: solve_blocks_matfree(
                    h.apply, *adapted_basis(miller, ops, Ws, np.zeros(3))[:2])[0],
                repeats=2)
            err = float(np.abs(evA - evB).max())
            print(f"{npw:5d} {len(sizes):4d} {max(sizes):6d} {1/R:6.3f} "
                  f"{tA*1e3:8.1f} {tB*1e3:7.1f} {tC*1e3:8.1f} "
                  f"{tA/tB:5.1f}× {tA/tC:5.1f}× {err:8.1e}")
        print()


if __name__ == "__main__":
    main()
