"""DG-ALB global-solver scaling — dense eigh vs O(N) density-matrix solves.

The crossover study (bench_dgalb.py) showed the reduced global solve `t_glob`
is cheap at 10^2-10^3 atoms (the ALB *build* dominates), but it is a dense
`eigh` — O(D^3) in the reduced dimension D = M_elem x N_elem. This study asks
the forward question toward *enormous* N:

  (1) At what atom count does the dense global eigh overtake the (linear) ALB
      build and become the bottleneck?  -> the ceiling of the dense-eigh strategy.
  (2) Does an O(N) density-matrix solver on the BLOCK-SPARSE ALB Hamiltonian
      (Chebyshev Fermi-operator expansion, or McWeeny purification) push that
      ceiling out?  -> the route to 10^4+ atoms.

The DG-ALB Hamiltonian is block-sparse: N_elem elements, an M_elem x M_elem
block each, coupled only to spatial neighbours (interior-penalty faces). So:

  dense eigh    REAL, measured. `torch.linalg.eigvalsh` on the full D x D matrix
                (gated to skip when it would exceed the memory budget).
  FOE / purify  Cost model grounded in a REAL measurement. Both build f(H) (the
                density matrix) by repeated block-sparse matmuls of the SAME
                sparsity pattern. The dominant cost per matmul is a batch of
                small M_elem x M_elem block products; we MEASURE that batched
                `bmm` at the right count and multiply by the iteration count
                (n_cheby for FOE; 2 matmuls x purify_iters for purification).
                Batched bmm is how you'd really run it -> a realistic, mildly
                conservative estimate (a truncated sparse-sparse product with
                locality would do fewer block products, not more).

Block-product count per density-matrix matmul, C = A@B truncated to the pattern:
each of the (N_elem x avg_deg) output blocks sums over ~avg_deg shared
neighbours -> ~N_elem x avg_deg^2 block multiplies. avg_deg = 1 (self) + 6 faces
= 7 for a 3D element grid (fewer on the surface; we use the bulk value, again
conservative). The count is LINEAR in N_elem -> the O(N) signature, vs the
dense eigh's O(D^3) = O(N^3).

What is REAL vs MODELLED:
  * dense eigh timings and the per-step block-bmm timing are measured on-box.
  * n_cheby and purify_iters are inputs; f(H) accuracy is NOT computed here
    (this is a solver-cost study). n_cheby for FOE scales ~ spectral-width /
    (k_B T); pick it to bracket cold vs warm metals.
  * The block-sparse step ignores the (cheap, O(N)) index bookkeeping and
    assumes the truncated pattern stays fixed — the standard O(N) assumption.

Self-contained: torch only. Runs anywhere; use asus for the large-D dense eigh.

Run:  uv run python benchmarks/dgalb_crossover/bench_dgalb_solver.py \
          --atoms 64,216,512,1000,4000,10000 --m-elem 64 --n-cheby 500 --threads 8
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

CDTYPE = torch.complex128


def _cubeish(cells: int):
    best = (cells, 1, 1)
    for nx in range(1, cells + 1):
        if cells % nx:
            continue
        rem = cells // nx
        for ny in range(1, rem + 1):
            if rem % ny:
                continue
            nz = rem // ny
            if max(nx, ny, nz) - min(nx, ny, nz) < max(best) - min(best):
                best = (nx, ny, nz)
    return best


def _time_bmm(count, m, budget_gb, reps=3):
    """Wall time of a batch of `count` complex128 M x M @ M x M products, run
    in memory-bounded chunks (this is the dominant per-matmul cost of a
    block-sparse density-matrix step). Returns seconds for the full `count`."""
    per = 3 * m * m * 16 / 1e9                        # A, B, C blocks
    chunk = max(1, int(budget_gb / max(per, 1e-12)))
    torch.manual_seed(0)
    # warmup
    a = torch.randn(min(chunk, count), m, m, dtype=CDTYPE)
    b = torch.randn(min(chunk, count), m, m, dtype=CDTYPE)
    torch.bmm(a, b)
    t = 0.0
    for _ in range(reps):
        done = 0
        t0 = time.perf_counter()
        while done < count:
            c = min(chunk, count - done)
            a = torch.randn(c, m, m, dtype=CDTYPE)
            b = torch.randn(c, m, m, dtype=CDTYPE)
            torch.bmm(a, b)
            done += c
        t += time.perf_counter() - t0
    return t / reps


def _time_dense_eigh(D, budget_gb):
    if 3 * D * D * 16.0 / 1e9 > budget_gb:
        return float("nan"), "OOM(est)"
    torch.manual_seed(1)
    A = torch.randn(D, D, dtype=CDTYPE)
    A = A + A.conj().T
    t0 = time.perf_counter()
    torch.linalg.eigvalsh(A)
    dt = time.perf_counter() - t0
    del A
    return dt, ""


def run(atoms, atoms_per_elem, m_elem, avg_deg, n_cheby, purify_iters,
        t_alb_per_elem, budget_gb):
    n_elem = max(1, round(atoms / atoms_per_elem))
    D = m_elem * n_elem

    # one density-matrix matmul ~ N_elem * avg_deg^2 block products
    n_block = n_elem * avg_deg * avg_deg
    t_step = _time_bmm(n_block, m_elem, budget_gb)
    t_foe = n_cheby * t_step
    t_purify = purify_iters * 2 * t_step

    t_dense, dense_note = _time_dense_eigh(D, budget_gb)

    t_alb = t_alb_per_elem * n_elem          # measured-linear ALB build (bench_dgalb.py)

    # global-solve bottleneck onset: dense global overtakes the ALB build
    dense_dominates = (not np.isnan(t_dense)) and t_dense > t_alb
    return {
        "atoms": n_elem * atoms_per_elem, "n_elem": n_elem, "D": D,
        "t_alb": t_alb, "t_dense": t_dense, "dense_note": dense_note,
        "t_step": t_step, "t_foe": t_foe, "t_purify": t_purify,
        "dense_dominates": dense_dominates,
        "best_global": min([x for x in [t_dense if not np.isnan(t_dense) else None,
                                        t_foe, t_purify] if x is not None]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", default="64,216,512,1000,4000,10000,30000",
                    help="atom counts to sweep")
    ap.add_argument("--atoms-per-elem", type=int, default=8, help="core atoms per DG element")
    ap.add_argument("--m-elem", type=int, default=64, help="ALBs per element (D = m_elem*N_elem)")
    ap.add_argument("--avg-deg", type=int, default=7, help="blocks/row: self + 6 faces")
    ap.add_argument("--n-cheby", type=int, default=500, help="Chebyshev terms for FOE")
    ap.add_argument("--purify-iters", type=int, default=25, help="McWeeny purification iters")
    ap.add_argument("--t-alb-per-elem", type=float, default=3.5,
                    help="measured ALB build s/element (bench_dgalb.py: ~3.5)")
    ap.add_argument("--mem-budget-gb", type=float, default=9.0)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    print(f"# threads={torch.get_num_threads()}  M_elem={args.m_elem}  avg_deg={args.avg_deg}  "
          f"n_cheby={args.n_cheby}  purify_iters={args.purify_iters}  "
          f"t_alb/elem={args.t_alb_per_elem}s  (fp64)", flush=True)
    print("# t_dense=dense eigh (REAL); t_foe/t_purify=block-sparse DM solve "
          "(measured bmm x iters); t_alb=measured-linear ALB build", flush=True)
    hdr = ("atoms", "n_elem", "D", "t_alb_s", "t_dense_s", "t_step_ms", "t_foe_s",
           "t_purify_s", "dense>build?", "best_global_s")
    print(("{:>7} {:>6} {:>7} {:>9} {:>10} {:>9} {:>9} {:>10} {:>12} {:>13}").format(*hdr),
          flush=True)
    for tok in args.atoms.split(","):
        r = run(int(tok), args.atoms_per_elem, args.m_elem, args.avg_deg,
                args.n_cheby, args.purify_iters, args.t_alb_per_elem, args.mem_budget_gb)
        dense = r["dense_note"] or f"{r['t_dense']:.2f}"
        print(("{:>7} {:>6} {:>7} {:>9.1f} {:>10} {:>9.2f} {:>9.1f} {:>10.1f} {:>12} {:>13.1f}").format(
            r["atoms"], r["n_elem"], r["D"], r["t_alb"], dense, r["t_step"] * 1e3,
            r["t_foe"], r["t_purify"], "YES" if r["dense_dominates"] else "no",
            r["best_global"]), flush=True)


if __name__ == "__main__":
    main()
