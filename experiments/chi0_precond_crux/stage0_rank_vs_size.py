"""Stage 0 — rank-vs-size diagnostic for the χ₀ preconditioning crux.

On a growing inhomogeneous series (Al(100) slabs at FIXED vacuum, plus fcc-Al
bulk as the homogeneous control) converge the SCF, then extract the top of the
eigenspectrum of the screening operator M = K_Hxc·χ₀ (the Jacobian of the outer
fixed point) via a single-vector Arnoldi factorization of the SHIPPED
``screening_apply`` matvec. Count

    n_dangerous = #{ |λ_i(M)| > 0.7 }

- GO signal:  n_dangerous SATURATES at O(tens) as N_atoms grows — a low-rank
  operator can capture the slow subspace, so a Woodbury sketch keeps up.
- NO-GO signal: n_dangerous grows ∝ N_atoms — the volume rank-explosion; no
  fixed-rank sketch can track it.

Krylov is the natural extractor for the extremal (largest-|λ|) eigenpairs, which
are exactly the "dangerous" ones. We report the count at two krylov budgets so a
count pinned at the budget (still rising) is visible as such.

Usage (asus only):
    uv run python experiments/chi0_precond_crux/stage0_rank_vs_size.py \
        --krylov 64 --out experiments/chi0_precond_crux/results/stage0.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.chi0_precond_crux.common import (
    al_bulk_system,
    al_slab_system,
    xc_for,
)
from gradwave.scf.loop import scf
from gradwave.scf.soft_mode import screening_apply
from gradwave.solvers.deflation import (
    _rand_field_like,
    arnoldi_factorization,
)


def ritz_spectrum(res, xc, *, krylov: int, seed: int, chi0_tol: float):
    """Ritz eigenvalues of M = K_Hxc·χ₀ from a krylov-step Arnoldi."""
    m = screening_apply(res, xc, chi0_tol=chi0_tol)
    v0 = _rand_field_like(res.rho, seed)
    v_list, h, reached = arnoldi_factorization(m, v0, krylov=krylov)
    evals = torch.linalg.eigvals(h[:reached, :reached])
    return evals, reached


def count_dangerous(evals: torch.Tensor, thresh: float) -> int:
    return int((evals.abs() > thresh).sum())


def analyze(res, xc, *, krylov: int, krylov_hi: int, seed: int, chi0_tol: float):
    evals, reached = ritz_spectrum(res, xc, krylov=krylov, seed=seed,
                                   chi0_tol=chi0_tol)
    evals_hi, reached_hi = ritz_spectrum(res, xc, krylov=krylov_hi, seed=seed,
                                         chi0_tol=chi0_tol)
    mags = evals.abs()
    reals = sorted((float(e.real) for e in evals), reverse=True)
    return {
        "krylov_lo": reached,
        "krylov_hi": reached_hi,
        "spectral_radius": float(mags.max()),
        "n_gt_0.5_lo": count_dangerous(evals, 0.5),
        "n_gt_0.7_lo": count_dangerous(evals, 0.7),
        "n_gt_0.9_lo": count_dangerous(evals, 0.9),
        "n_gt_0.5_hi": count_dangerous(evals_hi, 0.5),
        "n_gt_0.7_hi": count_dangerous(evals_hi, 0.7),
        "n_gt_0.9_hi": count_dangerous(evals_hi, 0.9),
        "max_real": reals[0],
        "top5_real": reals[:5],
    }


def run_case(name: str, system, natoms: int, nspin: int, *, krylov: int,
             krylov_hi: int, seed: int, chi0_tol: float, etol: float,
             rhotol: float, max_iter: int):
    xc = xc_for(nspin)
    t0 = time.time()
    res = scf(system, xc, smearing="gaussian", width=0.1, etol=etol,
              rhotol=rhotol, max_iter=max_iter, nspin=nspin, verbose=False)
    t_scf = time.time() - t0
    t1 = time.time()
    spec = analyze(res, xc, krylov=krylov, krylov_hi=krylov_hi, seed=seed,
                   chi0_tol=chi0_tol)
    t_arn = time.time() - t1
    row = {
        "name": name,
        "natoms": natoms,
        "nspin": nspin,
        "scf_iters": int(res.n_iter),
        "scf_converged": bool(res.converged),
        "energy": float(res.energies.total),
        "t_scf_s": round(t_scf, 1),
        "t_arnoldi_s": round(t_arn, 1),
        **spec,
    }
    print(f"{name:24s} N={natoms:2d}  ρ(M)={spec['spectral_radius']:5.2f}  "
          f"n>0.7={spec['n_gt_0.7_lo']:2d}(K{spec['krylov_lo']})/"
          f"{spec['n_gt_0.7_hi']:2d}(K{spec['krylov_hi']})  "
          f"max_real={spec['max_real']:+.3f}  "
          f"conv={res.converged} {res.n_iter}it  {t_scf:.0f}s+{t_arn:.0f}s",
          flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--krylov", type=int, default=64)
    ap.add_argument("--krylov-hi", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chi0-tol", type=float, default=1e-5)
    ap.add_argument("--etol", type=float, default=1e-9)
    ap.add_argument("--rhotol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--ecut", type=float, default=25.0)
    ap.add_argument("--kmesh-z", type=int, default=1)
    ap.add_argument("--kmesh-xy", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--with-bulk", action="store_true")
    ap.add_argument("--out", type=str,
                    default="experiments/chi0_precond_crux/results/stage0.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def save():
        out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))

    kw = dict(krylov=args.krylov, krylov_hi=args.krylov_hi, seed=args.seed,
              chi0_tol=args.chi0_tol, etol=args.etol, rhotol=args.rhotol,
              max_iter=args.max_iter)

    if args.with_bulk:
        system, natoms = al_bulk_system(ecut_ry=args.ecut,
                                        kmesh=(8, 8, 8), use_symmetry=False)
        rows.append(run_case("fcc Al bulk (homog.)", system, natoms, 1, **kw))
        save()

    for nlay in args.layers:
        system, natoms = al_slab_system(
            nlay, ecut_ry=args.ecut,
            kmesh=(args.kmesh_xy, args.kmesh_xy, args.kmesh_z),
            use_symmetry=False)
        rows.append(run_case(f"Al(100) slab {nlay} layer", system, natoms, 1,
                             **kw))
        save()

    print(f"\nsaved {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
