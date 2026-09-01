"""Stage 1 — teacher-ceiling A/B for the χ₀ preconditioning crux.

Wire the EXACT inverse dielectric ε_ρ⁻¹ = (1 − χ₀·K_Hxc)⁻¹ (the shipped
``apply_chi0``/``apply_k_hxc`` + an ``anderson_solve``) as ``PulayMixer``'s
``precond_op`` and A/B the iterations-to-fp64 against the full shipped
production mixer. A truncated/sketched low-rank student can never beat the
exact operator, so this iteration cut BOUNDS the whole low-rank cluster.

Three runs from an IDENTICAL cold start, identical convergence gate:

  baseline_prod  johnson + local-TF (+ Stoner spin-precond for nspin=2)
                 — the real production mixer, the thing to beat;
  baseline_ctrl  pulay   + local-TF (+ Stoner) — SAME mixer as the teacher,
                 isolates the preconditioner from the mixer algorithm;
  teacher        pulay   + exact-dielectric precond_op (frozen at the
                 converged density; NO Stoner — the exact χ₀ subsumes it).

The frozen-at-solution χ₀ is the STRONGEST dielectric preconditioner (exact at
the fixed point). A cold start spends its first iterations outside the linear-
response regime where a frozen χ₀ is representative, so we report BOTH the total
iteration count and the asymptotic-tail count (iterations after the residual
first drops below 1e-3 — the linear regime a quasi-Newton preconditioner acts
in), the latter being the fair ceiling.

Checks (mandated): both runs converge to the IDENTICAL fixed point
(dE small (eV), d|ρ| ≤ 1e-4), and the FFT counter confirms the exact-operator
path's extra cost (a cheap truncated student would add only ~2·n_band window
FFTs per step, not a full Sternheimer solve).

Usage (asus only):
    uv run python experiments/chi0_precond_crux/stage1_teacher_ceiling.py \
        --system fe --kmesh 4 --out .../results/stage1_fe.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.chi0_precond_crux.common import (
    ExactDielectricPrecond,
    al_slab_system,
    fe_fm_system,
    xc_for,
)
from gradwave.core import opcount
from gradwave.scf.loop import scf


def _resid_logger():
    """mixer_hook that records the relative density residual each step."""
    hist: list[float] = []

    def hook(it, vin, vout):
        num = float(torch.linalg.norm(vout - vin))
        den = max(1.0, float(torch.linalg.norm(vin)))
        hist.append(num / den)

    return hook, hist


def tail_iters(hist: list[float], thresh: float = 1e-3) -> int:
    """Iterations after the residual first fell below ``thresh`` (linear tail)."""
    for i, r in enumerate(hist):
        if r < thresh:
            return max(0, len(hist) - i)
    return len(hist)


def _run(tag, system_fn, xc, scf_kw, *, precond_op=None):
    hook, hist = _resid_logger()
    prev = opcount.snapshot()
    t0 = time.time()
    res = scf(system_fn(), xc, mixer_hook=hook, precond_op=precond_op, **scf_kw)
    dt = time.time() - t0
    ffts = opcount.since(prev)["fft"]
    row = {
        "tag": tag,
        "iters": int(res.n_iter),
        "converged": bool(res.converged),
        "energy_eV": float(res.energies.total),
        "mag_total": (None if getattr(res, "mag_total", None) is None
                      else float(res.mag_total)),
        "tail_iters": tail_iters(hist),
        "fft_launches": int(ffts),
        "fft_per_iter": round(ffts / max(1, res.n_iter), 1),
        "resid_hist": [float(x) for x in hist],
        "wall_s": round(dt, 1),
    }
    print(f"  {tag:16s} {res.n_iter:3d} iters (tail {row['tail_iters']:2d})  "
          f"conv={res.converged}  E={res.energies.total:.9f}  "
          f"fft={ffts}  {dt:.0f}s", flush=True)
    return res, row


def build_system_fn(args):
    if args.system == "fe":
        return (lambda: fe_fm_system(ecut_ry=args.ecut,
                                     kmesh=(args.kmesh,) * 3,
                                     use_symmetry=False)[0], 2,
                dict(smearing="gaussian", width=0.1, nspin=2, start_mag=[0.4]))
    if args.system == "al_slab":
        return (lambda: al_slab_system(args.layers, ecut_ry=args.ecut,
                                       kmesh=(args.kmesh, args.kmesh, 1),
                                       use_symmetry=False)[0], 1,
                dict(smearing="gaussian", width=0.1, nspin=1))
    raise ValueError(args.system)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["fe", "al_slab"], required=True)
    ap.add_argument("--kmesh", type=int, default=4)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ecut", type=float, default=45.0)
    ap.add_argument("--etol", type=float, default=1e-9)
    ap.add_argument("--rhotol", type=float, default=1e-7)
    ap.add_argument("--entol", type=float, default=1e-8)
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--teacher-alpha", type=float, default=0.7)
    ap.add_argument("--chi0-tol", type=float, default=1e-6)
    ap.add_argument("--inner-tol", type=float, default=1e-6)
    ap.add_argument("--inner-max", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-prod", action="store_true")
    ap.add_argument("--ref-scheme", choices=["pulay", "johnson", "broyden"],
                    default="pulay")  # frozen-χ₀ reference converger
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    opcount.enable()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    system_fn, nspin, phys_kw = build_system_fn(args)
    xc = xc_for(nspin)
    conv_kw = dict(etol=args.etol, rhotol=args.rhotol, max_iter=args.max_iter,
                   energy_metric=True, entol=args.entol, verbose=False)
    base_kw = {**phys_kw, **conv_kw}

    rows: list[dict] = []
    meta = {"args": vars(args), "nspin": nspin}

    def save():
        out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    # ---- reference: converge once (use_symmetry=False) → frozen χ₀ + fixed pt
    print(f"=== {args.system} (kmesh={args.kmesh}, ecut={args.ecut}) ===",
          flush=True)
    print("building frozen reference (converged χ₀ + fixed point)...", flush=True)
    t0 = time.time()
    # the frozen χ₀ must come from a CONVERGED density; use the robust converger
    # for the reference (johnson can stall on bcc-Fe FM at tight tol). A
    # non-converged reference makes the frozen operator meaningless.
    ref = scf(system_fn(), xc, mixing_scheme=args.ref_scheme, precond="local_tf",
              spin_precond=(nspin == 2), mixing_alpha=args.alpha, **base_kw)
    meta["ref"] = {"iters": int(ref.n_iter), "converged": bool(ref.converged),
                   "scheme": args.ref_scheme,
                   "energy_eV": float(ref.energies.total),
                   "mag_total": (None if getattr(ref, "mag_total", None) is None
                                 else float(ref.mag_total)),
                   "wall_s": round(time.time() - t0, 1)}
    print(f"  reference[{args.ref_scheme}] {ref.n_iter} iters  "
          f"conv={ref.converged}  E={ref.energies.total:.9f}  "
          f"{meta['ref']['wall_s']:.0f}s", flush=True)
    if not ref.converged:
        print("  WARNING: reference did NOT converge — frozen χ₀ is unreliable",
              flush=True)
    save()

    precond = ExactDielectricPrecond(ref, xc, chi0_tol=args.chi0_tol,
                                     inner_tol=args.inner_tol,
                                     inner_max_iter=args.inner_max)

    # ---- A: production baseline
    if not args.skip_prod:
        res_prod, row = _run("baseline_prod", system_fn, xc,
                             {**base_kw, "mixing_scheme": "johnson",
                              "precond": "local_tf", "mixing_alpha": args.alpha,
                              "spin_precond": (nspin == 2)})
        rows.append(row)
        save()

    # ---- B: controlled baseline (same mixer as teacher, TF precond)
    res_ctrl, row = _run("baseline_ctrl", system_fn, xc,
                         {**base_kw, "mixing_scheme": "pulay",
                          "precond": "local_tf", "mixing_alpha": args.alpha,
                          "spin_precond": (nspin == 2)})
    rows.append(row)
    save()

    # ---- C: teacher (exact dielectric precond_op)
    res_teach, row = _run("teacher_exact", system_fn, xc,
                         {**base_kw, "mixing_scheme": "pulay",
                          "mixing_alpha": args.teacher_alpha}, precond_op=precond)
    row["precond_calls"] = precond.n_calls
    row["inner_iters"] = precond.inner_iters
    rows.append(row)
    save()

    # ---- fixed-point identity + iteration-cut verdict
    # energies are in eV; two runs converged to etol should agree to ~etol
    dE = float(abs(res_teach.energies.total - res_ctrl.energies.total))
    drho = float((res_teach.rho - res_ctrl.rho).abs().max())
    ctrl_it = res_ctrl.n_iter
    teach_it = res_teach.n_iter
    ctrl_tail = tail_iters(_resid_hist(rows, "baseline_ctrl"))
    teach_tail = tail_iters(_resid_hist(rows, "teacher_exact"))
    meta["identity"] = {"dE_eV": dE, "dmax_rho": drho,
                        "same_fixed_point": bool(dE <= 1e-6 and drho <= 1e-4)}
    meta["iteration_cut"] = {
        "ctrl_iters": int(ctrl_it), "teach_iters": int(teach_it),
        "cut_total_vs_ctrl": round(ctrl_it / max(1, teach_it), 3),
        "cut_tail_vs_ctrl": round(ctrl_tail / max(1, teach_tail), 3),
    }
    if not args.skip_prod:
        prod_it = rows[0]["iters"]
        prod_tail = rows[0]["tail_iters"]
        meta["iteration_cut"]["prod_iters"] = int(prod_it)
        meta["iteration_cut"]["cut_total_vs_prod"] = round(
            prod_it / max(1, teach_it), 3)
        meta["iteration_cut"]["cut_tail_vs_prod"] = round(
            prod_tail / max(1, teach_tail), 3)
    save()

    print("\n--- VERDICT INPUTS ---")
    print(f"  fixed-point identity: dE={dE:.2e} Ha  d|ρ|={drho:.2e}  "
          f"SAME={meta['identity']['same_fixed_point']}")
    print(f"  iteration cut (total, teacher vs ctrl): "
          f"{meta['iteration_cut']['cut_total_vs_ctrl']:.2f}x  "
          f"(tail: {meta['iteration_cut']['cut_tail_vs_ctrl']:.2f}x)")
    if not args.skip_prod:
        print(f"  iteration cut (total, teacher vs prod):  "
              f"{meta['iteration_cut']['cut_total_vs_prod']:.2f}x  "
              f"(tail: {meta['iteration_cut']['cut_tail_vs_prod']:.2f}x)")
    print(f"\nsaved -> {out}")


def _resid_hist(rows, tag):
    for r in rows:
        if r["tag"] == tag:
            return r["resid_hist"]
    return []


if __name__ == "__main__":
    main()
