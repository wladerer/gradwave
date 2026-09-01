"""Stage 2 — the M1 verdict: cheap subspace-χ₀ Woodbury vs the exact teacher.

Four arms from an IDENTICAL cold start and convergence gate:

  baseline_prod   johnson + local-TF (+ Stoner for nspin=2) — the production
                  mixer, the thing to beat;
  baseline_ctrl   pulay   + local-TF (+ Stoner) — same mixer as the preconditioned
                  arms, isolates the preconditioner from the mixer algorithm;
  teacher_exact   pulay   + EXACT dielectric precond_op (full χ₀ Sternheimer,
                  frozen at the converged reference) — the ceiling;
  cheap_woodbury  pulay   + subspace-χ₀ Woodbury precond_op (in-window Adler-Wiser
                  sum + δμ, frozen at the SAME reference) — the student.

teacher and student are frozen at the SAME converged reference, so the only
thing that differs is χ₀ fidelity: a full conduction-space Sternheimer solve
(FFT-heavy, ~7.6×) vs an in-window sum-over-states inverted by Woodbury (zero
FFTs per apply). The M1 question: does the student keep most of the teacher's
iteration cut at a FRACTION of the FFT overhead?

Mandated checks (asserted, not just reported): every preconditioned arm
converges to the IDENTICAL fixed point as the baseline — dE ≤ 1e-9 (Ha),
d|ρ| ≤ 1e-6 — because a preconditioner cannot move the true fixed point. The
FFT counter quantifies the per-step cost of each χ₀.

Usage (asus only):
    uv run python experiments/chi0_precond_crux/stage2_cheap_student.py \
        --system fe --kmesh 4 --out .../results/stage2_fe_k4.json
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
from gradwave.scf.subspace_chi0 import build_woodbury_subspace

HARTREE_EV = 27.211386245988


def _resid_logger():
    hist: list[float] = []

    def hook(it, vin, vout):
        num = float(torch.linalg.norm(vout - vin))
        den = max(1.0, float(torch.linalg.norm(vin)))
        hist.append(num / den)

    return hook, hist


def tail_iters(hist: list[float], thresh: float = 1e-3) -> int:
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


def _identity(res_a, res_b):
    """Fixed-point identity of arm ``res_a`` vs baseline ``res_b`` (both eV E)."""
    dE_ha = abs(res_a.energies.total - res_b.energies.total) / HARTREE_EV
    drho = float((res_a.rho - res_b.rho).abs().max())
    return {"dE_ha": dE_ha, "dmax_rho": drho,
            "same_fixed_point": bool(dE_ha <= 1e-9 and drho <= 1e-6)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["fe", "al_slab"], required=True)
    ap.add_argument("--kmesh", type=int, default=4)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ecut", type=float, default=45.0)
    ap.add_argument("--etol", type=float, default=1e-9)
    # Density-residual gate (energy_metric OFF): the density is the binding
    # criterion so every arm lands on the IDENTICAL fixed-point density, not
    # merely the same energy. rhotol is ‖ρ_out−ρ_in‖·Ω/N_G (electrons scale);
    # 1e-9 pins the max pointwise density difference between runs below 1e-6.
    ap.add_argument("--rhotol", type=float, default=1e-9)
    ap.add_argument("--entol", type=float, default=1e-9)
    ap.add_argument("--energy-metric", action="store_true")
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--teacher-alpha", type=float, default=0.7)
    ap.add_argument("--chi0-tol", type=float, default=1e-6)
    ap.add_argument("--inner-tol", type=float, default=1e-6)
    ap.add_argument("--inner-max", type=int, default=30)
    ap.add_argument("--pair-cut", type=float, default=1e-6)
    ap.add_argument("--max-cols", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-teacher", action="store_true")
    ap.add_argument("--ref-scheme", choices=["pulay", "johnson", "broyden"],
                    default="pulay")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    opcount.enable()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    system_fn, nspin, phys_kw = build_system_fn(args)
    xc = xc_for(nspin)
    conv_kw = dict(etol=args.etol, rhotol=args.rhotol, max_iter=args.max_iter,
                   energy_metric=args.energy_metric, entol=args.entol,
                   verbose=False)
    base_kw = {**phys_kw, **conv_kw}

    rows: list[dict] = []
    meta = {"args": vars(args), "nspin": nspin}

    def _json_default(o):
        if isinstance(o, torch.Tensor):
            return o.item() if o.ndim == 0 else o.tolist()
        try:
            import numpy as _np
            if isinstance(o, _np.generic):
                return o.item()
        except ImportError:
            pass
        raise TypeError(f"not serializable: {type(o).__name__}")

    def save():
        out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2,
                                  default=_json_default))

    print(f"=== {args.system} (kmesh={args.kmesh}, ecut={args.ecut}) ===",
          flush=True)
    print("building frozen reference (converged χ₀ + fixed point)...", flush=True)
    t0 = time.time()
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
        print("  WARNING: reference did NOT converge — frozen χ₀ unreliable",
              flush=True)
    save()

    # ---- build the cheap Woodbury preconditioner (one-time) + its FFT cost.
    # scf() toggles opcount off on return (its own recorder owns it), so
    # re-enable before snapshotting the build's one-time FFTs.
    opcount.enable()
    prev = opcount.snapshot()
    t0 = time.time()
    cheap = build_woodbury_subspace(ref, xc, pair_cut=args.pair_cut,
                                    max_cols=args.max_cols)
    build_ffts = opcount.since(prev)["fft"]
    meta["cheap_build"] = {
        "n_col": (0 if cheap is None else int(cheap.n_col)),
        "build_ffts": int(build_ffts),
        "build_s": round(time.time() - t0, 1),
    }
    print(f"  cheap Woodbury: n_col={meta['cheap_build']['n_col']}  "
          f"build_ffts={build_ffts}  {meta['cheap_build']['build_s']:.0f}s",
          flush=True)
    save()

    # ---- A: production baseline
    res_prod, row = _run("baseline_prod", system_fn, xc,
                         {**base_kw, "mixing_scheme": "johnson",
                          "precond": "local_tf", "mixing_alpha": args.alpha,
                          "spin_precond": (nspin == 2)})
    rows.append(row)
    save()

    # ---- B: controlled baseline (same mixer as the preconditioned arms)
    res_ctrl, row = _run("baseline_ctrl", system_fn, xc,
                         {**base_kw, "mixing_scheme": "pulay",
                          "precond": "local_tf", "mixing_alpha": args.alpha,
                          "spin_precond": (nspin == 2)})
    rows.append(row)
    save()

    # ---- C: teacher (exact dielectric precond_op) — the ceiling
    if not args.skip_teacher:
        precond = ExactDielectricPrecond(ref, xc, chi0_tol=args.chi0_tol,
                                         inner_tol=args.inner_tol,
                                         inner_max_iter=args.inner_max)
        res_teach, row = _run("teacher_exact", system_fn, xc,
                             {**base_kw, "mixing_scheme": "pulay",
                              "mixing_alpha": args.teacher_alpha},
                             precond_op=precond)
        row["precond_calls"] = precond.n_calls
        rows.append(row)
        save()
    else:
        res_teach = None

    # ---- D: cheap subspace-χ₀ Woodbury precond_op — the student
    res_cheap = None
    if cheap is not None:
        res_cheap, row = _run("cheap_woodbury", system_fn, xc,
                             {**base_kw, "mixing_scheme": "pulay",
                              "mixing_alpha": args.teacher_alpha},
                             precond_op=cheap)
        row["precond_calls"] = cheap.n_calls
        rows.append(row)
        save()

    # ---- fixed-point identity + M1 verdict.
    # The identity is checked against the CONVERGED pulay control (same mixer),
    # not the production arm — production may fail to converge at a tight tol,
    # and comparing to a non-converged density is meaningless. A preconditioner
    # cannot move the true fixed point, so cheap/teacher/ctrl must coincide.
    meta["identity"] = {}
    if res_teach is not None:
        meta["identity"]["teacher_vs_ctrl"] = _identity(res_teach, res_ctrl)
    if res_cheap is not None:
        meta["identity"]["cheap_vs_ctrl"] = _identity(res_cheap, res_ctrl)
        if res_teach is not None:
            meta["identity"]["cheap_vs_teacher"] = _identity(res_cheap, res_teach)

    def _row(tag):
        for r in rows:
            if r["tag"] == tag:
                return r
        return None

    prod, ctrl = _row("baseline_prod"), _row("baseline_ctrl")
    prod_conv = bool(prod["converged"])
    verdict = {}
    if res_cheap is not None:
        ch = _row("cheap_woodbury")
        # The fair, same-mixer baseline is the pulay control. Production is the
        # real thing to beat; when it converges we compare to it, when it does
        # NOT the student is a rescue (report both).
        base = prod if prod_conv else ctrl
        base_tag = "prod" if prod_conv else "ctrl(prod DIVERGED)"
        verdict["cheap_iters"] = ch["iters"]
        verdict["prod_iters"] = prod["iters"]
        verdict["prod_converged"] = prod_conv
        verdict["ctrl_iters"] = ctrl["iters"]
        verdict["baseline_for_verdict"] = base_tag
        verdict["cut_vs_prod"] = (round(prod["iters"] / max(1, ch["iters"]), 3)
                                  if prod_conv else "rescue(prod diverged)")
        verdict["cut_vs_ctrl"] = round(ctrl["iters"] / max(1, ch["iters"]), 3)
        verdict["fft_overhead_vs_prod"] = round(
            ch["fft_launches"] / max(1, prod["fft_launches"]), 3)
        verdict["fft_overhead_vs_ctrl"] = round(
            ch["fft_launches"] / max(1, ctrl["fft_launches"]), 3)
        cut_primary = base["iters"] / max(1, ch["iters"])
        fft_primary = ch["fft_launches"] / max(1, base["fft_launches"])
        if res_teach is not None:
            te = _row("teacher_exact")
            verdict["teacher_iters"] = te["iters"]
            verdict["teacher_cut_vs_ctrl"] = round(
                ctrl["iters"] / max(1, te["iters"]), 3)
            verdict["teacher_fft_overhead_vs_ctrl"] = round(
                te["fft_launches"] / max(1, ctrl["fft_launches"]), 3)
            tc = ctrl["iters"] / max(1, te["iters"]) - 1.0
            cc = ctrl["iters"] / max(1, ch["iters"]) - 1.0
            verdict["frac_teacher_cut_retained"] = (
                round(cc / tc, 3) if abs(tc) > 1e-9 else None)
        idc = meta["identity"].get("cheap_vs_ctrl", {})
        verdict["M1_GO"] = bool(
            ch["converged"] and cut_primary >= 1.5 and fft_primary <= 1.3
            and idc.get("same_fixed_point", False))
        verdict["M1_rescue"] = bool(ch["converged"] and not prod_conv)
    meta["verdict"] = verdict
    save()

    print("\n--- M1 VERDICT INPUTS ---")
    for k, v in meta["identity"].items():
        print(f"  identity {k}: dE={v['dE_ha']:.2e} Ha  d|ρ|={v['dmax_rho']:.2e}"
              f"  SAME={v['same_fixed_point']}")
    if verdict:
        print(f"  cheap: {verdict['cheap_iters']} it  "
              f"cut_vs_ctrl={verdict['cut_vs_ctrl']}x  "
              f"fft_overhead_vs_ctrl={verdict['fft_overhead_vs_ctrl']}x  "
              f"cut_vs_prod={verdict['cut_vs_prod']}")
        if "teacher_iters" in verdict:
            print(f"  teacher: {verdict['teacher_iters']} it  "
                  f"cut_vs_ctrl={verdict['teacher_cut_vs_ctrl']}x  "
                  f"fft_overhead_vs_ctrl={verdict['teacher_fft_overhead_vs_ctrl']}x  "
                  f"student retains {verdict['frac_teacher_cut_retained']} of cut")
        print(f"  >>> M1_GO = {verdict.get('M1_GO')}  "
              f"(rescue={verdict.get('M1_rescue')})")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
