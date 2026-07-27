"""SCF iteration-count study (perf/scf-cycle-study).

Measures SCF ITERATION COUNTS (the trustworthy solver-logic metric, not wall
time) across the mixing axes on the norm-conserving screening set, reusing the
solver_battery.run SYSTEMS definitions so the systems match the #133 matrix.

Axes (select with `--axes a,b,...`; default all):
  scheme   : pulay|broyden|johnson x history {8,12,16}
  alpha    : damping {0.5,0.7,0.9} at the metal-recommended scheme (johnson)
  q0       : Kerker screening q0 {0.6,0.8,1.1,1.5,2.0} via precond_op (single-pole
             MultipoleKerkerPrecond) at pulay+kerker
  precond  : kerker vs local_tf (position-dependent TF) on the metals
  diago    : adaptive_diago_tol schedule variants (monkeypatched): first_tol and
             the quadratic coefficient, looser/default/tighter early

Every run records n_iter, converged, free-energy/atom, final |drho|, wall. A
"win" is judged downstream on iteration count with an identical fixed point
(<1e-8 eV/atom drift). Small systems get the full grid; mediums are confirm-only.

Usage (asus, canonical venv):
  PYTHONPATH=src .venv/bin/python benchmarks/scf_cycle_study.py \
      --systems si_insulator,mgo_insulator,al_metal,cu_metal,fe_fm_metal \
      --axes scheme,alpha,q0,precond,diago --out /tmp/scf_cycle_nc.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np  # noqa: F401  (imported by SYSTEMS build lambdas' closure)
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import gradwave.scf.loop as loop_mod  # noqa: E402
from gradwave.scf.common import adaptive_diago_tol as _ADT  # noqa: E402
from gradwave.scf.learned_precond import MultipoleKerkerPrecond  # noqa: E402
from gradwave.scf.loop import scf  # noqa: E402
from solver_battery.run import SYSTEMS  # noqa: E402

torch.set_num_threads(8)
sys.stdout.reconfigure(line_buffering=True)

# tolerances: energy-gated, matching the battery (etol/rhotol tight so the
# iteration count is the true convergence count, not a loose early stop)
COMMON = dict(etol=1e-9, rhotol=1e-8, verbose=False)


def _dens_g2(system):
    g = system.grid
    return g.g2.reshape(-1)[g.dens_mask.reshape(-1)].to(torch.float64)


def run_one(name, overrides, diago_patch=None):
    """Build a fresh System for `name` and run scf() with `overrides` merged
    over the battery spec. `overrides` may contain a 'q0' sentinel -> builds a
    single-pole Kerker precond_op. `diago_patch`=(first_tol, coeff) monkeypatches
    the adaptive diago-tol schedule for this run only."""
    spec = SYSTEMS[name]
    kw = dict(COMMON)
    kw.update(spec["scf"])
    xc = kw.pop("xc")
    ov = dict(overrides)
    q0 = ov.pop("q0", None)
    system = spec["build"]()
    natoms = int(system.positions.shape[0])
    if q0 is not None:
        ov["precond_op"] = MultipoleKerkerPrecond.kerker(_dens_g2(system), q0).detach_()
    kw.update(ov)

    orig = loop_mod.adaptive_diago_tol
    if diago_patch is not None:
        ft, coeff = diago_patch

        def patched(it, history, diago_tol, n_electrons, *, schedule, first_tol=1e-3):
            if it == 1:
                return max(diago_tol, ft)
            r_prev = history[-1]["res"]
            return max(diago_tol, min(1e-3, coeff * r_prev * r_prev / n_electrons))

        loop_mod.adaptive_diago_tol = patched
    try:
        t0 = time.perf_counter()
        res = scf(system, xc, **kw)
        wall = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        loop_mod.adaptive_diago_tol = orig
        return dict(error=f"{type(exc).__name__}: {exc}")
    finally:
        loop_mod.adaptive_diago_tol = orig

    fr = float(res.energies.free_energy)
    final_res = float(res.history[-1]["res"]) if res.history else None
    return dict(
        n_iter=int(res.n_iter), converged=bool(res.converged),
        free_energy=fr, free_energy_per_atom=fr / natoms,
        final_res=final_res, wall_s=round(wall, 2), natoms=natoms,
        mag_total=(float(res.mag_total) if getattr(res, "mag_total", None) else None),
    )


def scheme_configs(full):
    hists = (8, 12, 16) if full else (8, 12)
    cfgs = {"baseline": {}}
    for sch in ("pulay", "broyden", "johnson"):
        for h in hists:
            cfgs[f"{sch}/h{h}"] = dict(mixing_scheme=sch, mixing_history=h)
    return cfgs


def alpha_configs():
    return {f"johnson/a{a}": dict(mixing_scheme="johnson", mixing_alpha=a)
            for a in (0.5, 0.7, 0.9)}


def q0_configs():
    return {f"pulay+kerker/q0={q}": dict(mixing_scheme="pulay", q0=q)
            for q in (0.6, 0.8, 1.1, 1.5, 2.0)}


def precond_configs():
    return {"johnson+kerker": dict(mixing_scheme="johnson", precond="kerker"),
            "johnson+local_tf": dict(mixing_scheme="johnson", precond="local_tf")}


DIAGO_VARIANTS = {  # (first_tol, quadratic coeff); default is (1e-3, 0.1)
    "default(1e-3,0.1)": (1e-3, 0.1),
    "loose_early(1e-2,0.3)": (1e-2, 0.3),
    "tight_early(1e-4,0.033)": (1e-4, 0.033),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="si_insulator,mgo_insulator,al_metal,"
                    "cu_metal,fe_fm_metal")
    ap.add_argument("--axes", default="scheme,alpha,q0,precond,diago")
    ap.add_argument("--full-scheme", action="store_true",
                    help="full history grid + all systems for the scheme axis")
    ap.add_argument("--out", default="/tmp/scf_cycle_nc.json")
    args = ap.parse_args()
    systems = args.systems.split(",")
    axes = set(args.axes.split(","))
    metals = [s for s in systems if s in ("al_metal", "cu_metal", "cu_metal_m")]

    results = {}

    def record(sysname, label, rec):
        results.setdefault(sysname, {})[label] = rec
        it = rec.get("n_iter", "ERR")
        conv = rec.get("converged")
        epa = rec.get("free_energy_per_atom")
        extra = "" if rec.get("mag_total") is None else f" m={rec['mag_total']:+.3f}"
        epas = "" if epa is None else f" E/at={epa:.8f}"
        err = rec.get("error", "")
        print(f"  {sysname:16s} {label:26s} it={it!s:>4} conv={conv!s:>5}"
              f"{epas}{extra} {rec.get('wall_s','')}s {err}", flush=True)

    if "scheme" in axes:
        print("\n=== SCHEME x HISTORY ===")
        for s in systems:
            for label, ov in scheme_configs(args.full_scheme).items():
                record(s, f"scheme:{label}", run_one(s, ov))
    if "alpha" in axes:
        print("\n=== DAMPING alpha (johnson) ===")
        for s in metals:
            for label, ov in alpha_configs().items():
                record(s, f"alpha:{label}", run_one(s, ov))
    if "q0" in axes:
        print("\n=== KERKER q0 (pulay+kerker) ===")
        for s in metals:
            for label, ov in q0_configs().items():
                record(s, f"q0:{label}", run_one(s, ov))
    if "precond" in axes:
        print("\n=== PRECOND kerker vs local_tf ===")
        for s in metals:
            for label, ov in precond_configs().items():
                record(s, f"precond:{label}", run_one(s, ov))
    if "diago" in axes:
        print("\n=== DIAGO-TOL schedule variants ===")
        dsys = [s for s in ("si_insulator", "cu_metal") if s in systems] or systems[:1]
        for s in dsys:
            for label, patch in DIAGO_VARIANTS.items():
                record(s, f"diago:{label}", run_one(s, {}, diago_patch=patch))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
