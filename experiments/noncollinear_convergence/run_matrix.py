"""Baseline convergence matrix for the non-collinear / SOC SCF.

Runs a set of named cases (anchor / Ni / Fe / canted), each with the
NCConvergenceProbe attached, at identical tolerances, and writes:
  results/<host>/summary.jsonl   one row per run (iteration count, converged,
                                 final F, moment, per-channel floor stats)
  results/<host>/trace_<name>.jsonl  the full per-iteration probe trace

Groups are generators and the summary row is flushed as soon as each run
finishes, so a killed job loses at most the in-flight run.

Usage:
  uv run python experiments/noncollinear_convergence/run_matrix.py --group ni_stock
  uv run python experiments/noncollinear_convergence/run_matrix.py --group all
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

# repo root on sys.path so the experiments.* namespace package imports resolve
# regardless of the launching cwd (the script dir alone is added by default)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gradwave.core.xc.noncollinear import NoncollinearXC  # noqa: E402
from gradwave.core.xc.spin import SpinPBE  # noqa: E402
from gradwave.scf.loop import scf  # noqa: E402
from gradwave.scf.noncollinear import scf_noncollinear  # noqa: E402

from experiments.noncollinear_convergence import systems as sysmod  # noqa: E402
from experiments.noncollinear_convergence.probe import NCConvergenceProbe  # noqa: E402

# identical tolerances for every run (tight enough that a limit-cycling run
# exhausts the cap rather than passing on a loose gate)
TOL = dict(smearing="gaussian", width=0.1, max_iter=80, etol=1e-6, rhotol=1e-5,
           diago_tol=1e-9, verbose=False)
# the merged #79 recipe (spin_precond on the longitudinal m channel + the
# collinear quadratic diago schedule + a gentler magnetization step)
FIX = dict(spin_precond=True, mag_mixer="pulay", mag_diago_schedule="quadratic",
           mag_mixing_alpha=0.3)


def _floor_stats(records, keys, last=10):
    """mean of each key over the last `last` iterations (the residual floor)."""
    tail = records[-last:] if len(records) >= last else records
    out = {}
    for k in keys:
        vals = [r[k] for r in tail if k in r and r[k] == r[k]]  # drop NaN
        out[f"{k}_floor"] = float(np.mean(vals)) if vals else float("nan")
    return out


def run_nc(name, system, mag_vec_init, nonmagnetic=False, outdir=None, **kw):
    xc = NoncollinearXC(SpinPBE())
    probe = NCConvergenceProbe(system, nonmagnetic=nonmagnetic)
    opts = dict(TOL)
    opts.update(kw)
    t0 = time.perf_counter()
    res = scf_noncollinear(system, xc, mag_vec_init=mag_vec_init,
                           nonmagnetic=nonmagnetic, mixer_hook=probe, **opts)
    wall = time.perf_counter() - t0
    if outdir is not None:
        with open(outdir / f"trace_{name}.jsonl", "w") as f:
            for rec in probe.records:
                f.write(json.dumps(rec) + "\n")
    row = dict(
        name=name, kind="nc", nonmagnetic=nonmagnetic,
        converged=bool(res.converged), n_iter=int(res.n_iter),
        free_energy=float(res.energies.free_energy),
        mag_vec=[round(float(x), 4) for x in res.mag_vec],
        mag_abs=round(float(res.mag_abs), 4),
        seed=[round(float(x), 3) for x in np.asarray(mag_vec_init).ravel()],
        wall_s=round(wall, 1),
        opts={k: repr(v) for k, v in kw.items()},
    )
    if not nonmagnetic:
        row.update(_floor_stats(probe.records,
                                ["dn", "dm", "dm_x", "dm_y", "dm_z", "dtheta_out"]))
        row["final_dtheta"] = probe.records[-1].get("dtheta_out")
    else:
        row.update(_floor_stats(probe.records, ["dn"]))
    return row


def run_collinear(name, system, start_mag):
    t0 = time.perf_counter()
    res = scf(system, SpinPBE(), nspin=2, start_mag=[start_mag],
              smearing="gaussian", width=0.1, max_iter=80, etol=1e-6,
              rhotol=1e-5, diago_tol=1e-9, verbose=False)
    wall = time.perf_counter() - t0
    return dict(
        name=name, kind="collinear",
        converged=bool(getattr(res, "converged", False)),
        n_iter=int(getattr(res, "n_iter", -1)),
        free_energy=float(res.energies.free_energy),
        mag_total=round(float(getattr(res, "mag_total", float("nan"))), 4),
        seed=[start_mag], wall_s=round(wall, 1),
    )


def z(s):
    return [[0.0, 0.0, float(s)]]


def tilt(s):
    v = np.array([1.0, 1.0, 1.0])
    v = v / np.linalg.norm(v) * s
    return [list(map(float, v))]


def build_groups(outdir):
    def anchor():
        yield run_nc("pt_soc_nm", sysmod.pt_fcc(), z(0.0),
                     nonmagnetic=True, outdir=outdir)

    def ni_stock():
        for s in (0.3, 0.6, 1.0):
            yield run_nc(f"ni_soc_stock_s{s}_z", sysmod.ni_fcc(True), z(s),
                         outdir=outdir)
        # second seed DIRECTION: does the final moment axis track the seed?
        yield run_nc("ni_soc_stock_s0.6_tilt", sysmod.ni_fcc(True), tilt(0.6),
                     outdir=outdir)

    def ni_fix():
        for s in (0.3, 0.6, 1.0):
            yield run_nc(f"ni_soc_fix_s{s}_z", sysmod.ni_fcc(True), z(s),
                         outdir=outdir, **FIX)
        yield run_nc("ni_soc_fix_s0.6_tilt", sysmod.ni_fcc(True), tilt(0.6),
                     outdir=outdir, **FIX)

    def ni_alt():
        # johnson alternative to the Stoner precond (the collinear cure)
        yield run_nc("ni_soc_johnson_s0.6_z", sysmod.ni_fcc(True), z(0.6),
                     outdir=outdir, mag_mixer="johnson")
        # long-cap johnson: the fixed-point oracle attempt at tight tolerance
        yield run_nc("ni_soc_johnson_long_s0.6_z", sysmod.ni_fcc(True), z(0.6),
                     outdir=outdir, mag_mixer="johnson", max_iter=200)
        # SOC-free non-collinear vs the collinear path (cost of going NC)
        yield run_nc("ni_socfree_s0.6_z", sysmod.ni_fcc(False), z(0.6),
                     outdir=outdir)
        yield run_collinear("ni_collinear_s0.6", sysmod.ni_fcc(False), 0.6)

    def fe():
        yield run_nc("fe_soc_stock_s0.6_z", sysmod.fe_bcc(True), z(0.6),
                     outdir=outdir)
        yield run_nc("fe_soc_fix_s0.6_z", sysmod.fe_bcc(True), z(0.6),
                     outdir=outdir, **FIX)
        yield run_nc("fe_socfree_s0.6_z", sysmod.fe_bcc(False), z(0.6),
                     outdir=outdir)
        yield run_collinear("fe_collinear_s0.6", sysmod.fe_bcc(False), 0.6)

    def canted():
        # two moments seeded 90 deg apart (m0 along z, m1 along x); the
        # unconstrained exchange should align them. Watch the direction
        # dynamics on the way.
        for soc in (False, True):
            tag = "soc" if soc else "socfree"
            system = sysmod.fe_bcc_2atom(soc)
            mvi = [[0.0, 0.0, 0.6], [0.6, 0.0, 0.0]]
            yield run_nc(f"fe2_canted90_{tag}", system, mvi, outdir=outdir)

    return {"anchor": anchor, "ni_stock": ni_stock, "ni_fix": ni_fix,
            "ni_alt": ni_alt, "fe": fe, "canted": canted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all",
                    help="comma-separated group names, or 'all'")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--host", default=platform.node().split(".")[0])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    outdir = Path(__file__).resolve().parent / "results" / args.host
    outdir.mkdir(parents=True, exist_ok=True)
    groups = build_groups(outdir)
    names = list(groups) if args.group == "all" else args.group.split(",")

    summary_path = outdir / "summary.jsonl"
    n = 0
    for gname in names:
        for row in groups[gname]():
            n += 1
            print(json.dumps(row), flush=True)
            with open(summary_path, "a") as f:
                f.write(json.dumps(row) + "\n")
    print(f"\nDONE group={args.group} host={args.host} runs={n}", flush=True)


if __name__ == "__main__":
    main()
