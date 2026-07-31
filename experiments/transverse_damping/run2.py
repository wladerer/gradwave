"""Variant B (step-wrap) matrix: damp the mixer's total update, not the residual.

Same tolerances, systems, and probes as run.py. Each run installs StepWrapPatch
around scf_noncollinear via monkeypatch (no src edits). Groups:

  nisf2   SOC-free Ni: q0 scan + lab-frame control + flat null (stepwrap form)
  nis2    Ni + SOC stock pulay: qmid + flat
  best2   campaign best arm (johnson + quadratic + 0.3) with stepwrap qmid
  canted2 canted 2-atom Fe kill-criterion with stepwrap (qmid, qlo) + SOC

Usage: uv run python experiments/transverse_damping/run2.py --group nisf2
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.transverse_damping import systems as sysmod  # noqa: E402
from experiments.transverse_damping.damping2 import StepWrapPatch  # noqa: E402
from experiments.transverse_damping.probe import NCConvergenceProbe  # noqa: E402
from experiments.transverse_damping.run import BEST, TOL, _floor_stats, z  # noqa: E402
from gradwave.core.xc.noncollinear import NoncollinearXC  # noqa: E402
from gradwave.core.xc.spin import SpinPBE  # noqa: E402
from gradwave.scf.noncollinear import scf_noncollinear  # noqa: E402


def run_wrapped(name, system, mag_vec_init, outdir, wrap=None,
                track_per_atom=False, **kw):
    xc = NoncollinearXC(SpinPBE())
    probe = NCConvergenceProbe(system, nonmagnetic=False,
                               track_per_atom=track_per_atom)
    opts = dict(TOL)
    opts.update(kw)
    t0 = time.perf_counter()
    if wrap is None:
        res = scf_noncollinear(system, xc, mag_vec_init=mag_vec_init,
                               mixer_hook=probe, **opts)
        patch = None
    else:
        patch = StepWrapPatch(**wrap)
        with patch:
            res = scf_noncollinear(system, xc, mag_vec_init=mag_vec_init,
                                   mixer_hook=probe, **opts)
    wall = time.perf_counter() - t0
    with open(outdir / f"trace_{name}.jsonl", "w") as f:
        for rec in probe.records:
            f.write(json.dumps(rec) + "\n")
    if patch is not None:
        with open(outdir / f"damp_{name}.jsonl", "w") as f:
            for rec in patch.applied:
                f.write(json.dumps(rec) + "\n")
    res_hist = [h.get("res") for h in res.history if h.get("res") is not None]
    res_floor = float(np.mean(res_hist[-10:])) if res_hist else float("nan")
    row = dict(
        name=name, converged=bool(res.converged), n_iter=int(res.n_iter),
        free_energy=float(res.energies.free_energy),
        mag_vec=[round(float(x), 4) for x in res.mag_vec],
        mag_abs=round(float(res.mag_abs), 4),
        seed=[round(float(x), 3) for x in np.asarray(mag_vec_init).ravel()],
        wall_s=round(wall, 1), res_gate_floor=res_floor,
        damp=({"variant": "stepwrap", **wrap} if wrap is not None else None),
        opts={k: repr(v) for k, v in kw.items()},
    )
    row.update(_floor_stats(probe.records,
                            ["dn", "dm", "dm_x", "dm_y", "dm_z", "dtheta_out"]))
    if track_per_atom:
        row.update(_floor_stats(probe.records, ["pair_angle"]))
        row["pair_angle_traj"] = [r.get("pair_angle") for r in probe.records[:8]]
        row["m_abs_g0_traj"] = [r.get("m_abs_g0") for r in probe.records[:8]]
    row["dm_final"] = probe.records[-1].get("dm")
    return row


def build_groups(outdir, q0lo, q0mid, q0hi):
    NI_F = lambda: sysmod.ni_fcc(False)   # noqa: E731
    NI_S = lambda: sysmod.ni_fcc(True)    # noqa: E731

    def K(q0, frame="moment"):
        return dict(q0=q0, frame=frame, form="kerker")

    def nisf2():
        yield run_wrapped("nisf2_qlo", NI_F(), z(0.6), outdir, wrap=K(q0lo))
        yield run_wrapped("nisf2_qmid", NI_F(), z(0.6), outdir, wrap=K(q0mid))
        yield run_wrapped("nisf2_qhi", NI_F(), z(0.6), outdir, wrap=K(q0hi))
        yield run_wrapped("nisf2_qmid_lab", NI_F(), z(0.6), outdir,
                          wrap=K(q0mid, frame="lab"))
        yield run_wrapped("nisf2_flat03", NI_F(), z(0.6), outdir,
                          wrap=dict(form="flat", alpha_perp=0.3, frame="moment"))
        yield run_wrapped("nisf2_flat01", NI_F(), z(0.6), outdir,
                          wrap=dict(form="flat", alpha_perp=0.1, frame="moment"))

    def nis2():
        yield run_wrapped("nis2_qmid", NI_S(), z(0.6), outdir, wrap=K(q0mid))
        yield run_wrapped("nis2_qlo", NI_S(), z(0.6), outdir, wrap=K(q0lo))
        yield run_wrapped("nis2_flat03", NI_S(), z(0.6), outdir,
                          wrap=dict(form="flat", alpha_perp=0.3, frame="moment"))

    def best2():
        yield run_wrapped("nisbest2_qmid", NI_S(), z(0.6), outdir,
                          wrap=K(q0mid), **BEST)
        yield run_wrapped("nisfbest2_qmid", NI_F(), z(0.6), outdir,
                          wrap=K(q0mid), **BEST)

    def canted2():
        mvi = [[0.0, 0.0, 0.6], [0.6, 0.0, 0.0]]
        yield run_wrapped("fe2b_qmid", sysmod.fe_bcc_2atom(False), mvi, outdir,
                          track_per_atom=True, wrap=K(4.0))
        yield run_wrapped("fe2b_qlo", sysmod.fe_bcc_2atom(False), mvi, outdir,
                          track_per_atom=True, wrap=K(2.0))
        yield run_wrapped("fe2bsoc_qmid", sysmod.fe_bcc_2atom(True), mvi, outdir,
                          track_per_atom=True, wrap=K(4.0))

    return {"nisf2": nisf2, "nis2": nis2, "best2": best2, "canted2": canted2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--host", default=platform.node().split(".")[0])
    ap.add_argument("--q0lo", type=float, default=2.0)
    ap.add_argument("--q0mid", type=float, default=4.0)
    ap.add_argument("--q0hi", type=float, default=8.0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    outdir = Path(__file__).resolve().parent / "results" / args.host
    outdir.mkdir(parents=True, exist_ok=True)
    groups = build_groups(outdir, args.q0lo, args.q0mid, args.q0hi)
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
