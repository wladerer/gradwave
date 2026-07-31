"""Transverse-damping study driver.

Runs the measurement matrix from the study brief: baseline (no damping) vs
reverse-Kerker transverse damping (q0 scan, moment vs lab frame) vs a flat
transverse step reduction (the null hypothesis), on the campaign's Ni/Fe cells
at the campaign's tolerances. Every run carries an NCConvergenceProbe recording
the RAW (undamped) residual, so the transverse amplification is measured on the
true map; the driver's own rhotol gate reads that same raw residual (computed
before the mixer_hook fires), so convergence stays honest.

Writes:
  results/<host>/summary.jsonl        one row per run
  results/<host>/trace_<name>.jsonl   per-iteration probe trace
  results/<host>/damp_<name>.jsonl    per-iteration damping diagnostic

Usage:
  uv run python experiments/transverse_damping/run.py --inspect
  uv run python experiments/transverse_damping/run.py --group ni_socfree
  uv run python experiments/transverse_damping/run.py --group all
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
from experiments.transverse_damping.damping import TransverseDampingHook  # noqa: E402
from experiments.transverse_damping.probe import NCConvergenceProbe  # noqa: E402
from gradwave.core.xc.noncollinear import NoncollinearXC  # noqa: E402
from gradwave.core.xc.spin import SpinPBE  # noqa: E402
from gradwave.scf.noncollinear import scf_noncollinear  # noqa: E402

# campaign tolerances, identical across runs
TOL = dict(smearing="gaussian", width=0.1, max_iter=80, etol=1e-6, rhotol=1e-5,
           diago_tol=1e-9, verbose=False)
# the campaign's best arm (recommendation 2 measured floor): johnson + quadratic
# schedule + a gentler magnetization step held the moment at the lowest floor.
BEST = dict(mag_mixer="johnson", mag_diago_schedule="quadratic",
            mag_mixing_alpha=0.3)


def _floor_stats(records, keys, last=10):
    tail = records[-last:] if len(records) >= last else records
    out = {}
    for k in keys:
        vals = [r[k] for r in tail if k in r and r[k] == r[k]]
        out[f"{k}_floor"] = float(np.mean(vals)) if vals else float("nan")
    return out


def run_nc(name, system, mag_vec_init, outdir, damp=None, track_per_atom=False,
           **kw):
    """One spinor run. `damp` is a dict of TransverseDampingHook kwargs or None."""
    xc = NoncollinearXC(SpinPBE())
    probe = NCConvergenceProbe(system, nonmagnetic=False,
                               track_per_atom=track_per_atom)
    if damp is None:
        hook = probe
        damp_obj = None
    else:
        damp_obj = TransverseDampingHook(system, inner_probe=probe, **damp)
        hook = damp_obj
    opts = dict(TOL)
    opts.update(kw)
    t0 = time.perf_counter()
    res = scf_noncollinear(system, xc, mag_vec_init=mag_vec_init,
                           mixer_hook=hook, **opts)
    wall = time.perf_counter() - t0
    with open(outdir / f"trace_{name}.jsonl", "w") as f:
        for rec in probe.records:
            f.write(json.dumps(rec) + "\n")
    if damp_obj is not None:
        with open(outdir / f"damp_{name}.jsonl", "w") as f:
            for rec in damp_obj.applied:
                f.write(json.dumps(rec) + "\n")
    # residual gate floor: the driver gates on ||vout-vin||*vol over ALL channels
    res_hist = [h.get("res") for h in res.history if h.get("res") is not None]
    res_floor = float(np.mean(res_hist[-10:])) if res_hist else float("nan")
    row = dict(
        name=name, converged=bool(res.converged), n_iter=int(res.n_iter),
        free_energy=float(res.energies.free_energy),
        mag_vec=[round(float(x), 4) for x in res.mag_vec],
        mag_abs=round(float(res.mag_abs), 4),
        seed=[round(float(x), 3) for x in np.asarray(mag_vec_init).ravel()],
        wall_s=round(wall, 1),
        res_gate_floor=res_floor,
        damp=(damp if damp is not None else None),
        opts={k: repr(v) for k, v in kw.items()},
    )
    row.update(_floor_stats(probe.records,
                            ["dn", "dm", "dm_x", "dm_y", "dm_z", "dtheta_out"]))
    if track_per_atom:
        row.update(_floor_stats(probe.records, ["pair_angle"]))
        # alignment trajectory: pair_angle over the first iterations
        row["pair_angle_traj"] = [r.get("pair_angle") for r in probe.records[:8]]
        row["m_abs_g0_traj"] = [r.get("m_abs_g0") for r in probe.records[:8]]
    row["dm_final"] = probe.records[-1].get("dm")
    return row


def z(s):
    return [[0.0, 0.0, float(s)]]


def inspect():
    """Print the density-sphere |G| extent and shell edges so q0 can be picked
    to bracket the observed soft-sector (bottom 2 of 12 shells)."""
    for label, sysfn in [("ni_soc", lambda: sysmod.ni_fcc(True)),
                         ("ni_socfree", lambda: sysmod.ni_fcc(False)),
                         ("fe2", lambda: sysmod.fe_bcc_2atom(False))]:
        system = sysfn()
        grid = system.grid
        mask = grid.dens_mask.reshape(-1)
        gmag = torch.sqrt(grid.g2.reshape(-1)[mask].clamp_min(0.0))
        gmax = float(gmag.max())
        n_shells = 12
        edge = gmax / n_shells
        print(f"{label}: ng={int(mask.sum())} gmax={gmax:.3f} 1/Ang  "
              f"shell_width={edge:.3f}  bottom-2-shell edge={2*edge:.3f}  "
              f"q0 bracket ~ [{edge:.3f}, {2*edge:.3f}, {4*edge:.3f}]", flush=True)


def build_groups(outdir, q0lo, q0mid, q0hi):
    NI_F = lambda: sysmod.ni_fcc(False)   # noqa: E731
    NI_S = lambda: sysmod.ni_fcc(True)    # noqa: E731

    def K(q0, frame="moment"):
        return dict(q0=q0, frame=frame, form="kerker")

    def ni_socfree():
        # SOC-free Ni: the cleanest transverse-amplification signal (m_x,m_y
        # start at machine zero). baseline, q0 scan (moment frame), lab frame,
        # and the flat null.
        yield run_nc("nisf_baseline", NI_F(), z(0.6), outdir)
        yield run_nc("nisf_kerker_qlo", NI_F(), z(0.6), outdir, damp=K(q0lo))
        yield run_nc("nisf_kerker_qmid", NI_F(), z(0.6), outdir, damp=K(q0mid))
        yield run_nc("nisf_kerker_qhi", NI_F(), z(0.6), outdir, damp=K(q0hi))
        yield run_nc("nisf_kerker_qmid_lab", NI_F(), z(0.6), outdir,
                     damp=K(q0mid, frame="lab"))
        yield run_nc("nisf_flat_a03", NI_F(), z(0.6), outdir,
                     damp=dict(form="flat", alpha_perp=0.3, frame="moment"))
        yield run_nc("nisf_flat_a01", NI_F(), z(0.6), outdir,
                     damp=dict(form="flat", alpha_perp=0.1, frame="moment"))

    def ni_soc():
        # Ni + SOC, stock pulay: transverse seeded at ~1e-3 by SOC.
        yield run_nc("nis_baseline", NI_S(), z(0.6), outdir)
        yield run_nc("nis_kerker_qmid", NI_S(), z(0.6), outdir, damp=K(q0mid))
        yield run_nc("nis_kerker_qlo", NI_S(), z(0.6), outdir, damp=K(q0lo))
        yield run_nc("nis_flat_a03", NI_S(), z(0.6), outdir,
                     damp=dict(form="flat", alpha_perp=0.3, frame="moment"))

    def ni_soc_best():
        # does damping stack with the campaign best arm (johnson+quad+alpha0.3)?
        yield run_nc("nisbest_baseline", NI_S(), z(0.6), outdir, **BEST)
        yield run_nc("nisbest_kerker_qmid", NI_S(), z(0.6), outdir,
                     damp=K(q0mid), **BEST)
        # and the socfree best arm (its floor was the campaign's lowest)
        yield run_nc("nisfbest_baseline", NI_F(), z(0.6), outdir, **BEST)
        yield run_nc("nisfbest_kerker_qmid", NI_F(), z(0.6), outdir,
                     damp=K(q0mid), **BEST)

    def canted():
        # THE kill-criterion: two moments seeded 90 deg apart must still align.
        mvi = [[0.0, 0.0, 0.6], [0.6, 0.0, 0.0]]
        yield run_nc("fe2_baseline", sysmod.fe_bcc_2atom(False), mvi, outdir,
                     track_per_atom=True)
        yield run_nc("fe2_kerker_qmid", sysmod.fe_bcc_2atom(False), mvi, outdir,
                     track_per_atom=True, damp=K(q0mid))
        yield run_nc("fe2_kerker_qlo", sysmod.fe_bcc_2atom(False), mvi, outdir,
                     track_per_atom=True, damp=K(q0lo))
        # SOC canted too
        yield run_nc("fe2soc_baseline", sysmod.fe_bcc_2atom(True), mvi, outdir,
                     track_per_atom=True)
        yield run_nc("fe2soc_kerker_qmid", sysmod.fe_bcc_2atom(True), mvi, outdir,
                     track_per_atom=True, damp=K(q0mid))

    return {"ni_socfree": ni_socfree, "ni_soc": ni_soc,
            "ni_soc_best": ni_soc_best, "canted": canted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--host", default=platform.node().split(".")[0])
    ap.add_argument("--inspect", action="store_true")
    # soft-sector edge (bottom 2 of 12 |G| shells) is ~4 1/Ang for these cells;
    # bracket it below / at / above (see --inspect).
    ap.add_argument("--q0lo", type=float, default=2.0)
    ap.add_argument("--q0mid", type=float, default=4.0)
    ap.add_argument("--q0hi", type=float, default=8.0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.inspect:
        inspect()
        return
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
