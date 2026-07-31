#!/usr/bin/env python3
"""Bundle the H100 results and emit a comparison table.

  uv run python benchmarks/h100/collect.py

Does three things:
  1. tars benchmarks/results/<hostname>/ into one archive;
  2. prints the scp line to pull it back to a workstation;
  3. writes comparison.md -- a table skeleton pre-filled with the recorded
     RTX 3050 / asus-CPU reference numbers (quoted verbatim from
     benchmarks/minerals/README.md and benchmarks/delta_gauge/README.md), with
     the H100 columns filled from whatever JSON records this run produced. A
     missing run just leaves its H100 cell blank -- partial results still
     tabulate.

Reference numbers are quoted, not recomputed, so the table stays honest about
what was measured where.
"""

from __future__ import annotations

import json
import socket
import tarfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HOST = socket.gethostname()
RESULTS = REPO / "benchmarks" / "results" / HOST

# --- reference numbers, quoted verbatim from the committed READMEs --------- #
# benchmarks/minerals/README.md "Results" table (asus, Intel 22-core, gradwave =
# 8 torch threads, PBE ONCV NC). gradwave wall = setup + SCF on CPU.
MINERAL_REF = {
    # mineral: (atoms, asus_cpu_wall_s, iters, dE_elec_meV_per_atom)
    "eskolaite": (10, 1128.5, 16, -0.011),
    "hematite": (10, 2274.9, 17, -0.013),
}
# hematite GPU reference from docs/manual/performance.md: RTX 3050 GPU 586.8 s
# vs CPU 2274.9 s (3.9x) at this shape.
HEMATITE_GPU_REF_3050_S = 586.8

# benchmarks/delta_gauge: Δ_pdojo (PseudoDojo published Δ-factor for the same
# standard pseudo) from cases.py dfact=..., the reference the gradwave Δ tracks.
DELTA_REF_PDOJO = {"Al": 0.542, "Cu": 0.533, "Si": 0.146, "Ge": 0.485}
# delta_gauge/README.md device note, quoted: "the RTX 3050 beats the 22-core CPU
# 2.6-5.6x (Al/Cu/Si)"; "An 8-core laptop CPU ... ran Ge at 435 s/vol vs ~8
# s/vol on the GPU." per-element RTX 3050 s/vol from results/timing.json.
DELTA_REF_3050_SPER_VOL = {"Al": 14.0, "Cu": 96.9, "Si": 45.8, "Ge": 68.7}


def _load_records() -> dict[str, dict]:
    """name-prefix -> most recent record (records are <name>-<sha>-<ts>.json)."""
    out: dict[str, dict] = {}
    if not RESULTS.exists():
        return out
    for path in sorted(RESULTS.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        name = rec.get("name") or path.stem.rsplit("-", 2)[0]
        # keep the newest by filename order (timestamps sort lexically)
        out[name] = rec
    return out


def _load_eos(el: str) -> dict | None:
    p = RESULTS / f"lane1_eos_{el}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _fmt(v, spec: str = "") -> str:
    if v is None:
        return ""
    try:
        return format(v, spec) if spec else str(v)
    except (TypeError, ValueError):
        return str(v)


def _mineral_wall(rec: dict | None) -> tuple[str, str]:
    """(wall_s, n_iter) from a lane-0 mineral capture's own result.json, if any."""
    if not rec:
        return "", ""
    outdir = rec.get("outdir")
    if outdir:
        # run_bench writes result.json into <outdir>/<mineral>/
        for sub in Path(outdir).glob("*/result.json"):
            try:
                row = json.loads(sub.read_text())
                gw = row.get("gradwave", {})
                return _fmt(gw.get("wall_s"), ".1f"), _fmt(gw.get("n_iter"))
            except Exception:
                pass
    return _fmt(rec.get("wall_s"), ".1f"), ""


def build_markdown(recs: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append(f"# H100 benchmark pack -- results ({HOST})")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).isoformat()} from "
                 f"`benchmarks/results/{HOST}/`.")
    lines.append("")

    env = RESULTS / "torch_env.txt"
    if env.exists():
        lines.append("**Environment**: `" + env.read_text().strip() + "`")
        lines.append("")

    # ---- Lane 0: bench_scf ------------------------------------------------ #
    lines.append("## Lane 0 -- bench_scf (Si LDA, 30 Ry, 4x4x4)")
    lines.append("")
    lines.append("| device | H100 wall (s) | H100 iters | status |")
    lines.append("|---|---:|---:|---|")
    for dev, key in (("GPU", "lane0_bench_scf_gpu"), ("CPU", "lane0_bench_scf_cpu")):
        r = recs.get(key, {})
        lines.append(f"| {dev} | {_fmt(r.get('wall_s'), '.1f')} | | "
                     f"{r.get('status', '')} |")
    lines.append("")
    lines.append("*Reported line (energy/iters) is in each run's JSON `reported` field.*")
    lines.append("")

    # ---- Lane 0: minerals ------------------------------------------------- #
    lines.append("## Lane 0 -- minerals (Cr2O3, Fe2O3; skip-QE)")
    lines.append("")
    lines.append("Reference = asus 22-core CPU, gradwave 8 threads "
                 "(benchmarks/minerals/README.md). NiO omitted: its "
                 "Ni_ONCV_PBE-1.2 pseudo is not in tests/fixtures.")
    lines.append("")
    lines.append("| mineral | atoms | asus-CPU ref wall (iters) | H100 GPU wall | "
                 "H100 CPU wall | H100 GPU iters |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for mineral, (nat, ref_wall, ref_it, _de) in MINERAL_REF.items():
        gpu_wall, gpu_it = _mineral_wall(recs.get(f"lane0_mineral_{mineral}_cuda"))
        cpu_wall, _ = _mineral_wall(recs.get(f"lane0_mineral_{mineral}_cpu"))
        ref = f"{ref_wall:.1f} s ({ref_it})"
        if mineral == "hematite":
            ref += f"; 3050-GPU {HEMATITE_GPU_REF_3050_S:.1f} s"
        lines.append(f"| {mineral} | {nat} | {ref} | {gpu_wall} | {cpu_wall} | {gpu_it} |")
    lines.append("")

    # ---- Lane 1: delta-gauge --------------------------------------------- #
    lines.append("## Lane 1 -- delta-gauge subset (Al, Cu, Si, Ge)")
    lines.append("")
    lines.append("Reference: Delta_pdojo is PseudoDojo's published Delta-factor "
                 "for the same standard pseudo (the axis the gradwave Delta "
                 "tracks). RTX 3050 s/vol from delta_gauge/results/timing.json; "
                 'the README notes "the RTX 3050 beats the 22-core CPU 2.6-5.6x '
                 '(Al/Cu/Si)".')
    lines.append("")
    lines.append("| element | struct | Delta_pdojo ref | RTX3050 ref s/vol | "
                 "H100 GPU s/vol | H100 min iters |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for el in ("Al", "Cu", "Si", "Ge"):
        eos = _load_eos(el)
        struct = eos.get("struct", "") if eos else ""
        h100_sper = ""
        h100_it = ""
        # H100 GPU s/vol: MEASURED from this lane's own subprocess wall time
        # (the lane1_delta_<el> record), divided by the number of EOS volumes,
        # to match the RTX 3050 reference's total_s / 7 definition in
        # results/timing.json. The eos_<el>.json "sec" field times only the SCF
        # (excluding per-run setup/import), so it is NOT comparable to the
        # reference and must not fill this column.
        rec = recs.get(f"lane1_delta_{el}", {})
        n_vol = len(eos["sec"]) if eos and eos.get("sec") else 7
        wall = rec.get("wall_s")
        if wall and n_vol:
            h100_sper = f"{wall / n_vol:.1f}"
        if eos and eos.get("it"):
            its = [v for v in eos["it"].values()]
            if its:
                h100_it = f"{min(its)}"
        lines.append(f"| {el} | {struct} | {DELTA_REF_PDOJO[el]:.3f} | "
                     f"{DELTA_REF_3050_SPER_VOL[el]:.1f} | {h100_sper} | {h100_it} |")
    lines.append("")
    lines.append("*Run `uv run python benchmarks/delta_gauge/fit_delta.py` after "
                 "pulling results back for the fitted H100 Delta vs WIEN2k.*")
    lines.append("")

    # ---- Lane 2: A/B matrix ---------------------------------------------- #
    lines.append("## Lane 2 -- A/B matrix (Cr2O3 eskolaite, GPU)")
    lines.append("")
    lines.append("| pair | arm | wall (s) | iters | E_total (eV) | converged |")
    lines.append("|---|---|---:|---:|---:|---|")
    for key in ("lane2a_qr_offload_on", "lane2a_qr_offload_off",
                "lane2b_davidson", "lane2b_chebyshev"):
        r = recs.get(key, {})
        lines.append(f"| {r.get('pair', '')} | {r.get('arm', key)} | "
                     f"{_fmt(r.get('wall_s'), '.1f')} | {_fmt(r.get('n_iter'))} | "
                     f"{_fmt(r.get('etot_eV'), '.6f')} | {_fmt(r.get('converged'))} |")
    oracle = recs.get("lane2_energy_oracle", {})
    if oracle:
        lines.append("")
        lines.append(f"Energy oracle: {oracle.get('n_converged_arms', 0)} converged "
                     f"arms, max spread "
                     f"{_fmt(oracle.get('max_spread_meV_per_atom'), '.3g')} meV/atom, "
                     f"pass={oracle.get('oracle_pass')}.")
    rr = recs.get("lane2c_rrgemm_microbench", {})
    if rr:
        ms_keys = ("fp64_gpu_ms", "fp64_cpu_resident_ms",
                   "fp32compute_gpu_ms", "fp64_cpu_ms")
        ms = {k: v for k, v in rr.items() if k in ms_keys}
        lines.append("")
        lines.append(f"RR-GEMM microbench (diagnostic): {json.dumps(ms)}")
    lines.append("")

    # ---- Lane 3: ladder + mgpu ------------------------------------------- #
    lines.append("## Lane 3 -- Si memory-ceiling ladder + multi-GPU probe")
    lines.append("")
    lines.append("Reference (docs/ideas.md): on the 6 GB RTX 3050 peak memory "
                 "scaled roughly linearly to ~96 atoms, then a hard cliff at 128 "
                 "atoms from a single ~37 GB O(npw^2) allocation.")
    lines.append("")
    lines.append("| atoms | npw_max | peak GB | wall/iter (s) | iters | status |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for nat in (8, 16, 32, 64, 96, 128, 160, 216):
        r = recs.get(f"lane3_ladder_Si{nat}", {})
        lines.append(f"| {nat} | {_fmt(r.get('npw_max'))} | "
                     f"{_fmt(r.get('peak_gb'), '.3f')} | "
                     f"{_fmt(r.get('wall_per_iter_s'), '.3f')} | "
                     f"{_fmt(r.get('n_iter'))} | {r.get('status', '')} |")
    oom = recs.get("lane3_ladder_oom_point", {})
    if oom.get("oom_at_natoms"):
        lines.append("")
        lines.append(f"OOM point: {oom['oom_at_natoms']} atoms.")
    lines.append("")
    lines.append("### Multi-GPU probe (UNTESTED -- diagnostic)")
    probe = recs.get("lane3_mgpu_probe", {})
    speed = recs.get("lane3_mgpu_speedup", {})
    if probe:
        lines.append(f"- status: **{probe.get('status', '?')}** "
                     f"(nproc={probe.get('nproc', '?')}, "
                     f"gpu_count={probe.get('gpu_count', '?')})")
        if probe.get("status") != "ok":
            tail = probe.get("stderr_tail") or probe.get("stdout_tail") or []
            lines.append("- verbatim tail:")
            lines.append("```")
            lines.extend(tail[-10:])
            lines.append("```")
    if speed.get("speedup_vs_1gpu"):
        lines.append(f"- speedup vs 1 GPU: **{speed['speedup_vs_1gpu']}x** "
                     f"({speed['wall_1gpu_s']} s -> {speed['wall_mgpu_s']} s)")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not RESULTS.exists():
        print(f"[collect] no results dir at {RESULTS} -- nothing to bundle")
        return 1

    recs = _load_records()
    md = build_markdown(recs)
    md_path = RESULTS / "comparison.md"
    md_path.write_text(md)
    print(f"[collect] wrote {md_path} ({len(recs)} records)")

    tar_path = REPO / "benchmarks" / f"h100_results_{HOST}_{ts}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(RESULTS, arcname=f"results_{HOST}")
    size_mb = tar_path.stat().st_size / 1e6
    print(f"[collect] bundled -> {tar_path} ({size_mb:.1f} MB)")
    print()
    print("Pull it back with (edit host/port to your vast.ai instance):")
    print(f"  scp -P <PORT> root@<VAST_HOST>:{tar_path} .")
    print()
    print("Then locally:")
    print(f"  tar xzf {tar_path.name} && cat results_{HOST}/comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
