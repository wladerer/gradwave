#!/usr/bin/env python3
"""Cloud-GPU benchmark orchestrator for the 4x H100 pack.

Four lanes, each pinned to one GPU by ``CUDA_VISIBLE_DEVICES`` (set by
``run_all.sh`` before this script starts), each writing one JSON record per run
into ``benchmarks/results/<hostname>/`` -- the same directory convention
``benchmarks/_capture.py`` already uses, so ``collect.py`` and any future
dashboard read one tree.

    orchestrator.py --lane {0,1,2,3}      run one lane's runs sequentially
    orchestrator.py --lane 0 --dry-run    print the lane's commands, run nothing
    orchestrator.py --smoke               tiny CPU Si SCF through the JSON path

Resilience contract (every run must satisfy it, because lanes run unattended):
  * each sub-run is wrapped in its own timeout and try/except -- one crash never
    kills the rest of the lane;
  * a partial or failed run still writes a JSON record (status TIMEOUT/FAIL/ERR)
    so ``collect.py`` bundles evidence of every attempt, not just the winners.

Nothing here mutates ``src/`` defaults. The one arm that toggles an internal
knob (lane 2, the CPU-offloaded QR from PR #174) monkeypatches the module
global for the duration of that arm only, then restores it.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "benchmarks" / "results"
HOST = socket.gethostname()
RY = 13.605693122994

# CPU thread cap for the on-box CPU arms. H100 rentals ship many vCPUs, but the
# SCF thread-scaling plateaus early (see MEMORY: "Cap SCF threads at ~8 on
# asus"), so cap rather than grab every core.
CPU_THREADS = min(os.cpu_count() or 8, int(os.environ.get("H100_CPU_THREADS", "16")))

# Per-run wall ceilings (seconds). Generous for an H100 but finite so a wedged
# run reports TIMEOUT instead of hanging the whole session.
T_SCF = int(os.environ.get("H100_T_SCF", "1800"))
T_MINERAL = int(os.environ.get("H100_T_MINERAL", "3600"))
T_DELTA_EL = int(os.environ.get("H100_T_DELTA", "3600"))
T_MGPU = int(os.environ.get("H100_T_MGPU", "900"))  # 15 min hard cap on the probe


def _sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


SHA = _sha()


def _write_record(name: str, record: dict) -> Path:
    outdir = RESULTS / HOST
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = outdir / f"{name}-{SHA}-{ts}.json"
    path.write_text(json.dumps(record, indent=1))
    print(f"[rec] {name}: {record.get('status', '?')} -> {path}", flush=True)
    return path


def _base_record(name: str, extra: dict | None = None) -> dict:
    rec = {
        "name": name, "host": HOST, "git_sha": SHA,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_threads_cap": CPU_THREADS,
    }
    if extra:
        rec.update(extra)
    return rec


def _run_subprocess(name: str, cmd: list[str], timeout_s: int,
                    env: dict | None = None, extra: dict | None = None) -> dict:
    """Run cmd under a timeout, capture output, always write a JSON record."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=str(REPO), env=env,
        )
        status = "ok" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        status = "TIMEOUT"
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        rc = 124
    except Exception as e:  # noqa: BLE001 -- never let one run kill the lane
        status, stdout, stderr, rc = f"ERR({type(e).__name__})", "", repr(e), 1
    wall = time.time() - t0
    lines = [ln for ln in (stdout or "").splitlines() if ln.strip()]
    reported = lines[-1] if lines else (
        (stderr or "").strip().splitlines()[-1:] or [""])[0]
    rec = _base_record(name, {
        "cmd": cmd, "wall_s": round(wall, 2), "status": status, "returncode": rc,
        "reported": reported, "stdout_tail": lines[-15:],
        "stderr_tail": (stderr or "").splitlines()[-15:],
    })
    if extra:
        rec.update(extra)
    _write_record(name, rec)
    return rec


def _child_env(**overrides: str) -> dict:
    env = dict(os.environ)
    env.update(overrides)
    return env


# --------------------------------------------------------------------------- #
# Silicon supercell builder (lane 3 memory ladder)
# --------------------------------------------------------------------------- #
_SI_CONV_A = 5.43  # Å, conventional diamond-cubic cell
_SI_CONV_FRAC = np.array([
    [0.00, 0.00, 0.00], [0.00, 0.50, 0.50], [0.50, 0.00, 0.50], [0.50, 0.50, 0.00],
    [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
])
# (nx, ny, nz) tilings of the 8-atom conventional cell -> the requested ladder.
_SI_LADDER = [
    (1, 1, 1),   # 8
    (2, 1, 1),   # 16
    (2, 2, 1),   # 32
    (2, 2, 2),   # 64
    (3, 2, 2),   # 96
    (4, 2, 2),   # 128
    (5, 2, 2),   # 160
    (3, 3, 3),   # 216
]


def _si_supercell(tiling: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, int]:
    nx, ny, nz = tiling
    cell = np.diag([_SI_CONV_A * nx, _SI_CONV_A * ny, _SI_CONV_A * nz])
    frac = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = np.array([i, j, k], dtype=float)
                frac.append((_SI_CONV_FRAC + shift) / np.array([nx, ny, nz]))
    frac_all = np.concatenate(frac, axis=0)
    pos = frac_all @ cell
    return cell, pos, len(pos)


# ==========================================================================  #
# LANE 0 -- bench_scf + minerals, GPU and CPU
# ==========================================================================  #
def lane0(dry_run: bool) -> None:
    py = sys.executable
    minerals = REPO / "benchmarks" / "minerals" / "run_bench.py"
    pdir = str(REPO / "tests" / "fixtures" / "qe" / "pseudos")

    # bench_scf: (device, threads, sym)
    scf_runs = [
        ("lane0_bench_scf_gpu", [py, str(REPO / "benchmarks" / "bench_scf.py"),
                                 "cuda", str(CPU_THREADS), "nosym"], T_SCF, None),
        ("lane0_bench_scf_cpu", [py, str(REPO / "benchmarks" / "bench_scf.py"),
                                 "cpu", str(CPU_THREADS), "nosym"], T_SCF, None),
    ]
    # minerals whose ONCV pseudos ship in tests/fixtures (NiO's Ni_ONCV_PBE-1.2
    # is NOT in fixtures, so it is skipped here -- documented in README).
    mineral_runs = []
    for mineral in ("eskolaite", "hematite"):
        for dev in ("cuda", "cpu"):
            name = f"lane0_mineral_{mineral}_{dev}"
            outdir = RESULTS / HOST / f"{name}"
            cmd = [py, str(minerals), mineral, "--pseudo-dir", pdir,
                   "--outdir", str(outdir), "--threads", str(CPU_THREADS),
                   "--device", dev, "--skip-qe"]
            mineral_runs.append((name, cmd, T_MINERAL, {"outdir": str(outdir)}))

    all_runs = scf_runs + mineral_runs
    if dry_run:
        for name, cmd, t, _ in all_runs:
            print(f"[lane0] {name} (timeout {t}s):\n    {' '.join(cmd)}")
        return
    for name, cmd, t, extra in all_runs:
        _run_subprocess(name, cmd, t, extra=extra)


# ==========================================================================  #
# LANE 1 -- delta-gauge subset Al, Cu, Si, Ge on the GPU
# ==========================================================================  #
def lane1(dry_run: bool) -> None:
    py = sys.executable
    run_gw = REPO / "benchmarks" / "delta_gauge" / "run_gw.py"
    eos_dir = REPO / "benchmarks" / "delta_gauge" / "results"
    elements = ["Al", "Cu", "Si", "Ge"]
    env = _child_env(GW_DEVICE="cuda", GW_THREADS=str(CPU_THREADS))

    if dry_run:
        for el in elements:
            print(f"[lane1] lane1_delta_{el} (timeout {T_DELTA_EL}s):\n"
                  f"    GW_DEVICE=cuda GW_THREADS={CPU_THREADS} "
                  f"{py} {run_gw} run {el}")
        return

    # Each element is its own subprocess (its own JSON, its own timeout) so a
    # crash in one leaves the others intact -- the run_gw contract already
    # writes one file per element.
    for el in elements:
        name = f"lane1_delta_{el}"
        rec = _run_subprocess(name, [py, str(run_gw), "run", el], T_DELTA_EL, env=env)
        # copy the per-element EOS json into the results tree so collect.py
        # bundles it without needing to reach into delta_gauge/results.
        src = eos_dir / f"eos_{el}.json"
        if src.exists():
            dst = RESULTS / HOST / f"lane1_eos_{el}.json"
            dst.write_text(src.read_text())
            rec["eos_copied_to"] = str(dst)


# ==========================================================================  #
# LANE 2 -- A/B matrix of 3050-era optimizations on one midweight system
# ==========================================================================  #
# System: Cr2O3 eskolaite (corundum AFM, 10 atoms, nspin=2, 4x4x4). Justified:
#   * its Cr+O ONCV pseudos ship in tests/fixtures (self-contained);
#   * nspin=2 doubles the band block, so the subspace is wide enough that the
#     Rayleigh-Ritz GEMM and the tall-skinny QR (the two ops the 3050-era
#     offload tricks target) are actually exercised -- the same regime the
#     RR-GEMM case study in docs/manual/performance.md used (hematite/eskolaite
#     shapes), not a toy 2-atom cell where those ops are negligible;
#   * it is the fastest-converging of the 10-atom corundum minerals (1128 s CPU
#     on asus vs hematite's 2275 s), keeping six SCF arms inside the budget.
def _build_eskolaite():
    import structures as S  # noqa: PLC0415

    from gradwave.pseudo.upf import parse_upf  # noqa: PLC0415
    m = S.build("eskolaite")
    pdir = REPO / "tests" / "fixtures" / "qe" / "pseudos"
    upfs = [parse_upf(pdir / p) for p in m.pseudos]
    return m, upfs


def _eskolaite_system(device: str):
    from gradwave.scf.loop import setup_system  # noqa: PLC0415
    m, upfs = _build_eskolaite()
    system = setup_system(
        m.cell, m.positions, m.species, upfs, ecut=m.ecut_ry * RY,
        kmesh=m.kmesh, use_symmetry=True, magmoms=m.magmoms,
        collinear_magnetic=True,
    )
    if device != "cpu":
        system = system.to(device)
    return system, m


def _run_eskolaite_scf(system, m, eigensolver: str, device: str, max_iter: int):
    import torch  # noqa: PLC0415

    from gradwave.core.xc.spin import SpinPBE  # noqa: PLC0415
    from gradwave.scf.loop import scf  # noqa: PLC0415
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    r = scf(system, SpinPBE(), nspin=2, start_mag=m.start_mag,
            smearing="gaussian", width=m.degauss_ry * RY, etol=1e-8,
            rhotol=1e-7, max_iter=max_iter, mixing_scheme="johnson",
            eigensolver=eigensolver, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9 if device != "cpu" else 0.0)
    return {
        "wall_s": round(wall, 2), "n_iter": int(r.n_iter),
        "converged": bool(r.converged), "etot_eV": float(r.energies.total),
        "peak_gb": round(peak_gb, 3),
    }


def lane2(dry_run: bool) -> None:
    if dry_run:
        print("[lane2] in-process A/B matrix on Cr2O3 eskolaite (device cuda):")
        print("    (a) CPU-offloaded batched QR (PR #174): on vs off "
              "(monkeypatch davidson._QR_CPU_MAX_COLS 16<->0)")
        print("    (b) eigensolver: davidson vs chebyshev")
        print("    (c) RR-GEMM residency microbench: fp64 GPU vs CPU-resident "
              "vs fp32-compute (diagnostic, no SCF)")
        return

    sys.path.insert(0, str(REPO / "benchmarks" / "minerals"))
    import importlib  # noqa: PLC0415

    import torch  # noqa: PLC0415

    # gradwave.solvers.__init__ re-exports a `davidson` FUNCTION, which shadows
    # the submodule attribute, so `import ... as dv` would bind the function.
    # import_module returns the real module from sys.modules.
    dv = importlib.import_module("gradwave.solvers.davidson")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_iter = int(os.environ.get("H100_LANE2_MAXITER", "60"))
    arms: list[dict] = []

    # --- Pair (a): QR CPU-offload on vs off ------------------------------- #
    # PR #174's offload is a CUDA-only, hardcoded device-conditional in
    # davidson._qr_offload, gated by the module global _QR_CPU_MAX_COLS (16).
    # There is no env-var knob in src, so the benchmark harness toggles it by
    # monkeypatching the module global: 16 = offload ON (default), 0 = OFF
    # (cols <= 0 is never true, so QR stays on-GPU). src defaults are restored
    # after. Applies only on CUDA (the offload is a no-op on CPU).
    orig_qr = dv._QR_CPU_MAX_COLS
    for label, max_cols in (("qr_offload_on", orig_qr), ("qr_offload_off", 0)):
        name = f"lane2a_{label}"
        try:
            dv._QR_CPU_MAX_COLS = max_cols
            system, m = _eskolaite_system(device)
            res = _run_eskolaite_scf(system, m, "davidson", device, max_iter)
            res.update(arm=label, pair="qr_offload", eigensolver="davidson",
                       device=device, qr_cpu_max_cols=max_cols, status="ok")
        except Exception as e:  # noqa: BLE001
            res = {"arm": label, "pair": "qr_offload", "status": f"ERR({type(e).__name__})",
                   "error": repr(e), "trace": traceback.format_exc()[-1500:], "device": device}
        finally:
            dv._QR_CPU_MAX_COLS = orig_qr
        arms.append(res)
        _write_record(name, _base_record(name, res))

    # --- Pair (b): davidson vs chebyshev ---------------------------------- #
    for solver in ("davidson", "chebyshev"):
        name = f"lane2b_{solver}"
        try:
            system, m = _eskolaite_system(device)
            res = _run_eskolaite_scf(system, m, solver, device, max_iter)
            res.update(arm=solver, pair="eigensolver", eigensolver=solver,
                       device=device, status="ok")
        except Exception as e:  # noqa: BLE001
            res = {"arm": solver, "pair": "eigensolver", "status": f"ERR({type(e).__name__})",
                   "error": repr(e), "trace": traceback.format_exc()[-1500:], "device": device}
        arms.append(res)
        _write_record(name, _base_record(name, res))

    # --- Identical-energy oracle across SCF arms -------------------------- #
    energies = [a["etot_eV"] for a in arms if a.get("etot_eV") is not None
                and a.get("converged")]
    oracle = {"arm": "oracle", "n_converged_arms": len(energies)}
    if len(energies) >= 2:
        spread = max(energies) - min(energies)
        oracle.update(max_energy_spread_eV=spread,
                      max_spread_meV_per_atom=spread / 10 * 1000,  # 10 atoms
                      oracle_pass=bool(spread < 1e-4))
    _write_record("lane2_energy_oracle", _base_record("lane2_energy_oracle", oracle))

    # --- Pair (c): RR-GEMM residency microbench (diagnostic) -------------- #
    # Per docs/manual/performance.md: an ISOLATED GEMM comparison (not in-situ),
    # at the eskolaite/hematite subspace shape. Times the subspace build
    # s = v.conj() @ hv.mT in fp64 on GPU, fp64 CPU-resident, and fp32-compute
    # / fp64-accumulate on GPU. Diagnostic only -- the doc already found the
    # residency redesign does not pay off and fp32-RR breaks convergence.
    name = "lane2c_rrgemm_microbench"
    try:
        res = _rrgemm_microbench(device)
        res.update(status="ok")
    except Exception as e:  # noqa: BLE001
        res = {"status": f"ERR({type(e).__name__})", "error": repr(e),
               "trace": traceback.format_exc()[-1500:]}
    _write_record(name, _base_record(name, res))


def _rrgemm_microbench(device: str, nk: int = 13, dim: int = 240,
                       npw: int = 6746, reps: int = 20) -> dict:
    import torch  # noqa: PLC0415

    def _time(fn) -> float:
        for _ in range(3):  # warmup
            fn()
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if device != "cpu":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1e3  # ms/call

    out = {"shape": {"nk": nk, "dim": dim, "npw": npw}, "reps": reps}
    if device == "cpu":
        v = torch.randn(nk, dim, npw, dtype=torch.complex128)
        hv = torch.randn(nk, dim, npw, dtype=torch.complex128)
        out["fp64_cpu_ms"] = round(_time(lambda: v.conj() @ hv.mT), 3)
        out["note"] = "CPU-only host: GPU-resident arms skipped"
        return out

    v = torch.randn(nk, dim, npw, dtype=torch.complex128, device="cuda")
    hv = torch.randn(nk, dim, npw, dtype=torch.complex128, device="cuda")
    v_c = v.cpu()
    hv_c = hv.cpu()
    v32 = v.to(torch.complex64)
    hv32 = hv.to(torch.complex64)
    out["fp64_gpu_ms"] = round(_time(lambda: v.conj() @ hv.mT), 3)
    out["fp64_cpu_resident_ms"] = round(_time(lambda: v_c.conj() @ hv_c.mT), 3)
    out["fp32compute_gpu_ms"] = round(
        _time(lambda: (v32.conj() @ hv32.mT).to(torch.complex128)), 3)
    return out


# ==========================================================================  #
# LANE 3 -- memory-ceiling ladder + multi-GPU curiosity probe
# ==========================================================================  #
def lane3(dry_run: bool) -> None:
    ecut_ry = float(os.environ.get("H100_LANE3_ECUT_RY", "30"))
    max_iter = int(os.environ.get("H100_LANE3_MAXITER", "20"))
    if dry_run:
        print(f"[lane3] Si supercell memory ladder (Gamma-only, ecut={ecut_ry} Ry, "
              f"max_iter={max_iter}):")
        for tiling in _SI_LADDER:
            _, _, nat = _si_supercell(tiling)
            print(f"    Si x{tiling} -> {nat} atoms")
        print(f"[lane3] multi-GPU probe: torchrun --nproc-per-node 4 (Gloo, "
              f"distributed:true, cuda), hard {T_MGPU}s cap [UNTESTED]")
        return

    _lane3_ladder(ecut_ry, max_iter)
    _lane3_multigpu_probe()


def _lane3_ladder(ecut_ry: float, max_iter: int) -> None:
    import torch  # noqa: PLC0415

    from gradwave.core.xc.pbe import PBE  # noqa: PLC0415
    from gradwave.pseudo.upf import parse_upf  # noqa: PLC0415
    from gradwave.scf.loop import scf, setup_system  # noqa: PLC0415

    if not torch.cuda.is_available():
        _write_record("lane3_ladder_skip", _base_record(
            "lane3_ladder_skip", {"status": "SKIP", "reason": "no CUDA device"}))
        return

    pdir = REPO / "tests" / "fixtures" / "qe" / "pseudos"
    si = parse_upf(pdir / "Si_ONCV_PBE-1.2.upf")

    for tiling in _SI_LADDER:
        cell, pos, nat = _si_supercell(tiling)
        name = f"lane3_ladder_Si{nat}"
        rec = _base_record(name, {"natoms": nat, "tiling": list(tiling),
                                  "ecut_ry": ecut_ry, "kmesh": [1, 1, 1]})
        try:
            system = setup_system(cell, pos, [0] * nat, [si], ecut=ecut_ry * RY,
                                  kmesh=(1, 1, 1), use_symmetry=False)
            npw = max((int(s.mask.sum()) for s in system.spheres
                       if hasattr(s, "mask")), default=0)
            system = system.to("cuda")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            r = scf(system, PBE(), smearing="gaussian", width=0.1, etol=1e-8,
                    rhotol=1e-7, max_iter=max_iter, verbose=False)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            rec.update(status="ok", oom=False, npw_max=npw, wall_s=round(wall, 2),
                       n_iter=int(r.n_iter), converged=bool(r.converged),
                       wall_per_iter_s=round(wall / max(r.n_iter, 1), 3),
                       peak_gb=round(peak_gb, 3), etot_eV=float(r.energies.total))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            is_oom = "out of memory" in str(e).lower() or isinstance(
                e, torch.cuda.OutOfMemoryError)
            rec.update(status="OOM" if is_oom else f"ERR({type(e).__name__})",
                       oom=is_oom, error=repr(e)[:800])
            torch.cuda.empty_cache()
            _write_record(name, rec)
            if is_oom:
                # larger cells will only OOM harder -- stop the ladder here.
                _write_record("lane3_ladder_oom_point", _base_record(
                    "lane3_ladder_oom_point",
                    {"status": "ok", "oom_at_natoms": nat, "tiling": list(tiling)}))
                return
            continue
        except Exception as e:  # noqa: BLE001
            rec.update(status=f"ERR({type(e).__name__})", error=repr(e)[:800],
                       trace=traceback.format_exc()[-1200:])
        _write_record(name, rec)


def _physical_gpu_count() -> int:
    """Number of GPUs on the box, IGNORING any ``CUDA_VISIBLE_DEVICES`` pin.

    run_all.sh pins each lane to a single GPU, and by the time this runs the
    lane-3 ladder has already initialised CUDA under that pin — so
    ``torch.cuda.device_count()`` is both pin-limited and cached at 1, which
    made the multi-GPU probe self-skip (``gpu_count=1``). Count the physical
    devices with ``nvidia-smi -L`` instead (unaffected by the pin), falling
    back to the pinned torch count only if nvidia-smi is unavailable.
    """
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=10)
        n = sum(1 for ln in out.stdout.splitlines()
                if ln.strip().startswith("GPU "))
        if n:
            return n
    except Exception:  # noqa: BLE001 -- no nvidia-smi -> fall back to torch
        pass
    import torch  # noqa: PLC0415

    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _lane3_multigpu_probe() -> None:
    """torchrun 4-rank k-sharded SCF on 4 GPUs. UNTESTED end to end (no H100
    here). Captures whether it runs and the verbatim error if not, plus the
    speedup vs a 1-GPU single-process run of the same input."""
    # Count physical GPUs ignoring the lane's single-GPU pin (see
    # _physical_gpu_count) — the pinned torch count would falsely trip the
    # >=2-GPU guard below and self-skip the probe.
    ngpu = _physical_gpu_count()
    inp = HERE / "inputs" / "si_dist.yaml"
    wrap = HERE / "mgpu_wrap.sh"
    py = sys.executable

    if ngpu < 2:
        _write_record("lane3_mgpu_probe", _base_record(
            "lane3_mgpu_probe", {"status": "SKIP", "gpu_count": ngpu,
                                 "reason": "need >=2 GPUs for the probe"}))
        return

    # This lane is pinned to a single GPU by run_all.sh; the probe needs ALL of
    # them, so it deliberately clears CUDA_VISIBLE_DEVICES and lets mgpu_wrap.sh
    # re-pin each rank by LOCAL_RANK.
    env = _child_env(GW_PY=py)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    nproc = min(ngpu, 4)

    # 1-GPU baseline (same input, single process -> WORLD_SIZE unset -> ordinary
    # path). Pin to GPU 0.
    base_env = _child_env(CUDA_VISIBLE_DEVICES="0")
    outdir_base = RESULTS / HOST / "lane3_mgpu_1gpu"
    base = _run_subprocess(
        "lane3_mgpu_1gpu",
        [py, "-m", "gradwave.cli", "run", str(inp), "-o", str(outdir_base)],
        T_MGPU, env=base_env)

    # 4-rank torchrun probe.
    cmd = [py, "-m", "torch.distributed.run", "--nproc-per-node", str(nproc),
           "--no-python", str(wrap), str(inp)]
    probe = _run_subprocess("lane3_mgpu_probe", cmd, T_MGPU, env=env,
                            extra={"nproc": nproc, "gpu_count": ngpu,
                                   "untested": True})

    if base.get("status") == "ok" and probe.get("status") == "ok" \
            and base.get("wall_s") and probe.get("wall_s"):
        speedup = base["wall_s"] / probe["wall_s"]
        _write_record("lane3_mgpu_speedup", _base_record(
            "lane3_mgpu_speedup",
            {"status": "ok", "nproc": nproc, "wall_1gpu_s": base["wall_s"],
             "wall_mgpu_s": probe["wall_s"], "speedup_vs_1gpu": round(speedup, 3)}))


# ==========================================================================  #
# smoke -- tiny CPU Si SCF through the JSON-writing path
# ==========================================================================  #
def smoke() -> int:
    import torch  # noqa: PLC0415

    from gradwave.core.xc.lda_pw92 import LDA_PW92  # noqa: PLC0415
    from gradwave.pseudo.upf import parse_upf  # noqa: PLC0415
    from gradwave.scf.loop import scf, setup_system  # noqa: PLC0415
    torch.set_num_threads(2)
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    si = parse_upf(REPO / "tests" / "fixtures" / "qe" / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    t0 = time.perf_counter()
    system = setup_system(cell, pos, [0, 0], [si], ecut=12 * RY, kmesh=(2, 2, 2),
                          use_symmetry=False)
    r = scf(system, LDA_PW92(), smearing="none", etol=1e-6, rhotol=1e-5,
            max_iter=40, verbose=False)
    wall = time.perf_counter() - t0
    rec = _base_record("smoke_si_cpu", {
        "status": "ok", "device": "cpu", "wall_s": round(wall, 2),
        "n_iter": int(r.n_iter), "converged": bool(r.converged),
        "etot_eV": float(r.energies.total), "nk": len(system.spheres),
    })
    path = _write_record("smoke_si_cpu", rec)
    print(f"[smoke] E={rec['etot_eV']:.6f} eV  it={rec['n_iter']}  "
          f"conv={rec['converged']}  {rec['wall_s']}s  -> {path}")
    return 0


LANES = {0: lane0, 1: lane1, 2: lane2, 3: lane3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", type=int, choices=sorted(LANES))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the lane's commands without running them")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU Si SCF through the JSON path (no GPU)")
    a = ap.parse_args()

    if a.smoke:
        return smoke()
    if a.lane is None:
        ap.error("one of --lane or --smoke is required")
    print(f"[orchestrator] lane {a.lane} on {HOST} (sha {SHA}), "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
          f"dry_run={a.dry_run}", flush=True)
    LANES[a.lane](a.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
