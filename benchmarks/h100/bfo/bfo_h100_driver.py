#!/usr/bin/env python
"""BiFeO3 PAW+U on an H100 — de-risk timing then the deflation benchmark.

Phased and defensive so a rented hour is never wasted:

  0. env snapshot (device, torch, GPU).
  1. SMOKE: a tiny BiFeO3+U SCF on the GPU. The USPP/PAW(+U) path has never run
     on CUDA before, and carries extra device-sensitive tensors (augmentation,
     one-center, +U projectors). If this errors, the run STOPS and says so — a
     real finding that saves the hour.
  2. DE-RISK: time ONE composite response apply (chi-tilde) and one SCF iteration
     on CPU vs GPU. The apply is the batched generalized Sternheimer — the fp64
     GEMM/FFT the H100 is for — so its speedup is the number that decides whether
     the response scales here.
  3. BENCH: converge BiFeO3+U on the GPU at a GPU-appropriate k-mesh, then run
     the fxc=1.5 deflation comparison (baseline Anderson vs deflated), with peak
     VRAM.

Every phase writes results/bfo_h100.json + a human log incrementally, wrapped in
try/except, so a later crash keeps the earlier numbers. Env knobs:
  BFO_PSEUDO_DIR  (dir holding the 3 PAW UPFs; default: repo fixtures)
  BFO_KMESH       (main-run k-mesh per axis; default 3)  BFO_ECUT / BFO_ECUTRHO (Ry)
  BFO_U / BFO_J   (+U, +J on Fe 3d in eV; default 4.0 / 1.0)
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from gradwave.constants import RY_EV
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.uspp_softmode import build_uspp_screening
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_hubbard import HubbardManifold
from gradwave.solvers.deflation import (
    anderson_solve,
    deflated_solve,
    soft_subspace_from_operator,
)

# --------------------------------------------------------------------------- #
HAS_CUDA = torch.cuda.is_available()
REPO = Path(__file__).resolve().parents[3]
PS = Path(os.environ.get("BFO_PSEUDO_DIR", REPO / "tests/fixtures/qe/pseudos"))
KMESH = int(os.environ.get("BFO_KMESH", "3"))
ECUT = float(os.environ.get("BFO_ECUT", "45"))
ECUTRHO = float(os.environ.get("BFO_ECUTRHO", "360"))
U = float(os.environ.get("BFO_U", "4.0"))
J = float(os.environ.get("BFO_J", "1.0"))
CPU_THREADS = int(os.environ.get("BFO_THREADS", str(min(16, os.cpu_count() or 8))))

OUT = REPO / "benchmarks/results" / os.uname().nodename
OUT.mkdir(parents=True, exist_ok=True)
JSON = OUT / "bfo_h100.json"
RES: dict = {"config": {"kmesh": KMESH, "ecut": ECUT, "ecutrho": ECUTRHO,
                        "U": U, "J": J, "has_cuda": HAS_CUDA,
                        "gpu": torch.cuda.get_device_name(0) if HAS_CUDA else None,
                        "torch": torch.__version__, "cpu_threads": CPU_THREADS}}

A = 3.965
CELL = A * np.eye(3)
FRAC = np.array([[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
POS = FRAC @ CELL
SPECIES = [0, 1, 2, 2, 2]  # Bi, Fe, O, O, O
HUB = [HubbardManifold(species=1, l=2, u=U, j=J)]


def log(msg):
    print(msg, flush=True)
    RES.setdefault("log", []).append(msg)
    JSON.write_text(json.dumps(RES, indent=2, default=str))


def sync():
    if HAS_CUDA:
        torch.cuda.synchronize()


def make_system(kmesh: int, device: str):
    bi = parse_upf_paw(PS / "Bi.pbe-dn-kjpaw_psl.1.0.0.UPF")
    fe = parse_upf_paw(PS / "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF")
    o = parse_upf_paw(PS / "O.pbe-n-kjpaw_psl.1.0.0.UPF")
    sysm = setup_uspp(CELL, POS, SPECIES, [bi, fe, o], ecut=ECUT * RY_EV,
                      ecutrho=ECUTRHO * RY_EV, kmesh=(kmesh, kmesh, kmesh))
    if device == "cuda":
        sysm = sysm.to("cuda")
    return sysm


def converge(sysm, max_iter=250):
    return scf_uspp(sysm, SpinPBE(), smearing="gaussian", width=0.05, nspin=2,
                    start_mag=[0.0, 4.0, 0.0], hubbard=HUB, etol=1e-8,
                    rhotol=1e-7, verbose=False, max_iter=max_iter,
                    mixing_scheme="pulay")


# --------------------------------------------------------------------------- #
# Phase 1: smoke — does USPP/PAW+U run on CUDA at all?
def phase_smoke():
    if not HAS_CUDA:
        log("[smoke] no CUDA visible — CPU-only box; skipping GPU phases")
        return False
    log(f"[smoke] USPP/PAW+U on {RES['config']['gpu']}: tiny 2x2x2 SCF (5 iters)...")
    try:
        sysm = make_system(2, "cuda")
        t0 = time.time()
        res = converge(sysm, max_iter=5)  # not converged; just exercise the path
        sync()
        m = float((res["rho_spin"][0] - res["rho_spin"][1]).sum()) \
            * sysm.grid.volume / sysm.grid.n_points
        log(f"[smoke] OK — 5 GPU SCF iters in {time.time()-t0:.1f}s, m={m:+.2f} "
            f"(path runs on CUDA)")
        RES["smoke_ok"] = True
        return True
    except Exception as e:  # noqa: BLE001
        log(f"[smoke] FAILED — USPP/PAW+U is not GPU-ready: {e!r}")
        log("[smoke] " + traceback.format_exc().replace("\n", " | "))
        RES["smoke_ok"] = False
        return False


# --------------------------------------------------------------------------- #
# Phase 2: de-risk — CPU vs GPU on one chi-tilde apply and one SCF iter
def _time_apply(device, kmesh=2):
    torch.set_num_threads(CPU_THREADS)
    sysm = make_system(kmesh, device)
    t0 = time.time()
    res = converge(sysm, max_iter=6)  # a few iters is enough for a frozen state
    sync()
    scf_periter = (time.time() - t0) / max(1, res["n_iter"])
    sc = build_uspp_screening(res, SpinPBE(), fxc_scale=1.0, cg_tol=1e-5)
    g = torch.Generator(device=sc.ref.device).manual_seed(0)
    u = torch.randn(sc.ref.shape, dtype=sc.ref.dtype, device=sc.ref.device, generator=g)
    sc.apply(u)  # warm
    sync()
    t0 = time.time()
    for _ in range(3):
        sc.apply(u)
    sync()
    apply_s = (time.time() - t0) / 3
    return scf_periter, apply_s


def phase_derisk(gpu_ok):
    log("[derisk] timing one chi-tilde apply + one SCF iter, CPU vs GPU (2x2x2)")
    out = {}
    for dev in (["cpu", "cuda"] if gpu_ok else ["cpu"]):
        try:
            scf_pi, ap = _time_apply(dev)
            out[dev] = {"scf_s_per_iter": scf_pi, "apply_s": ap}
            log(f"[derisk] {dev}: SCF {scf_pi:.2f}s/iter | apply {ap:.2f}s")
        except Exception as e:  # noqa: BLE001
            log(f"[derisk] {dev} FAILED: {e!r}")
            out[dev] = {"error": repr(e)}
    if "cpu" in out and "cuda" in out and "apply_s" in out["cuda"]:
        sp = out["cpu"]["apply_s"] / out["cuda"]["apply_s"]
        out["apply_speedup_gpu"] = sp
        log(f"[derisk] >>> chi-tilde apply GPU speedup: {sp:.2f}x  "
            f"(this is the number that decides if the response scales)")
    RES["derisk"] = out


# --------------------------------------------------------------------------- #
# Phase 3: the deflation benchmark on GPU at the main k-mesh
def phase_bench(gpu_ok):
    device = "cuda" if gpu_ok else "cpu"
    log(f"[bench] BiFeO3+U on {device}, kmesh={KMESH}x{KMESH}x{KMESH}, "
        f"ecut={ECUT}/{ECUTRHO} Ry")
    try:
        if HAS_CUDA:
            torch.cuda.reset_peak_memory_stats()
        sysm = make_system(KMESH, device)
        t0 = time.time()
        res = converge(sysm)
        sync()
        assert res["converged"], "SCF did not converge"
        m = float((res["rho_spin"][0] - res["rho_spin"][1]).sum()) \
            * sysm.grid.volume / sysm.grid.n_points
        log(f"[bench] SCF converged in {res['n_iter']} iters, {time.time()-t0:.0f}s, "
            f"m={m:+.3f} muB")
        RES["bench"] = {"scf_iters": res["n_iter"], "scf_s": time.time() - t0,
                        "moment": m}

        xc = SpinPBE()
        sh = sysm.grid.shape
        g = torch.Generator(device=res["rho"].device).manual_seed(1)
        vg = torch.randn(sh, dtype=res["rho"].dtype, device=res["rho"].device,
                         generator=g)
        vg = vg - vg.mean()
        sc = build_uspp_screening(res, xc, fxc_scale=1.5, cg_tol=1e-5)
        vbar = sc.pack(vg)
        t0 = time.time()
        base = anderson_solve(sc.apply, vbar, beta=0.2, tol=1e-6, max_iter=80)
        sync()
        t_base = time.time() - t0
        sub = soft_subspace_from_operator(sc.apply, sc.ref, krylov=24, n_modes=6,
                                          seed=0)
        eig = [round(float(v.real), 3) for v in sub.values]
        crit = [q for v, q in zip(sub.values, sub.vectors, strict=True)
                if v.real > 0.75] or sub.vectors[:1]
        t0 = time.time()
        defl = deflated_solve(sc.apply, vbar, crit, method="post", beta=0.2,
                              tol=1e-6, max_iter=80)
        sync()
        t_defl = time.time() - t0
        peak = (torch.cuda.max_memory_allocated() / 1e9) if HAS_CUDA else 0.0
        RES["bench"].update({
            "fxc": 1.5, "soft_eigs": eig,
            "baseline": {"converged": base.converged, "iters": base.n_iter,
                         "wall_s": t_base},
            "deflated": {"converged": defl.converged, "iters": defl.n_iter,
                         "wall_s": t_defl, "k_deflated": len(crit)},
            "peak_vram_gb": peak})
        log(f"[bench] fxc=1.5 eigs={eig}")
        log(f"[bench] baseline: {'OK' if base.converged else 'STALL'}:"
            f"{base.n_iter} ({t_base:.0f}s)  |  deflated: "
            f"{'OK' if defl.converged else 'STALL'}:{defl.n_iter} "
            f"({t_defl:.0f}s, k={len(crit)})  |  peak VRAM {peak:.1f} GB")
    except Exception as e:  # noqa: BLE001
        log(f"[bench] FAILED: {e!r}")
        log("[bench] " + traceback.format_exc().replace("\n", " | "))
        RES.setdefault("bench", {})["error"] = repr(e)


def main():
    log(f"=== BiFeO3 H100 driver — device={'cuda' if HAS_CUDA else 'cpu'} "
        f"({RES['config']['gpu']}) torch {torch.__version__} ===")
    gpu_ok = phase_smoke()
    phase_derisk(gpu_ok)
    phase_bench(gpu_ok)
    log("=== BFO_H100_DONE ===")
    JSON.write_text(json.dumps(RES, indent=2, default=str))
    print(f"\nRESULTS: {JSON}", flush=True)


if __name__ == "__main__":
    main()
