"""H100 Test 0 — CALIBRATION BASELINE (run FIRST; gates every other H100 test).

Single metal SCF, fp64, CPU vs CUDA, across cell sizes. Answers the two load-bearing
questions before any datacenter-GPU feature is built:

  (1) Does H100 fp64 actually BEAT the 22-core CPU for the small/medium cells gradwave
      runs? (On the consumer RTX 3050 it did NOT — fp64 is 1/64 there. H100 SXM is full-rate.)
  (2) Is the SCF LAUNCH/DISPATCH-BOUND at these sizes (low % of wall in FFT/GEMM kernels)?
      If yes, batching/fusing many small ops into fewer larger kernels (spin-batch, config-
      batch, CUDA graphs) is the lever. If it's already throughput-bound, those won't help.

DO NOT infer perf from a consumer card — this must run on H100 SXM.

Run (tomorrow, on the H100 box):
    uv run python benchmarks/h100/test0_calibration.py
    uv run python benchmarks/h100/test0_calibration.py --sizes 1,2,3 --ecut 30 --reps 3
"""
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]


def build_al(nrep: int, ecut_ry: float, kmesh: tuple[int, int, int]):
    """fcc-Al conventional 4-atom cell tiled nrep**3 (4*nrep**3 atoms), nonmag metal."""
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system

    al = parse_upf(_ROOT / "benchmarks/delta_gauge/pseudos/Al.upf")
    a = 4.05
    basis = a * np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cell = a * nrep * np.eye(3)
    pos = np.concatenate([
        basis + a * np.array([i, j, k])
        for i in range(nrep) for j in range(nrep) for k in range(nrep)
    ])
    system = setup_system(cell, pos, [0] * len(pos), [al],
                          ecut=ecut_ry * RY, kmesh=kmesh, use_symmetry=False)
    return system, len(pos)


def run_scf(system, device: str):
    from gradwave.core.xc.lda_pw92 import LDA_PW92
    from gradwave.scf.loop import scf

    sysd = system.to(device) if device != "cpu" else system
    if device != "cpu":
        torch.cuda.synchronize()
    t = time.perf_counter()
    res = scf(sysd, LDA_PW92(), smearing="cold", width=0.2,
              etol=1e-8, rhotol=1e-7, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    return time.perf_counter() - t, int(res.n_iter)


def launch_fraction(system, device: str) -> float:
    """One profiled SCF; return the fraction of self time in real compute kernels
    (FFT + GEMM/eigh/QR). Low fraction => launch/dispatch-bound => batching helps."""
    from torch.profiler import ProfilerActivity, profile

    from gradwave.core.xc.lda_pw92 import LDA_PW92
    from gradwave.scf.loop import scf

    sysd = system.to(device) if device != "cpu" else system
    acts = [ProfilerActivity.CPU]
    if device != "cpu":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts) as prof:
        scf(sysd, LDA_PW92(), smearing="cold", width=0.2, etol=1e-8, rhotol=1e-7, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()

    def self_time(e):
        return getattr(e, "self_device_time_total", 0) if device != "cpu" else e.self_cpu_time_total
    ka = prof.key_averages()
    total = sum(self_time(e) for e in ka) or 1.0
    _compute_ops = ("fft", "mm", "bmm", "matmul", "addmm", "gemm", "einsum",
                    "eigh", "qr", "linalg", "cublas", "cufft")
    compute = sum(self_time(e) for e in ka
                  if any(k in e.key.lower() for k in _compute_ops))
    return 100.0 * compute / total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="1,2,3",
                    help="comma nrep list; atoms = 4*nrep**3 (1->4, 2->32, 3->108)")
    ap.add_argument("--ecut", type=float, default=30.0, help="Ry")
    ap.add_argument("--kmesh", type=int, default=4,
                    help="k per axis (drops to Gamma above --gamma-above atoms)")
    ap.add_argument("--gamma-above", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2, help="median over this many timed SCFs (warm)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    has_cuda = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if has_cuda else "no-CUDA"
    print(f"# H100 Test 0 — calibration | CUDA={has_cuda} ({dev_name}) | "
          f"cpu-threads={args.threads} | ecut={args.ecut} Ry")
    if not has_cuda:
        print("# WARNING: no CUDA visible — this run only characterizes CPU. Run on the H100 box.")
    print(f"{'atoms':>6} {'npw':>7} {'nb':>4} {'nk':>4} {'cpu[s]':>9} {'gpu[s]':>9} "
          f"{'gpu/cpu':>8} {'cpu%cmp':>8} {'gpu%cmp':>8}")

    for nrep in [int(x) for x in args.sizes.split(",")]:
        km = 1 if 4 * nrep**3 > args.gamma_above else args.kmesh
        system, natoms = build_al(nrep, args.ecut, (km, km, km))
        npw = max(int(s.npw) for s in system.spheres)
        nb = int(system.nbands)
        nk = len(system.spheres)

        cpu_walls = [run_scf(system, "cpu")[0] for _ in range(args.reps)]
        cpu = statistics.median(cpu_walls)
        cpu_frac = launch_fraction(system, "cpu")

        if has_cuda:
            # first rep also warms cuda kernels/caches; median absorbs it
            gpu_walls = [run_scf(system, "cuda")[0] for _ in range(args.reps)]
            gpu = statistics.median(gpu_walls)
            gpu_frac = launch_fraction(system, "cuda")
            ratio = f"{gpu / cpu:.2f}x"
            gpu_s, gpu_fs = f"{gpu:.3f}", f"{gpu_frac:.1f}"
        else:
            gpu_s, ratio, gpu_fs = "-", "-", "-"

        print(f"{natoms:>6} {npw:>7} {nb:>4} {nk:>4} {cpu:>9.3f} {gpu_s:>9} "
              f"{ratio:>8} {cpu_frac:>7.1f} {gpu_fs:>8}")

    print("\n# READ: gpu/cpu < 1 => H100 wins for that size. gpu%cmp LOW (say <50%) => "
          "launch-bound => batching/fusing (spin-batch, config-batch, CUDA graphs) is the lever.")
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
