"""Phase A: GPU-resident fp64 SCF profile on the RTX 3050 vs CPU arms.

Usage: uv run python experiments/gpu_resident_scf/profile_scf.py <case> <device> <mode> [threads]
  case:   al4 | si8
  device: cpu | cuda
  mode:   time    (clean e2e wall, no profiler)
          profile (torch.profiler kernel decomposition; e2e NOT clean)
  threads: CPU threads (default 8)

al4 = Al conventional cubic cell a=4.05 (4 atoms), PBE, 40 Ry, 4x4x4 MP,
      fermi-dirac width 0.1 eV, nbands=24, symmetry on.  ("Al-4 conv 4^3 FD")
si8 = diamond-Si conventional cell a=5.43 (8 atoms), LDA, 30 Ry, 2x2x2 MP,
      no smearing, symmetry on.  (bench_matrix "si8")
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

case = sys.argv[1]
device = sys.argv[2]
mode = sys.argv[3]
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
torch.set_num_threads(threads)

RY = 13.605693122994
root = Path(__file__).parents[2]
PSE = root / "tests/fixtures/qe/pseudos"

if case == "al4":
    a = 4.05
    cell = a * np.eye(3)
    frac = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    pos = frac @ cell
    upfs = [parse_upf(PSE / "Al_ONCV_PBE-1.2.upf")]
    soa = [0, 0, 0, 0]
    kw = dict(ecut=40 * RY, kmesh=(4, 4, 4), nbands=24, use_symmetry=True)
    scf_kw = dict(smearing="fermi-dirac", width=0.1)
    xc = PBE()
elif case == "si8":
    a = 5.43
    cell = a * np.eye(3)
    frac = np.array(
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
         [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75],
         [0.25, 0.75, 0.75]])
    pos = frac @ cell
    upfs = [parse_upf(PSE / "Si_ONCV_PBE-1.2.upf")]
    soa = [0] * 8
    kw = dict(ecut=30 * RY, kmesh=(2, 2, 2), nbands=None, use_symmetry=True)
    scf_kw = dict(smearing="none", width=0.1)
    xc = LDA_PW92()
elif case in ("si64", "al32"):
    # exact /tmp/gw_size.py configs (comparable to the band-parallel agent's CPU arms)
    import itertools
    if case == "si64":
        a = 10.86
        base = np.array(
            [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
             [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])
        upfs = [parse_upf(PSE / "Si_ONCV_PBE-1.2.upf")]
        scf_kw = dict(smearing="none", width=0.1)
    else:
        a = 8.10
        base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        upfs = [parse_upf(PSE / "Al_ONCV_PBE-1.2.upf")]
        scf_kw = dict(smearing="fermi-dirac", width=0.1)
    frac = np.array([(p + np.array([i, j, k])) / 2
                     for i, j, k in itertools.product(range(2), repeat=3)
                     for p in base])
    cell = a * np.eye(3)
    pos = frac @ cell
    soa = [0] * len(frac)
    kw = dict(ecut=30 * RY, kmesh=(2, 2, 2))  # setup defaults, as in gw_size.py
    xc = PBE()
else:
    raise SystemExit(f"unknown case {case}")

t0 = time.time()
system = setup_system(cell, pos, soa, upfs, **kw)
t_setup = time.time() - t0
if device != "cpu":
    system = system.to(device)
    torch.cuda.reset_peak_memory_stats()

print(f"case={case} device={device} mode={mode} threads={threads} "
      f"nk={len(system.spheres)} npw={system.spheres[0].npw} "
      f"grid={tuple(system.grid.shape)} ne={system.n_electrons}")

CATS = [
    ("fft", re.compile(r"fft", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|cublas|gemv|dot_kernel|splitKreduce", re.I)),
    ("eigh/qr/chol", re.compile(
        r"syevd|heevd|stedc|steqr|potrf|geqrf|ormqr|ungqr|unmqr|orgqr|trsm|"
        r"getrf|laswp|cusolver|syevj|hegst|lansy|lacpy|larft|larfb|tridiag|"
        r"bidiag|householder|eig", re.I)),
    ("elemwise/copy/reduce", re.compile(
        r"elementwise|vectorized|reduce|copy|CatArray|index|scatter|gather|"
        r"fill|where|masked|Memcpy|Memset|unrolled|softmax|sort|cub::|"
        r"cumsum|arange|triu|tril", re.I)),
]

def classify(name):
    for cat, rx in CATS:
        if rx.search(name):
            return cat
    return "other"

MAX_ITER = 3 if mode in ("smoke", "profile3") else 100

def run():
    import os
    mp = os.environ.get("GW_PROBE_MP", "") == "1"
    res = scf(system, xc, etol=1e-8, rhotol=1e-7, verbose=False,
              max_iter=MAX_ITER, mixed_precision=mp, **scf_kw)
    if device != "cpu":
        torch.cuda.synchronize()
    return res

if mode == "smoke":
    t0 = time.time()
    res = run()
    print(f"SMOKE (3 iters, untimed conditions): {time.time()-t0:.1f}s wall")
    if device != "cpu":
        print(f"peak CUDA mem: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
elif mode == "time":
    t0 = time.time()
    res = run()
    t = time.time() - t0
    print(f"setup {t_setup:.1f}s  SCF {t:.2f}s  iters={res.n_iter} conv={res.converged}")
    print(f"F = {float(res.energies.free_energy):.8f} eV")
    if device != "cpu":
        print(f"peak CUDA mem: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
else:
    from torch.profiler import ProfilerActivity, profile
    # profile3 on cuda: CUDA activity ONLY — the CPU-op trace of a large-cell
    # 3-iter run post-processes at >12 GB host RSS and gets OOM-killed.
    if device != "cpu" and mode == "profile3":
        acts = [ProfilerActivity.CUDA]
    else:
        acts = [ProfilerActivity.CPU]
        if device != "cpu":
            acts.append(ProfilerActivity.CUDA)
    t0 = time.time()
    with profile(activities=acts) as prof:
        res = run()
    t = time.time() - t0
    print(f"profiled SCF wall {t:.2f}s (profiler overhead included)  "
          f"iters={res.n_iter} conv={res.converged}  "
          f"F={float(res.energies.free_energy):.6f}")
    ka = prof.key_averages()
    key = "self_device_time_total" if device != "cpu" else "self_cpu_time_total"
    tot = {}
    rows = []
    for ev in ka:
        dt = getattr(ev, key, 0) or 0
        if dt <= 0:
            continue
        c = classify(ev.key)
        tot[c] = tot.get(c, 0) + dt
        rows.append((dt, ev.key, c))
    total = sum(tot.values())
    lbl = "CUDA kernel" if device != "cpu" else "CPU self"
    print(f"\n== {lbl} time by category (total {total/1e6:.2f}s) ==")
    for c, v in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"  {c:22s} {v/1e6:8.2f}s  {100*v/total:5.1f}%")
    print(f"\n== top 20 kernels ==")
    for dt, name, c in sorted(rows, reverse=True)[:20]:
        print(f"  {dt/1e6:8.3f}s  [{c:>20s}]  {name[:90]}")
    if device != "cpu":
        print(f"peak CUDA mem: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
print("DONE")
