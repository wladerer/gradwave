"""USPP/PAW SCF benchmark and profiler, fcc Pt reference.

One place to measure every performance change against QE's 3.2 s for the same
1-atom fcc Pt point (psl kjpaw, PBE, 40/400 Ry, 12x12x12, gaussian 0.2 eV).

    python bench/pt_uspp_bench.py                       # time (default 6x6x6)
    python bench/pt_uspp_bench.py --kmesh 12 12 12      # the QE-matched point
    python bench/pt_uspp_bench.py --profile             # cProfile one SCF
    python bench/pt_uspp_bench.py --device cuda

Reports iterations, wall, per-iteration, and the free energy so a change that
trades iteration count for per-iteration cost is visible in both columns. The
energy is the correctness guard, it must not move beyond SCF tolerance.
"""

import argparse
import cProfile
import io
import os
import pstats
import time

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

RY = 13.605693122994
PT = "tests/fixtures/qe/pseudos/Pt.pbe-n-kjpaw_psl.1.0.0.UPF"


def build(a, kmesh, ecut, ecutrho, nbands, device):
    paw = parse_upf_paw(PT)
    cell = 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    s = setup_uspp(cell, [[0.0, 0, 0]], [0], [paw], ecut=ecut * RY,
                   ecutrho=ecutrho * RY, kmesh=tuple(kmesh), nbands=nbands,
                   use_symmetry=True)
    return s.to(device)


def run(system, smearing, width):
    t = time.time()
    r = scf_uspp(system, PBE(), smearing=smearing, width=width, etol=1e-8,
                 rhotol=1e-7, max_iter=80, verbose=False)
    return time.time() - t, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmesh", type=int, nargs=3, default=[6, 6, 6])
    ap.add_argument("--ecut", type=float, default=40.0)
    ap.add_argument("--ecutrho", type=float, default=400.0)
    ap.add_argument("--nbands", type=int, default=14)
    ap.add_argument("--a", type=float, default=3.97)
    ap.add_argument("--smearing", default="gaussian")
    ap.add_argument("--width", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("GW_THREADS", "8")))
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--topn", type=int, default=25)
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    system = build(args.a, args.kmesh, args.ecut, args.ecutrho, args.nbands,
                   args.device)
    nk = len(system.spheres)
    print(f"fcc Pt a={args.a} | {tuple(args.kmesh)} -> {nk} irr k | "
          f"fft {tuple(system.grid.shape)} | {args.ecut:.0f}/{args.ecutrho:.0f} Ry "
          f"| {args.device} threads {torch.get_num_threads()}", flush=True)

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        dt, r = run(system, args.smearing, args.width)
        pr.disable()
        st = pstats.Stats(pr, stream=io.StringIO())
        print(f"\nSCF: {r['n_iter']} it, {dt:.1f}s, {dt/r['n_iter']:.2f} s/it, "
              f"E={float(r['energies'].free_energy):.6f} eV\n", flush=True)
        for sort in ("tottime", "cumtime"):
            buf = io.StringIO()
            pstats.Stats(pr, stream=buf).sort_stats(sort).print_stats(args.topn)
            print(f"===== top {args.topn} by {sort} =====", flush=True)
            print(buf.getvalue(), flush=True)
    else:
        dt, r = run(system, args.smearing, args.width)
        print(f"SCF: {r['n_iter']} it | {dt:.1f}s | {dt/r['n_iter']:.2f} s/it | "
              f"E={float(r['energies'].free_energy):.6f} eV | conv {r['converged']}",
              flush=True)


if __name__ == "__main__":
    main()
