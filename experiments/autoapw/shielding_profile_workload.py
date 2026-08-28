"""Deep-profile ONE isolated shielding kernel — a single CG Sternheimer solve.

Profiling a FULL GIPAW shielding eval is impossible on a 14 GB box in ANY
torch.profiler config: the eval issues millions of aten ops (per-k x per-q x
per-band x per-site CG), and the profiler's in-RAM event buffer (even with
``profile_memory`` only, ``with_stack``/``record_shapes`` off — the ``light``
mode) grew past 12-13 GB and the box OOM-killed the process (measured; see
``shielding_profile_note.md`` and ``docs/profiling.md``).

The RIGHT granularity for "where do a response solve's cycles and bytes go" is
ONE kernel invocation, not the whole multi-k/q/band eval. This profiles a
single :class:`~gradwave.postscf.kgeometry_nmr._SMetricResolventCG` ``apply`` —
one matrix-free S-metric CG Sternheimer resolvent solve, THE inner operator the
full shielding drives thousands of times. One solve is a few thousand ops:
fits trivially, and is more informative for the kernel-level question (which
ops dominate a single resolvent solve, and where its transient memory lives).

The context (a cheap MgO SCF + one dense per-k S-metric context) is built ONCE,
unprofiled; only the single CG apply is timed and profiled. MgO is the cell the
CG reroute was validated on (PR #401); the numbers here are structural, not
converged (the RHS is a random conduction-space source of the right shape, so
the op/byte breakdown is faithful while the solve itself is not physical).

The Mg PAW pseudo is not in the committed fixtures; fetch it once into
``tests/fixtures/qe/pseudos/`` from the QE pseudo library
(``Mg.pbe-n-kjpaw_psl.0.3.0.UPF``).

Run modes::

    python shielding_profile_workload.py --single   # one CG solve, standalone timing
    python shielding_profile_workload.py            # driver: deep_profile + summary.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from gradwave.constants import RY_EV

PSEUDOS = Path("tests/fixtures/qe/pseudos").resolve()
MG_PSEUDO = "Mg.pbe-n-kjpaw_psl.0.3.0.UPF"  # light non-semicore Mg PAW (3s, 2 e-)
O_PSEUDO = "O.pbe-n-kjpaw_psl.1.0.0.UPF"

ECUT_RY = 25.0
ECUTRHO_RY = 100.0
KMESH = (2, 2, 1)
NBANDS = 10
THREADS = 8
# The single CG solve's iteration budget: a random RHS will not converge, so it
# runs the full budget — a representative sample of the resolvent op mix.
CG_TOL = 1e-8
CG_MAX_ITER = 100


def _mgo_cell_pos() -> tuple[np.ndarray, np.ndarray]:
    a = 4.21
    cell = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ cell
    return cell, pos


def build_cg_kernel() -> tuple[Any, Any, dict[str, int]]:
    """Cheap MgO SCF + one dense S-metric k-context → a ready
    ``_SMetricResolventCG`` and a random conduction-space RHS ``z`` (npw, nocc).
    Returns ``(res_cg, z, sizes)``; everything here is the UNPROFILED setup."""
    import torch

    from gradwave.constants import HBAR2_2M
    from gradwave.core.xc.pbe import PBE
    from gradwave.dtypes import CDTYPE
    from gradwave.postscf import kgeometry_nmr as kg
    from gradwave.postscf.kgeometry_nmr import ShieldingDq, _SMetricResolventCG
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import scf_uspp, setup_uspp

    torch.set_num_threads(THREADS)
    paw_mg = parse_upf_paw(str(PSEUDOS / MG_PSEUDO))
    paw_o = parse_upf_paw(str(PSEUDOS / O_PSEUDO))
    cell, pos = _mgo_cell_pos()
    system = setup_uspp(
        cell, pos, [0, 1], [paw_mg, paw_o],
        ecut=ECUT_RY * RY_EV, kmesh=KMESH, ecutrho=ECUTRHO_RY * RY_EV,
        nbands=NBANDS)
    res = scf_uspp(system, PBE(), etol=1e-7, rhotol=1e-6, diago_tol=1e-8,
                   verbose=False, max_iter=80)
    ctx = kg.build_uspp_response_ctx(res, PBE())

    # Force the CG backend; build ONE dense per-k S-metric context and lift the
    # exact _SMetricResolventCG the full shielding builds per k (lines that make
    # res_cg in ShieldingDq._accumulate_k).
    eng = ShieldingDq(res, uspp=ctx, response_backend="cg",
                      cg_tol=CG_TOL, max_iter=CG_MAX_ITER)
    dk = eng._build_ctx(eng._kfs[0], eng._weights[0])
    nocc = eng.nocc
    uo = dk.u[:, :nocc]
    wo = dk.w[:nocc]
    t_kin = HBAR2_2M * (dk.kpg * dk.kpg).sum(-1)
    res_cg = _SMetricResolventCG(
        dk.hk.h(dk.kc), dk.s0, uo, wo, t_kin, tol=CG_TOL, max_iter=CG_MAX_ITER)

    npw = int(uo.shape[0])
    torch.manual_seed(0)
    z = torch.randn(npw, nocc, dtype=CDTYPE)
    return res_cg, z, {"npw": npw, "nocc": int(nocc), "max_iter": CG_MAX_ITER}


def build_workload(res_cg: Any, z: Any, sizes: dict[str, int]):
    from gradwave.profiling import Workload

    spec = {
        "system": "MgO (rocksalt) — one S-metric CG Sternheimer solve",
        "kernel": "_SMetricResolventCG.apply (matrix-free CG resolvent, PR #401)",
        "npw": sizes["npw"],
        "nocc": sizes["nocc"],
        "cg_max_iter": sizes["max_iter"],
        "ecut_Ry": ECUT_RY,
        "kmesh": list(KMESH),
        "profiled": ("ONE CG resolvent solve — the inner kernel the full "
                     "shielding drives thousands of times (structural, not "
                     "converged)"),
    }
    # command=None: the process-level samplers profile a fresh SCF-and-solve
    # subprocess, not this in-process pre-built kernel; the op/memory table from
    # torch.profiler is the deliverable here.
    return Workload(
        name="shielding-cg-kernel", spec=spec,
        run=lambda: res_cg.apply(z), command=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single", action="store_true",
        help="build the kernel, run one CG solve, and exit (standalone timing)")
    args = parser.parse_args()

    import time

    import torch

    torch.set_num_threads(THREADS)

    print("building CG kernel (cheap MgO SCF + one k-context, unprofiled)...",
          flush=True)
    res_cg, z, sizes = build_cg_kernel()
    print(f"  kernel ready: npw={sizes['npw']} nocc={sizes['nocc']} "
          f"max_iter={sizes['max_iter']}", flush=True)

    if args.single:
        t0 = time.time()
        _ = res_cg.apply(z)
        print(f"single CG solve done in {time.time() - t0:.3f}s", flush=True)
        return 0

    from gradwave.profiling import deep_profile, write_summary

    wl = build_workload(res_cg, z, sizes)
    result = deep_profile(wl, n_timed=5, warmup=1, threads=THREADS, light=True)
    path = write_summary(result)
    print(f"\nSUMMARY_HTML {path}", flush=True)
    print(f"OUT_DIR {result.out_dir}", flush=True)
    if result.notes:
        print("NOTES:", flush=True)
        for note in result.notes:
            print(f"  - {note}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
