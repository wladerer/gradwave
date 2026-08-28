"""Deep-profile one MgO GIPAW shielding evaluation (the profiling proxy).

The memory/time BREAKDOWN of the GIPAW shielding pipeline — where bytes and
cycles go across the CG response resolvent, the FFT boxes, and the band buffers
— is a STRUCTURAL property of the code paths, not of the material. alpha-quartz
(9 atoms) costs ~3 h per shielding eval; MgO (2-atom rocksalt) exercises the
IDENTICAL hard-augmentation code paths — it is the cell the S-metric CG reroute
was validated on (``mgo_cg_ecut_validate.py``, PR #401): hard-O PAW
augmentation, the ill-conditioned overlap, the matrix-free CG resolvent, the
same FFT boxes at ecutrho and the same band buffers — cheaply.

Two things make this profileable on a 14 GB box:

* **Low-level, CG-forced, iteration-capped eval.** ``single_eval`` calls
  ``kgeometry_nmr.sigma_shielding_gipaw`` directly with
  ``response_backend='cg'`` (so the CG resolvent is exercised regardless of the
  cond(S) gate) and a LOW ``max_iter`` / loose ``cg_tol``. The default driver
  runs CG to 1e-9 (hundreds of iterations, millions of tensor allocations);
  ``torch.profiler(profile_memory=True)`` records every allocation, so the
  full run OOMs the box (measured 12-13 GB). Capping the CG iteration count
  caps the allocation count and keeps the profiled pass in RAM. The shielding
  NUMBER is meaningless at this cap — this is a structural profile, not an
  accuracy run; use ``ssnmr_quartz_gate.py`` / ``quartz_sigma_cg.py`` for real
  numbers.
* **light-mode profiler.** ``deep_profile(..., light=True)`` drops
  ``with_stack`` / ``record_shapes`` (a full call stack + input shapes per op)
  while keeping ``profile_memory`` — the per-op memory column that answers
  "where do the bytes go". ``command=None`` skips py-spy/memray (their
  allocation tracking OOMs on the same allocation count).

Settings: ecut 25 Ry (ecutrho 100 Ry), k 2x2x1 (the cheapest mesh the q->0
assembly accepts — >=2 axes with n>1), ``chunk_k=1``, all-PAW pseudos (light
non-semicore Mg + O ``kjpaw``).

The Mg PAW pseudo is not in the committed fixtures; fetch it once into
``tests/fixtures/qe/pseudos/`` from the QE pseudo library
(``Mg.pbe-n-kjpaw_psl.0.3.0.UPF``).

Run modes::

    python shielding_profile_workload.py --single   # one capped eval, standalone
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
# Light non-semicore Mg PAW (3s valence only, 2 e-): the profile is a STRUCTURAL
# breakdown, not an accuracy run, so we skip the semicore spnl (10 e-, ~5x the
# band cost). The O site is the hard-augmentation one whose CG resolvent the
# profile is meant to exercise.
MG_PSEUDO = "Mg.pbe-n-kjpaw_psl.0.3.0.UPF"  # fetch into PSEUDOS from the QE library
O_PSEUDO = "O.pbe-n-kjpaw_psl.1.0.0.UPF"

ECUT_RY = 25.0
ECUTRHO_RY = 100.0
KMESH = (2, 2, 1)
NBANDS = 10  # 1 Mg (2 e-) + 1 O (6 e-) = 8 e- -> 4 occupied + buffer
THREADS = 8
# CG iteration cap: bounds the allocation count so profile_memory fits in RAM.
# The resolvent/FFT/band code paths are the SAME at any cap; only the number of
# iterations (hence the converged number) changes.
CG_TOL = 1e-3
CG_MAX_ITER = 20
SCF_MAX_ITER = 80


def _mgo_cell_pos() -> tuple[np.ndarray, np.ndarray]:
    """2-atom rocksalt MgO (a = 4.21 Å): fcc primitive cell, Mg at 0, O at (1/2)³."""
    a = 4.21
    cell = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ cell
    return cell, pos


def single_eval() -> dict[str, Any]:
    """One CG-forced, iteration-capped GIPAW shielding solve on MgO.

    Low-level so the CG backend and its iteration cap are explicit — this is the
    resolvent / FFT / band pipeline the profile attributes, run cheaply enough
    that ``profile_memory`` stays in RAM."""
    import torch

    from gradwave.core.xc.pbe import PBE
    from gradwave.postscf import kgeometry_nmr as kg
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
                   verbose=False, max_iter=SCF_MAX_ITER)
    ctx = kg.build_uspp_response_ctx(res, PBE())
    out = kg.sigma_shielding_gipaw(
        res, ctx, [paw_mg, paw_o], response_backend="cg",
        cg_tol=CG_TOL, max_iter=CG_MAX_ITER, use_symmetry=False, chunk_k=1)
    return {"n_sites": int(out["total"].shape[0])}


def build_workload():
    from gradwave.profiling import Workload

    spec = {
        "system": "MgO (rocksalt, profiling proxy for quartz hard-O paths)",
        "n_atoms": 2,
        "composition": "1 Mg + 1 O",
        "ecut_Ry": ECUT_RY,
        "ecutrho_Ry": ECUTRHO_RY,
        "kmesh": list(KMESH),
        "pseudos": "PAW (light non-semicore Mg + O kjpaw)",
        "shielding_level": "gipaw (low-level, CG-forced)",
        "response_backend": f"cg (forced); cg_tol={CG_TOL}, max_iter={CG_MAX_ITER}",
        "chunk_k": 1,
        "profiled": ("one iteration-capped GIPAW shielding eval (structural "
                     "breakdown, NOT a converged number)"),
    }
    # command=None: py-spy/memray SKIPPED — their allocation tracking OOMs on a
    # shielding eval's allocation count. The op-level memory breakdown comes from
    # torch.profiler profile_memory (light mode) instead.
    return Workload(
        name="mgo-shielding", spec=spec, run=single_eval, command=None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single", action="store_true",
        help="run one capped shielding eval and exit (standalone timing)")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(THREADS)

    if args.single:
        block = single_eval()
        print(f"single_eval done: {block['n_sites']} sites", flush=True)
        return 0

    from gradwave.profiling import deep_profile, write_summary

    wl = build_workload()
    # light=True: torch.profiler without per-op stacks/shapes (those balloon the
    # trace); profile_memory stays on so the op table keeps the per-op memory
    # column. The CG iteration cap in single_eval keeps that column's allocation
    # count in RAM.
    result = deep_profile(wl, n_timed=2, warmup=0, threads=THREADS, light=True)
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
