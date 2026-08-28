"""Deep-profile one MgO GIPAW shielding evaluation (the profiling proxy).

The memory/time BREAKDOWN of the GIPAW shielding pipeline — where bytes and
cycles go across the CG response resolvent, the FFT boxes, and the band buffers
— is a STRUCTURAL property of the code paths, not of the material. alpha-quartz
(9 atoms) costs ~3 h per shielding eval, and ``deep_profile`` runs the workload
~4x (2 timed + torch.profiler + py-spy + memray), so profiling quartz directly
is a ~15 h job and memray never finishes. MgO (2-atom rocksalt) exercises the
IDENTICAL hard-augmentation code paths — it is literally the cell the S-metric
CG reroute was validated on (``mgo_cg_ecut_validate.py``, PR #401): hard-O PAW
augmentation, the ill-conditioned overlap that trips the cond(S) gate onto the
matrix-free CG resolvent, the same FFT boxes at ecutrho and the same band
buffers — but one eval is ~1-3 min, so the full deep_profile with memray is
tractable (~30-60 min).

Settings: ecut 25 Ry (ecutrho 100 Ry), k 2x2x1 (the cheapest mesh the q->0
assembly accepts — >=2 axes with n>1), ``chunk_k=1``, all-PAW pseudos (light
non-semicore Mg + O ``kjpaw``) so ``shielding_level`` auto-selects
``gipaw`` and ``response_backend='auto'`` cond(S)-routes the hard O site onto
the ecut-stable CG resolvent (PR #401). This is the SAME regime the quartz O
sites live in; the profile answers "where do the bytes/cycles go", not a
converged MgO shielding number.

The Mg PAW pseudo is not in the committed fixtures; fetch it once into
``tests/fixtures/qe/pseudos/`` from the QE pseudo library
(``Mg.pbe-n-kjpaw_psl.0.3.0.UPF``).

Run modes::

    python quartz_shielding_profile_workload.py --single   # one eval (sampler target)
    python quartz_shielding_profile_workload.py            # driver: deep_profile + summary.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from gradwave.constants import RY_EV
from gradwave.inputs import Input, KPointsParams, NmrParams

PSEUDOS = Path("tests/fixtures/qe/pseudos").resolve()
# Light non-semicore Mg PAW (3s valence only, 2 e-): the profile is a STRUCTURAL
# breakdown, not an accuracy run, so we skip the semicore spnl (10 e-, ~5x the
# band cost, >10 min/eval). The O site is the hard-augmentation one that trips
# the cond(S) gate onto CG — the code path the profile is meant to exercise.
MG_PSEUDO = "Mg.pbe-n-kjpaw_psl.0.3.0.UPF"  # fetch into PSEUDOS from the QE library
O_PSEUDO = "O.pbe-n-kjpaw_psl.1.0.0.UPF"

ECUT_RY = 25.0
ECUTRHO_RY = 100.0
KMESH = (2, 2, 1)
THREADS = 8


def mgo_rocksalt() -> Atoms:
    """2-atom rocksalt MgO primitive cell (a = 4.21 Å): Mg at 0, O at (1/2)³."""
    a = 4.21
    cell = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return Atoms(
        symbols=["Mg", "O"],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=cell, pbc=True)


def _build_input(atoms: Atoms) -> Input:
    return Input(
        atoms=atoms, pseudo_dir=PSEUDOS,
        pseudo_map={"Mg": MG_PSEUDO, "O": O_PSEUDO},
        ecut=ECUT_RY * RY_EV, ecutrho=ECUTRHO_RY * RY_EV, xc="pbe",
        kpoints=KPointsParams(mesh=KMESH), task="nmr", symmetry=False,
        nmr=NmrParams(
            task="shielding", shielding_level="gipaw",
            # response_backend defaults to 'auto': the cond(S) gate routes the
            # hard O site onto the matrix-free CG resolvent (PR #401).
            chunk_k=1),
        verbose=False)


def single_eval() -> dict[str, Any]:
    """One GIPAW shielding solve on MgO. Returns the shielding block."""
    from gradwave.api import run_nmr

    return run_nmr(_build_input(mgo_rocksalt()), verbose=False)


def build_workload():
    from gradwave.profiling import Workload

    spec = {
        "system": "MgO (rocksalt, profiling proxy for quartz hard-O paths)",
        "n_atoms": 2,
        "composition": "1 Mg + 1 O",
        "ecut_Ry": ECUT_RY,
        "ecutrho_Ry": ECUTRHO_RY,
        "kmesh": list(KMESH),
        "pseudos": "PAW (Mg spnl semicore + O kjpaw)",
        "shielding_level": "gipaw",
        "response_backend": "auto (cond(S) gate -> CG on the hard O site, PR #401)",
        "chunk_k": 1,
        "profiled": "one GIPAW magnetic-shielding evaluation (both sites)",
    }
    command = [sys.executable, str(Path(__file__).resolve()), "--single"]
    return Workload(
        name="mgo-shielding", spec=spec, run=single_eval, command=command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single", action="store_true",
        help="run one shielding eval and exit (py-spy / memray subprocess target)")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(THREADS)

    if args.single:
        block = single_eval()
        n = block.get("n_sites", len(block.get("sites", [])))
        print(f"single_eval done: {n} sites", flush=True)
        return 0

    from gradwave.profiling import deep_profile, write_summary

    wl = build_workload()
    result = deep_profile(wl, n_timed=2, warmup=0, threads=THREADS)
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
