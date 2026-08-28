"""Deep-profile one alpha-quartz GIPAW shielding evaluation.

This is BOTH the in-process ``Workload.run`` target and the subprocess
``Workload.command`` target that py-spy / memray drive, so the op table, the
py-spy flamegraph, and the memray allocation flamegraph all describe the SAME
single shielding solve.

Settings are deliberately CHEAP, not converged: ecut 25 Ry (ecutrho 100 Ry),
k 2x2x1 (the cheapest mesh the q->0 shielding assembly accepts — it needs >=2
mesh axes with n>1), ``chunk_k=1`` (stream the dense per-k response contexts,
O(1)-in-nk peak memory), all-PAW pseudos so ``shielding_level`` auto-selects
``gipaw`` and the ``response_backend='auto'`` cond(S) gate routes the
hard-augmentation O sites onto the ecut-stable matrix-free CG resolvent (PR
#401). The memory/time BREAKDOWN across resolvent / FFT boxes / bands is a
STRUCTURAL property visible at any convergence level, so these cheap settings
still exercise the real hard-augmentation code paths — at production settings
(ecut 40 / k 2x2x2) one eval is ~3 h, and deep_profile runs it ~4x, so cheap
settings are required to profile at all. NOT a converged shielding number; use
``ssnmr_quartz_gate.py`` for that.

Run modes::

    python quartz_shielding_profile_workload.py --single   # one eval (sampler target)
    python quartz_shielding_profile_workload.py            # driver: deep_profile + summary.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ase.spacegroup import crystal

from gradwave.constants import RY_EV
from gradwave.inputs import Input, KPointsParams, NmrParams

PSEUDOS = Path("tests/fixtures/qe/pseudos").resolve()

ECUT_RY = 25.0
ECUTRHO_RY = 100.0
KMESH = (2, 2, 1)
THREADS = 8


def alpha_quartz():
    """alpha-quartz (P3_2 21, sg 152): 3 Si + 6 O in the primitive cell."""
    a, c = 4.9134, 5.4052
    return crystal(
        symbols=["Si", "O"],
        basis=[(0.4697, 0.0, 1.0 / 3.0), (0.4135, 0.2669, 0.1191)],
        spacegroup=152, cellpar=[a, a, c, 90, 90, 120])


def _build_input(atoms) -> Input:
    return Input(
        atoms=atoms, pseudo_dir=PSEUDOS,
        pseudo_map={"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
                    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF"},
        ecut=ECUT_RY * RY_EV, ecutrho=ECUTRHO_RY * RY_EV, xc="pbe",
        kpoints=KPointsParams(mesh=KMESH), task="nmr", symmetry=False,
        nmr=NmrParams(
            task="shielding", shielding_level="gipaw",
            # k-streaming memory route; response_backend defaults to 'auto',
            # so the cond(S) gate auto-routes the hard O sites onto CG.
            chunk_k=1),
        verbose=False)


def single_eval() -> dict[str, Any]:
    """One GIPAW shielding solve on alpha-quartz. Returns the shielding block."""
    from gradwave.api import run_nmr

    return run_nmr(_build_input(alpha_quartz()), verbose=False)


def build_workload():
    from gradwave.profiling import Workload

    spec = {
        "system": "alpha-quartz (SiO2)",
        "n_atoms": 9,
        "composition": "3 Si + 6 O",
        "ecut_Ry": ECUT_RY,
        "ecutrho_Ry": ECUTRHO_RY,
        "kmesh": list(KMESH),
        "pseudos": "PAW (kjpaw psl 1.0.0)",
        "shielding_level": "gipaw",
        "response_backend": "auto (cond(S) gate -> CG on hard O sites, PR #401)",
        "chunk_k": 1,
        "profiled": "one GIPAW magnetic-shielding evaluation (all sites)",
    }
    command = [sys.executable, str(Path(__file__).resolve()), "--single"]
    return Workload(
        name="quartz-shielding", spec=spec, run=single_eval, command=command)


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
    # One eval is heavy even at cheap settings, so n_timed=1 / warmup=0
    # (single-shot timing — LOW-N, no median; noted in the report header).
    result = deep_profile(wl, n_timed=1, warmup=0, threads=THREADS)
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
