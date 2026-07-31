"""Truth harness for the Pulay pressure-error estimator (issue #227).

Measures ratio = estimate / true on the sheared-Si cell from
tests/integration/test_stress_error.py, over a grid of (ecut, kmesh).

true error  = P_raw(45 Ry) - P_raw(ecut), with P_raw = -tr(sigma)/3 from
              postscf.stress.stress(symmetrize=False), converted to GPa.
estimate    = estimate_pressure_error(res_ecut, ..., ecut_large=45 Ry), the
              variant kwargs selected on the command line.

Run locally, small SCFs. Example:

    uv run python benchmarks/pulay_accuracy/measure.py --variants baseline
    uv run python benchmarks/pulay_accuracy/measure.py \
        --variants baseline,extrap,cg,extrap+cg --out benchmarks/pulay_accuracy/RESULTS.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# repo root on sys.path so `tests.helpers` (si_upf, RY) imports when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.stress import stress
from gradwave.postscf.stress_error import estimate_pressure_error
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_upf

EV_A3_TO_GPA = 160.2176634

# sheared Si cell (matches tests/integration/test_stress_error.py)
CELL = np.array([
    [0.01086, 2.70414, 2.7285750],
    [2.74215, 0.01629, 2.7231450],
    [2.75301, 2.70957, 0.0054300],
])
POS = np.array([[0.0, 0.0, 0.0], [1.426505, 1.3275, 1.3842875]])

REF_ECUT = 45.0  # Ry, treated as the converged reference

# variant name -> kwargs passed to estimate_pressure_error
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "extrap": {"extrapolate": True},
    "cg": {"solver": "cg"},
    "extrap+cg": {"extrapolate": True, "solver": "cg"},
}


def _run(ecut_ry: float, kmesh: tuple[int, int, int]):
    upf = si_upf()
    system = setup_system(CELL, POS, [0, 0], [upf], ecut=ecut_ry * RY,
                          kmesh=kmesh, use_symmetry=False)
    res = scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
              verbose=False)
    assert res.converged
    return res


def _p_raw_gpa(res) -> float:
    sig = stress(res, PBE(), symmetrize=False).cpu().numpy()
    return -np.trace(sig) / 3.0 * EV_A3_TO_GPA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="baseline",
                    help="comma-separated subset of " + ",".join(VARIANTS))
    ap.add_argument("--ecuts", default="10,12,14,16")
    ap.add_argument("--kmeshes", default="2,3")
    ap.add_argument("--out", default=None, help="markdown file to write/append")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v!r}; choose from {list(VARIANTS)}")
    ecuts = [float(e) for e in args.ecuts.split(",")]
    ks = [int(k) for k in args.kmeshes.split(",")]

    rows = []
    for kk in ks:
        kmesh = (kk, kk, kk)
        res_ref = _run(REF_ECUT, kmesh)
        p_ref = _p_raw_gpa(res_ref)
        for ecut in ecuts:
            res = _run(ecut, kmesh)
            p_true = p_ref - _p_raw_gpa(res)
            ratios: dict[str, float] = {}
            for v in variants:
                t0 = time.perf_counter()
                # default annulus (factor*ecut); the estimator drops the tail
                # beyond it, which is exactly what rung 1 recovers. The true
                # error is still measured against the 45 Ry reference.
                out = estimate_pressure_error(res, PBE(), **VARIANTS[v])
                dt = time.perf_counter() - t0
                p_est = float(out["pressure_error_eV_A3"]) * EV_A3_TO_GPA
                ratios[v] = p_est / p_true
                nh = out.get("n_h_apply")
                print(f"kmesh={kmesh} ecut={ecut:>4} Ry  {v:<10} "
                      f"P_true={p_true:7.3f} GPa  P_est={p_est:7.3f} GPa  "
                      f"ratio={ratios[v]:6.3f}  ({dt:5.2f}s"
                      + (f", {nh} H-applies)" if nh is not None else ")"),
                      flush=True)
            rows.append((kk, ecut, p_true, ratios))

    # markdown table
    hdr = "| kmesh | ecut (Ry) | P_true (GPa) | " + \
        " | ".join(variants) + " |"
    sep = "|" + "---|" * (3 + len(variants))
    lines = [hdr, sep]
    for kk, ecut, p_true, ratios in rows:
        cells = " | ".join(f"{ratios[v]:.3f}" for v in variants)
        lines.append(f"| {kk}x{kk}x{kk} | {ecut:g} | {p_true:.3f} | {cells} |")
    table = "\n".join(lines)
    print("\n" + table)

    if args.out:
        stamp = time.strftime("%Y-%m-%d %H:%M")
        with open(args.out, "a") as f:
            f.write(f"\n## Run {stamp} (variants: {', '.join(variants)})\n\n")
            f.write("Ratios are estimate / true; 1.0 is exact.\n\n")
            f.write(table + "\n")
        print(f"\nappended to {args.out}")


if __name__ == "__main__":
    main()
