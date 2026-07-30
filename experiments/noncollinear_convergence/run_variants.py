"""Floor-origin discriminators for the Ni + SOC residual floor.

The baseline matrix shows every mixer arm (pulay stock, the #79 fix recipe,
johnson) flooring at the same composite residual ~2e-3 on fcc Ni + SOC, with
the moment locked and no direction drift. A shared floor across mixers points
away from the mixer. These runs separate the remaining suspects:

  diag11    fix recipe + diago_tol=1e-11: if the floor is Davidson noise
            amplified by near-degenerate Fermi states, tightening the
            eigensolve two decades should drop it.
  noadapt   stock + adaptive=False: is the backoff churn (step x0.1, history
            dropped every 6 stalled iterations) holding the tail?
  width02   stock + width=0.2 eV: a smoother Fermi surface shrinks the
            occupation response of near-degenerate states.
  best      johnson + quadratic schedule + spin_precond + diago 1e-11 +
            adaptive off: does ANY configuration reach rhotol 1e-5?

Usage: uv run python experiments/noncollinear_convergence/run_variants.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.noncollinear_convergence import systems as sysmod  # noqa: E402
from experiments.noncollinear_convergence.run_matrix import (  # noqa: E402
    FIX,
    run_nc,
    z,
)

VARIANTS = {
    "diag11": dict(FIX, diago_tol=1e-11),
    "noadapt": dict(adaptive=False),
    "width02": dict(width=0.2),
    "best": dict(mag_mixer="johnson", mag_diago_schedule="quadratic",
                 spin_precond=True, diago_tol=1e-11, adaptive=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--host", default=platform.node().split(".")[0])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    outdir = Path(__file__).resolve().parent / "results" / args.host
    outdir.mkdir(parents=True, exist_ok=True)
    names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    summary_path = outdir / "summary.jsonl"
    for vname in names:
        row = run_nc(f"ni_soc_var_{vname}_s0.6_z", sysmod.ni_fcc(True), z(0.6),
                     outdir=outdir, **VARIANTS[vname])
        print(json.dumps(row), flush=True)
        with open(summary_path, "a") as f:
            f.write(json.dumps(row) + "\n")
    print(f"\nDONE variants={names}", flush=True)


if __name__ == "__main__":
    main()
