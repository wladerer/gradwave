"""GAP 3 — verify the zero-FFT vacuum-fraction pre-gate makes the RIGHT
engage/abstain call, and report the FFT it saves vs the ρ(M) power iteration.

Converge three references and run the PRODUCTION gate (``Chi0PrecondCache.update``)
on each, reporting vacuum_fraction, the decision, and the one-time gate+build FFT:

  * fcc-Al bulk (metal, homogeneous)   → must ABSTAIN, at ZERO gate FFT
  * Al(100) 4-layer slab (metal+vacuum) → must ENGAGE, at ZERO gate FFT
  * Si bulk (insulator, homogeneous)    → must ABSTAIN

The old gate paid a Sternheimer-χ₀ power iteration (~700-1700 FFT) on every one
of these before deciding; the pre-gate decides bulk (abstain) and slab (engage)
from the converged density alone. We print both so the saving is explicit.

Usage (asus only):
    uv run python experiments/chi0_precond_crux/gap3_gate_calib.py --out .../gap3_gate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk

from experiments.chi0_precond_crux.common import (
    PSEUDO,
    RY,
    al_bulk_system,
    al_slab_system,
    xc_for,
)
from gradwave.core import opcount
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.subspace_chi0 import (
    _VAC_ABSTAIN_BELOW,
    _VAC_ENGAGE_ABOVE,
    Chi0PrecondCache,
    vacuum_fraction,
)

BASE = dict(smearing="gaussian", width=0.1, nspin=1, etol=1e-9, rhotol=1e-8,
            max_iter=150, verbose=False)


def si_bulk_system(ecut_ry=25.0, kmesh=(4, 4, 4)):
    upf = parse_upf(PSEUDO / "Si_ONCV_PBE-1.2.upf")
    si = bulk("Si", "diamond", a=5.43)
    n = len(si)
    return setup_system(np.array(si.cell), si.get_positions(), [0] * n, [upf],
                        ecut=ecut_ry * RY, kmesh=kmesh, nbands=12,
                        use_symmetry=False), n


def converge(system):
    xc = xc_for(1)
    res = scf(system, xc, mixing_scheme="pulay", mixing_alpha=0.7,
              precond="local_tf", **BASE)
    return res, xc


def decide(name, system, expect_engage):
    opcount.enable()
    res, xc = converge(system)
    prev = opcount.snapshot()
    cache = Chi0PrecondCache(verbose=True)
    cache.update(res, xc, 1)
    total_one_time = int(opcount.since(prev)["fft"])
    opcount.disable()
    vac = vacuum_fraction(res)
    row = {
        "system": name,
        "vacuum_fraction": round(vac, 4),
        "engaged": bool(cache.engaged),
        "rho_M": cache.rho,                    # None when the pre-gate decided
        "gate_ffts": int(cache.gate_ffts),     # ρ(M) power-iteration FFT (0 if skipped)
        "build_ffts": int(cache.build_ffts),
        "one_time_ffts": total_one_time,
        "expect_engage": expect_engage,
        "correct": bool(cache.engaged) == expect_engage,
    }
    print(f"  [{name}] vac={vac:.4f} engaged={cache.engaged} "
          f"rho(M)={cache.rho} gate_ffts={cache.gate_ffts} "
          f"build_ffts={cache.build_ffts}  correct={row['correct']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"thresholds: abstain<{_VAC_ABSTAIN_BELOW}  engage>{_VAC_ENGAGE_ABOVE}",
          flush=True)
    rows = []
    al_b, _ = al_bulk_system(kmesh=(6, 6, 6))
    rows.append(decide("fcc-Al bulk (metal)", al_b, expect_engage=False))
    al_s, _ = al_slab_system(4)
    rows.append(decide("Al(100) 4-layer slab (metal)", al_s, expect_engage=True))
    si_b, _ = si_bulk_system()
    rows.append(decide("Si bulk (insulator)", si_b, expect_engage=False))

    allok = all(r["correct"] for r in rows)
    out.write_text(json.dumps({"thresholds": {
        "abstain_below": _VAC_ABSTAIN_BELOW, "engage_above": _VAC_ENGAGE_ABOVE},
        "rows": rows, "all_correct": allok}, indent=2))
    print(f"\nall_correct={allok}   saved -> {out}")


if __name__ == "__main__":
    main()
