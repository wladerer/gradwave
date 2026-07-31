"""Validate the Pulay pressure estimator on relaxed alpha-quartz (issue #227).

Uses the relaxed geometry from benchmarks/results/sio2-elastic/out_relax/relax.json
(read-only), Si/O ONCV PBE pseudos, 65 Ry, kmesh (3,3,3), use_symmetry=False,
smearing none. Reports estimate/true for the diagonal and cg solvers against the
ground-truth error 7.653 GPa (P_raw(130 Ry) - P_raw(65 Ry), given, not rerun).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.stress import stress
from gradwave.postscf.stress_error import estimate_pressure_error
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo

EV_A3_TO_GPA = 160.2176634
TRUE_ERR_GPA = 7.653  # given: P_raw(130 Ry) - P_raw(65 Ry) on relaxed quartz
# benchmarks/results/ is not tracked in worktrees; read it from the primary checkout
RELAX_JSON = Path.home() / "github/gradwave/benchmarks/results/sio2-elastic/out_relax/relax.json"
PSEUDO_OF = {"Si": "Si_ONCV_PBE-1.2.upf", "O": "O_ONCV_PBE-1.2.upf"}


def main() -> None:
    torch.set_num_threads(4)
    d = json.load(open(RELAX_JSON))["relax"]
    cell = np.asarray(d["cell_ang"], dtype=float)
    pos = np.asarray(d["positions_ang"], dtype=float)
    species = list(d["species"])
    names = sorted(set(species))
    upfs = [parse_upf(pseudo(PSEUDO_OF[n])) for n in names]
    idx = [names.index(s) for s in species]
    print(f"quartz: {len(species)} atoms {species}, species {names}", flush=True)

    t0 = time.perf_counter()
    system = setup_system(cell, pos, idx, upfs, ecut=65 * RY, kmesh=(3, 3, 3),
                          use_symmetry=False)
    res = scf(system, PBE(), smearing="none", etol=1e-10, rhotol=1e-9,
              verbose=False)
    assert res.converged
    print(f"65 Ry SCF converged in {time.perf_counter()-t0:.0f}s", flush=True)

    sig = stress(res, PBE(), symmetrize=False).cpu().numpy()
    p_raw = -np.trace(sig) / 3.0 * EV_A3_TO_GPA
    print(f"P_raw(65 Ry) = {p_raw:.3f} GPa (ref P_raw(65) given as -3.779)",
          flush=True)

    for solver in ("diagonal", "cg"):
        t1 = time.perf_counter()
        out = estimate_pressure_error(res, PBE(), solver=solver)
        p_est = float(out["pressure_error_eV_A3"]) * EV_A3_TO_GPA
        dt = time.perf_counter() - t1
        print(f"{solver:<9} P_est={p_est:7.3f} GPa  ratio={p_est/TRUE_ERR_GPA:6.3f}"
              f"  (true={TRUE_ERR_GPA} GPa, {dt:.1f}s, {out['n_h_apply']} H-applies)",
              flush=True)


if __name__ == "__main__":
    main()
