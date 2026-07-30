"""Part 2 - trace-driven auto-remediation for charge-sloshing on Al slabs.

The structural finding from Part 1: diagnose() is post-hoc (it reads the last-5
iterations), so it cannot drive a mid-run decision, and on a converged run it
stays quiet. A mid-run detector that reads the same |G|-shell decomposition CAN
catch sloshing while it is happening. This experiment prototypes that detector
and a detect-plus-restart remediation, then measures it honestly against the
two fixed policies.

Per slab (4- and 6-layer Al(100), NC) we compare four runs:
  A robust-from-start : precond=local_tf         (the documented best policy)
  B kerker-from-start : precond=kerker (default) (the sloshing-prone policy)
  C detect+restart    : run kerker; if the mid-run detector flags sloshing at
                        iteration k, warm-restart (start_from) with local_tf and
                        a fresh mixer. Total iters = k + restart iters.
The honest question: does C (detection + restart, paying the wasted k-iteration
prefix) beat A (just always using the robust preconditioner)?

Prototype only. No production paths touched (owner ruling). The mid-run detector
lives here, not in scf.recorder.

  uv run python experiments/convergence_cases/run_part2.py [--threads N] [--detect-k K]
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from _common import FIX, OUT, RY, al_slab, dump_trace, low2_series

SLOSH_LOW2 = 0.5       # mean 2-lowest-shell fraction over the detection window
SLOSH_RES_FLOOR = 1e-4  # only flag while the residual is still meaningfully above noise


def build_slab(nlay):
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import setup_system

    al = parse_upf(FIX / "Al_ONCV_PBE-1.2.upf")
    cell, pos, spec = al_slab(nlay)
    return setup_system(cell, pos, spec, [al], ecut=25 * RY, kmesh=(4, 4, 1),
                        nbands=6 * len(spec), use_symmetry=True)


def midrun_sloshing(recorder, res_floor=SLOSH_RES_FLOOR) -> bool:
    """Mid-run detector: over the iterations whose residual is still above the
    noise floor, is the long-wavelength (2 lowest |G|-shell) weight persistently
    dominant? Unlike diagnose() this does not require non-monotonicity - a good
    mixer converges a sloshing-prone cell monotonically but slowly, and that
    slow long-wavelength grind is exactly what we want to catch mid-run."""
    active = [i for i in recorder.iters if i["drho"] > res_floor]
    if len(active) < 3:
        return False
    low2 = [sum(i["shell_frac"][:2]) for i in active]
    return (sum(low2) / len(low2)) > SLOSH_LOW2


def run(nlay, detect_k):
    from gradwave.core.xc.pbe import PBE
    from gradwave.scf.loop import scf

    system = build_slab(nlay)
    xc = PBE()
    common = dict(smearing="gaussian", width=0.1, etol=1e-8, rhotol=1e-7,
                  verbose=False)
    out = {"nlayers": nlay, "detect_k": detect_k}

    # A: robust from start (local_tf)
    t0 = time.time()
    rA = scf(system, xc, max_iter=80, precond="local_tf", **common)
    out["A_local_tf"] = {"n_iter": rA.n_iter, "conv": rA.converged,
                         "E": round(float(rA.energies.total), 6),
                         "wall_s": round(time.time() - t0, 1)}
    dump_trace(f"part2_slab{nlay}_A_local_tf", rA.recorder)

    # B: kerker from start (default, sloshing-prone)
    t0 = time.time()
    rB = scf(system, xc, max_iter=80, precond="kerker", **common)
    out["B_kerker"] = {"n_iter": rB.n_iter, "conv": rB.converged,
                       "E": round(float(rB.energies.total), 6),
                       "wall_s": round(time.time() - t0, 1),
                       "low2_series": low2_series(rB.recorder)}
    dump_trace(f"part2_slab{nlay}_B_kerker", rB.recorder)

    # C: detect + restart. Phase 1: kerker for detect_k iters (the "wasted
    # prefix"). Inspect the mid-run detector. If it flags sloshing, warm-restart
    # with local_tf and a fresh mixer (start_from carries the density, not the
    # Pulay history).
    t0 = time.time()
    r1 = scf(system, xc, max_iter=detect_k, precond="kerker", **common)
    flagged = midrun_sloshing(r1.recorder)
    phase1_iters = r1.n_iter
    if flagged:
        r2 = scf(system, xc, max_iter=80, precond="local_tf",
                 start_from=r1, **common)
        total = phase1_iters + r2.n_iter
        conv = r2.converged
        E = round(float(r2.energies.total), 6)
        dump_trace(f"part2_slab{nlay}_C_phase2", r2.recorder)
    else:
        # detector did not flag -> continue on kerker (no remediation)
        r2 = scf(system, xc, max_iter=80, precond="kerker", start_from=r1, **common)
        total = phase1_iters + r2.n_iter
        conv = r2.converged
        E = round(float(r2.energies.total), 6)
    out["C_detect_restart"] = {"phase1_iters": phase1_iters, "flagged": flagged,
                               "phase2_iters": r2.n_iter, "total_iters": total,
                               "conv": conv, "E": E,
                               "wall_s": round(time.time() - t0, 1)}
    dump_trace(f"part2_slab{nlay}_C_phase1", r1.recorder)

    # verdict: does detect+restart beat robust-from-start?
    out["verdict"] = {
        "A_iters": rA.n_iter, "B_iters": rB.n_iter, "C_iters": total,
        "C_beats_A": total < rA.n_iter,
        "energies_match": abs(rA.energies.total - E) < 1e-3,
    }
    print(f"slab{nlay}: A(local_tf)={rA.n_iter}  B(kerker)={rB.n_iter}  "
          f"C(detect@{detect_k}+restart)={total} (flagged={flagged})  "
          f"C_beats_A={out['verdict']['C_beats_A']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--detect-k", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    results = []
    for nlay in (4, 6):
        results.append(run(nlay, a.detect_k))
        (OUT / "results_part2.json").write_text(json.dumps(results, indent=1))
    print("WROTE", OUT / "results_part2.json")


if __name__ == "__main__":
    main()
