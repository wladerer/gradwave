"""Part 1 (positive controls) - the patched tags MUST still fire on genuine
pathologies. After guarding moment-collapse and charge-sloshing against a
converged tail (recorder fix on feat/scf-flight-recorder), every converged case
in run_part1 goes quiet. These constructed pathologies confirm the guards did
not neuter the detectors.

  uv run python experiments/convergence_cases/run_positives.py [--threads N]

- moment-collapse: bcc Cr, FM seed, collapses to the nonmagnetic branch (m->0).
  The tag must fire (final |M| ~ 0 < the 0.1 muB floor).
- charge-sloshing: Al(100) 6-layer slab with the Kerker preconditioner disabled
  and a starved mixer, so the long-wavelength charge sloshes and the residual
  stalls high (> the 1e-4 floor). The tag must fire.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import torch
from _common import FIX, OUT, RY, al_slab, bcc_cell, case_summary, dump_trace

RESULTS: list[dict] = []


def pos_moment_collapse():
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system

    cr = parse_upf(FIX / "Cr_ONCV_PBE-1.2.upf")
    system = setup_system(bcc_cell(2.88), [[0.0, 0, 0]], [0], [cr], ecut=50 * RY,
                          kmesh=(6, 6, 6), nbands=14, use_symmetry=True)
    name = "pos_cr_collapse"
    t0 = time.time()
    r = scf(system, SpinPBE(), smearing="gaussian", width=0.05, etol=1e-8,
            rhotol=1e-6, max_iter=60, verbose=False, nspin=2, start_mag=[0.7])
    rec = r.recorder
    dump_trace(name, rec)
    RESULTS.append(case_summary(
        name, converged=r.converged, n_iter=r.n_iter, energy=float(r.energies.total),
        mag_abs=r.mag_abs, recorder=rec,
        extra={"expect": "moment-collapse", "seed_moment": rec.seed_moment,
               "wall_s": round(time.time() - t0, 1)}))
    print(f"  {name}: conv={r.converged} n={r.n_iter} mag={r.mag_abs:.3f} "
          f"tags={[t for t,_ in rec.diagnose()]}")


def pos_charge_sloshing():
    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system

    al = parse_upf(FIX / "Al_ONCV_PBE-1.2.upf")
    cell, pos, spec = al_slab(6)
    system = setup_system(cell, pos, spec, [al], ecut=25 * RY, kmesh=(4, 4, 1),
                          nbands=6 * len(spec), use_symmetry=True)
    name = "pos_al_slab_sloshing"
    t0 = time.time()
    # Kerker off + starved mixer (short history, aggressive alpha): the
    # long-wavelength (vacuum) modes are no longer screened, so the density
    # sloshes and the residual stalls high instead of converging.
    r = scf(system, PBE(), smearing="gaussian", width=0.1, etol=1e-8,
            rhotol=1e-7, max_iter=40, verbose=False, kerker=False,
            mixing_alpha=0.9, mixing_history=2)
    rec = r.recorder
    dump_trace(name, rec)
    RESULTS.append(case_summary(
        name, converged=r.converged, n_iter=r.n_iter, energy=float(r.energies.total),
        mag_abs=None, recorder=rec,
        extra={"expect": "charge-sloshing", "kerker": False,
               "mixing_alpha": 0.9, "mixing_history": 2,
               "wall_s": round(time.time() - t0, 1)}))
    print(f"  {name}: conv={r.converged} n={r.n_iter} "
          f"final_drho={rec.iters[-1]['drho']:.2e} "
          f"tags={[t for t,_ in rec.diagnose()]}")


CASES = {"collapse": pos_moment_collapse, "sloshing": pos_charge_sloshing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for k in (a.cases or list(CASES)):
        print(f"=== {k} ===", flush=True)
        try:
            CASES[k]()
        except Exception:
            traceback.print_exc()
            RESULTS.append({"case": k, "error": traceback.format_exc()})
        (OUT / "results_positives.json").write_text(json.dumps(RESULTS, indent=1))
    print("WROTE", OUT / "results_positives.json")


if __name__ == "__main__":
    main()
