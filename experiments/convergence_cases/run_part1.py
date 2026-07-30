"""Part 1 - validate the flight-recorder tags against the documented hard cases.

Runs each pathology from docs/manual/{wisdom,performance}.md with the recorder
on, dumps the per-iteration trace, and records whether the expected tag fires.
Self-contained: writes traces/<case>.json and results_part1.json. Every case is
wrapped so one failure does not sink the batch.

  uv run python experiments/convergence_cases/run_part1.py [cases...]

With no args runs all cases. Case keys: ni_nc, ni_uspp, fe_uspp, al_slab,
coarse_al, pt_paw.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import torch

from _common import (
    DG,
    FIX,
    OUT,
    RY,
    al_slab,
    bcc_cell,
    case_summary,
    dump_trace,
    fcc_cell,
)

RESULTS: list[dict] = []


def _record(res_obj, is_uspp: bool):
    """Normalize NC SCFResult vs USPP dict-like result to (conv, niter, E, mag, rec)."""
    if is_uspp:
        conv = res_obj["converged"]
        niter = res_obj["n_iter"]
        energy = float(res_obj["energies"].total)
        mag = res_obj.get("mag_abs", None)
        if mag is None:
            mag = getattr(res_obj.recorder.iters[-1] if res_obj.recorder.iters else None,
                          "get", lambda *_: None)
        rec = res_obj.recorder
    else:
        conv = res_obj.converged
        niter = res_obj.n_iter
        energy = float(res_obj.energies.total)
        mag = getattr(res_obj, "mag_abs", None)
        rec = res_obj.recorder
    return conv, niter, energy, mag, rec


# --------------------------------------------------------------------------- #
# Case 1: fcc Ni near the Stoner boundary (NC path, nspin=2)                   #
# --------------------------------------------------------------------------- #
def case_ni_nc():
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system

    ni = parse_upf(DG / "Ni.upf")
    cell, pos = fcc_cell(3.52), [[0.0, 0, 0]]
    system = setup_system(cell, pos, [0], [ni], ecut=45 * RY, kmesh=(8, 8, 8),
                          nbands=12, use_symmetry=True)

    variants = [
        # name,        start_mag, scheme,    expect
        ("ni_nc_underseed_pulay", 0.02, "pulay",   "moment-collapse"),
        ("ni_nc_underseed_johnson", 0.02, "johnson", "?"),
        ("ni_nc_seeded_johnson",  0.6,  "johnson", "none"),
    ]
    for name, sm, scheme, expect in variants:
        t0 = time.time()
        r = scf(system, SpinPBE(), smearing="gaussian", width=0.1, etol=1e-8,
                rhotol=1e-6, max_iter=80, verbose=False, nspin=2,
                start_mag=[sm], mixing_scheme=scheme)
        conv, niter, E, mag, rec = _record(r, False)
        dump_trace(name, rec)
        RESULTS.append(case_summary(
            name, converged=conv, n_iter=niter, energy=E, mag_abs=mag,
            recorder=rec, extra={"expect": expect, "start_mag": sm,
                                 "scheme": scheme, "seed_moment": rec.seed_moment,
                                 "wall_s": round(time.time() - t0, 1)}))
        print(f"  {name}: conv={conv} n={niter} m={mag:.3f} tags={[t for t,_ in rec.diagnose()]}")


# --------------------------------------------------------------------------- #
# Case 1b: fcc Ni USPP/PAW path -- johnson vs pulay (wisdom: 18 vs 27)         #
# --------------------------------------------------------------------------- #
def case_ni_uspp():
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp_loop import scf_uspp
    from gradwave.scf.uspp_setup import setup_uspp

    paw = parse_upf_paw(FIX / "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF")
    cell, pos = fcc_cell(3.52), [[0.0, 0, 0]]
    system = setup_uspp(cell, pos, [0], [paw], ecut=45 * RY, ecutrho=360 * RY,
                        kmesh=(8, 8, 8), nbands=12, use_symmetry=True)
    for scheme in ("johnson", "pulay"):
        name = f"ni_uspp_{scheme}"
        t0 = time.time()
        r = scf_uspp(system, SpinPBE(), smearing="gaussian", width=0.1,
                     etol=1e-8, rhotol=1e-6, max_iter=80, verbose=False,
                     nspin=2, start_mag=[0.6], mixing_scheme=scheme)
        conv, niter, E, mag, rec = _record(r, True)
        mag = rec.iters[-1]["mag_abs"] if rec.iters else None
        dump_trace(name, rec)
        RESULTS.append(case_summary(
            name, converged=conv, n_iter=niter, energy=E, mag_abs=mag,
            recorder=rec, extra={"expect_none_tag": True, "scheme": scheme,
                                 "documented_iters": {"johnson": 18, "pulay": 27}[scheme],
                                 "wall_s": round(time.time() - t0, 1)}))
        print(f"  {name}: conv={conv} n={niter} tags={[t for t,_ in rec.diagnose()]}")


# --------------------------------------------------------------------------- #
# Case 2: bcc Fe USPP with johnson forced (documented 29->93 blowup)           #
# --------------------------------------------------------------------------- #
def case_fe_uspp():
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp_loop import scf_uspp
    from gradwave.scf.uspp_setup import setup_uspp

    paw = parse_upf_paw(FIX / "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF")
    cell, pos = bcc_cell(2.87), [[0.0, 0, 0]]
    system = setup_uspp(cell, pos, [0], [paw], ecut=45 * RY, ecutrho=360 * RY,
                        kmesh=(8, 8, 8), nbands=12, use_symmetry=True)
    for scheme in ("pulay", "johnson"):
        name = f"fe_uspp_{scheme}"
        t0 = time.time()
        r = scf_uspp(system, SpinPBE(), smearing="gaussian", width=0.1,
                     etol=1e-8, rhotol=1e-6, max_iter=120, verbose=False,
                     nspin=2, start_mag=[0.7], mixing_scheme=scheme)
        conv, niter, E, mag, rec = _record(r, True)
        mag = rec.iters[-1]["mag_abs"] if rec.iters else None
        dump_trace(name, rec)
        RESULTS.append(case_summary(
            name, converged=conv, n_iter=niter, energy=E, mag_abs=mag,
            recorder=rec, extra={"scheme": scheme,
                                 "documented_iters": {"pulay": 29, "johnson": 93}[scheme],
                                 "note": "johnson forced -> becsum-driven blowup probe",
                                 "wall_s": round(time.time() - t0, 1)}))
        print(f"  {name}: conv={conv} n={niter} tags={[t for t,_ in rec.diagnose()]}")


# --------------------------------------------------------------------------- #
# Case 3: Al(100) slabs 4 & 6 layer (NC) -- charge-sloshing + local_tf remedy  #
# --------------------------------------------------------------------------- #
def case_al_slab():
    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system

    al = parse_upf(FIX / "Al_ONCV_PBE-1.2.upf")
    for nlay in (4, 6):
        cell, pos, spec = al_slab(nlay)
        system = setup_system(cell, pos, spec, [al], ecut=25 * RY,
                              kmesh=(4, 4, 1), nbands=6 * len(spec),
                              use_symmetry=True)
        for precond in ("kerker", "local_tf"):
            name = f"al_slab{nlay}_{precond}"
            t0 = time.time()
            r = scf(system, PBE(), smearing="gaussian", width=0.1, etol=1e-8,
                    rhotol=1e-7, max_iter=80, verbose=False, precond=precond)
            conv, niter, E, mag, rec = _record(r, False)
            dump_trace(name, rec)
            RESULTS.append(case_summary(
                name, converged=conv, n_iter=niter, energy=E, mag_abs=None,
                recorder=rec, extra={"nlayers": nlay, "precond": precond,
                                     "expect": "charge-sloshing" if precond == "kerker" else "none",
                                     "wall_s": round(time.time() - t0, 1)}))
            print(f"  {name}: conv={conv} n={niter} tags={[t for t,_ in rec.diagnose()]} "
                  f"low2_final={sum(rec.iters[-1]['shell_frac'][:2]):.3f}")


# --------------------------------------------------------------------------- #
# Case 4a: coarse-mesh Al (no fractional occupations trap)                     #
# --------------------------------------------------------------------------- #
def case_coarse_al():
    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system

    al = parse_upf(FIX / "Al_ONCV_PBE-1.2.upf")
    cell, pos = fcc_cell(4.05), [[0.0, 0, 0]]
    # 2x2x2 mesh, tiny smearing -> occupations noisy / near-integer (the trap)
    for width, tag in ((0.01, "tiny"), (0.5, "wide")):
        system = setup_system(cell, pos, [0], [al], ecut=30 * RY,
                              kmesh=(2, 2, 2), nbands=8, use_symmetry=True)
        name = f"coarse_al_{tag}"
        t0 = time.time()
        r = scf(system, PBE(), smearing="gaussian", width=width, etol=1e-8,
                rhotol=1e-7, max_iter=80, verbose=False)
        conv, niter, E, mag, rec = _record(r, False)
        dump_trace(name, rec)
        RESULTS.append(case_summary(
            name, converged=conv, n_iter=niter, energy=E, mag_abs=None,
            recorder=rec, extra={"width_eV": width, "expect_level_crossing": "quiet",
                                 "wall_s": round(time.time() - t0, 1)}))
        print(f"  {name}: conv={conv} n={niter} reorder_total={sum(i['reorder'] for i in rec.iters)} "
              f"tags={[t for t,_ in rec.diagnose()]}")


# --------------------------------------------------------------------------- #
# Case 4b: fcc Pt PAW 40/400 Ry (level-crossing where reordering is real)      #
# --------------------------------------------------------------------------- #
def case_pt_paw():
    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp_loop import scf_uspp
    from gradwave.scf.uspp_setup import setup_uspp

    paw = parse_upf_paw(FIX / "Pt.pbe-n-kjpaw_psl.1.0.0.UPF")
    cell, pos = fcc_cell(3.92), [[0.0, 0, 0]]
    system = setup_uspp(cell, pos, [0], [paw], ecut=40 * RY, ecutrho=400 * RY,
                        kmesh=(6, 6, 6), nbands=16, use_symmetry=True)
    name = "pt_paw_johnson"
    t0 = time.time()
    r = scf_uspp(system, PBE(), smearing="gaussian", width=0.2, etol=1e-8,
                 rhotol=1e-7, max_iter=60, verbose=False, mixing_scheme="johnson")
    conv, niter, E, mag, rec = _record(r, True)
    dump_trace(name, rec)
    RESULTS.append(case_summary(
        name, converged=conv, n_iter=niter, energy=E, mag_abs=None,
        recorder=rec, extra={"note": "smeared PAW metal, real level crossings possible",
                             "wall_s": round(time.time() - t0, 1)}))
    print(f"  {name}: conv={conv} n={niter} reorder_total={sum(i['reorder'] for i in rec.iters)} "
          f"tags={[t for t,_ in rec.diagnose()]}")


CASES = {
    "ni_nc": case_ni_nc,
    "ni_uspp": case_ni_uspp,
    "fe_uspp": case_fe_uspp,
    "al_slab": case_al_slab,
    "coarse_al": case_coarse_al,
    "pt_paw": case_pt_paw,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="subset of case keys (default all)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    keys = a.cases if a.cases else list(CASES)
    for k in keys:
        if k not in CASES:
            print(f"unknown case {k}")
            continue
        print(f"=== {k} ===", flush=True)
        try:
            CASES[k]()
        except Exception:
            traceback.print_exc()
            RESULTS.append({"case": k, "error": traceback.format_exc()})
        (OUT / "results_part1.json").write_text(json.dumps(RESULTS, indent=1))
    print("WROTE", OUT / "results_part1.json")


if __name__ == "__main__":
    main()
