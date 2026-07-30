"""USPP/PAW nspin=2 mixing-default benchmark: johnson vs pulay vs broyden.

Re-measures the mixing-scheme choice behind the USPP/PAW nspin==2 default
(pulay) against the two systems documented in docs/manual/wisdom.md (bcc Fe,
fcc Ni) plus a two-sublattice AFM case. Each (system, scheme, start_mag) run
uses identical SCF settings across schemes so iteration counts are comparable,
records the flight-recorder diagnostics, and checks the converged fixed point
(free energy + moment) so a scheme that silently picks a different branch is
flagged rather than averaged in.

Settings for Fe and Ni match the 2026-07-30 convergence case-study campaign
(archive branch research/convergence-case-studies), which itself matched the
wisdom.md-era numbers (Fe pulay 30, Ni pulay 27, Ni johnson 18).

  uv run python experiments/uspp_mixing_default/run_bench.py [cases...] [--threads 8]

Case keys: fe, ni, afm_fe. No args runs fe and ni (the documented pair).
Self-contained: writes traces/<case>.json and results.json.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

RY = 13.605693122994

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "qe" / "pseudos"
OUT = Path(__file__).resolve().parent
TRACES = OUT / "traces"

SCHEMES = ("pulay", "johnson", "broyden")
RESULTS: list[dict] = []


def fcc_cell(a: float) -> np.ndarray:
    return a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def bcc_cell(a: float) -> np.ndarray:
    return a / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])


def dump_trace(name: str, recorder) -> None:
    TRACES.mkdir(parents=True, exist_ok=True)
    (TRACES / f"{name}.json").write_text(json.dumps(recorder.to_trace_dict(), indent=1))


def _run(name, system, start_mag, scheme, *, max_iter, extra=None):
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.scf.uspp_loop import scf_uspp

    t0 = time.time()
    r = scf_uspp(
        system, SpinPBE(), smearing="gaussian", width=0.1, etol=1e-8,
        rhotol=1e-6, max_iter=max_iter, verbose=False, nspin=2,
        start_mag=start_mag, mixing_scheme=scheme,
    )
    wall = round(time.time() - t0, 1)
    rec = r.recorder
    mag = rec.iters[-1]["mag_abs"] if rec.iters else None
    dump_trace(name, rec)
    row = {
        "case": name,
        "scheme": scheme,
        "start_mag": start_mag,
        "converged": bool(r.converged),
        "n_iter": int(r.n_iter),
        "energy_eV": round(float(r.energies.total), 6),
        "mag_abs_muB": None if mag is None else round(float(mag), 4),
        "wall_s": wall,
        "tags": [{"tag": t, "reason": rr} for t, rr in rec.diagnose()],
        "scf_diagnostics": rec.summarize(),
        "drho_series": [round(i["drho"], 6) for i in rec.iters],
    }
    if extra:
        row.update(extra)
    RESULTS.append(row)
    print(f"  {name}: scheme={scheme} conv={r.converged} n={r.n_iter} "
          f"E={row['energy_eV']} m={row['mag_abs_muB']} wall={wall}s "
          f"tags={[t for t, _ in rec.diagnose()]}", flush=True)
    return row


def case_fe():
    """bcc Fe PAW, robust ferromagnet. Documented pulay=29/30, johnson forced
    was 93 (wisdom.md) but 16 in the 2026-07-30 case study."""
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp_setup import setup_uspp

    paw = parse_upf_paw(FIX / "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF")
    cell, pos = bcc_cell(2.87), [[0.0, 0, 0]]
    system = setup_uspp(cell, pos, [0], [paw], ecut=45 * RY, ecutrho=360 * RY,
                        kmesh=(8, 8, 8), nbands=12, use_symmetry=True)
    for sm in (0.7, 0.3):
        for scheme in SCHEMES:
            _run(f"fe_sm{sm}_{scheme}", system, [sm], scheme, max_iter=120,
                 extra={"system": "bcc Fe PAW", "doc": "pulay 29/30"})


def case_ni():
    """fcc Ni PAW, near the Stoner boundary. Documented johnson=18, pulay=27.
    Three start_mag so branch behavior is visible, not one lucky seed."""
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp_setup import setup_uspp

    paw = parse_upf_paw(FIX / "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF")
    cell, pos = fcc_cell(3.52), [[0.0, 0, 0]]
    system = setup_uspp(cell, pos, [0], [paw], ecut=45 * RY, ecutrho=360 * RY,
                        kmesh=(8, 8, 8), nbands=12, use_symmetry=True)
    for sm in (0.6, 0.3, 0.02):
        for scheme in SCHEMES:
            _run(f"ni_sm{sm}_{scheme}", system, [sm], scheme, max_iter=80,
                 extra={"system": "fcc Ni PAW", "doc": "johnson 18 / pulay 27"})


def case_afm_fe():
    """Two-sublattice AFM: simple-cubic Fe (B2/CsCl geometry) with the two
    Fe atoms seeded opposite (+0.7, -0.7). A genuine per-atom opposite-spin
    nspin=2 PAW SCF that reuses the trusted Fe PAW (no O, no Hubbard U). No
    PAW/USPP Cr or Co pseudo exists in-repo (only NC), so this stands in for
    the documented AFM Cr 2-atom case. Same fixed point across schemes is the
    oracle; a scheme that collapses one sublattice is a branch flip, recorded
    as such."""
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp_setup import setup_uspp

    paw = parse_upf_paw(FIX / "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF")
    a = 2.87
    cell = np.eye(3) * a
    pos = [[0.0, 0.0, 0.0], [0.5 * a, 0.5 * a, 0.5 * a]]
    # Collinear nspin=2 cannot use magnetic (magmoms) symmetry, and the plain
    # crystallographic spacegroup of B2 would symmetrize the two Fe equal and
    # kill the AFM order, so this runs on the full BZ (use_symmetry=False) at a
    # coarse 4x4x4 mesh to stay affordable.
    system = setup_uspp(cell, pos, [0, 0], [paw], ecut=45 * RY, ecutrho=360 * RY,
                        kmesh=(4, 4, 4), nbands=20, use_symmetry=False)
    for scheme in SCHEMES:
        _run(f"afm_fe_{scheme}", system, [0.7, -0.7], scheme, max_iter=60,
             extra={"system": "AFM Fe 2-atom (B2)", "doc": "no PAW Cr in repo"})


CASES = {"fe": case_fe, "ni": case_ni, "afm_fe": case_afm_fe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", help="subset (default: fe ni)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results.json", help="output filename")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    keys = a.cases if a.cases else ["fe", "ni"]
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
        (OUT / a.out).write_text(json.dumps(RESULTS, indent=1))
    print("WROTE", OUT / a.out, flush=True)


if __name__ == "__main__":
    main()
