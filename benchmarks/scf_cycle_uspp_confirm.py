"""USPP/PAW johnson-vs-pulay confirmation: the two cases the small-cell sweep
(scf_cycle_uspp.py) left open before flipping the scf_uspp default to johnson.

1. nspin=2 (magnetic) PAW — the default flip is unconditional, so a magnetic
   PAW must not regress. FM bcc Fe and FM fcc Ni, pulay vs johnson, same seed.
2. a medium PAW insulator (8-atom cubic diamond Si) — does the nspin=1 win
   hold at a larger cell.

Reports n_iter, free energy/atom, and (nspin=2) the converged moment, so the
"same fixed point, fewer iterations" claim is checkable per system.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.core.xc.spin import SpinPBE  # noqa: E402
from gradwave.pseudo.upf_paw import parse_upf_paw  # noqa: E402
from gradwave.scf.uspp import scf_uspp, setup_uspp  # noqa: E402

torch.set_num_threads(8)
sys.stdout.reconfigure(line_buffering=True)
RY = 13.605693122994
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
BCC = np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
PSE = ROOT / "tests/fixtures/qe/pseudos"


def _paw(name):
    return parse_upf_paw(PSE / name)


def fe_fm_paw():
    a = 2.87
    cell = a / 2 * BCC
    return setup_uspp(cell, np.zeros((1, 3)), [0],
                      [_paw("Fe.pbe-spn-kjpaw_psl.1.0.0.UPF")],
                      ecut=45 * RY, ecutrho=360 * RY, kmesh=(4, 4, 4), nbands=16,
                      use_symmetry=True)


def ni_fm_paw():
    a = 3.52
    cell = a / 2 * FCC
    return setup_uspp(cell, np.zeros((1, 3)), [0],
                      [_paw("Ni.pbe-spn-kjpaw_psl.1.0.0.UPF")],
                      ecut=45 * RY, ecutrho=360 * RY, kmesh=(4, 4, 4), nbands=16,
                      use_symmetry=True)


def si_paw_m():
    # 8-atom conventional cubic diamond Si
    a = 5.43
    cell = a * np.eye(3)
    frac = np.array([
        [0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
        [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25],
    ])
    pos = frac * a
    return setup_uspp(cell, pos, [0] * 8, [_paw("Si.pbe-n-kjpaw_psl.1.0.0.UPF")],
                      ecut=30 * RY, ecutrho=120 * RY, kmesh=(2, 2, 2),
                      use_symmetry=True)


SYSTEMS = {
    "fe_fm_paw": dict(build=fe_fm_paw, cls="FM metal (PAW, nspin=2)",
                      xc=SpinPBE(), scf=dict(nspin=2, start_mag=[0.5],
                                             smearing="gaussian", width=0.1)),
    "ni_fm_paw": dict(build=ni_fm_paw, cls="FM metal (PAW, nspin=2)",
                      xc=SpinPBE(), scf=dict(nspin=2, start_mag=[0.3],
                                             smearing="gaussian", width=0.1)),
    "si_paw_m": dict(build=si_paw_m, cls="insulator (PAW, medium 8-atom)",
                     xc=PBE(), scf=dict(smearing="none")),
}

COMMON = dict(etol=1e-9, rhotol=1e-8, verbose=False, batched=True, max_iter=100)


def run_one(name, scheme):
    spec = SYSTEMS[name]
    kw = dict(COMMON)
    kw.update(spec["scf"])
    system = spec["build"]()
    natoms = int(system.positions.shape[0])
    try:
        t0 = time.perf_counter()
        res = scf_uspp(system, spec["xc"], mixing_scheme=scheme, **kw)
        wall = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return dict(error=f"{type(exc).__name__}: {exc}")
    fr = float(res["free_energy"]) if "free_energy" in res else float(
        res["energies"].free_energy)
    it = int(res["n_iter"])
    conv = bool(res["converged"])
    final_res = float(res["history"][-1]["res"]) if res.get("history") else None
    mag = res.get("mag_total")
    return dict(n_iter=it, converged=conv, free_energy=fr,
                free_energy_per_atom=fr / natoms, final_res=final_res,
                mag_total=(None if mag is None else float(mag)),
                wall_s=round(wall, 2), natoms=natoms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="fe_fm_paw,ni_fm_paw,si_paw_m")
    ap.add_argument("--out", default="/tmp/scf_uspp_confirm.json")
    args = ap.parse_args()
    results = {}
    for name in args.systems.split(","):
        results[name] = {}
        for scheme in ("pulay", "johnson"):
            rec = run_one(name, scheme)
            results[name][scheme] = rec
            epa = rec.get("free_energy_per_atom")
            epas = "" if epa is None else f" E/at={epa:.8f}"
            mag = rec.get("mag_total")
            ms = "" if mag is None else f" m={mag:+.4f}"
            print(f"  {name:10s} {scheme:8s} it={rec.get('n_iter','ERR')!s:>4} "
                  f"conv={rec.get('converged')!s:>5}{epas}{ms} "
                  f"{rec.get('wall_s','')}s {rec.get('error','')}", flush=True)
            Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
