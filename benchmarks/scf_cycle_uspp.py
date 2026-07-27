"""USPP/PAW SCF iteration-count study: does johnson beat pulay as the scf_uspp
default? (perf/scf-cycle-study — the options.py MixerOptions.scheme question.)

scf_uspp defaults mixing_scheme="pulay" for ALL nspin (uspp_loop.py:730), while
loop.py's NC path already auto-picks johnson for nspin==2. The docs record fcc Pt
PAW johnson 13 vs pulay 17 (docs/manual/performance.md). This confirms the
scheme effect on fresh PAW runs across a PAW insulator (no-regression check) and
two PAW metals, at the same fixed point.

Usage (asus, canonical venv):
  PYTHONPATH=src .venv/bin/python benchmarks/scf_cycle_uspp.py --out /tmp/scf_cycle_uspp.json
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
from gradwave.pseudo.upf_paw import parse_upf_paw  # noqa: E402
from gradwave.scf.uspp import scf_uspp, setup_uspp  # noqa: E402

torch.set_num_threads(8)
sys.stdout.reconfigure(line_buffering=True)
RY = 13.605693122994
FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
PSE = ROOT / "tests/fixtures/qe/pseudos"


def _paw(name):
    return parse_upf_paw(PSE / name)


# label: (build -> USPPSystem, scf kwargs, class)
def si_paw():
    a = 5.43
    cell = a / 2 * FCC
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    return setup_uspp(cell, pos, [0, 0], [_paw("Si.pbe-n-kjpaw_psl.1.0.0.UPF")],
                      ecut=30 * RY, ecutrho=120 * RY, kmesh=(4, 4, 4),
                      use_symmetry=True)


def cu_paw():
    a = 3.61
    cell = a / 2 * FCC
    return setup_uspp(cell, np.zeros((1, 3)), [0],
                      [_paw("Cu.pbe-dn-kjpaw_psl.1.0.0.UPF")],
                      ecut=45 * RY, ecutrho=360 * RY, kmesh=(6, 6, 6), nbands=20,
                      use_symmetry=True)


def pt_paw():
    a = 3.92
    cell = a / 2 * FCC
    return setup_uspp(cell, [[0.0, 0, 0]], [0],
                      [_paw("Pt.pbe-n-kjpaw_psl.1.0.0.UPF")],
                      ecut=40 * RY, ecutrho=400 * RY, kmesh=(6, 6, 6), nbands=18,
                      use_symmetry=True)


SYSTEMS = {
    "si_paw": dict(build=si_paw, cls="insulator (PAW)",
                   scf=dict(smearing="none")),
    "cu_paw": dict(build=cu_paw, cls="noble metal (PAW)",
                   scf=dict(smearing="gaussian", width=0.1)),
    "pt_paw": dict(build=pt_paw, cls="noble metal (PAW)",
                   scf=dict(smearing="gaussian", width=0.2)),
}

COMMON = dict(etol=1e-9, rhotol=1e-8, verbose=False, batched=True, max_iter=80)


def run_one(name, scheme):
    spec = SYSTEMS[name]
    kw = dict(COMMON)
    kw.update(spec["scf"])
    system = spec["build"]()
    natoms = int(system.positions.shape[0])
    try:
        t0 = time.perf_counter()
        res = scf_uspp(system, PBE(), mixing_scheme=scheme, **kw)
        wall = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        return dict(error=f"{type(exc).__name__}: {exc}")
    fr = float(res["free_energy"]) if "free_energy" in res else float(
        res["energies"].free_energy)
    it = int(res["n_iter"])
    conv = bool(res["converged"])
    final_res = float(res["history"][-1]["res"]) if res.get("history") else None
    return dict(n_iter=it, converged=conv, free_energy=fr,
                free_energy_per_atom=fr / natoms, final_res=final_res,
                wall_s=round(wall, 2), natoms=natoms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="si_paw,cu_paw,pt_paw")
    ap.add_argument("--out", default="/tmp/scf_cycle_uspp.json")
    args = ap.parse_args()
    results = {}
    for name in args.systems.split(","):
        results[name] = {}
        for scheme in ("pulay", "broyden", "johnson"):
            rec = run_one(name, scheme)
            results[name][scheme] = rec
            epa = rec.get("free_energy_per_atom")
            epas = "" if epa is None else f" E/at={epa:.8f}"
            print(f"  {name:8s} {scheme:8s} it={rec.get('n_iter','ERR')!s:>4} "
                  f"conv={rec.get('converged')!s:>5}{epas} "
                  f"{rec.get('wall_s','')}s {rec.get('error','')}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
