"""M2 end-to-end: does the subspace-χ₀ Woodbury net win survive the REAL driver?

Rattle a relaxed Al(100) slab and relax it through ``api.run_relax`` (the nested
BFGS+SCF engine, the production path), feature ON vs OFF:

  OFF  scf.mixing.precond = local_tf, chi0_precond = false     (the M1 baseline)
  ON   scf.mixing.precond = local_tf, chi0_precond = true      (build-once/reuse)

ON builds the Woodbury subspace ONCE at the first converged geometry (behind the
auto-abstain ρ(M) gate) and reuses it as precond_op on every later ionic step.
We count total FFT launches (process opcount, which each SCF re-enables — so the
sum spans every step's SCF plus the one-time gate+build) and wall time both ways,
and assert the relaxed energy + geometry MATCH (a preconditioner cannot move the
fixed point). This is the deliverable: the ~1.9× amortized FFT win must hold
through the driver, not just the experiment harness.

Usage (asus only):
    uv run python experiments/chi0_precond_crux/m2_end_to_end.py \
        --layers 4 --kmesh 4 --steps 8 --rattle 0.08 --out .../m2_e2e.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.build import fcc100

from gradwave.core import opcount
from gradwave.inputs import load_input

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]
PSEUDO = _ROOT / "tests" / "fixtures" / "qe" / "pseudos"


def _write_input(tmp: Path, atoms, *, kmesh, ecut_ry, steps, fmax, chi0, nbands):
    cell = atoms.cell.array.tolist()
    pos = atoms.get_positions().tolist()
    species = atoms.get_chemical_symbols()
    body = f"""
structure:
  cell: {cell}
  positions: {{cart: {pos}}}
  species: {species}
symmetry: false
nbands: {nbands}
pseudopotentials:
  dir: {PSEUDO}
  map: {{Al: Al_ONCV_PBE-1.2.upf}}
ecut: {ecut_ry * RY}
kpoints: {{mesh: [{kmesh}, {kmesh}, 1]}}
smearing: {{type: gaussian, width: 0.1}}
scf:
  max_iter: 120
  etol: 1.0e-9
  rhotol: 1.0e-8
  mixing: {{scheme: pulay, alpha: 0.7, precond: local_tf, chi0_precond: {str(chi0).lower()}}}
task: relax
relax: {{optimizer: bfgs, fmax: {fmax}, max_steps: {steps}, method: nested,
        extrapolation: reuse}}
"""
    p = tmp / f"in_{'on' if chi0 else 'off'}.yaml"
    p.write_text(body)
    return load_input(p)


def _run(tag, inp):
    from gradwave.api import run_relax

    opcount.enable()
    prev = opcount.snapshot()
    t0 = time.time()
    relax, atoms, _frames = run_relax(inp, verbose=True)
    dt = time.time() - t0
    ffts = int(opcount.since(prev)["fft"])
    cache = getattr(atoms.calc, "_chi0_cache", None)
    row = {
        "tag": tag,
        "converged": bool(relax["converged"]),
        "n_steps": int(relax["n_steps"]),
        "energy_eV": float(relax["energy_eV"]),
        "fmax_eV_ang": float(relax["fmax_eV_ang"]),
        "scf_total_iter": int(relax.get("scf_total_iter", -1)),
        "scf_iter_per_step": relax.get("scf_iter_per_step"),
        "total_fft": ffts,
        "wall_s": round(dt, 1),
        "positions_ang": relax["positions_ang"],
        "chi0": {
            "engaged": bool(getattr(cache, "engaged", False)),
            "rho": (None if cache is None else cache.rho),
            "n_col": (None if getattr(cache, "precond", None) is None
                      else int(cache.precond.n_col)),
            "gate_ffts": int(getattr(cache, "gate_ffts", 0)),
            "build_ffts": int(getattr(cache, "build_ffts", 0)),
            "precond_calls": (None if getattr(cache, "precond", None) is None
                              else int(cache.precond.n_calls)),
        } if cache is not None else None,
    }
    print(f"\n>>> {tag}: {relax['n_steps']} steps  conv={relax['converged']}  "
          f"E={relax['energy_eV']:.8f} eV  scf_iters={row['scf_total_iter']}  "
          f"FFT={ffts}  {dt:.0f}s\n", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--kmesh", type=int, default=4)
    ap.add_argument("--ecut", type=float, default=25.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--rattle", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent

    slab = fcc100("Al", size=(1, 1, args.layers), a=4.05, vacuum=8.0)
    slab.rattle(stdev=args.rattle, seed=args.seed)
    natoms = len(slab)
    nbands = 6 * natoms

    meta = {"args": vars(args), "natoms": natoms, "nbands": nbands}
    print(f"=== Al(100) {args.layers}-layer slab, {natoms} atoms, "
          f"kmesh={args.kmesh}, rattle={args.rattle} Å ===", flush=True)

    rows = []
    for chi0 in (False, True):
        inp = _write_input(tmp, slab.copy(), kmesh=args.kmesh, ecut_ry=args.ecut,
                           steps=args.steps, fmax=args.fmax, chi0=chi0,
                           nbands=nbands)
        rows.append(_run("ON" if chi0 else "OFF", inp))
        out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    off, on = rows[0], rows[1]
    dE = abs(on["energy_eV"] - off["energy_eV"])
    dpos = float(np.abs(np.array(on["positions_ang"])
                        - np.array(off["positions_ang"])).max())
    verdict = {
        "dE_eV": dE,
        "d_pos_ang": dpos,
        "same_answer": bool(dE < 1e-6 and dpos < 1e-4),
        "fft_off": off["total_fft"],
        "fft_on": on["total_fft"],
        "fft_ratio_off_over_on": round(off["total_fft"]
                                       / max(1, on["total_fft"]), 3),
        "wall_ratio_off_over_on": round(off["wall_s"]
                                        / max(1e-9, on["wall_s"]), 3),
        "scf_iters_off": off["scf_total_iter"],
        "scf_iters_on": on["scf_total_iter"],
        "chi0_engaged": on["chi0"]["engaged"] if on["chi0"] else False,
    }
    meta["verdict"] = verdict
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    print("\n===== M2 END-TO-END VERDICT =====")
    print(f"  faithfulness: dE={dE:.2e} eV  d|pos|={dpos:.2e} Å  "
          f"SAME={verdict['same_answer']}")
    print(f"  FFT: OFF={off['total_fft']}  ON={on['total_fft']}  "
          f"→ {verdict['fft_ratio_off_over_on']}× fewer with ON")
    print(f"  wall: OFF={off['wall_s']}s  ON={on['wall_s']}s  "
          f"→ {verdict['wall_ratio_off_over_on']}×")
    print(f"  scf iters: OFF={off['scf_total_iter']}  ON={on['scf_total_iter']}")
    if on["chi0"]:
        c = on["chi0"]
        print(f"  chi0: engaged={c['engaged']} ρ={c['rho']} n_col={c['n_col']} "
              f"gate_ffts={c['gate_ffts']} build_ffts={c['build_ffts']} "
              f"precond_calls={c['precond_calls']}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
