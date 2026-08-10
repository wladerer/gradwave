"""Capstone: stack every relax win against a naive BFGS relax.

  naive       line_search=off, initial_hessian=identity, tol_ladder=off
              (plain fixed-step BFGS, constant SCF tolerance)
  everything  line_search=adaptive + warmup, initial_hessian=lindh, tol_ladder=on,
              n_workers>1 (parallel candidate SCFs)

Same system, same minimum. Reports ionic steps, total SCF iterations, and wall for
each — so the levers' product shows up as one number. Runs both configs on the GPU
(the algorithmic speedup); with --with-cpu-baseline it also runs naive on the CPU
for the full hardware+algorithm 'vs a laptop' headline.

Run on the H100 box (branch feat/relax-stack):
    uv run python benchmarks/relax_stack/combined_relax_stack.py --with-cpu-baseline
"""
import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]
_PDIR = _ROOT / "tests/fixtures/qe/pseudos"

NAIVE = dict(line_search="off", warmup=0, initial_hessian="identity", tol_ladder=False)
EVERY = dict(line_search="adaptive", warmup=2, initial_hessian="lindh", tol_ladder=True)


def build_mol_crystal():
    from ase import Atoms
    from ase.build import molecule

    m = molecule("CO2")
    a = Atoms(cell=[7.0, 7.0, 7.0], pbc=True)
    for shift in ([1.5, 1.5, 1.5], [4.5, 4.0, 4.5]):
        mm = m.copy()
        mm.translate(shift)
        a += mm
    a.set_positions(a.get_positions()
                    + np.random.default_rng(1).normal(0, 0.08, (len(a), 3)))
    return a, {"C": "C_ONCV_PBE-1.2.upf", "O": "O_ONCV_PBE-1.2.upf"}, (1, 1, 1), "none"


def build_defected_si(nrep):
    from ase import Atoms

    a0 = 5.43
    cell = (a0 / 2) * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    at = Atoms("Si2", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
               cell=cell, pbc=True).repeat((nrep, nrep, nrep))
    del at[0]
    at.rattle(0.06, seed=3)
    return at, {"Si": "Si_ONCV_PBE-1.2.upf"}, (2, 2, 2), "none"


def yaml_for(at, pmap, kpts, smear, *, cfg, ecut_ry, xc, device, outdir):
    ls = ""
    if cfg["line_search"] != "off":
        ls = (f"  line_search: {cfg['line_search']}\n"
              "  line_search_n_samples: 4\n  line_search_n_workers: 2\n"
              f"  line_search_warmup: {cfg['warmup']}\n  line_search_warmup_samples: 6\n")
    lad = ("  tol_ladder: true\n  tol_ladder_rhotol_start: 1.0e-4\n"
           "  tol_ladder_first_step: loose\n") if cfg["tol_ladder"] else ""
    pm = "{" + ", ".join(f"{s}: {p}" for s, p in pmap.items()) + "}"
    body = f"""
structure:
  cell: {np.asarray(at.cell.array).tolist()}
  positions: {{cart: {at.get_positions().tolist()}}}
  species: {at.get_chemical_symbols()}
pseudopotentials:
  dir: {_PDIR}
  map: {pm}
ecut: {ecut_ry * RY}
xc: {xc}
kpoints: {{mesh: {list(kpts)}}}
smearing: {{type: {smear}, width: 0.0}}
symmetry: false
task: relax
device: {device}
scf: {{etol: 1.0e-8, rhotol: 1.0e-7, max_iter: 200}}
relax:
  method: nested
  optimizer: bfgs
  fmax: 0.03
  max_steps: 80
  initial_hessian: {cfg['initial_hessian']}
{ls}{lad}output: {{dir: {outdir}}}
"""
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(body)
    f.close()
    return f.name


def run(at, pmap, kpts, smear, *, cfg, ecut_ry, xc, device, tmp, tag):
    import torch

    from gradwave.api import run_relax
    from gradwave.inputs import load_input

    inp = load_input(yaml_for(at.copy(), pmap, kpts, smear, cfg=cfg, ecut_ry=ecut_ry,
                              xc=xc, device=device, outdir=f"{tmp}/{tag}"))
    if device != "cpu":
        torch.cuda.synchronize()
    t = time.perf_counter()
    relax, _a, _f = run_relax(inp, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    return {"tag": tag, "wall": time.perf_counter() - t, "steps": relax["n_steps"],
            "scf_iter": relax.get("scf_total_iter"), "E": float(relax["energy_eV"]),
            "conv": relax["converged"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--si-nrep", type=int, default=2, help="defected-Si tiling; atoms = 2*n**3 - 1")
    ap.add_argument("--with-cpu-baseline", action="store_true",
                    help="also run naive on CPU (the full hardware+algorithm headline)")
    args = ap.parse_args()

    import torch
    gpu = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if gpu else "cpu-only"
    print(f"# Combined relax stack | CUDA={gpu} ({dev_name})")

    systems = [
        ("mol_crystal", *build_mol_crystal(), 18.0, "pbe"),
        (f"defected_Si_{2 * args.si_nrep**3 - 1}", *build_defected_si(args.si_nrep), 15.0, "lda"),
    ]
    tmp = tempfile.mkdtemp()
    for name, at, pmap, kpts, smear, ecut, xc in systems:
        print(f"\n===== {name} ({len(at)} atoms) =====", flush=True)
        dev = "cuda" if gpu else "cpu"
        rn = run(at, pmap, kpts, smear, cfg=NAIVE, ecut_ry=ecut, xc=xc, device=dev,
                 tmp=tmp, tag=f"{name}_naive_{dev}")
        re = run(at, pmap, kpts, smear, cfg=EVERY, ecut_ry=ecut, xc=xc, device=dev,
                 tmp=tmp, tag=f"{name}_every_{dev}")
        rows = [("naive (GPU)", rn), ("everything (GPU)", re)]
        if args.with_cpu_baseline:
            rc = run(at, pmap, kpts, smear, cfg=NAIVE, ecut_ry=ecut, xc=xc, device="cpu",
                     tmp=tmp, tag=f"{name}_naive_cpu")
            rows.insert(0, ("naive (CPU)", rc))
        for lab, r in rows:
            print(f"  {lab:18s} steps {str(r['steps']):>3}  scf_iter {str(r['scf_iter']):>4}  "
                  f"wall {r['wall']:8.1f}s  E={r['E']:.6f} eV  conv={r['conv']}")
        print(f"  -> everything vs naive (GPU, algorithmic): {rn['wall'] / re['wall']:.2f}x "
              f"| steps {rn['steps']}->{re['steps']} | scf_iter {rn['scf_iter']}->{re['scf_iter']} "
              f"| dE={abs(rn['E'] - re['E']) * 1e3:.3f} meV")
        if args.with_cpu_baseline:
            full = rows[0][1]["wall"] / re["wall"]
            print(f"  -> everything(GPU) vs naive(CPU, full stack): {full:.2f}x")
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
