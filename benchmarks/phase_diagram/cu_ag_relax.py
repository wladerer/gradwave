"""Variable-cell relaxation for the Cu-Ag mixing energy, GPU-capable.

The fixed-cell run overestimated the mixing energy because the size-mismatched
L1_0 alloy carried coherency strain. This relaxes the cell and ions of each
structure to zero stress and force (FrechetCellFilter + BFGS through the gradwave
ASE calculator), so the mixing energy is the relaxed one. Per-step timing is
printed unbuffered, and --max-steps caps a run for a timing estimate before
committing to the full relaxation.

  uv run python benchmarks/phase_diagram/cu_ag_relax.py \
      --pseudo-dir benchmarks/delta_gauge/pseudos --outdir ~/cuag_relax \
      --structures CuAg --device cuda --max-steps 6   # timing probe
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.build import bulk


def build_structures(a_a, a_b, a_ab, sym_a, sym_b):
    a4 = bulk(sym_a, "fcc", a=a_a, cubic=True)
    b4 = bulk(sym_b, "fcc", a=a_b, cubic=True)
    ab = bulk(sym_a, "fcc", a=a_ab, cubic=True)
    frac = ab.get_scaled_positions()
    ab.set_chemical_symbols([sym_a if p[2] < 0.25 else sym_b for p in frac])
    return {sym_a: a4, sym_b: b4, "CuAg": ab}


def relax(atoms, calc_kwargs, fmax, max_steps):
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    from gradwave.calculator import GradWave

    atoms = atoms.copy()
    atoms.calc = GradWave(**calc_kwargs)
    opt = BFGS(FrechetCellFilter(atoms), logfile=None)  # relaxes cell + ions
    t0 = time.perf_counter()
    per_step = []

    def _cb():
        e = float(atoms.get_potential_energy()) / len(atoms)
        fmax_now = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        per_step.append(time.perf_counter() - t0)
        print(f"  step {opt.nsteps:2d}  E={e:.5f} eV/atom  fmax={fmax_now:.4f}"
              f"  t={per_step[-1]:.1f}s", flush=True)

    opt.attach(_cb, interval=1)
    opt.run(fmax=fmax, steps=max_steps)
    wall = time.perf_counter() - t0
    return {"energy_per_atom": float(atoms.get_potential_energy()) / len(atoms),
            "volume_per_atom": float(atoms.get_volume()) / len(atoms),
            "n_steps": int(opt.nsteps), "wall_s": wall,
            "sec_per_step": wall / max(opt.nsteps, 1),
            "converged": bool(opt.nsteps < max_steps),
            "cell": atoms.cell.array.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--elements", nargs=2, default=["Cu", "Ag"])
    ap.add_argument("--a0", nargs=3, type=float, default=[3.63, 4.15, 3.90])
    ap.add_argument("--structures", nargs="+", default=None,
                    help="subset to relax, e.g. CuAg for a timing probe")
    ap.add_argument("--ecut-ry", type=float, default=55.0)
    ap.add_argument("--kmesh", type=int, nargs=3, default=[8, 8, 8])
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ry = 13.605693122994
    sym_a, sym_b = a.elements
    pdir = Path(a.pseudo_dir).resolve()
    calc_kwargs = dict(
        ecut=a.ecut_ry * ry,
        pseudopotentials={sym_a: str(pdir / f"{sym_a}.upf"),
                          sym_b: str(pdir / f"{sym_b}.upf")},
        kpts=tuple(a.kmesh), smearing="gaussian", width=0.1,
        use_symmetry=True, device=a.device, max_iter=200, verbose=False)

    structs = build_structures(a.a0[0], a.a0[1], a.a0[2], sym_a, sym_b)
    keys = a.structures if a.structures else list(structs)
    out = {"system": f"{sym_a}-{sym_b}", "device": a.device, "kmesh": a.kmesh,
           "ecut_ry": a.ecut_ry, "results": {}}
    for key in keys:
        print(f"=== relaxing {key} ({a.device}) ===", flush=True)
        out["results"][key] = relax(structs[key], calc_kwargs, a.fmax, a.max_steps)
        r = out["results"][key]
        print(f"{key}: E={r['energy_per_atom']:.5f} eV/atom  steps={r['n_steps']}"
              f"  {r['sec_per_step']:.1f} s/step  wall={r['wall_s']/60:.1f} min"
              f"  converged={r['converged']}", flush=True)

    outdir = Path(a.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "relax.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
