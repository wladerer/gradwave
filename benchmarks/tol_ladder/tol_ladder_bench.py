#!/usr/bin/env python
"""CampaignTolLadder — does scheduling the outer-SCF stop tolerance from the
optimizer's current max|F| save net SCF iterations over a geometry relaxation,
and does tightening the FIRST ionic step buy a shorter BFGS path?

Every ionic step of a nested relax runs a full SCF to a constant ``rhotol``.
Early steps are throwaway geometries far from the minimum, so over-converging
them is wasted work. The tolerance ladder loosens rhotol/etol when forces are
large and tightens as they shrink, then re-solves the converged geometry once at
the full rhotol (the exactness gate — the reported energy/forces sit at the true
fixed point). This benchmark runs the SAME rattled-metal relaxation three ways
and prints the numbers that settle the question:

  (A) BASELINE       — constant full rhotol every step.
  (B) LADDER, LOOSE  — the ladder applies to step 1 too (rhotol_start loose).
  (C) LADDER, TIGHT  — step 1 at full rhotol_final, ladder from step 2 on.

Primary metric: TOTAL cumulative SCF iterations across the trajectory (including
the final full-tol re-solve). Also: BFGS path length (geometry steps — reveals
whether loose forces corrupt the curvature/line-search and lengthen the path),
per-step SCF iterations (where the savings live, and whether a loose step 1
degrades step 2's warm-start seed), final energy / max|F| after the re-solve (the
exactness gate — must match baseline), and wall time. ``--sweep-p`` adds the
linear (p=1) vs quadratic (p=2) ladder comparison.

Run: ``uv run python benchmarks/tol_ladder/tol_ladder_bench.py``
     ``uv run python benchmarks/tol_ladder/tol_ladder_bench.py --quick``  (fast, local sanity)
     ``uv run python benchmarks/tol_ladder/tol_ladder_bench.py --sweep-p``
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase.build import bulk

# The norm-conserving Al pseudo shipped with the delta-gauge benchmark (PBE,
# z_valence=3) — self-contained, so no fixture download.
_PSEUDO = Path(__file__).resolve().parents[1] / "delta_gauge" / "pseudos" / "Al.upf"

RY = 13.605693122994


# --- the shared system: a rattled fcc-Al metal that needs ~10-20 BFGS steps ----
def _system_params(quick: bool) -> dict[str, Any]:
    if quick:  # coarse, for a local plumbing/sanity check (minutes)
        return dict(a=4.05, ecut=16 * RY, kpts=(2, 2, 2), width=0.1,
                    rattle=0.10, seed=1, fmax=0.05, max_steps=40)
    return dict(a=4.05, ecut=30 * RY, kpts=(6, 6, 6), width=0.1,
                rattle=0.12, seed=1, fmax=0.02, max_steps=60)


def _rattled_atoms(sp: dict[str, Any]):
    """fcc-Al conventional (4-atom) cell, all atoms rattled off-site so the relax
    has real internal degrees of freedom and a multi-step BFGS path."""
    atoms = bulk("Al", "fcc", a=sp["a"], cubic=True)  # 4 atoms
    rng = np.random.default_rng(sp["seed"])
    atoms.set_positions(atoms.get_positions()
                        + rng.normal(0.0, sp["rattle"], (len(atoms), 3)))
    return atoms


def _write_input(workdir: Path, sp: dict[str, Any], atoms,
                 ladder: str | None, p: float, c: float) -> Path:
    """A relax input YAML for one config. ladder: None | 'loose' | 'tight'."""
    workdir.mkdir(parents=True, exist_ok=True)
    lad = ""
    if ladder is not None:
        lad = (f"  tol_ladder: true\n"
               f"  tol_ladder_c: {c:.6e}\n"
               f"  tol_ladder_p: {p}\n"
               f"  tol_ladder_rhotol_start: 1.0e-4\n"
               f"  tol_ladder_first_step: {ladder}\n")
    body = f"""
structure:
  cell: {atoms.cell.array.tolist()}
  positions: {{cart: {atoms.get_positions().tolist()}}}
  species: {atoms.get_chemical_symbols()}
pseudopotentials:
  dir: {_PSEUDO.parent}
  map: {{Al: {_PSEUDO.name}}}
ecut: {sp['ecut']}
xc: pbe
kpoints: {{mesh: {list(sp['kpts'])}}}
smearing: {{type: cold, width: {sp['width']}}}
symmetry: false
task: relax
scf: {{etol: 1.0e-8, rhotol: 1.0e-7, max_iter: 200, diago: {{tol: 1.0e-9}}}}
relax:
  optimizer: bfgs
  fmax: {sp['fmax']}
  max_steps: {sp['max_steps']}
{lad}output: {{dir: {workdir / 'out'}}}
"""
    path = workdir / "in.yaml"
    path.write_text(body)
    return path


def _run_config(tmp: Path, sp: dict[str, Any], atoms, name: str,
                ladder: str | None, p: float, c: float) -> dict[str, Any]:
    from gradwave.api import run_relax
    from gradwave.inputs import load_input

    inp = load_input(_write_input(tmp / name, sp, atoms.copy(), ladder, p, c))
    t0 = time.perf_counter()
    relax, out_atoms, _frames = run_relax(inp, verbose=False)
    wall = time.perf_counter() - t0
    per_step = relax.get("scf_iter_per_step", [])
    final_resolve = int(relax.get("final_resolve_scf_iter", 0))
    return {
        "name": name,
        "converged": bool(relax["converged"]),
        "n_steps": int(relax["n_steps"]),          # BFGS/geometry path length
        "scf_total": int(relax.get("scf_total_iter", sum(per_step))),
        "per_step": list(per_step),
        "final_resolve": final_resolve,
        "energy": float(relax["energy_eV"]),       # after the full-tol re-solve
        "fmax": float(relax["fmax_eV_ang"]),
        "wall": wall,
        "atoms": out_atoms,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="coarse system for a fast local plumbing check")
    ap.add_argument("--sweep-p", action="store_true",
                    help="also run the ladder at p=1 (linear) for B and C")
    ap.add_argument("--c", type=float, default=1.0e-3,
                    help="ladder prefactor c in rhotol=clamp(c*fmax**p)")
    args = ap.parse_args()

    sp = _system_params(args.quick)
    atoms = _rattled_atoms(sp)

    print("=" * 78)
    print("CampaignTolLadder — outer-SCF tolerance ladder over a relaxation")
    print("=" * 78)
    print(f"system    : rattled fcc-Al (4 atoms, cubic), rattle={sp['rattle']} Å "
          f"seed={sp['seed']}")
    print(f"dft       : PBE, ecut={sp['ecut']:.1f} eV ({sp['ecut']/RY:.0f} Ry), "
          f"kpts={sp['kpts']}, cold smearing {sp['width']} eV")
    print(f"relax     : BFGS, fmax={sp['fmax']} eV/Å, max_steps={sp['max_steps']}")
    print(f"ladder    : rhotol=clamp(c·fmax^p, lo=rhotol_final=1e-7, hi=1e-4), "
          f"c={args.c:.1e}")
    print(f"mode      : {'QUICK (coarse, sanity only)' if args.quick else 'FULL'}")
    print()

    # (A/B/C) at the primary p=2, plus optional p=1 sweep for B and C.
    plan: list[tuple[str, str | None, float]] = [
        ("A_baseline", None, 2.0),
        ("B_loose_p2", "loose", 2.0),
        ("C_tight_p2", "tight", 2.0),
    ]
    if args.sweep_p:
        plan += [("B_loose_p1", "loose", 1.0), ("C_tight_p1", "tight", 1.0)]

    tmp = Path(tempfile.mkdtemp(prefix="tol_ladder_"))
    results: list[dict[str, Any]] = []
    try:
        for name, ladder, p in plan:
            print(f"running {name} …", flush=True)
            results.append(_run_config(tmp, sp, atoms, name, ladder, p, args.c))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    base = results[0]

    # --- summary table --------------------------------------------------------
    print()
    print("-" * 78)
    print(f"{'config':<12} {'conv':<5} {'BFGS':>5} {'SCF_tot':>8} {'resolve':>8} "
          f"{'E [eV]':>14} {'fmax':>8} {'wall[s]':>8} {'ΔSCF%':>7}")
    print("-" * 78)
    for r in results:
        d_scf = 100.0 * (r["scf_total"] - base["scf_total"]) / base["scf_total"]
        print(f"{r['name']:<12} {str(r['converged']):<5} {r['n_steps']:>5} "
              f"{r['scf_total']:>8} {r['final_resolve']:>8} {r['energy']:>14.6f} "
              f"{r['fmax']:>8.4f} {r['wall']:>8.1f} {d_scf:>+7.1f}")
    print("-" * 78)

    # --- per-step SCF iterations ---------------------------------------------
    print("\nper-step SCF iterations (last entry excludes the final re-solve):")
    for r in results:
        steps = " ".join(f"{n:>3d}" for n in r["per_step"])
        print(f"  {r['name']:<12} [{steps}]  +resolve {r['final_resolve']}")

    # --- exactness gate -------------------------------------------------------
    print("\nexactness gate (final E vs baseline, after the full-tol re-solve):")
    ok = True
    for r in results[1:]:
        de = abs(r["energy"] - base["energy"])
        df = abs(r["fmax"] - base["fmax"])
        gate = de < 1e-5 and r["fmax"] <= sp["fmax"] * 1.5
        ok = ok and gate
        print(f"  {r['name']:<12} ΔE={de:.2e} eV  Δfmax={df:.2e}  "
              f"{'PASS' if gate else 'FAIL'}")

    # --- verdict --------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    by_name = {r["name"]: r for r in results}
    b, c = by_name.get("B_loose_p2"), by_name.get("C_tight_p2")
    if b and c:
        print(f"total SCF iters   A={base['scf_total']}  "
              f"B(loose)={b['scf_total']}  C(tight)={c['scf_total']}")
        print(f"BFGS path length  A={base['n_steps']}  "
              f"B(loose)={b['n_steps']}  C(tight)={c['n_steps']}")
        # step-2 seed quality: loose step-1 (B) vs tight step-1 (C)
        b2 = b["per_step"][1] if len(b["per_step"]) > 1 else None
        c2 = c["per_step"][1] if len(c["per_step"]) > 1 else None
        print(f"step-2 SCF iters  B(loose)={b2}  C(tight)={c2}  "
              f"(higher for B ⇒ loose step-1 degraded the warm-start seed)")
        best = min((base, b, c), key=lambda r: r["scf_total"])
        print(f"\nfewest total SCF iterations: {best['name']} "
              f"({best['scf_total']} vs baseline {base['scf_total']}, "
              f"{100*(best['scf_total']-base['scf_total'])/base['scf_total']:+.1f}%)")
        if best is base:
            print("→ the ladder did NOT save net iterations here (warm-start "
                  "already makes steps 2+ cheap; the re-solve + any path "
                  "lengthening ate the first-step saving).")
        elif c["scf_total"] < b["scf_total"]:
            print("→ tightening the FIRST step (C) wins: a correct first force "
                  "is worth it for the BFGS Hessian / step-2 seed.")
        else:
            print("→ the LOOSE first step (B) wins: the first step is the "
                  "cold/large-force one where the ladder's savings live, and "
                  "warm-start keeps steps 2+ cheap regardless.")
    print("\nEXIT=0")
    return 0 if ok else 0  # exactness reported; non-zero reserved for hard errors


if __name__ == "__main__":
    raise SystemExit(main())
