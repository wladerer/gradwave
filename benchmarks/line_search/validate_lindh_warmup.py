"""Prove initial_hessian=lindh (+ warmup) removes the early BFGS overshoot bump.

Relaxes a rattled 2×CO₂ molecular crystal — the stiff-C=O / soft-intermolecular
system whose max force *rises* around ionic step 5 with the default identity
Hessian — three ways, and prints the per-step fmax so the bump is visible (or
gone). Same start, same minimum expected.

  baseline      : initial_hessian=identity, line_search=off  (shows the bump)
  lindh         : initial_hessian=lindh,    line_search=off  (fixes the direction)
  lindh+warmup  : initial_hessian=lindh,    line_search=adaptive, warmup=2

Run on the H100 box (branch feat/parallel-line-search, latest):
    uv run python benchmarks/line_search/validate_lindh_warmup.py --device cuda
"""
import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]
_PDIR = _ROOT / "tests/fixtures/qe/pseudos"


def build_mol_crystal():
    from ase import Atoms
    from ase.build import molecule

    m = molecule("CO2")
    a = Atoms(cell=[7.0, 7.0, 7.0], pbc=True)
    for shift in ([1.5, 1.5, 1.5], [4.5, 4.0, 4.5]):
        mm = m.copy()
        mm.translate(shift)
        a += mm
    rng = np.random.default_rng(1)
    a.set_positions(a.get_positions() + rng.normal(0, 0.08, a.get_positions().shape))
    return a


def yaml_for(at, *, hessian, line_search, warmup, device, outdir):
    ls = ""
    if line_search != "off":
        ls = (f"  line_search: {line_search}\n"
              "  line_search_n_samples: 4\n  line_search_n_workers: 1\n"
              f"  line_search_warmup: {warmup}\n  line_search_warmup_samples: 6\n")
    body = f"""
structure:
  cell: {np.asarray(at.cell.array).tolist()}
  positions: {{cart: {at.get_positions().tolist()}}}
  species: {at.get_chemical_symbols()}
pseudopotentials:
  dir: {_PDIR}
  map: {{C: C_ONCV_PBE-1.2.upf, O: O_ONCV_PBE-1.2.upf}}
ecut: {18 * RY}
xc: pbe
kpoints: {{mesh: [1, 1, 1]}}
smearing: {{type: none, width: 0.0}}
symmetry: false
task: relax
device: {device}
scf: {{etol: 1.0e-8, rhotol: 1.0e-7, max_iter: 120}}
relax:
  method: nested
  optimizer: bfgs
  fmax: 0.03
  max_steps: 60
  initial_hessian: {hessian}
{ls}output: {{dir: {outdir}}}
"""
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(body)
    f.close()
    return f.name


def run(at, *, hessian, line_search, warmup, device, tmp, tag):
    import torch

    from gradwave.api import run_relax
    from gradwave.inputs import load_input

    path = yaml_for(at.copy(), hessian=hessian, line_search=line_search,
                    warmup=warmup, device=device, outdir=f"{tmp}/{tag}")
    inp = load_input(path)
    if device != "cpu":
        torch.cuda.synchronize()
    t = time.perf_counter()
    relax, _atoms, _frames = run_relax(inp, verbose=False)
    if device != "cpu":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t
    fmax = [float(s["fmax_eV_ang"]) for s in relax["trajectory"] if "fmax_eV_ang" in s]
    return {"tag": tag, "wall": wall, "n_steps": relax["n_steps"],
            "energy": float(relax["energy_eV"]), "converged": relax["converged"],
            "fmax": fmax}


def early_bump(fmax, window=8):
    """Largest fmax increase between consecutive early steps (the overshoot)."""
    w = fmax[:window]
    return max((w[i + 1] - w[i] for i in range(len(w) - 1)), default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    at = build_mol_crystal()
    tmp = tempfile.mkdtemp()
    print(f"# Lindh+warmup proof | device={device} ({dev_name}) | 2xCO2 rattled | {len(at)} atoms")

    configs = [
        ("baseline", dict(hessian="identity", line_search="off", warmup=0)),
        ("lindh", dict(hessian="lindh", line_search="off", warmup=0)),
        ("lindh+warmup", dict(hessian="lindh", line_search="adaptive", warmup=2)),
    ]
    results = []
    for tag, kw in configs:
        r = run(at, device=device, tmp=tmp, tag=tag, **kw)
        results.append(r)
        print(f"\n=== {tag} ===  steps={r['n_steps']}  wall={r['wall']:.1f}s  "
              f"E={r['energy']:.6f} eV  conv={r['converged']}  "
              f"early-fmax-rise={early_bump(r['fmax']):+.4f} eV/Å")
        print("  fmax/step: " + " ".join(f"{f:.3f}" for f in r["fmax"][:12])
              + (" ..." if len(r["fmax"]) > 12 else ""))

    e0 = results[0]["energy"]
    print("\n# same minimum (dE vs baseline, meV): "
          + " ".join(f"{r['tag']}={abs(r['energy'] - e0) * 1e3:.3f}" for r in results[1:]))
    print("# early overshoot (max consecutive fmax rise in first 8 steps, eV/Å):")
    for r in results:
        print(f"#   {r['tag']:14s} {early_bump(r['fmax']):+.4f}")
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
