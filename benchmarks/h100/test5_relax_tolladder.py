"""H100 Test 5 — large-cell geometry relax with the tol-ladder ON vs OFF (on GPU).

The tol-ladder (#273) loosens each ionic step's SCF rho-tolerance when forces are
large (early, throwaway geometries) and tightens as forces shrink, then does one
exact full-tol re-solve at the accepted minimum. It should cut SCF iterations on the
early steps without moving the final geometry/energy. This measures that on a real
defected large cell (Si supercell + vacancy + rattle), relaxed on the H100:

  OFF : plain nested BFGS relax at fixed rhotol every step
  ON  : same relax, tol_ladder=true

Reports per-ionic-step SCF iteration counts, total SCF iterations, and wall for each,
plus confirmation that both land on the same energy.

REQUIRES the tol-ladder feature: run on branch feat/tol-ladder-relax (or main once
#273 has merged). The ladder is nested-engine + nspin only (api gates method=="nested").

Run on the H100 box:
    uv run python benchmarks/h100/test5_relax_tolladder.py --nrep 2
"""
import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]
_PSEUDO_DIR = _ROOT / "tests/fixtures/qe/pseudos"


def build_defected_si(nrep: int, rattle: float, seed: int):
    """Rattled diamond-Si supercell with a single vacancy (real internal forces)."""
    from ase import Atoms

    a = 5.43
    cell = (a / 2) * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    at = Atoms("Si2", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
               cell=cell, pbc=True).repeat((nrep, nrep, nrep))
    del at[0]                       # vacancy
    at.rattle(rattle, seed=seed)    # break symmetry -> nonzero forces to relax
    return np.asarray(at.cell.array), at.get_positions(), at.get_chemical_symbols()


def write_yaml(cell, pos, species, *, ecut_ry, kmesh, device, ladder, outdir):
    lad = ("  tol_ladder: true\n"
           "  tol_ladder_c: 1.0e-3\n"
           "  tol_ladder_p: 2.0\n"
           "  tol_ladder_rhotol_start: 1.0e-4\n"
           "  tol_ladder_first_step: loose\n") if ladder else ""
    body = f"""
structure:
  cell: {cell.tolist()}
  positions: {{cart: {pos.tolist()}}}
  species: {list(species)}
pseudopotentials:
  dir: {_PSEUDO_DIR}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {ecut_ry * RY}
xc: lda
kpoints: {{mesh: [{kmesh}, {kmesh}, {kmesh}]}}
symmetry: false
task: relax
device: {device}
scf: {{etol: 1.0e-8, rhotol: 1.0e-7, max_iter: 200, diago: {{tol: 1.0e-9}}}}
relax:
  method: nested
  optimizer: bfgs
  fmax: 0.02
  max_steps: 60
{lad}output: {{dir: {outdir}}}
"""
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(body)
    f.close()
    return f.name


def bench(path):
    import torch

    from gradwave.api import run_relax
    from gradwave.inputs import load_input

    inp = load_input(path)
    dev = inp.device
    if dev != "cpu":
        torch.cuda.synchronize()
    t = time.perf_counter()
    relax, _atoms, _frames = run_relax(inp, verbose=False)
    if dev != "cpu":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t
    return {
        "wall": wall,
        "n_steps": relax["n_steps"],
        "scf_iter_per_step": relax.get("scf_iter_per_step"),
        "scf_total_iter": relax.get("scf_total_iter"),
        "final_resolve": relax.get("final_resolve_scf_iter"),
        "energy_eV": relax["energy_eV"],
        "converged": relax["converged"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nrep", type=int, default=2, help="Si tiling; atoms = 2*nrep**3 - 1")
    ap.add_argument("--ecut", type=float, default=15.0, help="Ry")
    ap.add_argument("--kmesh", type=int, default=2)
    ap.add_argument("--rattle", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    cell, pos, species = build_defected_si(args.nrep, args.rattle, args.seed)
    nat = len(species)
    print(f"# H100 Test 5 — relax tol-ladder | device={device} ({dev_name}) | "
          f"Si {nat} atoms (vacancy+rattle) | ecut={args.ecut} Ry")

    tmp = tempfile.mkdtemp()
    off = bench(write_yaml(cell, pos, species, ecut_ry=args.ecut, kmesh=args.kmesh,
                           device=device, ladder=False, outdir=f"{tmp}/off"))
    on = bench(write_yaml(cell, pos, species, ecut_ry=args.ecut, kmesh=args.kmesh,
                          device=device, ladder=True, outdir=f"{tmp}/on"))

    def fmt(tag, r):
        print(f"{tag:>4}: wall {r['wall']:8.2f}s  steps {r['n_steps']:>3}  "
              f"scf_total_iter {str(r['scf_total_iter']):>5}  "
              f"final_resolve {str(r['final_resolve']):>4}  E={r['energy_eV']:.6f} eV  "
              f"conv={r['converged']}")
        if r["scf_iter_per_step"]:
            print(f"      per-step scf_iter: {r['scf_iter_per_step']}")

    fmt("OFF", off)
    fmt("ON", on)
    if off["scf_total_iter"] and on["scf_total_iter"]:
        speedup = off["wall"] / on["wall"]
        iters = off["scf_total_iter"] / on["scf_total_iter"]
        print(f"\n# tol-ladder: {iters:.2f}x fewer total SCF iterations, {speedup:.2f}x wall")
        print(f"# same minimum: dE = {abs(off['energy_eV'] - on['energy_eV']) * 1e3:.4f} meV")
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
