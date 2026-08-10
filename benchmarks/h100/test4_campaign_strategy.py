"""H100 Test 4 — campaign execution strategy on a GPU node (resolves the plan's Test 3).

For a hub-and-spoke forward campaign (here an EOS: N independent volume SCFs), which
wins on a datacenter node — running the spokes SERIALLY on the GPU, or SeedPool
CPU-process-parallel across the box's many cores? The two are mutually exclusive for
the same spokes; this picks the campaign strategy on H100-class hardware.

  Run A (GPU serial)   : device=cuda, eos.n_workers=1  → 1 GPU chews N volumes in turn
  Run B (CPU parallel) : device=cpu,  eos.n_workers=K  → K worker processes (SeedPool),
                         each pinned to ~8 threads (the SCF sweet spot)

The sweep found the H100 ~50x per LARGE-cell SCF, so GPU-serial is expected to beat
192-core CPU-parallel once each spoke is big enough — this measures the crossover.

REQUIRES the SeedPool feature: run on branch feat/seedpool-parallel-spokes (or main
once #276 has merged). Forward-only (E(V)); n_workers>1 is a pure forward lever.

Run on the H100 box:
    uv run python benchmarks/h100/test4_campaign_strategy.py --nrep 2 --n-workers 6
"""
import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]
_PSEUDO_DIR = _ROOT / "tests/fixtures/qe/pseudos"
SCALES = [0.96, 0.98, 1.00, 1.02, 1.04, 1.06]  # reference nearest 1.0 -> (len-1) spokes


def build_si(nrep: int):
    from ase import Atoms

    a = 5.43
    cell = (a / 2) * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    at = Atoms("Si2", scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
               cell=cell, pbc=True).repeat((nrep, nrep, nrep))
    return np.asarray(at.cell.array), at.get_positions(), at.get_chemical_symbols()


def write_yaml(cell, pos, species, *, ecut_ry, kmesh, device, n_workers, outdir):
    body = f"""
structure:
  cell: {cell.tolist()}
  positions: {{cart: {pos.tolist()}}}
  species: {list(species)}
pseudopotentials:
  dir: {_PSEUDO_DIR}
  map: {{Si: Si_ONCV_PBE-1.2.upf}}
ecut: {ecut_ry * RY}
xc: pbe
kpoints: {{mesh: [{kmesh}, {kmesh}, {kmesh}]}}
smearing: {{type: none, width: 0.0}}
symmetry: true
task: eos
eos:
  scales: {SCALES}
  energy: total_energy
  n_workers: {n_workers}
device: {device}
distributed: false
output: {{dir: {outdir}, checkpoint: false}}
"""
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(body)
    f.close()
    return f.name


def run_campaign(path):
    from gradwave.api import run_eos
    from gradwave.inputs import load_input

    inp = load_input(path)
    dev = inp.device
    if dev != "cpu":
        torch.cuda.synchronize()
    t = time.perf_counter()
    summ = run_eos(inp, verbose=False)
    if dev != "cpu":
        torch.cuda.synchronize()
    return time.perf_counter() - t, summ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nrep", type=int, default=2, help="Si supercell tiling; atoms = 2*nrep**3")
    ap.add_argument("--ecut", type=float, default=20.0, help="Ry")
    ap.add_argument("--kmesh", type=int, default=2)
    ap.add_argument("--n-workers", type=int, default=len(SCALES) - 1,
                    help="SeedPool worker processes for run B (CPU parallel)")
    ap.add_argument("--threads-per-worker", type=int, default=8)
    args = ap.parse_args()

    nat = 2 * args.nrep**3
    n_spokes = len(SCALES) - 1
    eff_workers = min(args.n_workers, n_spokes)
    has_cuda = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if has_cuda else "no-CUDA"
    print(f"# H100 Test 4 — campaign strategy | CUDA={has_cuda} ({dev_name}) | "
          f"Si {nat} atoms | {len(SCALES)} volumes ({n_spokes} spokes) | ecut={args.ecut} Ry")

    cell, pos, species = build_si(args.nrep)
    tmp = tempfile.mkdtemp()

    # Run A — GPU serial
    if has_cuda:
        pa = write_yaml(cell, pos, species, ecut_ry=args.ecut, kmesh=args.kmesh,
                        device="cuda", n_workers=1, outdir=f"{tmp}/a")
        wall_a, sa = run_campaign(pa)
        print(f"A  GPU-serial      : wall {wall_a:8.2f}s  b0={sa['b0_GPa']:.1f} GPa  "
              f"conv={sa['all_converged']}")
    else:
        wall_a, sa = float("nan"), None
        print("A  GPU-serial      : SKIPPED (no CUDA)")

    # Run B — CPU process-parallel (SeedPool); pin parent threads so each worker ~= sweet spot
    torch.set_num_threads(min(eff_workers * args.threads_per_worker, 192))
    pb = write_yaml(cell, pos, species, ecut_ry=args.ecut, kmesh=args.kmesh,
                    device="cpu", n_workers=args.n_workers, outdir=f"{tmp}/b")
    wall_b, sb = run_campaign(pb)
    print(f"B  CPU-parallel x{eff_workers:<3}: wall {wall_b:8.2f}s  b0={sb['b0_GPa']:.1f} GPa  "
          f"conv={sb['all_converged']}  ({args.threads_per_worker} threads/worker)")

    if has_cuda:
        print(f"\n# winner: {'GPU-serial' if wall_a < wall_b else 'CPU-parallel'} "
              f"({max(wall_a, wall_b) / min(wall_a, wall_b):.2f}x faster)")
        if sa and sb:
            print(f"# b0 agree (GPU vs CPU): {abs(sa['b0_GPa'] - sb['b0_GPa']):.3f} GPa "
                  f"(warm-start-from-reference, matches to SCF tol not bit-for-bit)")
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
