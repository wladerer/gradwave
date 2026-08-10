"""Graphite vdW phonon dispersion — the flagship demo (DisplacementStar + D3 forces).

AB-stacked graphite (4-atom hexagonal cell, P6/mmm) phonon dispersion along
Γ-M-K-Γ-A, computed TWICE: with the Grimme D3 dispersion forces folded into the
finite-displacement force constants (this branch's new wiring) and WITHOUT. The
in-plane covalent branches (E2g ~1580 cm⁻¹) are identical either way; the point is
the low-frequency INTERLAYER branches (rigid-layer shear ~40 cm⁻¹, ZO'/ZA along
Γ-A) — bare PBE under-binds the layers and leaves them too soft, and D3 stiffens
them. Plotting D3-off vs D3-on makes the van der Waals physics visible.

Also exercises DisplacementStar (#274): graphite's point group collapses the unique
finite displacements below the brute-force 3·N_prim (printed by run_phonons). On the
H100 the reduced displacement spokes run on the GPU serially.

Saves both dispersion blocks to JSON (plot locally with analysis.plot_phonons or the
companion plot step). Run on the H100 box (this branch: feat/graphite-vdw-phonon-demo):
    uv run python benchmarks/h100/graphite_phonon_demo.py --supercell 3,3,2
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

RY = 13.605693122994
_ROOT = Path(__file__).resolve().parents[2]
_PSEUDO_DIR = _ROOT / "tests/fixtures/qe/pseudos"


def build_graphite():
    """AB-stacked graphite, 4-atom hexagonal cell (a=2.46 Å, c=6.70 Å)."""
    from ase import Atoms

    a, c = 2.46, 6.70
    cell = [[a, 0.0, 0.0],
            [-a / 2, a * np.sqrt(3) / 2, 0.0],
            [0.0, 0.0, c]]
    frac = [[0.0, 0.0, 0.0], [1 / 3, 2 / 3, 0.0],       # A layer
            [0.0, 0.0, 0.5], [2 / 3, 1 / 3, 0.5]]        # B layer
    return Atoms("C4", scaled_positions=frac, cell=cell, pbc=True)


def write_yaml(at, *, ecut_ry, kmesh, supercell, dispersion, device, outdir):
    disp = ("dispersion:\n  enabled: true\n  method: d3\n  functional: pbe\n"
            if dispersion else "")
    body = f"""
structure:
  cell: {np.asarray(at.cell.array).tolist()}
  positions: {{cart: {at.get_positions().tolist()}}}
  species: {at.get_chemical_symbols()}
pseudopotentials:
  dir: {_PSEUDO_DIR}
  map: {{C: C_ONCV_PBE-1.2.upf}}
ecut: {ecut_ry * RY}
xc: pbe
kpoints: {{mesh: {list(kmesh)}}}
smearing: {{type: gaussian, width: 0.1}}
symmetry: false
task: phonons
device: {device}
scf: {{etol: 1.0e-8, rhotol: 1.0e-7, max_iter: 200}}
phonons:
  supercell: {list(supercell)}
  path: GMKGA
  npoints: 200
  displacement: 0.01
  use_displacement_symmetry: true
  dos_mesh: [0, 0, 0]
{disp}output: {{dir: {outdir}}}
"""
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(body)
    f.close()
    return f.name


def run(path):
    import torch

    from gradwave.api import run_phonons
    from gradwave.inputs import load_input

    inp = load_input(path)
    dev = inp.device
    if dev != "cpu":
        torch.cuda.synchronize()
    t = time.perf_counter()
    block = run_phonons(inp, verbose=True)
    if dev != "cpu":
        torch.cuda.synchronize()
    block["_wall_s"] = time.perf_counter() - t
    return block


def gamma_freqs(block):
    """Sorted phonon frequencies at Γ (first q-point on the path) [cm⁻¹]."""
    return np.sort(np.asarray(block["frequencies_cm1"])[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--supercell", default="3,3,2", help="phonon supercell, e.g. 3,3,2")
    ap.add_argument("--ecut", type=float, default=45.0, help="Ry")
    ap.add_argument("--kmesh", default="12,12,4", help="primitive k-mesh (folded by supercell)")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--out", default="graphite_phonons", help="JSON basename for the two blocks")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    supercell = [int(x) for x in args.supercell.split(",")]
    kmesh = [int(x) for x in args.kmesh.split(",")]
    at = build_graphite()
    print(f"# Graphite vdW phonons | device={device} ({dev_name}) | "
          f"supercell {supercell} ({4 * int(np.prod(supercell))} atoms) | ecut={args.ecut} Ry")

    tmp = tempfile.mkdtemp()
    results = {}
    for tag, disp in [("d3off", False), ("d3on", True)]:
        print(f"\n=== run {tag} (dispersion={'D3' if disp else 'none'}) ===", flush=True)
        block = run(write_yaml(at, ecut_ry=args.ecut, kmesh=kmesh, supercell=supercell,
                               dispersion=disp, device=device, outdir=f"{tmp}/{tag}"))
        results[tag] = block
        out = f"{args.out}_{tag}.json"
        Path(out).write_text(json.dumps(block))
        print(f"  saved {out}  (wall {block['_wall_s']:.1f}s, "
              f"min freq {block['min_frequency_cm1']:.1f} cm⁻¹)")

    # The money shot: Γ frequencies, D3-off vs D3-on. Low-freq interlayer modes
    # (the ~40 cm⁻¹ rigid-layer shear + ZO') should STIFFEN under D3; the ~1580 cm⁻¹
    # in-plane E2g should barely move.
    goff, gon = gamma_freqs(results["d3off"]), gamma_freqs(results["d3on"])
    print("\n# Γ-point frequencies [cm⁻¹] (sorted): D3-off -> D3-on")
    for i, (fo, fn) in enumerate(zip(goff, gon, strict=True)):
        band = "acoustic" if i < 3 else ("interlayer" if fo < 300 else "in-plane")
        print(f"  mode {i:2d}: {fo:8.1f} -> {fn:8.1f}   ({fn - fo:+6.1f})  {band}")
    lo_off = goff[3:][goff[3:] < 300]
    lo_on = gon[3:][goff[3:] < 300]
    if lo_off.size:
        print(f"\n# interlayer optical modes: mean {lo_off.mean():.1f} -> {lo_on.mean():.1f} cm⁻¹ "
              f"({lo_on.mean() - lo_off.mean():+.1f}, D3 stiffening)")
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
