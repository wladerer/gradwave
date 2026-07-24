"""Bader charge analysis: ionic charge transfer in rocksalt NaCl.

Runs an SCF on the 2-atom rocksalt cell and partitions ρ(r) into Bader (QTAIM)
basins with postscf.bader, which implements the on-grid steepest-ascent scheme
of Henkelman, Arnaldsson and Jonsson (Comput. Mater. Sci. 36, 354 (2006)). NaCl
is the textbook case: the density maxima sit on the Na and Cl nuclei, so the
zero-flux basins map cleanly onto atoms and the net charges report the ionic
transfer Na(+) / Cl(-). (For a homopolar covalent crystal such as Si the valence
pseudo-density peaks in the bonds instead, and per-atom Bader charges are not
meaningful without the augmented PAW density.)

The script prints the per-atom charge table, writes the density to a CHGCAR, and
renders a supercell isosurface with tinykit; add_core=True folds the partial-core
density back onto the grid to sharpen the nuclear maxima.

Fixtures: tests/fixtures/qe/pseudos/{Na,Cl}_ONCV_PBE_sr.upf (SG15 ONCV, PBE).
Runtime: ~1-2 min SCF on 8 CPU threads.

Run:
    uv run python examples/bader_nacl.py --outdir examples
    uv run python examples/bader_nacl.py --no-render     # CHGCAR + table only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf import volumetric as V
from gradwave.postscf.bader import bader
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

torch.set_num_threads(8)
RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"


def nacl_scf():
    na = parse_upf(f"{PSE}/Na_ONCV_PBE_sr.upf")
    cl = parse_upf(f"{PSE}/Cl_ONCV_PBE_sr.upf")
    a = 5.64  # Å, rocksalt lattice constant
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) @ (a * np.eye(3))
    system = setup_system(cell, pos, [0, 1], [na, cl], ecut=60 * RY,
                          kmesh=(4, 4, 4), nbands=16)
    return scf(system, PBE(), smearing="gaussian", width=0.01,
               etol=1e-8, rhotol=1e-7, verbose=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("SCF: NaCl rocksalt, 60 Ry, 4x4x4 k-mesh ...")
    res = nacl_scf()

    out = bader(res, add_core=True)
    labels = ["Na", "Cl"]
    print("\n  Bader charges (add_core=True):")
    print("    atom   Z_val   electrons   charge q [e]   volume [Å³]")
    for i, sym in enumerate(labels):
        print(f"    {sym:>4}   {out.valence[i]:5.1f}   {out.electrons[i]:9.3f}   "
              f"{out.charges[i]:+11.3f}   {out.volumes[i]:9.2f}")
    print(f"    total electrons ∫ρ dr = {out.total_electrons:.3f} "
          f"(Σ valence = {out.valence.sum():.1f}); "
          f"{out.n_attractors} attractors, {len(out.nonnuclear)} non-nuclear")

    chg = V.write_density(res, outdir / "nacl_CHGCAR")
    print(f"  wrote {Path(chg).name}")

    if not args.no_render and shutil.which("tk") is not None:
        png = outdir / "nacl_density.png"
        subprocess.run([
            "tk", "viz", str(chg), "-o", str(png),
            "--supercell", "2", "2", "2", "--isovalue", "0.18",
            "--iso-color", "0.55,0.80,0.75",
            "--rotation", "-65", "15", "0", "--width", "900", "--height", "700",
        ], check=True)
        print(f"  wrote {png.name}")


if __name__ == "__main__":
    main()
