"""Volumetric export: the valence charge density and ELF of diamond.

Runs a norm-conserving SCF on the 2-atom diamond cell, writes the total valence
density ρ(r) and the electron-localization function ELF(r) to VASP CHGCAR files,
and renders a 2x2x2 supercell isosurface of each with tinykit (POV-Ray). The
density isosurface traces the tetrahedral covalent bonds; the ELF isosurface
(Becke and Edgecombe, J. Chem. Phys. 92, 5397 (1990)) localizes on the bond
midpoints, the signature of covalent bonding.

gradwave stores ρ(r) on the FFT grid after any SCF; postscf.volumetric turns it
into the standard viewer formats. The CHGCAR writer stores ρ·Ω (VASP
convention), so ASE's VaspChargeDensity reader — and tinykit, which is built on
it — recover ρ(r) in e/Å³.

Fixtures: tests/fixtures/qe/pseudos/C_ONCV_PBE-1.2.upf (SG15 ONCV, PBE).
Runtime: ~30 s SCF on 8 CPU threads, plus a POV-Ray render per isosurface.

Run:
    uv run python examples/volumetric_density.py --outdir examples
    # skip the POV-Ray step (CHGCAR files only):
    uv run python examples/volumetric_density.py --no-render
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
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

torch.set_num_threads(8)
RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"


def diamond_scf():
    """Converged SCF for the 2-atom diamond primitive cell."""
    c = parse_upf(f"{PSE}/C_ONCV_PBE-1.2.upf")
    a = 3.567  # Å, experimental lattice constant
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
    system = setup_system(cell, pos, [0, 0], [c, c], ecut=50 * RY,
                          kmesh=(4, 4, 4), nbands=8)
    return scf(system, PBE(), etol=1e-8, rhotol=1e-7, verbose=False)


def render(chgcar: Path, png: Path, isovalue: float, color: str) -> bool:
    """Render a 2x2x2 supercell isosurface of a CHGCAR with tinykit."""
    if shutil.which("tk") is None:
        print(f"  [skip render] tk not on PATH; wrote {chgcar.name}")
        return False
    cmd = [
        "tk", "viz", str(chgcar), "-o", str(png),
        "--supercell", "2", "2", "2",
        "--isovalue", str(isovalue), "--iso-color", color,
        "--rotation", "-70", "10", "0", "--width", "900", "--height", "700",
    ]
    subprocess.run(cmd, check=True)
    print(f"  wrote {png.name}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("SCF: diamond, 50 Ry, 4x4x4 k-mesh ...")
    res = diamond_scf()

    rho = V.density(res)
    elf = V.elf(res)
    print(f"  rho: {rho.min():.3f}..{rho.max():.3f} e/Å³, "
          f"integral {rho.sum() * res.system.grid.volume / rho.size:.3f} e")
    print(f"  ELF: {elf.min():.3f}..{elf.max():.3f}")

    rho_chg = V.write_density(res, outdir / "diamond_CHGCAR")
    elf_chg = V.write_elf(res, outdir / "diamond_ELF_CHGCAR")
    print(f"  wrote {Path(rho_chg).name}, {Path(elf_chg).name}")

    if not args.no_render:
        render(Path(rho_chg), outdir / "diamond_density.png",
               isovalue=0.55, color="0.95,0.72,0.20")
        render(Path(elf_chg), outdir / "diamond_elf.png",
               isovalue=0.85, color="0.35,0.65,0.85")


if __name__ == "__main__":
    main()
