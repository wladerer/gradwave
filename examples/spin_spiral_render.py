"""Render the bcc-Fe spin spiral as magnetic-moment arrows (tinykit).

A visual aid for the spin-spiral dispersion in examples/fe_spin_spiral.py. That
run holds each Fe moment at its full magnitude (~2.22 μB from the converged
non-collinear SCF, examples/fe_spin_spiral.json) while rotating its direction;
the energy rises monotonically from the collinear ground state to the
antiparallel arrangement. Here we lay a frozen spiral of that moment on a bcc-Fe
supercell and draw the moments as arrows: the direction advances by a fixed angle
per layer along the stacking axis, which is the θ the dispersion scans.

The moment magnitude and the fact that a strong ferromagnet holds it while the
direction turns are the physical results; the linear stack is a schematic chosen
so the rotation reads cleanly (in-plane moments render as green arrows).

Runtime: a POV-Ray render, no SCF. Needs tinykit (`tk`) on PATH.

Run:
    uv run python examples/spin_spiral_render.py --outdir examples
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np

MOMENT_MUB = 2.22        # per-atom moment from fe_spin_spiral.json (vector mode)
TWIST_DEG = 50.0         # spiral advance per bcc layer along the stacking axis
A = 2.87                 # bcc Fe lattice constant [Å]
N_LAYERS = 8


def build_spiral():
    """Positions (Å) and in-plane moment vectors for a frozen bcc-Fe spiral."""
    positions, moments = [], []
    for n in range(N_LAYERS):
        phi = np.deg2rad(TWIST_DEG * n)
        m = MOMENT_MUB * np.array([np.cos(phi), np.sin(phi), 0.0])
        # two bcc sublattice sites per layer, stacked along z
        positions.append([0.0, 0.0, n * A])
        positions.append([0.5 * A, 0.5 * A, (n + 0.5) * A])
        # body-centre site carries the same layer phase (rigid spiral)
        moments.append(m)
        moments.append(MOMENT_MUB * np.array(
            [np.cos(phi + np.deg2rad(TWIST_DEG / 2)),
             np.sin(phi + np.deg2rad(TWIST_DEG / 2)), 0.0]))
    return np.array(positions), np.array(moments)


def write_poscar(path, positions):
    cell = np.diag([A, A, (N_LAYERS + 0.5) * A])
    frac = positions @ np.linalg.inv(cell)
    lines = ["Fe spin spiral (schematic)", "1.0"]
    for row in cell:
        lines.append(f"  {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
    lines += ["Fe", str(len(positions)), "Direct"]
    for f in frac:
        lines.append(f"  {f[0]:.6f} {f[1]:.6f} {f[2]:.6f}")
    Path(path).write_text("\n".join(lines) + "\n")


def _block(component, totals):
    head = f" magnetization ({component})\n\n# of ion     s       p       d       tot\n"
    head += "------------------------------------------\n"
    body = "".join(f"    {i + 1}     0.00   0.00   {t:6.3f}   {t:6.3f}\n"
                   for i, t in enumerate(totals))
    tot = sum(totals)
    foot = "--------------------------------------------------\n"
    foot += f"tot       0.00   0.00   {tot:6.3f}   {tot:6.3f}\n\n"
    return head + body + foot


def write_outcar(path, moments):
    """Non-collinear OUTCAR with magnetization (x/y/z) per-ion tables."""
    text = "".join(_block(c, moments[:, ax]) for ax, c in enumerate("xyz"))
    Path(path).write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # tinykit resolves --moments to a file named exactly OUTCAR, so use the
    # canonical VASP filenames in a dedicated directory.
    cell_dir = outdir / "fe_spiral"
    cell_dir.mkdir(parents=True, exist_ok=True)
    positions, moments = build_spiral()
    poscar = cell_dir / "POSCAR"
    outcar = cell_dir / "OUTCAR"
    write_poscar(poscar, positions)
    write_outcar(outcar, moments)
    print(f"  wrote {poscar}, {outcar} ({len(positions)} Fe atoms, "
          f"{MOMENT_MUB} μB, {TWIST_DEG}°/layer)")

    if shutil.which("tk") is None:
        print("  [skip render] tk not on PATH")
        return
    png = outdir / "fe_spin_spiral_moments.png"
    try:
        subprocess.run([
            "tk", "viz", str(poscar), "--moments", str(outcar),
            "-o", str(png), "--rotation", "-80", "0", "-15",
            "--width", "500", "--height", "820", "--radius-scale", "0.5",
        ], check=True)
        print(f"  wrote {png.name}")
    except subprocess.CalledProcessError:
        print("  [skip render] tk failed; needs a tinykit with the OUTCAR "
              "moment-vector parser (magviz.read_outcar_moment_vectors)")


if __name__ == "__main__":
    main()
