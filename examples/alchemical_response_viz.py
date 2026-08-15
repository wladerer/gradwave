"""Visualize the alchemical composition density response dρ/dλ.

The composition DFPT (scf.alchemical.alchemical_density_response) returns the
self-consistent first-order density response dρ/dλ on the real-space grid — the
charge that rearranges when the composition is nudged. Rendering its signed
isosurface (io.viz.field_isosurface) shows the screening cloud directly: the
positive and negative lobes are where electrons flow in and out as the halide
sublattice of CsPbI3 is transmuted toward Cl. This is precisely the response the
frozen (first-order APDFT) picture omits, and it is what carries the band-gap
gradient (see examples/perovskite_alchemical_bandgap.py).

    uv run --extra viz python examples/alchemical_response_viz.py [out_stem]

Writes ``<out_stem>.html`` (self-contained, interactive) and, if kaleido + a
browser are available, ``<out_stem>.png``.
"""

import sys
from pathlib import Path

import numpy as np

from gradwave.core.xc.pbe import PBE
from gradwave.io import viz
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.alchemical import alchemical_density_response, setup_alchemical_substitution
from gradwave.scf.loop import scf

RY = 13.605693122994
ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "qe" / "pseudos"
DG = ROOT / "benchmarks" / "delta_gauge" / "pseudos"


def main(out_stem="alchemical_response"):
    cs = parse_upf(FIX / "Cs_ONCV_PBE_sr.upf")
    pb = parse_upf(DG / "Pb.upf")
    iod = parse_upf(FIX / "I_ONCV_PBE_sr.upf")
    cl = parse_upf(FIX / "Cl_ONCV_PBE_sr.upf")

    a = 6.29
    cell = a * np.eye(3)
    frac = np.array([
        [0.0, 0.0, 0.0],   # Cs
        [0.5, 0.5, 0.5],   # Pb
        [0.5, 0.5, 0.0],   # I
        [0.5, 0.0, 0.5],   # I
        [0.0, 0.5, 0.5],   # I
    ])
    pos = frac @ cell
    symbols = ["Cs", "Pb", "I", "I", "I"]

    res = scf(
        setup_alchemical_substitution(
            cell, pos, [cs, pb, iod], [0, 1, 2, 2, 2], {2: cl, 3: cl, 4: cl},
            0.5, ecut=30 * RY, kmesh=(1, 1, 1), use_symmetry=False),
        PBE(), smearing="none", etol=1e-9, rhotol=1e-9, max_iter=300, verbose=False)
    assert res.converged

    # composition density response dρ/dλ (all three X sites -> Cl together)
    drho, _, _ = alchemical_density_response(res, PBE())
    drho = drho.detach().cpu().numpy()
    print(f"dρ/dλ: max|δρ| = {np.abs(drho).max():.4f} e/Å³   "
          f"∫δρ dV = {drho.sum() * res.system.grid.volume / drho.size:.2e} e (≈0)")

    fig = viz.field_isosurface(
        drho, res.system.grid.cell, res.system.positions.numpy(), symbols,
        iso_frac=0.25,
        title="CsPbI₃ → CsPbCl₃ composition response dρ/dλ (red: in, blue: out)")

    html = Path(f"{out_stem}.html")
    fig.write_html(html, include_plotlyjs=True, full_html=True)
    print(f"wrote {html} ({html.stat().st_size // 1024} KiB, self-contained)")
    try:
        fig.write_image(f"{out_stem}.png", width=900, height=800, scale=2)
        print(f"wrote {out_stem}.png")
    except Exception as e:  # kaleido/browser missing — HTML is the primary output
        print(f"(png skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "alchemical_response")
