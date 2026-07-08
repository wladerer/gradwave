"""Generate QE EOS inputs with FFT grids pinned to gradwave's."""
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from eos_cases import CASES, RY, SCALES, geometry  # noqa: E402

MASS = {"Si": 28.085, "C": 12.011, "Ga": 69.723, "As": 74.922,
        "Mg": 24.305, "O": 15.999, "Al": 26.982, "Cu": 63.546}
PSE = "/home/wladerer/github/QSuite/tests/fixtures/qe/pseudos"
dims = json.loads((SP / "eos_dims.json").read_text())
# fixed per-case grid (largest over volumes) — must match eos_gw.py run mode
fixed = {c: [max(dims[c][str(s)][i] for s in SCALES) for i in range(3)]
         for c in CASES}
outdir = SP / "eos_qe"
outdir.mkdir(exist_ok=True)

for case, cfg in CASES.items():
    for i, s in enumerate(SCALES):
        cell, _, elems = geometry(case, s)
        species = sorted(set(elems))
        n1, n2, n3 = fixed[case]
        occ = ("  occupations = 'fixed'" if cfg["smearing"] == "none" else
               "  occupations = 'smearing'\n  smearing = 'gaussian'\n"
               f"  degauss = {cfg['width'] / RY:.10f}")
        nbnd = f"\n  nbnd = {cfg['nbands']}" if cfg["nbands"] else ""
        lines = [
            "&control", "  calculation = 'scf'", f"  prefix = '{case}{i}'",
            "  outdir = './tmp'", f"  pseudo_dir = '{PSE}'", "  verbosity = 'low'",
            "/", "&system", "  ibrav = 0", f"  nat = {len(elems)}",
            f"  ntyp = {len(species)}", f"  ecutwfc = {cfg['ecut_ry']:.1f}",
            f"  nr1 = {n1}", f"  nr2 = {n2}", f"  nr3 = {n3}", occ + nbnd,
            "/", "&electrons", "  conv_thr = 1.0d-10", "/",
            "ATOMIC_SPECIES",
        ]
        for sp in species:
            lines.append(f"  {sp}  {MASS[sp]}  {sp}_ONCV_PBE-1.2.upf")
        lines.append("CELL_PARAMETERS angstrom")
        for row in cell:
            lines.append("  " + "  ".join(f"{x:.12f}" for x in row))
        lines.append("ATOMIC_POSITIONS crystal")
        for el, fr in zip(elems, cfg["frac"], strict=True):
            lines.append(f"  {el}  " + "  ".join(f"{x:.10f}" for x in np.array(fr)))
        k = cfg["kmesh"]
        lines += ["K_POINTS automatic", f"  {k[0]} {k[1]} {k[2]} 0 0 0", ""]
        (outdir / f"{case}{i}.in").write_text("\n".join(lines))
print(f"wrote {len(CASES) * len(SCALES)} inputs to {outdir}")
