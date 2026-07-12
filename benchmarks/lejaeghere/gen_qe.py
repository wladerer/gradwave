"""QE cross-check inputs for the Lejaeghere subset, FFT grids pinned to
gradwave's fixed per-case grid (run run_gw.py dims first)."""
import json
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from cases import CASES, RY, SCALES, geometry  # noqa: E402

MASS = {"Si": 28.085, "Ge": 72.630, "Al": 26.982, "Cu": 63.546, "Ni": 58.693}
PSE = "/home/wladerer/github/QSuite/tests/fixtures/qe/pseudos"
dims = json.loads((SP / "results" / "eos_dims.json").read_text())
outdir = SP / "qe_inputs"
outdir.mkdir(exist_ok=True)

n = 0
for case, cfg in CASES.items():
    if case not in dims:
        continue
    fixed = [max(dims[case][str(s)][i] for s in SCALES) for i in range(3)]
    for i, s in enumerate(SCALES):
        cell, _, elems = geometry(case, s)
        el = elems[0]
        occ = ("  occupations = 'fixed'" if cfg["smearing"] == "none" else
               "  occupations = 'smearing'\n  smearing = 'gaussian'\n"
               f"  degauss = {cfg['width'] / RY:.10f}")
        if cfg["nbands"]:
            occ += f"\n  nbnd = {cfg['nbands']}"
        if cfg["nspin"] == 2:
            occ += (f"\n  nspin = 2\n  starting_magnetization(1) = "
                    f"{cfg['start_mag'][0]:.2f}")
        lines = [
            "&control", "  calculation = 'scf'", f"  prefix = '{case}{i}'",
            "  outdir = './tmp'", f"  pseudo_dir = '{PSE}'",
            "  verbosity = 'low'", "/", "&system", "  ibrav = 0",
            f"  nat = {len(elems)}", "  ntyp = 1",
            f"  ecutwfc = {cfg['ecut_ry']:.1f}",
            f"  ecutrho = {cfg['ecutrho_ry']:.1f}",
            f"  nr1 = {fixed[0]}", f"  nr2 = {fixed[1]}",
            f"  nr3 = {fixed[2]}", occ, "/", "&electrons",
            "  conv_thr = 1.0d-10", "/", "ATOMIC_SPECIES",
            f"  {el}  {MASS[el]}  {cfg['pseudo']}",
            "CELL_PARAMETERS angstrom",
        ]
        for row in cell:
            lines.append("  " + "  ".join(f"{x:.12f}" for x in row))
        lines.append("ATOMIC_POSITIONS crystal")
        for e_, fr in zip(elems, cfg["frac"], strict=True):
            lines.append(f"  {e_}  " + "  ".join(f"{x:.10f}"
                                                 for x in np.array(fr, dtype=float)))
        k = cfg["kmesh"]
        lines += ["K_POINTS automatic", f"  {k[0]} {k[1]} {k[2]} 0 0 0", ""]
        (outdir / f"{case}{i}.in").write_text("\n".join(lines))
        n += 1
print(f"wrote {n} inputs to {outdir}")
