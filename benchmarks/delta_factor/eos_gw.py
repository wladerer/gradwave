"""gradwave side of the Δ-factor EOS.

Usage: eos_gw.py dims  -> writes eos_dims.json (natural FFT grid per case/volume)
       eos_gw.py run   -> writes eos_gw.json   (free energy per case/volume,
                          FIXED per-case grid = elementwise max over volumes)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(8)
sys.stdout.reconfigure(line_buffering=True)
SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from eos_cases import CASES, RY, SCALES, geometry  # noqa: E402

from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.pseudo.upf import parse_upf  # noqa: E402
from gradwave.scf.loop import scf, setup_system  # noqa: E402

PSE = Path(__file__).parents[2] / "tests/fixtures/qe/pseudos"
mode = sys.argv[1]


def build(case, scale, fft_shape=None):
    cell, pos, elems = geometry(case, scale)
    species = sorted(set(elems))
    upfs = [parse_upf(PSE / f"{s}_ONCV_PBE-1.2.upf") for s in species]
    soa = [species.index(s) for s in elems]
    cfg = CASES[case]
    return setup_system(cell, pos, soa, upfs, ecut=cfg["ecut_ry"] * RY,
                        kmesh=cfg["kmesh"], nbands=cfg["nbands"], use_symmetry=True,
                        fft_shape=fft_shape)


if mode == "dims":
    dims = {}
    for case in CASES:
        dims[case] = {}
        for s in SCALES:
            system = build(case, s)
            dims[case][str(s)] = list(system.grid.shape)
    (SP / "eos_dims.json").write_text(json.dumps(dims, indent=1))
    print("dims written")
else:
    # one FIXED grid per case (largest volume's) so E(V) is smooth — a
    # volume-varying minimal box shifts the absolute energy per grid change
    dims = json.loads((SP / "eos_dims.json").read_text())
    fixed = {c: [max(dims[c][str(s)][i] for s in SCALES) for i in range(3)]
             for c in CASES}
    out = {}
    for case in CASES:
        cfg = CASES[case]
        out[case] = {"natoms": len(cfg["elems"]), "E_eV": {},
                     "fft": fixed[case]}
        for s in SCALES:
            system = build(case, s, fft_shape=tuple(fixed[case]))
            t0 = time.time()
            res = scf(system, PBE(), smearing=cfg["smearing"], width=cfg["width"],
                      etol=1e-8, rhotol=1e-7, verbose=False, max_iter=150)
            assert res.converged, (case, s)
            e = float(res.energies.total)
            vol = float(np.abs(np.linalg.det(geometry(case, s)[0])))
            out[case]["E_eV"][str(s)] = e
            out[case].setdefault("V_A3", {})[str(s)] = vol
            print(f"{case:5s} v={s:.2f} E={e:+.8f} eV  ({time.time()-t0:.0f}s, "
                  f"it={res.n_iter})")
        (SP / "eos_gw.json").write_text(json.dumps(out, indent=1))
    print("done")
