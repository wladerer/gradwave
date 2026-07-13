"""gradwave PAW side of the Lejaeghere Δ-benchmark.

Usage: run_gw.py dims [cases...]  -> results/eos_dims.json (natural FFT grid
                                     per case/volume)
       run_gw.py run  [cases...]  -> results/eos_gw.json (free energy per
                                     case/volume, FIXED per-case grid =
                                     elementwise max over volumes)

Results merge into the existing JSON, so cases can run on different
machines (insulators on CPU, metals on the GPU box). Set GW_DEVICE=cuda
to run on GPU.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(8)
sys.stdout.reconfigure(line_buffering=True)
SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from cases import CASES, RY, SCALES, geometry  # noqa: E402

from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.core.xc.spin import SpinPBE  # noqa: E402
from gradwave.pseudo.upf_paw import parse_upf_paw  # noqa: E402
from gradwave.scf.uspp import scf_uspp, setup_uspp  # noqa: E402

PSE = Path(__file__).parents[2] / "tests/fixtures/qe/pseudos"
RES = SP / "results"
RES.mkdir(exist_ok=True)
DEV = os.environ.get("GW_DEVICE", "cpu")

mode = sys.argv[1]
sel = sys.argv[2:] or list(CASES)
_paw_cache = {}


def build(case, scale, fft_shape=None):
    cfg = CASES[case]
    if case not in _paw_cache:
        _paw_cache[case] = parse_upf_paw(PSE / cfg["pseudo"])
    cell, pos, elems = geometry(case, scale)
    return setup_uspp(cell, pos, [0] * len(elems), [_paw_cache[case]],
                      ecut=cfg["ecut_ry"] * RY, ecutrho=cfg["ecutrho_ry"] * RY,
                      kmesh=cfg["kmesh"], nbands=cfg["nbands"],
                      use_symmetry=True, fft_shape=fft_shape)


def merge_write(path, key, value):
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = value
    path.write_text(json.dumps(data, indent=1))


if mode == "dims":
    from gradwave.grids import build_fft_grid

    for case in sel:
        cfg = CASES[case]
        dims = {}
        for s in SCALES:
            grid = build_fft_grid(geometry(case, s)[0],
                                  cfg["ecutrho_ry"] * RY / 4.0,
                                  equal_dims=True)
            dims[str(s)] = list(grid.shape)
        merge_write(RES / "eos_dims.json", case, dims)
        print(f"{case}: {dims}")
    print("dims written")
elif mode == "run":
    dims = json.loads((RES / "eos_dims.json").read_text())
    for case in sel:
        cfg = CASES[case]
        # one FIXED grid per case (elementwise max over volumes) so E(V)
        # is smooth — a volume-varying minimal box steps the energy
        fixed = [max(dims[case][str(s)][i] for s in SCALES) for i in range(3)]
        xc = SpinPBE() if cfg["nspin"] == 2 else PBE()
        out = {"natoms": len(cfg["elems"]), "fft": fixed,
               "E_eV": {}, "V_A3": {}, "mag": {}}
        # resume: keep volumes already scanned on the SAME fixed grid
        gw_path = RES / "eos_gw.json"
        prev = (json.loads(gw_path.read_text()).get(case)
                if gw_path.exists() else None)
        if prev and prev.get("fft") == fixed:
            for key in ("E_eV", "V_A3", "mag"):
                out[key].update(prev.get(key, {}))
        prev_res = None  # warm-start chain: each volume from the last one
        for s in SCALES:
            if str(s) in out["E_eV"]:
                print(f"{case:4s} v={s:.2f} done (resume)")
                prev_res = None  # resumed points leave no state to chain
                continue
            system = build(case, s, fft_shape=tuple(fixed))
            if DEV != "cpu":
                system = system.to(torch.device(DEV))
            t0 = time.time()
            r = scf_uspp(system, xc, nspin=cfg["nspin"],
                         start_mag=cfg["start_mag"],
                         smearing=cfg["smearing"], width=cfg["width"],
                         mixing_alpha=cfg.get("mixing_alpha", 0.7),
                         criterion=cfg.get("criterion", "drho"),
                         etol=1e-9, rhotol=cfg.get("rhotol", 1e-8),
                         verbose=False, max_iter=150, start_from=prev_res)
            prev_res = r
            assert r["converged"], (case, s)
            if cfg["nspin"] == 2:
                assert abs(float(r["mag_total"])) > 0.1, \
                    (case, s, "moment collapsed to the NM branch")
            e = float(r["energies"].free_energy)
            out["E_eV"][str(s)] = e
            out["V_A3"][str(s)] = float(np.abs(np.linalg.det(
                geometry(case, s)[0])))
            if cfg["nspin"] == 2:
                out["mag"][str(s)] = float(r["mag_total"])
            merge_write(RES / "eos_gw.json", case, out)
            m = (f" m={out['mag'][str(s)]:+.3f}" if cfg["nspin"] == 2 else "")
            print(f"{case:4s} v={s:.2f} F={e:+.8f} eV{m}  "
                  f"({time.time() - t0:.0f}s)")
    print("done")
else:
    sys.exit(f"unknown mode {mode!r} (use: dims | run)")
