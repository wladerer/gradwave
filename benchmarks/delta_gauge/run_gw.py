"""gradwave side of the periodic-table Δ-gauge (norm-conserving, PBE).

Usage:
  run_gw.py dims [els...]   -> results/dims_<el>.json  (natural grid per volume)
  run_gw.py run  [els...]   -> results/eos_<el>.json   (free energy per volume,
                               fixed per-element grid = elementwise max)

Each element writes its OWN json, so elements parallelize across processes with
no shared-file race (wisdom.md: "Independent SCF points parallelize"). Volumes
within an element run serially and warm-start from the previous volume, which is
the cheap, branch-stable EOS chain. Device via GW_DEVICE (cpu|cuda); threads via
GW_THREADS. Resumes: a volume already present on the same fixed grid is skipped.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from cases import CASES, RY, geometry, nbands  # noqa: E402

torch.set_num_threads(int(os.environ.get("GW_THREADS", "8")))
sys.stdout.reconfigure(line_buffering=True)

from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.core.xc.spin import SpinPBE  # noqa: E402
from gradwave.pseudo.upf import parse_upf  # noqa: E402
from gradwave.scf.loop import scf, setup_system  # noqa: E402

PSE = SP / "pseudos"
RES = SP / "results"
RES.mkdir(exist_ok=True)
DEV = os.environ.get("GW_DEVICE", "cpu")
SCALES = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]
_upf_cache = {}


def build(elem, scale, fft_shape=None):
    c = CASES[elem]
    if elem not in _upf_cache:
        _upf_cache[elem] = parse_upf(PSE / f"{elem}.upf")
    cell, pos, elems = geometry(elem, scale)
    return setup_system(cell, pos, [0] * len(elems), [_upf_cache[elem]],
                        ecut=c["ecut"] * RY, kmesh=c["kmesh"],
                        nbands=nbands(elem), use_symmetry=True,
                        fft_shape=fft_shape)


def run_dims(els):
    for elem in els:
        dims = {}
        for s in SCALES:
            dims[str(s)] = list(build(elem, s).grid.shape)
        (RES / f"dims_{elem}.json").write_text(json.dumps(dims, indent=1))
        print(f"{elem}: {dims['1.0']} (grids over volumes)")


def run_eos(els):
    for elem in els:
        c = CASES[elem]
        dpath = RES / f"dims_{elem}.json"
        if not dpath.exists():
            run_dims([elem])
        dims = json.loads(dpath.read_text())
        fixed = [max(dims[str(s)][i] for s in SCALES) for i in range(3)]
        xc = SpinPBE() if c["nspin"] == 2 else PBE()
        opath = RES / f"eos_{elem}.json"
        out = (json.loads(opath.read_text()) if opath.exists() else None)
        if not out or out.get("fft") != fixed:
            out = {"struct": c["struct"], "nspin": c["nspin"], "fft": fixed,
                   "dfact_pdojo": c["dfact"], "E_eV": {}, "V_A3": {},
                   "mag": {}, "sec": {}, "it": {}, "gb": {}}
        prev = None
        for s in SCALES:
            if str(s) in out["E_eV"]:
                print(f"{elem} v={s:.2f} done (resume)")
                prev = None
                continue
            sysd = build(elem, s, fft_shape=tuple(fixed))
            if DEV != "cpu":
                sysd = sysd.to(torch.device(DEV))
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            r = scf(sysd, xc, nspin=c["nspin"], start_mag=c["start_mag"],
                    smearing=c["smear"], width=c["width"] * RY,
                    etol=1e-8, rhotol=1e-7, max_iter=200, verbose=False,
                    start_from=prev)
            dt = time.time() - t0
            gb = (torch.cuda.max_memory_allocated() / 1e9
                  if DEV == "cuda" else 0.0)
            if not r.converged:
                print(f"{elem} v={s:.2f} NOT CONVERGED it={r.n_iter} "
                      f"({dt:.0f}s) -- recording anyway")
            prev = r
            out["E_eV"][str(s)] = float(r.energies.total)
            out["V_A3"][str(s)] = float(np.abs(np.linalg.det(
                geometry(elem, s)[0])))
            out["sec"][str(s)] = round(dt, 1)
            out["it"][str(s)] = int(r.n_iter)
            out["gb"][str(s)] = round(gb, 2)
            mtag = ""
            if c["nspin"] == 2:
                m = float(getattr(r, "mag_total", getattr(r, "magnetization",
                                                          float("nan"))))
                out["mag"][str(s)] = m
                mtag = f" m={m:+.3f}"
            opath.write_text(json.dumps(out, indent=1))
            print(f"{elem} v={s:.2f} E={out['E_eV'][str(s)]:+.6f} eV{mtag} "
                  f"it={r.n_iter} ({dt:.0f}s, {gb:.2f}GB)")
    print("done")


if __name__ == "__main__":
    mode = sys.argv[1]
    els = sys.argv[2:] or list(CASES)
    if mode == "dims":
        run_dims(els)
    elif mode == "run":
        run_eos(els)
    else:
        sys.exit(f"unknown mode {mode!r} (use: dims | run)")
