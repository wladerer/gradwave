"""E_cut convergence of the EOS for one element.

Usage: GW_DEVICE=cpu ecut_scan.py <El> <ecut_ry> [<ecut_ry> ...]

Runs the full 7-volume EOS at each cutoff (fresh fixed grid per cutoff,
warm-started over volumes) and reports V0, B0, B1 and Δ vs WIEN2k. Diagnostic
only — writes results/scan_<El>.json, never the production eos_<El>.json.
The point is that B0/stress converge slower in the plane-wave basis than the
total energy, so the PseudoDojo energy hint can under-converge a hard-pseudo EOS.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from cases import CASES, RY, WIEN2K, geometry, nbands  # noqa: E402

torch.set_num_threads(int(os.environ.get("GW_THREADS", "8")))
sys.stdout.reconfigure(line_buffering=True)
from gradwave.core.xc.pbe import PBE  # noqa: E402
from gradwave.pseudo.upf import parse_upf  # noqa: E402
from gradwave.scf.loop import scf, setup_system  # noqa: E402

PSE, RES = SP / "pseudos", SP / "results"
DEV = os.environ.get("GW_DEVICE", "cpu")
SCALES = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]
EV_A3_TO_GPA = 160.2176634
el = sys.argv[1]
ecuts = [float(x) for x in sys.argv[2:]]
c = CASES[el]
upf = parse_upf(os.environ.get("GW_PSEUDO_OVERRIDE", str(PSE / f"{el}.upf")))
nat = 1 if c["struct"] != "diamond" else 2


def bm3(v, e0, v0, b0, b0p):
    x = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9 * v0 * b0 / 16 * ((x - 1) ** 3 * b0p + (x - 1) ** 2 * (6 - 4 * x))


def build(scale, ecut_ry, fft=None):
    cell, pos, elems = geometry(el, scale)
    return setup_system(cell, pos, [0] * len(elems), [upf], ecut=ecut_ry * RY,
                        kmesh=c["kmesh"], nbands=nbands(el), use_symmetry=True,
                        fft_shape=fft)


out = {}
for ec in ecuts:
    dims = [build(s, ec).grid.shape for s in SCALES]
    fixed = tuple(max(d[i] for d in dims) for i in range(3))
    es, vs, prev = [], [], None
    for s in SCALES:
        sysd = build(s, ec, fixed)
        if DEV != "cpu":
            sysd = sysd.to(torch.device(DEV))
        t0 = time.time()
        r = scf(sysd, PBE(), smearing=c["smear"], width=c["width"] * RY,
                etol=1e-8, rhotol=1e-7, max_iter=200, verbose=False, start_from=prev)
        prev = r
        es.append(float(r.energies.total) / nat)
        vs.append(float(np.abs(np.linalg.det(geometry(el, s)[0]))) / nat)
        print(f"  {el} ecut={ec:.0f} v={s:.2f} it={r.n_iter} ({time.time()-t0:.0f}s)")
    vs, es = np.array(vs), np.array(es)
    i = int(np.argmin(es))
    p, _ = curve_fit(bm3, vs, es, p0=[es[i], vs[i], 0.6, 4.0], maxfev=40000)
    v0w, b0w, b1w = WIEN2K[el]
    vv = np.linspace(0.94 * 0.5 * (p[1] + v0w), 1.06 * 0.5 * (p[1] + v0w), 1000)
    d = bm3(vv, 0.0, p[1], p[2], p[3]) - bm3(vv, 0.0, v0w, b0w / EV_A3_TO_GPA, b1w)
    delta = np.sqrt(np.trapezoid(d ** 2, vv) / (vv[-1] - vv[0])) * 1000
    out[str(ec)] = dict(fft=list(fixed), v0=p[1], b0=p[2] * EV_A3_TO_GPA,
                        b1=p[3], delta=delta)
    print(f"{el} ecut={ec:.0f} Ry (grid {fixed}): V0={p[1]:.3f} (AE {v0w:.3f}) "
          f"B0={p[2]*EV_A3_TO_GPA:.1f} (AE {b0w:.1f}) B1={p[3]:.2f} "
          f"Δ={delta:.3f} meV/atom")
(RES / f"scan_{el}.json").write_text(json.dumps(out, indent=1))
