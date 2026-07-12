"""Birch-Murnaghan fits + Δ-values: gradwave PAW vs the WIEN2k reference.

Δ follows calcDelta 3.0: both equations of state per atom, each shifted to
its own minimum, RMS of the difference over [0.94, 1.06] x V0_avg where
V0_avg is the mean of the two equilibrium volumes.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from cases import CASES, RY, SCALES, WIEN2K  # noqa: E402

EV_A3_TO_GPA = 160.2176634


def bm3(v, e0, v0, b0, b0p):
    x = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9 * v0 * b0 / 16 * ((x - 1) ** 3 * b0p + (x - 1) ** 2 * (6 - 4 * x))


def fit_points(vols, es):
    i = int(np.argmin(es))
    popt, _ = curve_fit(bm3, vols, es, p0=[es[i], vols[i], 0.6, 4.0],
                        maxfev=20000)
    return popt  # e0, v0, b0 (eV/Å^3/atom), b1


def delta_value(p1, p2):
    """calcDelta: RMS of the shifted-curve difference over the +/-6% window
    around the average equilibrium volume, in meV/atom."""
    v0av = 0.5 * (p1[1] + p2[1])
    vv = np.linspace(0.94 * v0av, 1.06 * v0av, 1000)
    d1 = bm3(vv, 0.0, *p1[1:])
    d2 = bm3(vv, 0.0, *p2[1:])
    return np.sqrt(np.trapezoid((d1 - d2) ** 2, vv) / (vv[-1] - vv[0])) * 1000


gw = json.loads((SP / "results" / "eos_gw.json").read_text())
qe = {c: {} for c in CASES}
qe_file = SP / "results" / "eos_qe_energies.txt"
if qe_file.exists():
    for line in qe_file.read_text().splitlines():
        m = re.match(r"([a-z]+)(\d)\.out:!.*=\s*(-[\d.]+)\s*Ry", line)
        if m:
            qe[m.group(1)][SCALES[int(m.group(2))]] = float(m.group(3)) * RY

print(f"{'case':5s} {'code':6s} {'V0 (Å³/at)':>11s} {'B0 (GPa)':>9s} "
      f"{'B1':>6s} {'Δ_wien2k':>9s} {'Δ_qe':>7s}  (meV/atom)")
for case in CASES:
    if case not in gw or len(gw[case]["E_eV"]) < len(SCALES):
        have = len(gw[case]["E_eV"]) if case in gw else 0
        print(f"{case:5s} -- gradwave data incomplete ({have}/{len(SCALES)}) --")
        continue
    nat = gw[case]["natoms"]
    vols = np.array([gw[case]["V_A3"][str(s)] for s in SCALES]) / nat
    es = np.array([gw[case]["E_eV"][str(s)] for s in SCALES]) / nat
    pg = fit_points(vols, es)

    v0w, b0w_gpa, b1w = WIEN2K[case]
    pw = (0.0, v0w, b0w_gpa / EV_A3_TO_GPA, b1w)
    d_w = delta_value(pg, pw)

    d_q = ""
    if len(qe[case]) == len(SCALES):
        eq = np.array([qe[case][s] for s in SCALES]) / nat
        pq = fit_points(vols, eq)
        d_q = f"{delta_value(pg, pq):7.4f}"
        off = (es - eq).mean() * 1000
        pt = np.abs(es - eq - off / 1000).max() * 1000
        extra = (f"   [vs QE: V0 {pq[1]:.4f}, B0 "
                 f"{pq[2] * EV_A3_TO_GPA:.2f}, B1 {pq[3]:.3f}; "
                 f"offset {off:+.3f}, pt-wise {pt:.4f} meV/at]")
    else:
        extra = ""

    print(f"{case:5s} {'gw':6s} {pg[1]:11.4f} {pg[2] * EV_A3_TO_GPA:9.2f} "
          f"{pg[3]:6.3f} {d_w:9.3f} {d_q:>7s}{extra}")
    print(f"{case:5s} {'wien2k':6s} {v0w:11.4f} {b0w_gpa:9.2f} {b1w:6.3f}")
