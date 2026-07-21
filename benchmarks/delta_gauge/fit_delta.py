"""Birch-Murnaghan fits + Δ-values for the periodic-table Δ-gauge.

Δ follows calcDelta 3.0 (Lejaeghere et al.): both EOS per atom, each shifted to
its own minimum, RMS of the difference over [0.94, 1.06] × V0_avg where V0_avg
is the mean of the two equilibrium volumes, in meV/atom.

Reads results/eos_<el>.json (any subset present), writes results/delta_summary.json
and prints a table. `dfact_pdojo` is PseudoDojo's own published Δ for the same
standard pseudo — gradwave's Δ_wien2k should track it, since both use the same
pseudopotential against the same all-electron reference.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from cases import CASES, WIEN2K  # noqa: E402

EV_A3_TO_GPA = 160.2176634
SCALES = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]
RES = SP / "results"


def bm3(v, e0, v0, b0, b0p):
    x = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9 * v0 * b0 / 16 * ((x - 1) ** 3 * b0p + (x - 1) ** 2 * (6 - 4 * x))


def fit(vols, es):
    i = int(np.argmin(es))
    popt, _ = curve_fit(bm3, vols, es, p0=[es[i], vols[i], 0.6, 4.0], maxfev=40000)
    return popt  # e0, v0(Å³/atom), b0(eV/Å³), b1


def delta_value(p1, p2):
    v0av = 0.5 * (p1[1] + p2[1])
    vv = np.linspace(0.94 * v0av, 1.06 * v0av, 1000)
    d = bm3(vv, 0.0, *p1[1:]) - bm3(vv, 0.0, *p2[1:])
    return np.sqrt(np.trapezoid(d ** 2, vv) / (vv[-1] - vv[0])) * 1000


def main():
    rows, summary = [], {}
    for el in CASES:
        p = RES / f"eos_{el}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if len(d["E_eV"]) < len(SCALES):
            rows.append((el, None))
            continue
        nat = 1 if d["struct"] != "diamond" else 2
        vols = np.array([d["V_A3"][str(s)] for s in SCALES]) / nat
        es = np.array([d["E_eV"][str(s)] for s in SCALES]) / nat
        pg = fit(vols, es)
        v0w, b0w, b1w = WIEN2K[el]
        pw = (0.0, v0w, b0w / EV_A3_TO_GPA, b1w)
        dwien = delta_value(pg, pw)
        summary[el] = dict(v0=pg[1], b0=pg[2] * EV_A3_TO_GPA, b1=pg[3],
                           delta_wien2k=dwien, dfact_pdojo=d["dfact_pdojo"],
                           v0_wien2k=v0w, b0_wien2k=b0w, b1_wien2k=b1w)
        rows.append((el, summary[el]))
    (RES / "delta_summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{'el':3} {'V0':>8} {'V0_AE':>8} {'B0':>7} {'B0_AE':>7} "
          f"{'B1':>5} {'Δ_gw':>6} {'Δ_pdojo':>7}   (Å³/at, GPa, meV/at)")
    good = []
    for el, s in rows:
        if s is None:
            print(f"{el:3}  -- incomplete --")
            continue
        print(f"{el:3} {s['v0']:8.3f} {s['v0_wien2k']:8.3f} {s['b0']:7.1f} "
              f"{s['b0_wien2k']:7.1f} {s['b1']:5.2f} {s['delta_wien2k']:6.3f} "
              f"{s['dfact_pdojo']:7.3f}")
        good.append(s['delta_wien2k'])
    if good:
        g = np.array(good)
        print(f"\nΔ vs WIEN2k over {len(g)} elements: "
              f"mean {g.mean():.3f}, median {np.median(g):.3f}, max {g.max():.3f} meV/atom")


if __name__ == "__main__":
    main()
