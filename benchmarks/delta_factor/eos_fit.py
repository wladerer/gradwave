"""Birch-Murnaghan fits + Δ-factor: gradwave vs QE."""
import json
import re
import sys
from pathlib import Path

import numpy as np

SP = Path(__file__).parent
sys.path.insert(0, str(SP))
from eos_cases import CASES, RY, SCALES  # noqa: E402

# the Birch-Murnaghan fit ships in the library (postscf.eos)
from gradwave.postscf.eos import EV_A3_TO_GPA, fit_bm3  # noqa: E402
from gradwave.postscf.eos import birch_murnaghan as bm3  # noqa: E402


def fit(v, e):
    f = fit_bm3(v, e)
    return (f.e0, f.v0, f.b0, f.b0_prime)  # e0, v0, b0 (eV/A^3), b0p


# gradwave energies
gw = json.loads((SP / "eos_gw.json").read_text())
# QE energies from "case<i>.out:!    total energy = X Ry"
qe = {c: {} for c in CASES}
for line in (SP / "eos_qe_energies.txt").read_text().splitlines():
    m = re.match(r"([a-z]+)(\d)\.out:!.*=\s*(-[\d.]+)\s*Ry", line)
    if m:
        qe[m.group(1)][SCALES[int(m.group(2))]] = float(m.group(3)) * RY

print(f"{'case':6s} {'code':4s} {'a0 (Å)':>9s} {'B0 (GPa)':>9s} {'B0prm':>6s} "
      f"{'ΔE_pt max':>10s} {'Δ (meV/at)':>10s}")
for case in CASES:
    nat = len(CASES[case]["elems"])
    vols = np.array([gw[case]["V_A3"][str(s)] for s in SCALES])
    e_gw = np.array([gw[case]["E_eV"][str(s)] for s in SCALES])
    e_qe = np.array([qe[case][s] for s in SCALES])

    # point-wise: residual after removing the constant offset
    off = (e_gw - e_qe).mean()
    pt = np.abs(e_gw - e_qe - off).max() / nat * 1000  # meV/atom

    res = {}
    for code, e in (("gw", e_gw), ("qe", e_qe)):
        e0, v0, b0, b0p = fit(vols, e)  # cell energy vs cell volume
        a0 = (4.0 * v0) ** (1.0 / 3.0)  # fcc primitive: V_cell = a^3/4
        res[code] = (e0, v0, b0, b0p, a0)

    # Δ (Lejaeghere): RMS difference of the fitted curves, each shifted to
    # its own E0, over the volume window, per atom
    vv = np.linspace(vols.min(), vols.max(), 400)
    d_gw = bm3(vv, *res["gw"][:4]) - res["gw"][0]
    d_qe = bm3(vv, *res["qe"][:4]) - res["qe"][0]
    delta = (np.sqrt(np.trapezoid((d_gw - d_qe) ** 2, vv)
                     / (vv.max() - vv.min())) / nat) * 1000

    for code in ("gw", "qe"):
        e0, v0, b0, b0p, a0 = res[code]
        extra = f" {pt:9.3f} {delta:10.4f}" if code == "gw" else ""
        print(f"{case:6s} {code:4s} {a0:9.5f} {b0 * EV_A3_TO_GPA:9.2f} "
              f"{b0p:6.3f}{extra}")
    print(f"{'':6s} off  abs offset {off / nat * 1000:+8.3f} meV/atom")
