"""Fallback headline: diamond-Si 29Si MAS spectrum through the full pipeline.

Si-only (no hard-augmentation O), so the GIPAW absolute sigma is on the validated
side of the #392 trust boundary (sigma_iso ~ 398 ppm). Referenced with the
documented sigma_ref(29Si, TMS) = 337.4 ppm used by the shipped shielding tests,
so the axis is a true ppm-vs-TMS scale. Two ecut rungs recorded for honesty.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ase.build import bulk

from gradwave.api import run_nmr
from gradwave.constants import RY_EV
from gradwave.inputs import Input, KPointsParams, NmrParams, NmrSpectrumParams

PSEUDOS = Path("tests/fixtures/qe/pseudos").resolve()
SIGMA_REF_TMS = 337.4  # documented absolute 29Si shielding of TMS (ppm)


def run(atoms, ecut_ry, ecutrho_ry, spectrum=None, sigma_ref=None):
    inp = Input(
        atoms=atoms, pseudo_dir=PSEUDOS,
        pseudo_map={"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF"},
        ecut=ecut_ry * RY_EV, ecutrho=ecutrho_ry * RY_EV, xc="pbe",
        kpoints=KPointsParams(mesh=(2, 2, 2)), task="nmr", symmetry=False,
        nmr=NmrParams(task="shielding", shielding_level="gipaw",
                      sigma_ref=sigma_ref, spectrum=spectrum or NmrSpectrumParams()),
        verbose=False)
    t0 = time.time()
    return run_nmr(inp, verbose=False), time.time() - t0


def main() -> int:
    atoms = bulk("Si", "diamond", a=5.43)
    print(f"diamond-Si: {len(atoms)} atoms", flush=True)
    means = {}
    for ec, er in [(12, 48), (40, 320)]:
        block, dt = run(atoms, ec, er)
        vals = [s["sigma_iso_ppm"] for s in block["sites"]]
        means[f"{ec}_{er}"] = float(np.mean(vals))
        print(f"  ecut={ec}/{er} Ry: sigma_iso(Si) = {[round(v,2) for v in vals]} "
              f"mean={np.mean(vals):.2f} spread={max(vals)-min(vals):.3f} ppm ({dt:.0f}s)",
              flush=True)
    drift = abs(means["40_320"] - means["12_48"])
    print(f"  ecut drift 12->40 Ry: {drift:.2f} ppm", flush=True)

    spec = NmrSpectrumParams(enabled=True, mode="mas", spin_rate_hz=5000.0,
                             larmor_mhz=79.5, broadening_ppm=1.0,
                             n_orientations=2000, n_points=4096)
    block, dt = run(atoms, 40, 320, spectrum=spec, sigma_ref={"Si": SIGMA_REF_TMS})
    d = [s["delta_iso_ppm"] for s in block["sites"]]
    sp = block["spectrum"]
    print(f"  delta_iso(Si) = {[round(x,2) for x in d]} ppm (ref TMS {SIGMA_REF_TMS})",
          flush=True)
    print(f"  spectrum: peak={sp['peak_ppm']:.2f} ppm ({dt:.0f}s)", flush=True)
    Path("diamond_spectrum.json").write_text(json.dumps({
        "ppm_axis": sp["ppm_axis"], "intensity": sp["intensity"],
        "delta_iso": d, "sigma_ref": SIGMA_REF_TMS, "peak_ppm": sp["peak_ppm"],
        "larmor_mhz": sp["larmor_mhz"], "spin_rate_hz": sp["spin_rate_hz"],
        "ecut_means": means}))
    print("  wrote diamond_spectrum.json\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
