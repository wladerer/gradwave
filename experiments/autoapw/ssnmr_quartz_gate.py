"""alpha-quartz 29Si shielding ecut-stability gate + MAS spectrum demo.

Two ecut rungs (Si sigma_iso must agree within a few ppm to trust the site,
because the analytic-USPP bare term diverges with ecut on hard-augmentation O
datasets, PR #392). If Si is stable, reference sigma_ref so <delta_iso(Si)> =
-107.4 ppm (exp) and synthesize the 29Si MAS spectrum.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.spacegroup import crystal

from gradwave.constants import RY_EV
from gradwave.inputs import Input, KPointsParams, NmrParams, NmrSpectrumParams

PSEUDOS = Path("tests/fixtures/qe/pseudos").resolve()
DELTA_EXP_SI = -107.4  # alpha-quartz 29Si isotropic shift vs TMS (ppm)


def alpha_quartz():
    a, c = 4.9134, 5.4052
    return crystal(
        symbols=["Si", "O"],
        basis=[(0.4697, 0.0, 1.0 / 3.0), (0.4135, 0.2669, 0.1191)],
        spacegroup=152, cellpar=[a, a, c, 90, 90, 120])


def run_rung(atoms, ecut_ry, ecutrho_ry, spectrum=None, sigma_ref=None):
    from gradwave.api import run_nmr

    inp = Input(
        atoms=atoms, pseudo_dir=PSEUDOS,
        pseudo_map={"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
                    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF"},
        ecut=ecut_ry * RY_EV, ecutrho=ecutrho_ry * RY_EV, xc="pbe",
        kpoints=KPointsParams(mesh=(2, 2, 2)), task="nmr", symmetry=False,
        nmr=NmrParams(
            task="shielding", shielding_level="gipaw",
            sigma_ref=sigma_ref,
            spectrum=spectrum or NmrSpectrumParams(),
            # k-streaming (PR #375/#376 route): quartz's full-mesh dense
            # contexts OOM a 14 GB box; chunk_k=1 is bit-identical, O(1) in nk.
            chunk_k=1),
        verbose=False)
    t0 = time.time()
    block = run_nmr(inp, verbose=False)
    dt = time.time() - t0
    return block, dt


def si_stats(block):
    si = [s for s in block["sites"] if s["species"] == "Si"]
    vals = [s["sigma_iso_ppm"] for s in si]
    return float(np.mean(vals)), float(max(vals) - min(vals)), vals


def main():
    atoms = alpha_quartz()
    print(f"alpha-quartz: {len(atoms)} atoms "
          f"({sum(s=='Si' for s in atoms.get_chemical_symbols())} Si, "
          f"{sum(s=='O' for s in atoms.get_chemical_symbols())} O)", flush=True)

    rungs = [(40, 320), (60, 480)]
    results = {}
    for ec, er in rungs:
        print(f"\n=== rung ecut={ec} Ry / ecutrho={er} Ry ===", flush=True)
        block, dt = run_rung(atoms, ec, er)
        mean, spread, vals = si_stats(block)
        print(f"  Si sigma_iso: mean={mean:.2f} ppm  equiv-spread={spread:.3f} ppm  "
              f"vals={[round(v,2) for v in vals]}  ({dt:.0f}s)", flush=True)
        o = [s for s in block["sites"] if s["species"] == "O"]
        print(f"  O  sigma_iso: {[round(s['sigma_iso_ppm'],1) for s in o]}", flush=True)
        results[f"{ec}_{er}"] = {"si_mean": mean, "si_spread": spread,
                                 "si_vals": vals,
                                 "o_vals": [s["sigma_iso_ppm"] for s in o]}

    lo = results["40_320"]["si_mean"]
    hi = results["60_480"]["si_mean"]
    ecut_drift = abs(hi - lo)
    print(f"\n=== ecut gate ===\n  Si sigma_iso 40->60 Ry: {lo:.2f} -> {hi:.2f} ppm "
          f"(drift {ecut_drift:.2f} ppm)", flush=True)
    stable = ecut_drift < 5.0
    print(f"  STABLE (<5 ppm): {stable}", flush=True)

    # Reference off the trusted (higher) rung so <delta_iso(Si)> = DELTA_EXP_SI.
    sigma_ref = DELTA_EXP_SI + hi  # delta = sigma_ref - sigma_iso  =>  ref = delta_exp + sigma
    print(f"  sigma_ref(Si) chosen so <delta_iso> = {DELTA_EXP_SI}: "
          f"{sigma_ref:.2f} ppm", flush=True)

    if stable:
        print("\n=== 29Si MAS spectrum (referenced) ===", flush=True)
        spec = NmrSpectrumParams(
            enabled=True, mode="mas", spin_rate_hz=5000.0, larmor_mhz=79.5,
            broadening_ppm=1.0, lineshape="gauss", n_orientations=2000,
            n_points=4096)
        block, dt = run_rung(atoms, 60, 480, spectrum=spec,
                             sigma_ref={"Si": sigma_ref})
        si = [s for s in block["sites"] if s["species"] == "Si"]
        d = [s["delta_iso_ppm"] for s in si]
        print(f"  delta_iso(Si) = {[round(x,2) for x in d]} ppm  "
              f"(target {DELTA_EXP_SI})", flush=True)
        sp = block["spectrum"]
        print(f"  spectrum: kind={sp['kind']} peak={sp['peak_ppm']:.2f} ppm "
              f"axis=[{sp['ppm_range'][0]:.1f},{sp['ppm_range'][1]:.1f}] "
              f"npts={len(sp['ppm_axis'])} ({dt:.0f}s)", flush=True)
        Path("quartz_spectrum.json").write_text(json.dumps({
            "ppm_axis": sp["ppm_axis"], "intensity": sp["intensity"],
            "delta_iso": d, "sigma_ref": sigma_ref, "peak_ppm": sp["peak_ppm"],
            "larmor_mhz": sp["larmor_mhz"], "spin_rate_hz": sp["spin_rate_hz"],
        }))
        print("  wrote quartz_spectrum.json", flush=True)

    Path("quartz_gate.json").write_text(json.dumps(results, indent=1))
    print("\nDONE", flush=True)
    return 0 if stable else 2


if __name__ == "__main__":
    sys.exit(main())
