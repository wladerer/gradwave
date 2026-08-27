"""Synthesize the alpha-quartz 29Si MAS spectrum from the preserved rung-A
shielding (ecut 40 Ry / 320 Ry), local + cheap (no SCF).

Rung A gave sigma_iso(29Si) = [582.17, 582.17, 587.77] ppm (mean 584.04, equiv-
site spread 5.6 ppm — a k-mesh-convergence artifact; alpha-quartz has ONE
crystallographic Si site). sigma_ref is calibrated so <delta_iso> = the
experimental -107.4 ppm vs TMS, giving a true ppm-vs-TMS axis. alpha-quartz 29Si
has a small CSA, so at 5 kHz MAS the manifold is essentially the centerband; the
spectrum is driven entirely by the computed delta_iso (CSA set to zero — the full
tensor was not captured before the rung-B run was truncated; the CSA/sideband
path is validated separately in the unit tests).

Writes ssnmr_quartz_si29.png (via the io.analysis helper) + ssnmr_quartz_si29.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gradwave.io.analysis import plot_nmr_spectrum
from gradwave.postscf.nmr_spectrum import NMRSite, spectrum

SIGMA_ISO_SI = [582.17, 582.17, 587.77]  # rung-A ecut 40/320 (ppm)
DELTA_EXP = -107.4  # experimental alpha-quartz 29Si isotropic shift vs TMS (ppm)
LARMOR_MHZ = 79.5   # 29Si at 9.4 T
SPIN_RATE_HZ = 5000.0
# A single Gaussian wide enough that the 5.6 ppm k-noise spread reads as the
# line's convergence-limited width (alpha-quartz is one crystallographic Si
# site: one physical line at delta_iso), not as three resolved peaks.
BROADENING_PPM = 5.0


def main() -> int:
    mean_sigma = float(np.mean(SIGMA_ISO_SI))
    sigma_ref = DELTA_EXP + mean_sigma  # delta = sigma_ref - sigma  =>  <delta> = DELTA_EXP
    delta = [sigma_ref - s for s in SIGMA_ISO_SI]
    print(f"sigma_ref(29Si) = {sigma_ref:.2f} ppm  (calibrated to <delta>={DELTA_EXP})")
    print(f"delta_iso(29Si) = {[round(d, 2) for d in delta]} ppm  "
          f"mean={np.mean(delta):.2f}")

    sites = [NMRSite(delta_iso=d, delta_aniso=0.0, eta_csa=0.0, weight=1.0,
                     label=f"Si{i}") for i, d in enumerate(delta)]
    # Explicit 29Si window: with a small CSA the 5 kHz MAS manifold is the
    # centerband; the auto-axis would otherwise span every (zero-intensity)
    # sideband order. Sideband spacing here is nu_r/nu0 = 62.9 ppm.
    axis = np.linspace(-200.0, -20.0, 4096)
    ppm, inten = spectrum(
        sites, kind="mas", nu0_hz=LARMOR_MHZ * 1e6, nu_r_hz=SPIN_RATE_HZ,
        n_orientations=2000, axis_ppm=axis, fwhm_gauss=BROADENING_PPM)
    peak = float(ppm[int(np.argmax(inten))])
    print(f"MAS spectrum: peak={peak:.2f} ppm  "
          f"axis=[{ppm.min():.1f},{ppm.max():.1f}] npts={len(ppm)}")

    out = Path(__file__).parent
    (out / "ssnmr_quartz_si29.json").write_text(json.dumps({
        "material": "alpha-quartz", "nucleus": "29Si", "mode": "mas",
        "ecut_ry": 40, "ecutrho_ry": 320,
        "sigma_iso_ppm": SIGMA_ISO_SI, "sigma_ref_ppm": sigma_ref,
        "delta_iso_ppm": delta, "delta_exp_ppm": DELTA_EXP,
        "larmor_mhz": LARMOR_MHZ, "spin_rate_hz": SPIN_RATE_HZ,
        "broadening_ppm": BROADENING_PPM, "peak_ppm": peak,
        "ppm_axis": ppm.tolist(), "intensity": inten.tolist()}))
    label = f"$^{{29}}$Si MAS · $\\delta_{{iso}}$ = {np.mean(delta):.1f} ppm (α-quartz)"
    plot_nmr_spectrum(ppm, inten, path=str(out / "ssnmr_quartz_si29.png"), label=label)
    print(f"wrote {out / 'ssnmr_quartz_si29.png'} and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
