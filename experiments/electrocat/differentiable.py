"""STRETCH: differentiable electrocatalysis — descriptor gradients via autograd.

The thing no production DFT code does: gradients of a catalytic descriptor through
the converged SCF. Cleanest concrete demo here — the **biaxial-strain sensitivity
of the adsorption energy**,

    dE_ads/dε_bi = (σ_ads·Ω)_in-plane − (σ_clean·Ω)_in-plane ,

read straight from the autograd stress (postscf.stress, which differentiates the
converged energy w.r.t. strain). This is the strain–activity relation (d-band-centre
tuning) as an exact derivative, not a finite-difference scan — one calculation, not
a lattice sweep.

Frontier extensions (sketched, not run here — bigger builds):
- **d E_ads / d(composition)** via ``scf/alchemical.py`` — inverse catalyst design
  (which way to alloy Pt↔Au to hit a target ΔG). A differentiable alloy surface.
- **d E_ads / d(electrode potential)** via the constant-µ ESM we built
  (``boundary="open_z_metal", target_mu=…``) — the grand-canonical sensitivity, i.e.
  how binding responds to the applied potential, from ∂N/∂µ.

    uv run python differentiable.py Pt H
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

RESULTS = HERE / "results"


def _stress_inplane_times_vol(atoms) -> float:
    """(σ_xx + σ_yy)·Ω [eV] — the biaxial-strain derivative of the total energy,
    from gradwave's autograd stress."""
    atoms.calc = config.make_calc()
    atoms.get_potential_energy()          # converge the SCF
    s = atoms.get_stress(voigt=False)     # eV/Å³, 3×3 (ASE convention)
    vol = atoms.get_volume()
    return float((s[0, 0] + s[1, 1]) * vol)


def run(metal: str, ads: str) -> None:
    best = json.loads((RESULTS / f"{metal}_{ads}.json").read_text())
    site = best["adsorption"]["best_site"]
    clean = read(RESULTS / f"{metal}_slab_relaxed.xyz")
    adsl = read(RESULTS / f"{metal}_{ads}_{site}_relaxed.xyz")

    d_clean = _stress_inplane_times_vol(clean)
    d_ads = _stress_inplane_times_vol(adsl)
    sens = d_ads - d_clean  # dE_ads / dε_biaxial [eV per unit biaxial strain]

    print(f"\n*{ads} on {metal}(111), site {site}:")
    print(f"  dE_ads/dε_biaxial = {sens:+.3f} eV / unit strain   (autograd stress)")
    print(f"  → {sens * 0.01:+.4f} eV per +1% biaxial tensile strain")
    print("  (sign: negative ⇒ tensile strain strengthens binding — the usual "
          "d-band-centre trend for these metals)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("metal", choices=["Pt", "Au"])
    ap.add_argument("ads", choices=["H", "CO"])
    a = ap.parse_args()
    run(a.metal, a.ads)
