"""STRETCH: does the r2SCAN meta-GGA change the energetics?

Single-point r2SCAN energies on the PBE-relaxed best-site geometry (and the clean
slab + gas ref), forming the r2SCAN adsorption energy for comparison with PBE. No
re-relaxation — this isolates the functional's effect on the energetics, which is
the cheap, informative first look (a full r2SCAN relax is the follow-on).

Uses the NC (ONCV) pseudopotentials: meta-GGA (τ-dependent) runs on gradwave's
norm-conserving path only. Higher ecut than PAW, but these cells are small.

Note on ISDF/THC: gradwave's ISDF accelerates the *exact-exchange* (Fock) build,
so it applies to hybrids (incl. a hybrid meta-GGA like r2SCAN0), NOT to the bare
r2SCAN semilocal functional here. To exercise ISDF, run a hybrid — set
``xc="pbe0"`` (or a r2SCAN-hybrid if registered) and pass the fock/ISDF options;
that is a separate, heavier benchmark and is left as a flag below.

    uv run python r2scan_isdf.py Pt CO
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ase.io import read
from gradwave.constants import RY_EV as RY

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import che  # noqa: E402
import config  # noqa: E402

RESULTS = HERE / "results"
NC = {
    "Pt": "Pt_ONCV.upf", "Au": "Au_ONCV.upf", "C": "C_ONCV_PBE-1.2.upf",
    "O": "O_ONCV_PBE-1.2.upf", "H": "H_ONCV_PBE-1.2.upf",
}
NC_PSEUDOS = {el: str(HERE / "pseudos_nc" / f) for el, f in NC.items()}
# ONCV 5d metals need a high wavefunction cutoff
ECUT_NC = 80.0 * RY
GAS_FOR = {"H": "H2", "CO": "CO"}
GAS_SCALE = {"H": 0.5, "CO": 1.0}


def _sp(atoms, xc, is_gas, **fock):
    from gradwave.calculator import GradWave
    atoms.calc = GradWave(
        ecut=ECUT_NC, pseudopotentials=NC_PSEUDOS, xc=xc,
        kpts=config.KPTS_GAS if is_gas else config.KPTS_SLAB,
        smearing="none" if is_gas else config.SMEARING, width=config.WIDTH,
        device=config.DEVICE, use_symmetry=True, **fock)
    return float(atoms.get_potential_energy())


def run(metal: str, ads: str, xc: str = "r2scan") -> None:
    res = {}
    for label, tag, is_gas in [
        ("clean", f"{metal}_slab_relaxed.xyz", False),
        ("ads", None, False),           # best site, filled below
        ("gas", f"gas_{GAS_FOR[ads]}_relaxed.xyz", True),
    ]:
        if label == "ads":
            import json
            best = json.loads((RESULTS / f"{metal}_{ads}.json").read_text())
            tag = f"{metal}_{ads}_{best['adsorption']['best_site']}_relaxed.xyz"
        f = RESULTS / tag
        if not f.exists():
            raise SystemExit(f"missing {f} — run run_pair.py {metal} {ads} first")
        res[label] = _sp(read(f), xc, is_gas)
        print(f"  {xc} {label}: {res[label]:.4f} eV", flush=True)

    e_ads = che.adsorption_energy(res["ads"], res["clean"],
                                  GAS_SCALE[ads] * res["gas"])
    print(f"\n{xc.upper()} ΔE(*{ads}/{metal}) = {e_ads:+.3f} eV   "
          f"(compare to the PBE value in results/{metal}_{ads}.json)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("metal", choices=["Pt", "Au"])
    ap.add_argument("ads", choices=["H", "CO"])
    ap.add_argument("--xc", default="r2scan")
    a = ap.parse_args()
    run(a.metal, a.ads, a.xc)
