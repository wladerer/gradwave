"""Full pipeline for ONE adsorbate–surface pair (the run-through target).

  1. relax the clean slab                         → E_clean
  2. relax the adsorbate at each (111) site        → E(site)   [warm-started]
  3. relax the gas reference (H2 for *H, CO for *CO)→ E_gas
  4. adsorption energy + CHE free energy + report

Robust-by-default: every energy is written to results/<pair>.json as it lands, so
a mid-run drop loses nothing and a re-run skips finished pieces.

    uv run python run_pair.py Pt H            # one pair, production settings (GPU)
    uv run python run_pair.py Pt H --debug    # tiny/fast, CPU — for local debugging
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ase.io import read, write
from ase.optimize import BFGS

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import che  # noqa: E402
import config  # noqa: E402

STRUCT = HERE / "structures"
RESULTS = HERE / "results"
GAS_FOR = {"H": "H2", "CO": "CO"}
GAS_SCALE = {"H": 0.5, "CO": 1.0}  # ½H2 for *H, CO for *CO


def _relax(atoms, calc, fmax, max_steps, tag):
    atoms.calc = calc
    t = time.time()
    opt = BFGS(atoms, logfile=str(RESULTS / f"{tag}.log"))
    opt.run(fmax=fmax, steps=max_steps)
    e = float(atoms.get_potential_energy())
    write(RESULTS / f"{tag}_relaxed.xyz", atoms)  # for the r2SCAN/diff stretch
    print(f"    [{tag}] E={e:.4f} eV  ({opt.nsteps} steps, {time.time()-t:.0f}s)",
          flush=True)
    return e


def run_pair(metal: str, ads: str, dbg: dict | None = None) -> dict:
    RESULTS.mkdir(exist_ok=True)
    out_f = RESULTS / f"{metal}_{ads}.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {"metal": metal, "ads": ads}

    def save():
        out_f.write_text(json.dumps(res, indent=2))

    p = dbg or {}
    kw_slab = dict(kpts=p.get("kpts_slab"), device=p.get("device"),
                   ecut=p.get("ecut"), ecutrho=p.get("ecutrho"))
    fmax, steps = p.get("fmax", config.FMAX), p.get("max_steps", config.MAX_STEPS)

    # ONE calculator for the clean slab + all four adsorbate sites (same cell and
    # elements). Two wins over a fresh calc per stage:
    #   (1) the pseudopotential form factors (radial FT / sbt, the CPU-bound setup
    #       that dominates the first SCF of a fresh calc) are built once, not 5×.
    #   (2) each stage's SCF warm-starts off the previous stage's converged density
    #       — the metal is barely perturbed by one adsorbate, so the slab density is
    #       a far better seed than an atomic guess. The density seed is grid-based
    #       and always applies; the orbital seed is auto-dropped when the adsorbate
    #       changes the symmetry/nbands (_remap_coeffs_to_spheres bails to a fresh
    #       orbital start) — robust, never an error.
    # Gas uses its own calculator below (different cell).
    slab_calc = config.make_calc(**kw_slab)

    # 1. clean slab
    if "e_clean" not in res:
        slab = read(STRUCT / f"slab_{metal}.xyz")
        res["e_clean"] = _relax(slab, slab_calc, fmax, steps, f"{metal}_slab")
        save()

    # 2. adsorbate at each site (reusing slab_calc → warm-started from the slab)
    res.setdefault("sites", {})
    for site in ("ontop", "bridge", "fcc", "hcp"):
        if site in res["sites"]:
            continue
        a = read(STRUCT / f"{metal}_{ads}_{site}.xyz")
        res["sites"][site] = _relax(a, slab_calc, fmax, steps,
                                    f"{metal}_{ads}_{site}")
        save()

    # 3. gas reference
    gas = GAS_FOR[ads]
    if "e_gas" not in res:
        g = read(STRUCT / f"gas_{gas}.xyz")
        kw_gas = dict(is_gas=True, device=p.get("device"),
                      ecut=p.get("ecut"), ecutrho=p.get("ecutrho"))
        res["e_gas"] = _relax(g, config.make_calc(**kw_gas), fmax, steps,
                              f"gas_{gas}")
        save()

    # 4. thermodynamics
    e_gas_ref = GAS_SCALE[ads] * res["e_gas"]
    summary = che.summarize(metal, ads, res["sites"], res["e_clean"], e_gas_ref)
    res["adsorption"] = {
        "best_site": summary.best_site, "e_ads": summary.e_ads,
        "dg_ads": summary.dg_ads, "per_site_dE": summary.per_site,
    }
    save()
    print(che.report(summary), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("metal", choices=["Pt", "Au"])
    ap.add_argument("ads", choices=["H", "CO"])
    ap.add_argument("--debug", action="store_true", help="tiny CPU profile")
    args = ap.parse_args()
    run_pair(args.metal, args.ads, config.DEBUG if args.debug else None)
