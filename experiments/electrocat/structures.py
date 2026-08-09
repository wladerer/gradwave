"""Build all structures for the *H / *CO on Pt(111) & Au(111) study.

Clean slabs, adsorbate configurations at the four high-symmetry (111) sites
(top, bridge, fcc, hcp), and the gas-phase references (H2, H2O, CO) used by the
computational hydrogen electrode. Everything is written to structures/ as
extxyz (constraints preserved) plus a JSON manifest, so the GPU run just loads
them — no ASE model-building on the box.

Lattice constants are PBE values; a quick bulk relax (bulk.py) can refine them,
but adsorption ENERGIES are differences and largely insensitive to a ~1% a.
"""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms
from ase.build import add_adsorbate, fcc111, molecule
from ase.constraints import FixAtoms
from ase.io import write

HERE = Path(__file__).parent
OUT = HERE / "structures"

# PBE lattice constants [Å]
A_PBE = {"Pt": 3.968, "Au": 4.159}
SIZE = (2, 2, 4)          # 2×2 surface cell, 4 layers → 1/4 ML coverage
VACUUM = 7.5              # per side → ~15 Å total vacuum
N_FIXED_LAYERS = 2        # bottom layers frozen (bulk-like)
SITES = ("ontop", "bridge", "fcc", "hcp")
# initial adsorbate heights [Å] of the binding atom above the surface plane
H_HEIGHT = {"ontop": 1.55, "bridge": 1.05, "fcc": 0.95, "hcp": 0.95}
CO_HEIGHT = {"ontop": 1.85, "bridge": 1.45, "fcc": 1.35, "hcp": 1.35}
CO_BOND = 1.15           # C–O [Å], C-down


def _fix_bottom(slab: Atoms) -> Atoms:
    """Freeze the bottom N_FIXED_LAYERS layers (lowest-z atoms)."""
    z = slab.positions[:, 2]
    zt = sorted(set(round(v, 3) for v in z))
    fixed_z = set(zt[:N_FIXED_LAYERS])
    idx = [i for i, v in enumerate(z) if round(v, 3) in fixed_z]
    slab.set_constraint(FixAtoms(indices=idx))
    return slab


def clean_slab(metal: str) -> Atoms:
    slab = fcc111(metal, size=SIZE, a=A_PBE[metal], vacuum=VACUUM)
    return _fix_bottom(slab)


def with_adsorbate(metal: str, ads: str, site: str) -> Atoms:
    slab = fcc111(metal, size=SIZE, a=A_PBE[metal], vacuum=VACUUM)
    if ads == "H":
        add_adsorbate(slab, "H", height=H_HEIGHT[site], position=site)
    elif ads == "CO":
        co = Atoms("CO", positions=[[0, 0, 0], [0, 0, CO_BOND]])  # C down, O up
        add_adsorbate(slab, co, height=CO_HEIGHT[site], position=site)
    else:
        raise ValueError(ads)
    return _fix_bottom(slab)


def gas_references() -> dict[str, Atoms]:
    """H2, H2O, CO in a 12 Å box (Γ-point molecules for the CHE references)."""
    out = {}
    for name in ("H2", "H2O", "CO"):
        m = molecule(name)
        m.center(vacuum=6.0)
        m.pbc = True
        out[name] = m
    return out


def build_all() -> None:
    OUT.mkdir(exist_ok=True)
    manifest: dict[str, dict] = {"slabs": {}, "adsorbed": {}, "gas": {}}
    for metal in ("Pt", "Au"):
        f = OUT / f"slab_{metal}.xyz"
        write(f, clean_slab(metal))
        manifest["slabs"][metal] = f.name
        for ads in ("H", "CO"):
            for site in SITES:
                key = f"{metal}_{ads}_{site}"
                f = OUT / f"{key}.xyz"
                write(f, with_adsorbate(metal, ads, site))
                manifest["adsorbed"][key] = {
                    "file": f.name, "metal": metal, "ads": ads, "site": site,
                }
    for name, atoms in gas_references().items():
        f = OUT / f"gas_{name}.xyz"
        write(f, atoms)
        manifest["gas"][name] = f.name
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n = len(manifest["adsorbed"])
    print(f"wrote {n} adsorbate configs + 2 slabs + 3 gas refs to {OUT}")


if __name__ == "__main__":
    build_all()
