"""CO adsorption on Pt(111): binding energy at the top and fcc-hollow sites.

The PBE baseline for the classic "CO puzzle": PBE over-stabilizes the hollow
site (the CO 2π* sits too low, over-counting back-donation), so it predicts
fcc-hollow binding where experiment finds atop. This script computes the PBE
binding energies that a learned/linear-response Hubbard U is meant to correct.

First pass uses a RIGID substrate: the Pt slab is frozen at its bulk-truncated
geometry (a0 from the EOS) and only the CO molecule relaxes. This is the cheap
site-comparison; substrate relaxation is a later refinement.

    E_bind(site) = E(slab+CO) - E(slab) - E(CO_gas)     (negative = bound)

Runs on the asus GPU (PAW, 40/400 Ry). Resumable: each system's result is
written to slab_co.json as it finishes, and a re-run skips finished systems.

    LD_LIBRARY_PATH=/run/opengl-driver/lib \
      ~/.venvs/base/bin/python examples/co_pt/slab_co.py
"""

import json
import os
import sys
import time

import numpy as np
import torch
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from gradwave.calculator import GradWave

RY = 13.605693122994
FIX = "tests/fixtures/qe/pseudos"
PSEUDOS = {
    "Pt": f"{FIX}/Pt.pbe-n-kjpaw_psl.1.0.0.UPF",
    "C": f"{FIX}/C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O": f"{FIX}/O.pbe-n-kjpaw_psl.1.0.0.UPF",
}
OUT = "examples/co_pt/slab_co.json"

ECUT = 40 * RY
ECUTRHO = 400 * RY
WIDTH = 0.20
SLAB_K = (6, 6, 1)     # rhombic 2x2 cell keeps C3v: 6x6x1 -> 7 irreducible k
GAS_K = (1, 1, 1)
CO_D = 1.15            # initial C-O distance, Å
HEIGHT = 1.95         # initial C height above the site, Å
FMAX = 0.05           # eV/Å
MAXSTEP = 40
DEVICE = os.environ.get("GW_DEVICE", "cpu")   # CPU beats the laptop GPU here


def a0_from_eos(default=3.97):
    p = "examples/co_pt/bulk_pt_eos.json"
    if os.path.exists(p):
        return json.load(open(p))["a0_ang"]
    print(f"WARNING: {p} not found, using a0={default}", flush=True)
    return default


def co_molecule():
    """CO with C at the origin and O directly above (C binds the surface)."""
    return Atoms("CO", positions=[[0, 0, 0], [0, 0, CO_D]])


def build_slab(a0):
    slab = fcc111("Pt", size=(2, 2, 3), a=a0, vacuum=7.5, orthogonal=False)
    slab.set_constraint(FixAtoms(mask=[True] * len(slab)))  # rigid substrate
    return slab


def build_slab_co(a0, site):
    slab = fcc111("Pt", size=(2, 2, 3), a=a0, vacuum=7.5, orthogonal=False)
    n_pt = len(slab)
    add_adsorbate(slab, co_molecule(), height=HEIGHT, position=site, mol_index=0)
    # freeze all Pt, relax only C and O
    slab.set_constraint(FixAtoms(indices=list(range(n_pt))))
    return slab


def calc(kpts):
    return GradWave(ecut=ECUT, ecutrho=ECUTRHO, pseudopotentials=PSEUDOS,
                    xc="pbe", kpts=kpts, smearing="gaussian", width=WIDTH,
                    etol=1e-7, rhotol=1e-6, device=DEVICE, verbose=False)


def relax(atoms, kpts, tag, results, *, single_point=False):
    if tag in results:
        print(f"  [skip] {tag} (done: E={results[tag]['energy']:.4f})", flush=True)
        return
    atoms.calc = calc(kpts)
    t0 = time.time()
    if not single_point:
        BFGS(atoms, logfile=None).run(fmax=FMAX, steps=MAXSTEP)
    e = float(atoms.get_potential_energy())
    fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    results[tag] = dict(energy=e, fmax=fmax, n_atoms=len(atoms),
                        positions=atoms.get_positions().tolist(),
                        seconds=time.time() - t0, converged=fmax <= FMAX)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"  {tag}: E={e:.4f} eV  fmax={fmax:.3f}  {time.time()-t0:.0f}s", flush=True)


def main():
    if DEVICE == "cpu":
        torch.set_num_threads(int(os.environ.get("GW_THREADS", "16")))
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    a0 = a0_from_eos()
    print(f"device: {DEVICE}  threads: {torch.get_num_threads()}", flush=True)
    print(f"a0 = {a0:.4f} Å", flush=True)

    # gas-phase CO (relax the molecule)
    gas = co_molecule()
    gas.set_cell([10.0, 10.0, 10.0])
    gas.center()
    relax(gas, GAS_K, "CO_gas", results)

    # clean slab (rigid, single-point)
    relax(build_slab(a0), SLAB_K, "slab", results, single_point=True)

    # slab + CO at the two sites (relax CO only)
    for site in ("ontop", "fcc"):
        relax(build_slab_co(a0, site), SLAB_K, f"slab_CO_{site}", results)

    # binding energies
    if all(k in results for k in ("CO_gas", "slab", "slab_CO_ontop", "slab_CO_fcc")):
        e_gas = results["CO_gas"]["energy"]
        e_slab = results["slab"]["energy"]
        summary = {}
        for site in ("ontop", "fcc"):
            eb = results[f"slab_CO_{site}"]["energy"] - e_slab - e_gas
            summary[site] = eb
            print(f"E_bind({site}) = {eb:+.4f} eV", flush=True)
        pref = "ontop" if summary["ontop"] < summary["fcc"] else "fcc"
        results["binding_eV"] = summary
        results["preferred_site"] = pref
        results["site_gap_eV"] = abs(summary["ontop"] - summary["fcc"])
        json.dump(results, open(OUT, "w"), indent=2)
        print(f"\nPBE prefers: {pref}  (gap {results['site_gap_eV']:.3f} eV)", flush=True)
        print("experiment: atop; PBE's hollow preference is the CO puzzle", flush=True)


if __name__ == "__main__":
    sys.exit(main())
