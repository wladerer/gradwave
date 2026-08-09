"""Tersoff-Hamann STM of monolayer graphene (postscf/stm.py demo).

Runs a real gradwave SCF of graphene, then images the Fermi-level LDOS on a tip
plane ~2 A above the carbon sheet -> the iconic honeycomb STM pattern. Saves the
LDOS map (.npy) and a tiled PNG.
"""
from __future__ import annotations

import sys

import numpy as np

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.stm import stm_constant_height
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

RY = 13.605693122994
UPF = "tests/fixtures/qe/pseudos/C_ONCV_PBE-1.2.upf"

def main():
    a, c = 2.46, 16.0
    cell = np.array([[a, 0, 0], [a/2, a*np.sqrt(3)/2, 0], [0, 0, c]])
    frac = np.array([[1/3, 2/3, 0.5], [2/3, 1/3, 0.5]])   # honeycomb A/B sublattice
    pos = frac @ cell
    upf = parse_upf(UPF)
    ecut = float(sys.argv[1]) if len(sys.argv) > 1 else 35.0
    nk = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    print(f"# graphene STM: a={a} A, vacuum c={c} A, ecut={ecut} Ry, k=({nk},{nk},1)", flush=True)
    system = setup_system(cell, pos, [0, 0], [upf], ecut=ecut*RY, kmesh=(nk, nk, 1),
                          use_symmetry=True)
    print(f"# npw~{system.spheres[0].npw}, nk_ibz={len(system.spheres)}, nbands={system.nbands}",
          flush=True)
    res = scf(system, PBE(), smearing="gaussian", width=0.1, max_iter=120,
              etol=1e-7, rhotol=1e-6, verbose=True)
    print(f"# converged={res.converged} in {res.n_iter} iters, E_F={res.fermi:.4f} eV", flush=True)

    img, z_tip = stm_constant_height(res, height=2.0, energy=res.fermi, sigma=0.3)
    arr = img.detach().cpu().numpy()
    np.save("/tmp/graphene_stm.npy", arr)
    print(f"# STM map at z_tip={z_tip:.2f} A: shape={arr.shape}, "
          f"contrast max/min={arr.max()/max(arr.min(),1e-30):.2f}", flush=True)
    np.save("/tmp/graphene_cell.npy", cell)
    np.save("/tmp/graphene_pos.npy", pos)
    print("EXIT_OK", flush=True)

if __name__ == "__main__":
    main()
