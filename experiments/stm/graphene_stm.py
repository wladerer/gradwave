"""Tersoff-Hamann STM of monolayer graphene (postscf/stm.py demo).

Graphene is a semimetal with the Dirac point at E_F, so the density of states
vanishes there. A small negative bias images the occupied pi band, where the map
resolves the lattice. The SCF is symmetry-reduced (IBZ); ldos_grid symmetrizes the
map over the space group, so the result matches a full-BZ calculation.
"""

from __future__ import annotations

import numpy as np

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.stm import ldos_grid
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

RY = 13.605693122994
UPF = "tests/fixtures/qe/pseudos/C_ONCV_PBE-1.2.upf"


def main():
    a, c = 2.46, 16.0
    cell = np.array([[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
    frac = np.array([[1 / 3, 1 / 3, 0.5], [2 / 3, 2 / 3, 0.5]])  # honeycomb, 1.42 A bonds
    pos = frac @ cell
    upf = parse_upf(UPF)
    system = setup_system(cell, pos, [0, 0], [upf], ecut=40 * RY, kmesh=(18, 18, 1),
                          use_symmetry=True)
    print(f"# graphene: npw~{system.spheres[0].npw}, nk_ibz={len(system.spheres)}, "
          f"n_ops={system.sym.n_ops}", flush=True)
    res = scf(system, PBE(), smearing="gaussian", width=0.05, max_iter=150,
              etol=1e-7, rhotol=1e-6, verbose=False)
    print(f"# converged={res.converged} in {res.n_iter}, E_F={res.fermi:.4f} eV", flush=True)
    n3 = system.grid.shape[2]
    iz = round(10.0 / c * n3)                                   # tip plane 2 A above sheet
    occ = sum(ldos_grid(res, energy=res.fermi + de, sigma=0.3) for de in (-1.8, -1.4, -1.0, -0.6))
    np.save("/tmp/g_occ.npy", occ[:, :, iz].numpy())
    np.save("/tmp/g_cell.npy", cell)
    np.save("/tmp/g_pos.npy", pos)
    print("# saved occupied-pi STM map", flush=True)


if __name__ == "__main__":
    main()
