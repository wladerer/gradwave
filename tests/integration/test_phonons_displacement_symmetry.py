"""Point-group reduction of the FD phonon displacement set — self-oracle.

The reduced path (`use_displacement_symmetry=True`) runs only the point-group-
irreducible (home atom, axis) displacements and reconstructs the rest of the
force-constant matrix Φ by the crystal group action (the force-completion
transform in `SupercellDisplacementSymmetry`). For a symmorphic cubic monatomic
metal on a symmetry-commensurate grid this is EXACT: reduced Φ and phonon
frequencies must match the full 6·N_prim run to round-off, from the same SCF
force data (only permuted/rotated). fcc Al: 6→2 displacement SCFs.

Slow tier: a small fcc Al 2×2×2 supercell run.
"""

import numpy as np
import pytest
import torch

from gradwave.api import XC_REGISTRY
from gradwave.postscf.phonons_supercell import (
    SupercellDisplacementSymmetry,
    build_supercell,
    dispersion,
    force_constants_home,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.slow


def _al_force_constants(both: bool = True):
    torch.set_num_threads(8)
    a = 4.05
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0]])
    scmap = build_supercell(cell, pos, [0], (2, 2, 2))
    upf = parse_upf(str(PSEUDOS / "Al_ONCV_PBE-1.2.upf"))
    xc = XC_REGISTRY["pbe"]()
    ksuper = (4, 4, 4)

    def make_scf(pos_sc, start_from=None):
        system = setup_system(scmap.cell_super, pos_sc, scmap.species_super,
                              [upf], ecut=20 * RY, kmesh=ksuper,
                              use_symmetry=False)
        return scf(system, xc, nspin=1, smearing="cold", width=0.01,
                   etol=1e-9, rhotol=1e-8, start_from=start_from, verbose=False)

    phi_full = force_constants_home(make_scf, scmap, h=0.01, xc=xc,
                                    use_displacement_symmetry=False)
    phi_red = force_constants_home(make_scf, scmap, h=0.01, xc=xc,
                                   use_displacement_symmetry=True)
    return scmap, phi_full, phi_red


def test_fcc_al_reduced_matches_full():
    scmap, phi_full, phi_red = _al_force_constants()
    sym = SupercellDisplacementSymmetry(scmap)
    # 6→2 SCFs: one irreducible column of three
    assert len(sym.displacements) == 1
    assert len(sym.displacements) < 3 * scmap.n_prim

    # bit-level Φ agreement: same SCF forces, only permuted/rotated
    dphi = float(np.abs(phi_full - phi_red).max())
    assert dphi < 1e-7, f"max|dPhi|={dphi:.2e}"

    masses = np.array([26.98])
    qs = [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0.0], [0.25, 0.25, 0.25]]
    f_full = dispersion(phi_full, scmap, masses, qs)
    f_red = dispersion(phi_red, scmap, masses, qs)
    dfreq = float(np.abs(f_full - f_red).max())
    assert dfreq < 1e-4, f"max|dFreq|={dfreq:.2e} cm^-1"
