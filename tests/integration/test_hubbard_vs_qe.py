"""DFT+U validation: NiO AFM-II with Hubbard U on the Ni 3d manifold vs QE 7.5
(HUBBARD `atomic` projectors, same PseudoDojo pseudos, cell, ecut, k-mesh,
smearing). The physical +U observables — Hubbard energy, occupation matrix,
and magnetization — must match QE; the absolute total energy carries a
constant PseudoDojo NLCC/semicore reference offset that cancels in differences,
so it is NOT asserted here (see reference.json note).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.hubbard import HubbardManifold
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994


@pytest.mark.slow
def test_nio_afm_hubbard_vs_qe():
    torch.set_num_threads(8)
    ref = json.load(open(FIX / "nio_afm_u_ci" / "reference.json"))
    cell = np.array(ref["cell_angstrom"])
    frac = np.array(ref["positions_crystal"])
    ni = parse_upf(FIX / "pseudos" / ref["pseudos"]["Ni"])
    o = parse_upf(FIX / "pseudos" / ref["pseudos"]["O"])
    system = setup_system(cell, frac @ cell, [0, 0, 1, 1], [ni, o],
                          ecut=ref["ecutwfc_ry"] * RY, kmesh=tuple(ref["kmesh"]),
                          nbands=40)
    res = scf(system, SpinPBE(), smearing=ref["smearing"],
              width=ref["degauss_ry"] * RY, etol=1e-6, rhotol=1e-5, verbose=False,
              nspin=2, start_mag=[+0.5, -0.5, 0, 0],
              hubbard=[HubbardManifold(species=0, l=2, u=ref["U_eV"], j=0.0)],
              max_iter=100)
    assert res.converged

    e_u = float(res.energies.hubbard)
    assert abs(e_u - ref["hubbard_energy_eV"]) < 0.02, e_u  # E_U within 20 meV

    # occupation matrix: the two Ni are opposite spin (AFM), traces vs QE
    up1 = float(torch.trace(res.hub_occ[0][0]).real)
    dn1 = float(torch.trace(res.hub_occ[1][0]).real)
    up2 = float(torch.trace(res.hub_occ[0][1]).real)
    dn2 = float(torch.trace(res.hub_occ[1][1]).real)
    assert abs(up1 - ref["ni_d_occ_up"]) < 0.02
    assert abs(dn1 - ref["ni_d_occ_dn"]) < 0.02
    assert abs((up2 - dn2) + (up1 - dn1)) < 1e-3  # exactly antiferromagnetic
    assert abs((up1 + dn1) - ref["ni_d_total"]) < 0.02

    # magnetization (spin-density integral)
    assert abs(res.mag_abs - ref["abs_magnetization_uB"]) < 0.05
    assert abs(res.mag_total) < 1e-3  # AFM: net zero
