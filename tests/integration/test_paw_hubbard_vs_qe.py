"""DFT+U on PAW vs QE (Dudarev, S-metric atomic projection).

Si kjpaw with U = 2 eV on the 3p manifold — unphysical but a sharp
code-vs-code gate at fixed settings (QE 7.5 HUBBARD (atomic) card).
Observed: E_U to 0.008 meV, total to 0.31 meV/atom (the one-center
quadrature residual — the +U part adds nothing), occupations 0.8076 vs
0.808. Conventions that matter (each cost ~100 meV when wrong): RAW
PP_PSWFC orbitals (a PAW pseudo-orbital's plain norm is deliberately != 1;
S supplies the rest), and the QE msh (10 bohr) truncation of the atomic-wfc
radial integrals (psl meshes run to 53 A and the oscillating SBT tail
pollutes the form factors).

Also asserts U=0 reproduces the plain-PAW SCF bit-for-bit (plumbing inert).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_hubbard import HubbardManifold

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994
SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS = np.array([[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]])


@pytest.mark.slow
def test_paw_hubbard_vs_qe():
    torch.set_num_threads(8)
    ref = json.loads((FIX / "si_paw_hubbard_ci" / "reference.json").read_text())
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")

    def make():
        return setup_uspp(SI_CELL, SI_POS, [0, 0], [paw], ecut=45 * RY,
                          kmesh=(2, 2, 2), ecutrho=180 * RY,
                          fft_shape=ref["fft_dims"])

    r = scf_uspp(make(), PBE(), etol=1e-10, rhotol=1e-9, verbose=False,
                 max_iter=50, hubbard=[HubbardManifold(species=0, l=1, u=2.0)])
    assert r["converged"]
    de = abs(float(r["energies"].free_energy) - ref["etot_eV"]) / 2 * 1000
    assert de < 1.0, f"total off by {de:.3f} meV/atom"
    deu = abs(float(r["energies"].hubbard) - ref["hubbard_eV"]) * 1000
    assert deu < 1.0, f"E_U off by {deu:.3f} meV"
    n = r["hub_occ"][0][0]
    eigs = np.linalg.eigvalsh(n.cpu().numpy())
    assert np.abs(eigs - ref["occ_eig_per_spin"]).max() < 2e-3

    # U=0 must be the plain-PAW result exactly
    r0 = scf_uspp(make(), PBE(), etol=1e-10, rhotol=1e-9, verbose=False,
                  max_iter=50)
    ru0 = scf_uspp(make(), PBE(), etol=1e-10, rhotol=1e-9, verbose=False,
                   max_iter=50, hubbard=[HubbardManifold(species=0, l=1, u=0.0)])
    assert abs(float(ru0["energies"].free_energy)
               - float(r0["energies"].free_energy)) < 1e-10
