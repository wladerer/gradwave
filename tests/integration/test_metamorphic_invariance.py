"""Metamorphic invariances through the full SCF (Tier 1, docs/verification.md).

Exact identities of the theory under input transformations, no reference
code as oracle.

Permutation: relabeling atoms (and reordering the species list) leaves the
Hamiltonian identical up to summation order — the identity is exact to SCF
tolerance at ANY cutoff. Tested tight on heteropolar GaAs so a projector or
species-indexing bug cannot hide.

Translation: rigidly shifting all atoms by a grid-incommensurate vector
transforms every G-space term by exact phases; the only translation-variant
piece is the aliasing (egg-box) error of the nonlinear XC quadrature, since
ρ^(4/3) is not band-limited. Measured floors (LDA, rattled cells):
Si 5.6e-7 eV/atom at 14 Ry, 4.6e-6 at 20 Ry (non-monotonic — the minimal
FFT box changes shape with ecut); GaAs (semicore Ga-3d) 9.1e-5 at 25 Ry
→ 2.6e-5 at 40 Ry → 2.5e-6 at 60 Ry. A convention/phase bug would violate
invariance at orders of magnitude above these floors, so the test asserts
a bound just above the measured egg-box, not zero.
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.forces import forces as compute_forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994

torch.set_num_threads(4)

SCF_KW = dict(smearing="gaussian", width=0.1,
              etol=1e-10, rhotol=1e-9, diago_tol=1e-12, verbose=False)


def test_permutation_invariance_exact():
    a = 5.65
    lattice = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [1.48, 1.40, 1.39]])  # rattled, P1
    ga = parse_upf(FIX / "pseudos" / "Ga_ONCV_PBE-1.2.upf")
    ars = parse_upf(FIX / "pseudos" / "As_ONCV_PBE-1.2.upf")
    xc = LDA_PW92()

    res_a = scf(setup_system(lattice, pos, [0, 1], [ga, ars],
                             ecut=25 * RY, kmesh=(2, 1, 1)), xc, **SCF_KW)
    res_b = scf(setup_system(lattice, np.ascontiguousarray(pos[::-1]), [1, 0],
                             [ga, ars], ecut=25 * RY, kmesh=(2, 1, 1)),
                xc, **SCF_KW)
    assert res_a.converged and res_b.converged
    de = abs(float(res_a.energies.free_energy)
             - float(res_b.energies.free_energy)) / 2
    assert de < 5e-8, f"permutation changed E by {de:.3e} eV/atom"
    f_a = compute_forces(res_a).numpy()
    f_b = compute_forces(res_b).numpy()
    assert np.abs(f_b - f_a[::-1]).max() < 1e-6


def test_translation_invariance_at_egg_box_floor():
    lattice = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [1.45, 1.27, 1.41]])  # rattled, P1
    t = np.array([0.7137, -0.2911, 0.4302])  # grid-incommensurate
    si = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    xc = LDA_PW92()

    res_a = scf(setup_system(lattice, pos, [0, 0], [si],
                             ecut=20 * RY, kmesh=(2, 1, 1)), xc, **SCF_KW)
    res_b = scf(setup_system(lattice, pos + t, [0, 0], [si],
                             ecut=20 * RY, kmesh=(2, 1, 1)), xc, **SCF_KW)
    assert res_a.converged and res_b.converged
    de = abs(float(res_a.energies.free_energy)
             - float(res_b.energies.free_energy)) / 2
    # measured XC egg-box at this grid: 4.6e-6 eV/atom / 1.5e-4 eV/Å
    assert de < 2e-5, f"translation changed E by {de:.3e} eV/atom"
    f_a = compute_forces(res_a).numpy()
    f_b = compute_forces(res_b).numpy()
    assert np.abs(f_b - f_a).max() < 1e-3
