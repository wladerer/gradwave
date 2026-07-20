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
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.forces import forces as compute_forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


@pytest.fixture(autouse=True)
def _limit_threads():
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


def test_time_reversal_kmesh_reduction():
    """k → −k: on a shifted MP mesh every k has a distinct −k partner, and on
    a rattled P1 cell only time reversal (H(−k) = H(k)*, no spatial symmetry)
    relates them. The TR-halved mesh (2 k, doubled weights) must reproduce
    the full ±k mesh (4 k) exactly — a conjugation/phase-convention slip in
    the sphere, projector, or density assembly shows up at O(1)."""
    lattice = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [1.45, 1.27, 1.41]])  # rattled, P1
    si = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    xc = LDA_PW92()
    kw = dict(ecut=14 * RY, kmesh=(2, 2, 1), kshift=(1, 1, 0))

    sys_half = setup_system(lattice, pos, [0, 0], [si], time_reversal=True, **kw)
    sys_full = setup_system(lattice, pos, [0, 0], [si], time_reversal=False, **kw)
    assert len(sys_half.spheres) == 2 and len(sys_full.spheres) == 4

    res_h = scf(sys_half, xc, **SCF_KW)
    res_f = scf(sys_full, xc, **SCF_KW)
    assert res_h.converged and res_f.converged
    de = abs(float(res_h.energies.free_energy)
             - float(res_f.energies.free_energy)) / 2
    assert de < 5e-8, f"time-reversal reduction changed E by {de:.3e} eV/atom"
    df = np.abs(compute_forces(res_h).numpy()
                - compute_forces(res_f).numpy()).max()
    assert df < 1e-6, f"time-reversal reduction changed forces by {df:.3e}"


def test_cell_rotation_reparameterization():
    """Equivalent-cell re-parameterization: the same crystal described in a
    rigidly rotated Cartesian frame. Every |G|, |k+G|, and interatomic
    distance is unchanged, so E is invariant to solver tolerance and forces
    co-rotate — while every Cartesian intermediate (g_cart, the σ=|∇ρ|²
    chain, Ewald, projector Ylm's) is completely re-indexed. A frame-fixed
    convention leaking in (an axis hard-coded, a non-covariant contraction)
    breaks this at O(1)."""
    rng = np.random.default_rng(7)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    rot = q * np.sign(np.diag(r))[None, :]
    if np.linalg.det(rot) < 0:
        rot[:, 0] = -rot[:, 0]  # proper rotation

    lattice = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [1.45, 1.27, 1.41]])  # rattled, P1
    si = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    xc = LDA_PW92()

    res_a = scf(setup_system(lattice, pos, [0, 0], [si],
                             ecut=14 * RY, kmesh=(2, 1, 1)), xc, **SCF_KW)
    res_b = scf(setup_system(lattice @ rot.T, pos @ rot.T, [0, 0], [si],
                             ecut=14 * RY, kmesh=(2, 1, 1)), xc, **SCF_KW)
    assert res_a.converged and res_b.converged
    de = abs(float(res_a.energies.free_energy)
             - float(res_b.energies.free_energy)) / 2
    assert de < 5e-8, f"frame rotation changed E by {de:.3e} eV/atom"
    f_a = compute_forces(res_a).numpy()
    f_b = compute_forces(res_b).numpy()
    df = np.abs(f_b - f_a @ rot.T).max()
    assert df < 1e-6, f"forces do not co-rotate: {df:.3e} eV/Å"


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
