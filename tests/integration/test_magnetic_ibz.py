"""Magnetic-IBZ SCF equivalence: folded k + Shubnikov symmetrization must
reproduce the full-mesh SCF exactly (same functional, same basis — the fold
is a symmetry statement, not an approximation).

Three rungs, all self-referential:
1. Centrosymmetric L1_0 FePt (SOC), m ∥ [001]: magnetic IBZ = para+TR IBZ
   (inversion is unitary for axial vectors) — validates the unitary fold +
   (ρ, m⃗) symmetrization on the real MAE system. Measured: |dF| = 5.0e-11 eV,
   |M| identical to 5 decimals, same iteration count.
2. POLAR FePt (Pt off the midplane, P4mm — no inversion), m ∥ [001]: the
   anti-unitary ops fold k beyond the unitary set (27 → 6 vs 9 unitary-only
   at (3,3,3)) — validates the −W⁻ᵀ anti-unitary k-action with SOC.
3. Spinor PAW Si with the GREY group (magmoms = 0): every op is both unitary
   and anti-unitary; the spinor PAW loop with magnetic (ρ, m⃗, becsum)
   symmetrizers must match the symmetrized collinear scf_uspp — validates the
   PAW half (MagneticBecsumSymmetrizer incl. the anti-unitary conj path).
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_noncollinear import scf_uspp_noncollinear

RY = 13.605693122994
PSE = "tests/fixtures/qe/pseudos"


def _fept_energy(cell, pos, kmesh, moms, **setup_kw):
    fe = parse_upf(f"{PSE}/Fe_ONCV_PBE_FR-1.0.upf")
    pt = parse_upf(f"{PSE}/Pt_ONCV_PBE_FR-1.0.upf")
    system = setup_system(cell, pos, [0, 1], [fe, pt], ecut=30 * RY,
                          kmesh=kmesh, nbands=30, **setup_kw)
    res = scf_noncollinear(system, NoncollinearXC(LSDA_PW92()),
                           mag_vec_init=moms, smearing="gaussian", width=0.1,
                           etol=1e-9, rhotol=1e-7, max_iter=150,
                           mixing_alpha=0.3, mixing_history=12, verbose=False)
    assert res.converged
    return (float(res.energies.free_energy), np.array(res.mag_vec),
            len(system.spheres))


@pytest.mark.slow
def test_fept_soc_magnetic_ibz():
    a, c = 2.723, 3.712
    cell = np.diag([a, a, c])
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) @ cell
    moms = [[0, 0, 3.0], [0, 0, 0.4]]
    f_full, m_full, nk_full = _fept_energy(cell, pos, (2, 2, 2), moms,
                                           use_symmetry=False,
                                           time_reversal=False)
    f_mag, m_mag, nk_mag = _fept_energy(cell, pos, (2, 2, 2), moms,
                                        use_symmetry=True, magmoms=moms)
    assert (nk_full, nk_mag) == (8, 6)
    assert abs(f_full - f_mag) < 5e-8, f"|dF| = {abs(f_full - f_mag):.2e} eV"
    assert np.linalg.norm(m_full - m_mag) < 1e-3


@pytest.mark.slow
def test_polar_fept_anti_unitary_fold():
    a, c = 2.723, 3.712
    cell = np.diag([a, a, c])
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.45]]) @ cell  # P4mm, no inversion
    moms = [[0, 0, 3.0], [0, 0, 0.4]]
    f_full, m_full, nk_full = _fept_energy(cell, pos, (3, 3, 3), moms,
                                           use_symmetry=False,
                                           time_reversal=False)
    f_mag, m_mag, nk_mag = _fept_energy(cell, pos, (3, 3, 3), moms,
                                        use_symmetry=True, magmoms=moms)
    # 27 → 6 only with the anti-unitary {−W⁻ᵀ}; unitary C4 alone gives 9
    assert (nk_full, nk_mag) == (27, 6)
    assert abs(f_full - f_mag) < 5e-8, f"|dF| = {abs(f_full - f_mag):.2e} eV"
    assert np.linalg.norm(m_full - m_mag) < 1e-3


@pytest.mark.slow
def test_spinor_paw_si_grey_group():
    torch.set_num_threads(8)
    si = parse_upf_paw(f"{PSE}/Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    zeros = [[0.0, 0, 0], [0.0, 0, 0]]

    def system(**kw):
        return setup_uspp(cell, pos, [0, 0], [si], ecut=20 * RY,
                          kmesh=(2, 2, 2), nbands=8, **kw)

    ref = scf_uspp(system(use_symmetry=True), LDA_PW92(), smearing="gaussian",
                   width=0.05, etol=1e-8, rhotol=1e-7, verbose=False)
    nc = scf_uspp_noncollinear(system(use_symmetry=True, magmoms=zeros),
                               LSDA_PW92(), zeros, smearing="gaussian",
                               width=0.05, etol=1e-8, rhotol=1e-7,
                               verbose=False)
    assert ref["converged"] and nc["converged"]
    d = abs(float(nc["energies"].free_energy) - float(ref["energies"].free_energy))
    assert d < 5e-6, f"grey-group spinor PAW vs collinear: |dF| = {d:.2e} eV"
    assert max(abs(x) for x in nc["mag_vec"]) < 1e-6
