"""ISDF-compressed Fock exchange against a direct plane-wave Fock build, on the
orbitals of a converged Γ-point SCF.

The direct build is the O(N²) pair-FFT reference; the ISDF build compresses the
pairs onto interpolation vectors. Both use the same G=0-excluded Coulomb
convention, so their agreement isolates the ISDF rank truncation. This is the
milli-eV validation gate the exact-exchange work in docs/ideas.md is sequenced
behind.
"""

import pytest

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf import isdf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf


@pytest.fixture(scope="module")
def si_gamma():
    """Converged Γ-point diamond-Si primitive cell (4 occupied bands)."""
    cell, pos = si_fcc()
    upf = si_upf()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=20 * RY,
                          kmesh=(1, 1, 1), nbands=8)
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-9, rhotol=1e-8,
              verbose=False)
    assert res.converged
    occ = res.occupations[0] > 1e-6
    phi_r = isdf.orbitals_on_grid(res.coeffs[0][occ], system.spheres[0].flat_idx,
                                  system.grid.shape)
    return phi_r, system.grid.shape, system.grid.g2, system.grid.volume


def test_isdf_matches_direct_fock_at_saturation(si_gamma):
    """At a rank past the pair space, ISDF exchange ≡ the direct Fock build to
    far below a milli-eV (machine precision)."""
    phi_r, shape, g2, volume = si_gamma
    e_direct = float(isdf.exchange_energy_direct(phi_r, shape, g2, volume))
    ex = isdf.build_exchange(phi_r, shape, g2, volume, n_mu=32)
    e_isdf = float(ex.energy())
    # 4 real Γ orbitals -> co-density rank 4*5/2 = 10, saturated by 32 points
    assert ex.n_mu <= 10
    assert abs(e_isdf - e_direct) < 1e-6, f"ISDF {e_isdf} vs direct {e_direct}"


def test_isdf_rank_is_the_convergence_knob(si_gamma):
    """A rank below the pair space gives a finite error that vanishes once the
    rank saturates — the rank is the accuracy parameter."""
    phi_r, shape, g2, volume = si_gamma
    e_direct = float(isdf.exchange_energy_direct(phi_r, shape, g2, volume))
    coarse = isdf.build_exchange(phi_r, shape, g2, volume, n_mu=6)
    fine = isdf.build_exchange(phi_r, shape, g2, volume, n_mu=32)
    err_coarse = abs(float(coarse.energy()) - e_direct)
    err_fine = abs(float(fine.energy()) - e_direct)
    assert err_coarse > 1e-3          # truncated rank is visibly approximate
    assert err_fine < 1e-6            # saturated rank is exact
    assert err_fine < err_coarse


def test_direct_fock_exchange_is_negative_and_sized(si_gamma):
    """Sanity: the exchange energy is negative and of a physical magnitude."""
    phi_r, shape, g2, volume = si_gamma
    e_direct = float(isdf.exchange_energy_direct(phi_r, shape, g2, volume))
    assert e_direct < 0
    assert 1.0 < abs(e_direct) < 1e3
