"""Self-consistent hybrid (PBE0-form) SCF at Γ — exchange acting in the loop.

At α = 0 the hybrid SCF is exactly a PBE SCF (the reduction gate). At α > 0 it
converges, the Fock energy term is consistent with the ACE energy on the
converged orbitals, and the gap opens relative to PBE — the operator is affecting
the eigenvalues, i.e. exchange is self-consistent.
"""

import pytest

from gradwave.core.xc.pbe import PBE
from gradwave.postscf import exchange
from gradwave.postscf.hybrid import hybrid_scf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf


def _system():
    cell, pos = si_fcc()
    upf = si_upf()
    return setup_system(cell, pos, [0, 0], [upf], ecut=18 * RY, kmesh=(1, 1, 1), nbands=8)


@pytest.fixture(scope="module")
def pbe_ref():
    res = scf(_system(), PBE(), smearing="none", etol=1e-9, rhotol=1e-8, verbose=False)
    assert res.converged
    return res


def test_alpha_zero_reduces_to_pbe(pbe_ref):
    res_h = hybrid_scf(_system(), alpha=0.0, smearing="none", etol=1e-9, rhotol=1e-8,
                       verbose=False)
    assert res_h.converged
    assert float(res_h.energies.fock) == 0.0
    assert abs(float(res_h.energies.free_energy) - float(pbe_ref.energies.free_energy)) < 1e-8


def test_pbe0_converges_and_opens_gap(pbe_ref):
    system = _system()
    res_h = hybrid_scf(system, alpha=0.25, smearing="none", etol=1e-9, rhotol=1e-8,
                       verbose=False, max_iter=80)
    assert res_h.converged
    # the Fock term is negative and a physical fraction of the exchange
    assert float(res_h.energies.fock) < 0
    # PBE0 opens the Γ gap (4 occupied bands) relative to PBE
    gap_pbe = float(pbe_ref.eigenvalues[0][4] - pbe_ref.eigenvalues[0][3])
    gap_h = float(res_h.eigenvalues[0][4] - res_h.eigenvalues[0][3])
    assert gap_h > gap_pbe + 0.1


def test_fock_energy_matches_ace_on_converged_orbitals():
    system = _system()
    res_h = hybrid_scf(system, alpha=0.25, smearing="none", etol=1e-9, rhotol=1e-8,
                       verbose=False, max_iter=80)
    occ = res_h.occupations[0] > 1e-6
    psi = exchange.physical_orbitals(res_h.coeffs[0][occ], system.spheres[0].flat_idx,
                                     system.grid.shape, system.grid.volume)
    vx = exchange.exchange_operator_direct(psi, psi, system.grid.shape, system.grid.g2)
    ace = exchange.build_ace(psi, vx, system.grid.volume)
    # nspin=1: physical exchange is 2× the spatial-orbital ACE energy
    e_expected = 0.25 * 2.0 * float(ace.energy(psi))
    assert abs(float(res_h.energies.fock) - e_expected) < 1e-7
