"""Fock exchange operator, ISDF acceleration, and ACE on converged Γ Si orbitals.

Ties the operator/ACE layer (postscf/exchange.py) to a real SCF: the operator's
energy must match the direct energy build, the ISDF operator must saturate to
the direct operator, and ACE must reproduce V_x on the occupied subspace. This
is the operator that a hybrid-functional SCF would carry.
"""

import pytest

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf import exchange, isdf
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo, si_fcc


@pytest.fixture(scope="module")
def si_gamma():
    cell, pos = si_fcc()
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    system = setup_system(cell, pos, [0, 0], [upf], ecut=20 * RY,
                          kmesh=(1, 1, 1), nbands=8)
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-9, rhotol=1e-8,
              verbose=False)
    assert res.converged
    occ = res.occupations[0] > 1e-6
    coeffs = res.coeffs[0][occ]
    flat_idx, shape = system.spheres[0].flat_idx, system.grid.shape
    return coeffs, flat_idx, shape, system.grid.g2, system.grid.volume


def test_operator_energy_matches_energy_build(si_gamma):
    coeffs, flat_idx, shape, g2, vol = si_gamma
    psi = exchange.physical_orbitals(coeffs, flat_idx, shape, vol)
    f = isdf.orbitals_on_grid(coeffs, flat_idx, shape)
    vx = exchange.exchange_operator_direct(psi, psi, shape, g2)
    e_op = float(exchange.exchange_energy_from_operator(psi, vx, vol))
    e_build = float(isdf.exchange_energy_direct(f, shape, g2, vol))
    assert abs(e_op - e_build) < 1e-6


def test_isdf_operator_saturates_to_direct(si_gamma):
    coeffs, flat_idx, shape, g2, vol = si_gamma
    psi = exchange.physical_orbitals(coeffs, flat_idx, shape, vol)
    vx = exchange.exchange_operator_direct(psi, psi, shape, g2)
    points, zeta = exchange.build_exchange_operator_isdf(psi, shape, g2, n_mu=32)
    vx_isdf = exchange.exchange_operator_isdf(psi, psi, points, zeta, shape, g2)
    assert float((vx_isdf - vx).norm() / vx.norm()) < 1e-8


def test_ace_reproduces_and_energy_exact(si_gamma):
    coeffs, flat_idx, shape, g2, vol = si_gamma
    psi = exchange.physical_orbitals(coeffs, flat_idx, shape, vol)
    vx = exchange.exchange_operator_direct(psi, psi, shape, g2)
    ace = exchange.build_ace(psi, vx, vol)
    assert ace.rank == psi.shape[0]
    assert float((ace.apply(psi) - vx).abs().max()) < 1e-9 * float(vx.abs().max())
    e_op = float(exchange.exchange_energy_from_operator(psi, vx, vol))
    assert abs(float(ace.energy(psi)) - e_op) < 1e-6
