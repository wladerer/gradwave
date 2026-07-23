"""ISDF-K: compressed multi-k exchange against the direct multi-k build.

Reduces to the Γ ISDF build at one k-point, and reproduces the direct
``multik_exchange_energy`` on a full-BZ mesh once the shared interpolation rank
saturates — with the rank as the accuracy knob, and for the screened kernel too.
"""

import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf import exchange_multik as xk
from gradwave.postscf import isdf, isdf_k
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf


def _run(kmesh):
    cell, pos = si_fcc()
    upf = si_upf()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=14 * RY, kmesh=kmesh,
                          use_symmetry=False, time_reversal=False, nbands=8)
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-9, rhotol=1e-8, verbose=False)
    assert res.converged
    return system, res


@pytest.fixture(scope="module")
def si_gamma():
    return _run((1, 1, 1))


@pytest.fixture(scope="module")
def si_mesh():
    return _run((2, 1, 1))


def test_isdf_k_reduces_to_gamma_direct(si_gamma):
    system, res = si_gamma
    shape, g2, vol = system.grid.shape, system.grid.g2, system.grid.volume
    u, kc, kw = xk.occupied_periodic_orbitals(res, system)
    occ = res.occupations[0] > 1e-6
    f = isdf.orbitals_on_grid(res.coeffs[0][occ], system.spheres[0].flat_idx, shape)
    e_ref = float(isdf.exchange_energy_direct(f, shape, g2, vol))
    points, zeta = isdf_k.build_isdf_k(u, 64)
    e_k = float(isdf_k.isdf_k_exchange_energy(u, kc, kw, points, zeta,
                                              system.grid.g_cart, vol, mode="full"))
    assert abs(e_k - e_ref) < 1e-8


def test_isdf_k_matches_direct_multik_at_saturation(si_mesh):
    system, res = si_mesh
    u, kc, kw = xk.occupied_periodic_orbitals(res, system)
    gc, vol = system.grid.g_cart, system.grid.volume
    e_direct = float(xk.multik_exchange_energy(u, kc, kw, gc, vol, mode="full"))
    gen = torch.Generator().manual_seed(0)
    # below saturation: finite error; at saturation: exact
    p_lo, z_lo = isdf_k.build_isdf_k(u, 20, generator=gen)
    e_lo = float(isdf_k.isdf_k_exchange_energy(u, kc, kw, p_lo, z_lo, gc, vol, mode="full"))
    p_hi, z_hi = isdf_k.build_isdf_k(u, 160, generator=gen)
    e_hi = float(isdf_k.isdf_k_exchange_energy(u, kc, kw, p_hi, z_hi, gc, vol, mode="full"))
    assert abs(e_lo - e_direct) > 1e-3
    assert abs(e_hi - e_direct) < 1e-7
    assert abs(e_hi - e_direct) < abs(e_lo - e_direct)


def test_isdf_k_screened_matches_direct(si_mesh):
    system, res = si_mesh
    u, kc, kw = xk.occupied_periodic_orbitals(res, system)
    gc, vol = system.grid.g_cart, system.grid.volume
    omega = torch.tensor(0.3, dtype=torch.float64)
    e_direct = float(xk.multik_exchange_energy(u, kc, kw, gc, vol, mode="short_range", omega=omega))
    points, zeta = isdf_k.build_isdf_k(u, 160)
    e_k = float(isdf_k.isdf_k_exchange_energy(u, kc, kw, points, zeta, gc, vol,
                                              mode="short_range", omega=omega))
    assert abs(e_k - e_direct) < 1e-7
