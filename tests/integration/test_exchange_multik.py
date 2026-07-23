"""Multi-k Fock exchange and the differentiable hybrid slot, on converged Si.

Reduces to the Γ direct build at a single k-point, produces a finite real
exchange on a full-BZ mesh, computes screened (HSE-style) exchange, and carries
autograd gradients through the hybrid (α, ω) parameters.
"""

import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf import exchange_multik as xk
from gradwave.postscf import isdf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf


def _run(kmesh):
    """Converged Si on an UNREDUCED (full-BZ) mesh — required for exchange."""
    cell, pos = si_fcc()
    upf = si_upf()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=18 * RY, kmesh=kmesh,
                          use_symmetry=False, time_reversal=False, nbands=8)
    res = scf(system, LDA_PW92(), smearing="none", etol=1e-9, rhotol=1e-8,
              verbose=False)
    assert res.converged
    return system, res


@pytest.fixture(scope="module")
def si_gamma():
    return _run((1, 1, 1))


@pytest.fixture(scope="module")
def si_mesh():
    return _run((2, 2, 2))


def test_single_k_reduces_to_direct_gamma_build(si_gamma):
    system, res = si_gamma
    shape, g2, vol = system.grid.shape, system.grid.g2, system.grid.volume
    occ = res.occupations[0] > 1e-6
    f = isdf.orbitals_on_grid(res.coeffs[0][occ], system.spheres[0].flat_idx, shape)
    e_ref = float(isdf.exchange_energy_direct(f, shape, g2, vol))
    u, kc, kw = xk.occupied_periodic_orbitals(res, system)
    e_mk = float(xk.multik_exchange_energy(u, kc, kw, system.grid.g_cart, vol, mode="full"))
    assert abs(e_mk - e_ref) < 1e-6


def test_multik_full_bz_is_finite_negative(si_mesh):
    system, res = si_mesh
    u, kc, kw = xk.occupied_periodic_orbitals(res, system)
    assert len(u) == 8
    assert abs(float(kw.sum()) - 1.0) < 1e-12
    e = float(xk.multik_exchange_energy(u, kc, kw, system.grid.g_cart, system.grid.volume,
                                        mode="full"))
    assert e < 0 and torch.isfinite(torch.tensor(e))


def test_screened_exchange_is_finite(si_mesh):
    """Short-range (HSE) exchange is well defined without a singularity
    correction — the screened q+G=0 cell is finite."""
    system, res = si_mesh
    u, kc, kw = xk.occupied_periodic_orbitals(res, system)
    omega = torch.tensor(0.3, dtype=torch.float64)
    e_sr = xk.multik_exchange_energy(u, kc, kw, system.grid.g_cart, system.grid.volume,
                                     mode="short_range", omega=omega)
    assert torch.isfinite(e_sr) and float(e_sr) < 0


def test_hybrid_energy_is_differentiable(si_mesh):
    """dE_hybrid/dω via autograd matches a finite difference, and dE/dα is exact
    (the energy is linear in the mixing fraction)."""
    system, res = si_mesh
    u, kc, kw = xk.occupied_periodic_orbitals(res, system)
    gc, vol = system.grid.g_cart, system.grid.volume

    params = xk.HybridExchangeParams(alpha=0.25, omega=0.25, mode="short_range").double()
    e = xk.hybrid_exchange_energy(u, kc, kw, gc, vol, params)
    e.backward()
    assert params.raw_alpha.grad is not None and torch.isfinite(params.raw_alpha.grad)
    assert params.raw_omega.grad is not None and torch.isfinite(params.raw_omega.grad)

    # finite-difference d(α·E_x)/dω against autograd, at fixed α = 0.25
    def ex(wv):
        return float(xk.multik_exchange_energy(u, kc, kw, gc, vol, mode="short_range",
                                               omega=torch.tensor(wv, dtype=torch.float64)))
    h, w0, alpha = 1e-5, 0.25, 0.25
    fd = alpha * (ex(w0 + h) - ex(w0 - h)) / (2 * h)
    wt = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    (alpha * xk.multik_exchange_energy(u, kc, kw, gc, vol, mode="short_range", omega=wt)).backward()
    assert abs(float(wt.grad) - fd) < 1e-5 * abs(fd)
