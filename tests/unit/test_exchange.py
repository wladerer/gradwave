"""Fock exchange operator and ACE factorization (Layer C).

Synthetic orthonormal orbitals on a small Γ box exercise the operator algebra
without an SCF: the operator's energy must match the direct energy build, the
ISDF-accelerated operator must converge to the direct operator as the rank
grows, and the ACE low-rank factorization must reproduce V_x on the occupied
subspace to machine precision.
"""


import numpy as np
import torch

from gradwave.dtypes import CDTYPE
from gradwave.grids import build_fft_grid, build_gsphere
from gradwave.postscf import exchange, isdf
from tests.helpers import RY


def _synthetic(n_orb, ecut, a=6.0, seed=0):
    """Orthonormal random orbitals at Γ: returns (coeffs, flat_idx, shape, g2, vol)."""
    grid = build_fft_grid(a * np.eye(3), ecut)
    sphere = build_gsphere(grid, ecut, k_frac=(0.0, 0.0, 0.0))
    gen = torch.Generator().manual_seed(seed)
    c = torch.randn(sphere.npw, n_orb, dtype=CDTYPE, generator=gen)
    q, _ = torch.linalg.qr(c)                      # columns orthonormal in G
    return q.transpose(0, 1), sphere.flat_idx, grid.shape, grid.g2, grid.volume


def test_operator_energy_matches_energy_build():
    """½ Σ_i ⟨ψ_i|V_x|ψ_i⟩ from the direct operator equals the contracted
    ISDF-convention energy build."""
    coeffs, flat_idx, shape, g2, vol = _synthetic(n_orb=6, ecut=8 * RY)
    psi = exchange.physical_orbitals(coeffs, flat_idx, shape, vol)
    f = isdf.orbitals_on_grid(coeffs, flat_idx, shape)
    vx = exchange.exchange_operator_direct(psi, psi, shape, g2)
    e_op = float(exchange.exchange_energy_from_operator(psi, vx, vol))
    e_build = float(isdf.exchange_energy_direct(f, shape, g2, vol))
    assert abs(e_op - e_build) < 1e-9 * abs(e_build)


def test_physical_orbitals_are_orthonormal():
    coeffs, flat_idx, shape, g2, vol = _synthetic(n_orb=5, ecut=8 * RY)
    psi = exchange.physical_orbitals(coeffs, flat_idx, shape, vol)
    w = vol / psi.shape[1]
    gram = w * (psi.conj() @ psi.transpose(0, 1))
    eye = torch.eye(psi.shape[0], dtype=gram.dtype)
    assert float((gram - eye).abs().max()) < 1e-10


def test_isdf_operator_converges_to_direct():
    """The ISDF-accelerated operator approaches the direct operator as the
    interpolation rank grows, and matches it at saturation."""
    coeffs, flat_idx, shape, g2, vol = _synthetic(n_orb=8, ecut=9 * RY)
    psi = exchange.physical_orbitals(coeffs, flat_idx, shape, vol)
    vx = exchange.exchange_operator_direct(psi, psi, shape, g2)
    gen = torch.Generator().manual_seed(3)
    errs = []
    for n_mu in (30, 60):
        points, zeta = exchange.build_exchange_operator_isdf(psi, shape, g2, n_mu, generator=gen)
        vx_isdf = exchange.exchange_operator_isdf(psi, psi, points, zeta, shape, g2)
        errs.append(float((vx_isdf - vx).norm() / vx.norm()))
    assert errs[0] > errs[1]
    points, zeta = exchange.build_exchange_operator_isdf(psi, shape, g2, n_mu=200, generator=gen)
    vx_sat = exchange.exchange_operator_isdf(psi, psi, points, zeta, shape, g2)
    assert float((vx_sat - vx).norm() / vx.norm()) < 1e-8


def test_ace_reproduces_exchange_on_occupied():
    """V_x^ACE ψ_n = V_x ψ_n exactly for every occupied n (the ACE property),
    and the ACE exchange energy equals the operator energy."""
    coeffs, flat_idx, shape, g2, vol = _synthetic(n_orb=7, ecut=9 * RY)
    psi = exchange.physical_orbitals(coeffs, flat_idx, shape, vol)
    vx = exchange.exchange_operator_direct(psi, psi, shape, g2)
    ace = exchange.build_ace(psi, vx, vol)
    assert ace.rank == psi.shape[0]
    scale = float(vx.abs().max())
    assert float((ace.apply(psi) - vx).abs().max()) < 1e-10 * scale
    e_op = float(exchange.exchange_energy_from_operator(psi, vx, vol))
    assert abs(float(ace.energy(psi)) - e_op) < 1e-9 * abs(e_op)


def test_ace_from_isdf_operator_matches_direct_energy():
    """The full chain ISDF operator → ACE reproduces the direct exchange energy
    once the interpolation rank saturates."""
    coeffs, flat_idx, shape, g2, vol = _synthetic(n_orb=6, ecut=8 * RY)
    psi = exchange.physical_orbitals(coeffs, flat_idx, shape, vol)
    vx = exchange.exchange_operator_direct(psi, psi, shape, g2)
    e_direct = float(exchange.exchange_energy_from_operator(psi, vx, vol))
    gen = torch.Generator().manual_seed(4)
    points, zeta = exchange.build_exchange_operator_isdf(psi, shape, g2, n_mu=200, generator=gen)
    vx_isdf = exchange.exchange_operator_isdf(psi, psi, points, zeta, shape, g2)
    ace = exchange.build_ace(psi, vx_isdf, vol)
    assert abs(float(ace.energy(psi)) - e_direct) < 1e-6 * abs(e_direct)


def test_ace_apply_batched_and_single_agree():
    coeffs, flat_idx, shape, g2, vol = _synthetic(n_orb=5, ecut=8 * RY)
    psi = exchange.physical_orbitals(coeffs, flat_idx, shape, vol)
    vx = exchange.exchange_operator_direct(psi, psi, shape, g2)
    ace = exchange.build_ace(psi, vx, vol)
    batched = ace.apply(psi)
    single = torch.stack([ace.apply(psi[i]) for i in range(psi.shape[0])])
    assert torch.allclose(batched, single, atol=1e-12)
