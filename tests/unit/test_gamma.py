"""Gamma-real path gated against the complex path to machine precision.

The Gamma specialization stores the half plane-wave sphere with real
wavefunctions and runs the local term on a real FFT. Every test here compares
it against the existing complex machinery restricted to the Gamma point, so a
regression that changes the physics fails immediately.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.batch import BatchedHamiltonian, projectors_b
from gradwave.core.gamma import (
    build_gamma_basis,
    davidson_gamma,
    embed_real,
    full_to_half,
    GammaHamiltonian,
    half_to_full,
    metric_inner,
    unembed_real,
)
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.solvers.davidson import davidson_batched

pytestmark = pytest.mark.standard

RY = 13.605693122994
FIX = Path(__file__).parents[1] / "fixtures" / "qe"


@pytest.fixture(scope="module")
def o2_gamma():
    """O2 molecule in a box, Gamma-only, converged with an NC ONCV pseudo."""
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "O_ONCV_PBE-1.2.upf")
    a = 8.0
    cell = np.diag([a, a, a])
    d = 1.21
    pos = np.array([[a / 2, a / 2, a / 2 - d / 2], [a / 2, a / 2, a / 2 + d / 2]])
    system = setup_system(cell, pos, [0, 0], [upf], ecut=35 * RY,
                          kmesh=(1, 1, 1), nbands=8)
    res = scf(system, PBE(), smearing="gaussian", width=0.2, etol=1e-9,
              rhotol=1e-8, verbose=False, max_iter=60)
    assert res.converged
    grid, sphere, bk = system.grid, system.spheres[0], system.batch
    gb = build_gamma_basis(sphere, grid.shape)
    p_full = projectors_b(bk, system.positions)[0]
    return dict(system=system, res=res, grid=grid, sphere=sphere, bk=bk,
                gb=gb, veff=res.v_eff, p_full=p_full, dij=bk.dij_full)


def _herm_partner(sphere, shape):
    """Map each sphere index to the sphere index of -G (for Hermitian checks)."""
    miller = sphere.miller.cpu().numpy()
    box_shape = np.array(shape)
    lut = {tuple(m % box_shape): g for g, m in enumerate(miller)}
    return np.array([lut[tuple((-m) % box_shape)] for m in miller])


def test_basis_closure_and_sizes(o2_gamma):
    gb, sphere = o2_gamma["gb"], o2_gamma["sphere"]
    assert gb.nhalf == (gb.npw + 1) // 2
    assert gb.npw == sphere.npw
    # metric weight: 1 on the G=0 slot, 2 elsewhere
    assert float(gb.metric_w[0]) == 1.0
    assert torch.all(gb.metric_w[1:] == 2.0)


def test_half_full_roundtrip(o2_gamma):
    gb = o2_gamma["gb"]
    torch.manual_seed(1)
    chalf = torch.randn(5, gb.nhalf, dtype=torch.complex128)
    chalf[:, 0] = chalf[:, 0].real.to(torch.complex128)  # G=0 real
    back = full_to_half(gb, half_to_full(gb, chalf))
    assert torch.allclose(back, chalf, atol=1e-15)


def test_full_sphere_is_hermitian(o2_gamma):
    gb, sphere, grid = o2_gamma["gb"], o2_gamma["sphere"], o2_gamma["grid"]
    torch.manual_seed(2)
    chalf = torch.randn(3, gb.nhalf, dtype=torch.complex128)
    chalf[:, 0] = chalf[:, 0].real.to(torch.complex128)
    cfull = half_to_full(gb, chalf)
    partner = _herm_partner(sphere, grid.shape)
    err = (cfull - cfull[:, partner].conj()).abs().max()
    assert float(err) < 1e-15


def test_irfftn_matches_ifftn(o2_gamma):
    """The real half-box transform reproduces the complex box transform."""
    gb, sphere, grid = o2_gamma["gb"], o2_gamma["sphere"], o2_gamma["grid"]
    shape = grid.shape
    n = shape[0] * shape[1] * shape[2]
    torch.manual_seed(3)
    chalf = torch.randn(4, gb.nhalf, dtype=torch.complex128)
    chalf[:, 0] = chalf[:, 0].real.to(torch.complex128)
    cfull = half_to_full(gb, chalf)
    box = torch.zeros(4, n, dtype=torch.complex128)
    box.index_add_(1, sphere.flat_idx, cfull)
    box = box.reshape(4, *shape)
    psi_c = torch.fft.ifftn(box, dim=(-3, -2, -1))
    psi_r = torch.fft.irfftn(box[..., : gb.nh3], s=shape, dim=(-3, -2, -1))
    assert float(psi_c.imag.abs().max()) < 1e-14  # Hermitian => real
    assert float((psi_r - psi_c.real).abs().max()) < 1e-14


def test_apply_matches_complex(o2_gamma):
    """H-apply equivalence: the whole point of the specialization."""
    d = o2_gamma
    gb, bk, grid = d["gb"], d["bk"], d["grid"]
    torch.manual_seed(4)
    chalf = torch.randn(8, gb.nhalf, dtype=torch.complex128)
    chalf[:, 0] = chalf[:, 0].real.to(torch.complex128)
    cfull = half_to_full(gb, chalf)

    gh = GammaHamiltonian(gb, d["veff"], d["p_full"], d["dij"])
    hb = BatchedHamiltonian(bk, grid.shape, d["veff"], d["p_full"][None])
    out_gamma = gh.apply(chalf)
    out_full = hb.apply(cfull[None])[0]
    # the complex result must itself be Hermitian-symmetric, else the
    # half-sphere projection would lose information
    partner = _herm_partner(d["sphere"], grid.shape)
    assert float((out_full - out_full[:, partner].conj()).abs().max()) < 1e-11
    assert float((out_gamma - full_to_half(gb, out_full)).abs().max()) < 1e-11


def test_embed_roundtrip_and_metric(o2_gamma):
    """The real embedding is invertible and turns the metric into a dot product."""
    gb = o2_gamma["gb"]
    torch.manual_seed(5)
    a = torch.randn(6, gb.nhalf, dtype=torch.complex128)
    b = torch.randn(6, gb.nhalf, dtype=torch.complex128)
    a[:, 0] = a[:, 0].real.to(torch.complex128)
    b[:, 0] = b[:, 0].real.to(torch.complex128)
    assert torch.allclose(unembed_real(gb, embed_real(gb, a)), a, atol=1e-15)
    dot = embed_real(gb, a) @ embed_real(gb, b).T
    assert torch.allclose(dot, metric_inner(gb, a, b), atol=1e-13)


def test_eigenvalues_match_complex(o2_gamma):
    """Frozen-potential eigenvalues match the complex Davidson to ~machine eps."""
    d = o2_gamma
    gb, bk, grid = d["gb"], d["bk"], d["grid"]
    nb = d["system"].nbands
    gh = GammaHamiltonian(gb, d["veff"], d["p_full"], d["dij"])
    hb = BatchedHamiltonian(bk, grid.shape, d["veff"], d["p_full"][None])

    x0h = torch.zeros(nb, gb.nhalf, dtype=torch.complex128)
    order = torch.argsort(gb.t_half)
    for i in range(nb):
        x0h[i, order[i]] = 1.0
    gres = davidson_gamma(gh, x0h, tol=1e-9, max_iter=80)

    x0c = torch.zeros(1, nb, bk.npw_max, dtype=torch.complex128)
    torder = torch.argsort(bk.t[0])
    for i in range(nb):
        x0c[0, i, torder[i]] = 1.0
    cres = davidson_batched(hb.apply, x0c, bk.t, bk.mask, tol=1e-9, max_iter=80)

    assert float((gres.eigenvalues - cres.eigenvalues[0]).abs().max()) < 1e-8
    g = metric_inner(gb, gres.eigenvectors, gres.eigenvectors)
    assert float((g - torch.eye(nb, dtype=g.dtype)).abs().max()) < 1e-10
