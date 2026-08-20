"""Quantized invariants from link overlaps (postscf.kgeometry_topo).

- QWZ Chern insulator: FHS integers across the phase diagram, agreement of
  the three readouts (FHS plaquette sum, WCC winding, Berry-curvature
  integral from milestone-1 qgt) with one sign convention;
- BZ-boundary embedding: link overlaps invariant under integer translation
  of both ks (exact — pins the Miller re-index), FHS identical for a slice
  straddling the zone boundary;
- Si nulls: C = 0 and windingless WCC flow on a generic slice (20³ box, see
  test_kgeometry.py for the grid-parity story);
- differentiable WCC gap: autograd d(gap)/dk_perp matches FD on a 4-band
  model and on the Si Bloch provider;
- Weyl chirality: ±1 through a small cube around a synthetic node.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.kgeometry import qgt
from gradwave.postscf.kgeometry_topo import (
    BlochLinkStates,
    ModelLinkStates,
    chern_fhs,
    wcc_flow,
    wcc_gap,
    weyl_chirality,
)
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf

SX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
SY = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
SZ = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

E1, E2 = (1.0, 0.0), (0.0, 1.0)
VALENCE = [0, 1, 2, 3]
SLICE_ORIGIN = (0.03, 0.06, 0.17)  # generic: off every symmetry line


def qwz(u):
    """QWZ Chern insulator on fractional k: H = sin(2πk₁)σx + sin(2πk₂)σy +
    (u + cos 2πk₁ + cos 2πk₂)σz. Lower band: C = −1 (0<u<2), +1 (−2<u<0),
    0 (|u|>2) in this module's sign convention (C = (1/2π)∫Ω, Ω = −2 Im Q)."""

    def h(k):
        a, b = 2.0 * np.pi * k[0], 2.0 * np.pi * k[1]
        return (
            torch.sin(a).to(torch.complex128) * SX
            + torch.sin(b).to(torch.complex128) * SY
            + (u + torch.cos(a) + torch.cos(b)).to(torch.complex128) * SZ
        )

    return h


def double_qwz(k):
    """Two decoupled QWZ copies (u = 1 topological ⊕ u = 2.5 trivial): the
    two lowest bands are a globally isolated group with total Chern −1."""
    h1, h2 = qwz(1.0)(k), qwz(2.5)(k)
    z = torch.zeros(2, 2, dtype=torch.complex128)
    return torch.cat([torch.cat([h1, z], 1), torch.cat([z, h2], 1)], 0)


@pytest.fixture(scope="module")
def si_res():
    torch.set_num_threads(2)
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [si_upf()], ecut=12 * RY,
                          kmesh=(1, 1, 1), nbands=8, use_symmetry=False,
                          fft_shape=(20, 20, 20))
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    return res


# --------------------------------------------------------------------------- #
# QWZ model: integers, three readouts, one sign convention                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("u,c_expected", [(1.0, -1), (-1.0, 1), (3.0, 0)])
def test_qwz_fhs_chern_phases(u, c_expected):
    res = chern_fhs(ModelLinkStates(qwz(u), [0]), 12, e1=E1, e2=E2, origin=(0, 0))
    assert res.chern == c_expected
    assert res.residual < 1e-9  # exactly quantized (floating error only)


@pytest.mark.parametrize("u", [1.0, -1.0, 3.0])
def test_qwz_wcc_winding_matches_fhs(u):
    prov = ModelLinkStates(qwz(u), [0])
    c = chern_fhs(prov, 12, e1=E1, e2=E2, origin=(0, 0))
    f = wcc_flow(prov, e_loop=E1, e_perp=E2, origin=(0, 0), n_loop=12, n_perp=12)
    assert f.chern == c.chern
    assert f.residual < 1e-9


def test_qwz_chern_matches_berry_curvature_integral():
    # links milestone 1 to milestone 2: C = (1/2π)∫Ω d²k (Riemann on 12²)
    h3 = qwz(1.0)  # qgt passes a 3-vector; the model reads k[0], k[1]
    n = 12
    tot = 0.0
    for i in range(n):
        for j in range(n):
            k = torch.tensor([i / n, j / n, 0.0], dtype=torch.float64)
            tot += (-2.0 * qgt(h3, k, [0]).imag)[0, 1].item()
    c_berry = tot / n**2 / (2.0 * np.pi)
    c_fhs = chern_fhs(ModelLinkStates(h3, [0]), n, e1=E1, e2=E2, origin=(0, 0)).chern
    assert abs(c_berry - c_fhs) < 0.01


def test_fhs_straddling_origin_identical():
    # plaquettes crossing the zone boundary give the same exact integer
    c0 = chern_fhs(ModelLinkStates(qwz(1.0), [0]), 12, e1=E1, e2=E2, origin=(0, 0))
    cs = chern_fhs(
        ModelLinkStates(qwz(1.0), [0]), 12, e1=E1, e2=E2, origin=(0.63, 0.71)
    )
    assert cs.chern == c0.chern == -1
    assert cs.residual < 1e-9


def test_double_qwz_group_chern_adds():
    prov = ModelLinkStates(double_qwz, [0, 1])
    c = chern_fhs(prov, 12, e1=E1, e2=E2, origin=(0, 0))
    f = wcc_flow(prov, e_loop=E1, e_perp=E2, origin=(0, 0), n_loop=12, n_perp=12)
    assert c.chern == f.chern == -1  # C(u=1) + C(u=2.5) = −1 + 0


# --------------------------------------------------------------------------- #
# BZ-boundary embedding on real Si (Miller re-index)                          #
# --------------------------------------------------------------------------- #


def test_bloch_link_translation_invariance(si_res):
    # M(k1+G0, k2+G0) == M(k1, k2): the same folded states are re-labelled by
    # the same integer shift, so the contraction is identical — zero tolerance
    prov = BlochLinkStates(si_res, VALENCE)
    k1 = np.array([0.13, 0.07, 0.17])
    k2 = np.array([0.38, 0.07, 0.17])
    g0 = np.array([1.0, -2.0, 1.0])
    m_a = prov.overlap(k1, k2)
    m_b = prov.overlap(k1 + g0, k2 + g0)
    assert (m_a - m_b).abs().max().item() < 1e-15


def test_si_chern_null_and_boundary_straddle(si_res):
    prov = BlochLinkStates(si_res, VALENCE)
    c_a = chern_fhs(prov, 4, e1=(1, 0, 0), e2=(0, 1, 0), origin=SLICE_ORIGIN)
    c_b = chern_fhs(
        BlochLinkStates(si_res, VALENCE), 4, e1=(1, 0, 0), e2=(0, 1, 0),
        origin=(0.63, 0.71, 0.17),
    )
    assert c_a.chern == c_b.chern == 0  # trivial topology, boundary-independent
    assert c_a.residual < 1e-9 and c_b.residual < 1e-9


def test_si_wcc_flow_windingless(si_res):
    prov = BlochLinkStates(si_res, VALENCE)
    f = wcc_flow(prov, e_loop=(1, 0, 0), e_perp=(0, 1, 0), origin=SLICE_ORIGIN,
                 n_loop=4, n_perp=4)
    assert f.chern == 0
    assert f.residual < 1e-9
    assert f.wcc.shape == (4, len(VALENCE))
    assert np.all((f.wcc >= 0.0) & (f.wcc < 1.0))


# --------------------------------------------------------------------------- #
# differentiable WCC gap                                                      #
# --------------------------------------------------------------------------- #


def test_wcc_gap_gradient_matches_fd_model():
    prov = ModelLinkStates(double_qwz, [0, 1])
    kp = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    start = torch.stack([torch.zeros((), dtype=torch.float64), kp])
    gap = wcc_gap(prov, start, E1, n_loop=12)
    (grad,) = torch.autograd.grad(gap, kp)

    h = 1e-5

    def gap_at(x):
        s = torch.tensor([0.0, x], dtype=torch.float64)
        return wcc_gap(prov, s, E1, n_loop=12).item()

    fd = (gap_at(0.23 + h) - gap_at(0.23 - h)) / (2.0 * h)
    assert np.isfinite(grad.item())
    assert abs(grad.item() - fd) < 1e-6


def test_wcc_gap_gradient_matches_fd_bloch(si_res):
    prov = BlochLinkStates(si_res, VALENCE)
    kp = torch.tensor(0.06, dtype=torch.float64, requires_grad=True)
    start = torch.stack([torch.tensor(0.03, dtype=torch.float64), kp,
                         torch.tensor(0.17, dtype=torch.float64)])
    gap = wcc_gap(prov, start, (1, 0, 0), n_loop=4)
    (grad,) = torch.autograd.grad(gap, kp)

    h = 1e-5

    def gap_at(x):
        s = torch.tensor([0.03, x, 0.17], dtype=torch.float64)
        return wcc_gap(prov, s, (1, 0, 0), n_loop=4).item()

    fd = (gap_at(0.06 + h) - gap_at(0.06 - h)) / (2.0 * h)
    assert np.isfinite(grad.item())
    assert abs(grad.item() - fd) < 1e-6


def test_wcc_gap_single_band_is_full_circle():
    prov = ModelLinkStates(qwz(1.0), [0])
    start = torch.tensor([0.0, 0.23], dtype=torch.float64, requires_grad=True)
    assert wcc_gap(prov, start, E1, n_loop=12).item() == 1.0


# --------------------------------------------------------------------------- #
# Weyl chirality                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sign,q_expected", [(1.0, 1), (-1.0, -1)])
def test_weyl_chirality(sign, q_expected):
    def h(k):
        return sign * (
            k[0].to(torch.complex128) * SX
            + k[1].to(torch.complex128) * SY
            + k[2].to(torch.complex128) * SZ
        )

    q, resid = weyl_chirality(h, [0], (0, 0, 0), 0.3, n=6)
    assert q == q_expected
    assert resid < 1e-9
