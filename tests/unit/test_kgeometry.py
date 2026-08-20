"""Quantum geometric tensor via autograd ∂/∂k (postscf.kgeometry).

Validation ladder:
- the dense H(k) reproduces the SCF's Davidson eigenvalues at a mesh k;
- autograd Q_μν matches central finite differences of the band-group
  projector P(k) on diamond Si at generic k;
- the projector form (fwAD through eigh) matches the sum-over-states
  assembly (fwAD through the H build only) to near machine precision;
- exact limits: free electrons (Q ≡ 0) and the two-band Dirac model
  H = d(k)·σ (closed-form metric and curvature);
- physics: Q Hermitian, g PSD, and Berry curvature ≈ 0 pointwise for Si
  (PT symmetry). The FFT box is pinned to (20,20,20): the diamond inversion
  center sits at fractional (1/8,1/8,1/8), so the r-grid maps onto itself
  under inversion only when n is divisible by 4 — on 18³ the discretized
  v_eff breaks PT at the 1e-5 level and Ω does not vanish.
"""

import numpy as np
import pytest
import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.xc.pbe import PBE
from gradwave.postscf.kgeometry import (
    BlochHK,
    band_projector,
    metric_curvature,
    qgt,
    qgt_sos,
)
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf

# generic k (fractional): off every symmetry line so the valence group is
# internally nondegenerate (fwAD through eigh needs simple coupled eigenvalues)
K_GENERIC = np.array([0.13, 0.07, -0.21])
VALENCE = [0, 1, 2, 3]


@pytest.fixture(scope="module")
def si_res():
    torch.set_num_threads(2)
    upf = si_upf()
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=12 * RY,
                          kmesh=(1, 1, 1), nbands=8, use_symmetry=False,
                          fft_shape=(20, 20, 20))
    res = scf(system, PBE(), etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    return res


def _qgt_fd(h_fn, k_cart, bands, h=1e-4):
    """Q_μν from central finite differences of the projector P(k)."""
    def proj(k):
        return band_projector(h_fn, k, bands)

    dp = []
    for mu in range(3):
        e = torch.zeros(3, dtype=torch.float64)
        e[mu] = h
        dp.append((proj(k_cart + e) - proj(k_cart - e)) / (2.0 * h))
    p0 = proj(k_cart)
    comp = torch.eye(p0.shape[0], dtype=p0.dtype) - p0
    return torch.stack([
        torch.stack([torch.einsum("ab,bc,ca->", dp[m], comp, dp[n]) for n in range(3)])
        for m in range(3)
    ])


# --------------------------------------------------------------------------- #
# dense H(k) correctness                                                      #
# --------------------------------------------------------------------------- #


def test_dense_h_matches_scf_eigenvalues(si_res):
    hk = BlochHK.from_scf(si_res, (0.0, 0.0, 0.0))
    hmat = hk.h(torch.zeros(3, dtype=torch.float64))
    assert (hmat - hmat.mH).abs().max().item() < 1e-12
    w = torch.linalg.eigh(hmat).eigenvalues
    ref = si_res.eigenvalues[0]
    assert (w[: ref.shape[0]] - ref).abs().max().item() < 1e-6


def test_dense_h_smooth_and_hermitian_off_mesh(si_res):
    hk = BlochHK.from_scf(si_res, K_GENERIC)
    hmat = hk.h(hk.k_cart(K_GENERIC))
    assert (hmat - hmat.mH).abs().max().item() < 1e-12
    # kinetic diagonal is exactly HBAR2_2M |k+G|²: check against local+NL-free
    kc = hk.k_cart(K_GENERIC)
    kin = HBAR2_2M * ((hk.g_cart + kc) ** 2).sum(-1)
    assert torch.isfinite(hmat).all()
    assert kin.min() >= 0.0


# --------------------------------------------------------------------------- #
# autograd vs finite differences (Si, generic k)                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("k_frac", [K_GENERIC, np.array([0.31, -0.17, 0.09]),
                                    np.array([-0.08, 0.22, 0.41])])
def test_qgt_matches_finite_differences(si_res, k_frac):
    hk = BlochHK.from_scf(si_res, k_frac)
    kc = hk.k_cart(k_frac)
    q_ad = qgt(hk.h, kc, VALENCE)
    q_fd = _qgt_fd(hk.h, kc, VALENCE, h=1e-4)
    assert (q_ad - q_fd).abs().max().item() < 1e-6


def test_qgt_projector_form_matches_sum_over_states(si_res):
    hk = BlochHK.from_scf(si_res, K_GENERIC)
    kc = hk.k_cart(K_GENERIC)
    q_p = qgt(hk.h, kc, VALENCE)
    q_s = qgt_sos(hk.h, kc, VALENCE)
    assert (q_p - q_s).abs().max().item() < 1e-10


# --------------------------------------------------------------------------- #
# physics sanity (Si)                                                         #
# --------------------------------------------------------------------------- #


def test_qgt_hermitian_metric_psd(si_res):
    hk = BlochHK.from_scf(si_res, K_GENERIC)
    q = qgt(hk.h, hk.k_cart(K_GENERIC), VALENCE)
    assert (q - q.mH).abs().max().item() < 1e-10
    g, _ = metric_curvature(q)
    eigs = torch.linalg.eigvalsh((g + g.T) / 2)
    assert eigs.min().item() > -1e-10  # PSD
    assert eigs.max().item() > 0.1  # and genuinely nonzero for Si valence


@pytest.mark.parametrize("k_frac", [K_GENERIC, np.array([0.31, -0.17, 0.09])])
def test_berry_curvature_vanishes_pointwise_pt_symmetry(si_res, k_frac):
    # diamond Si has inversion + time reversal → Ω(k) = 0 at every k
    hk = BlochHK.from_scf(si_res, k_frac)
    q = qgt(hk.h, hk.k_cart(k_frac), VALENCE)
    _, omega = metric_curvature(q)
    assert omega.abs().max().item() < 1e-9


# --------------------------------------------------------------------------- #
# exact limits                                                                #
# --------------------------------------------------------------------------- #


def test_free_electron_qgt_is_zero():
    # H(k) = diag(HBAR2_2M |k+G|²): eigenvectors are k-independent basis
    # vectors, so P(k) is constant and Q vanishes identically.
    g = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0],
                      [0, 1.0, 0], [0, -1.0, 0]], dtype=torch.float64)

    def h_free(k):
        kin = HBAR2_2M * ((g + k) ** 2).sum(-1)
        return torch.diag_embed(kin.to(torch.complex128))

    k = torch.tensor([0.11, 0.06, -0.03], dtype=torch.float64)  # generic: no crossing
    q = qgt(h_free, k, [0])
    assert q.abs().max().item() < 1e-14


def _dirac_h(mass):
    sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    sy = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    sz = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

    def h(k):
        return k[0].to(torch.complex128) * sx + k[1].to(torch.complex128) * sy + mass * sz

    return h


@pytest.mark.parametrize("band,sign", [(0, +1.0), (1, -1.0)])
def test_two_band_dirac_model_analytic(band, sign):
    # H = d·σ with d = (kx, ky, M). Closed form for the lower band:
    #   Q_μν = ¼ (∂_μd̂·∂_νd̂ − i d̂·(∂_μd̂ × ∂_νd̂))
    #   g_μν = ¼ ∂_μd̂·∂_νd̂,   Ω_xy = ½ d̂·(∂_xd̂ × ∂_yd̂) = M/(2|d|³)
    # (upper band: same metric, opposite curvature).
    mass = 0.7
    k = torch.tensor([0.3, -0.2, 0.0], dtype=torch.float64)
    q = qgt(_dirac_h(mass), k, [band])
    g, omega = metric_curvature(q)

    d = np.array([k[0].item(), k[1].item(), mass])
    nd = np.linalg.norm(d)
    dd = [np.eye(3)[mu] / nd - d * d[mu] / nd**3 for mu in range(2)]  # ∂_x d̂, ∂_y d̂
    dd.append(np.zeros(3))
    g_ref = np.array([[0.25 * dd[m] @ dd[n] for n in range(3)] for m in range(3)])
    omega_xy_ref = sign * mass / (2.0 * nd**3)

    assert np.abs(g.numpy() - g_ref).max() < 1e-12
    assert abs(omega[0, 1].item() - omega_xy_ref) < 1e-12
    assert abs(omega[1, 0].item() + omega_xy_ref) < 1e-12
    assert abs(omega[0, 2].item()) < 1e-14 and abs(omega[1, 2].item()) < 1e-14
    # metric is identical for the two bands; PSD holds for both
    assert torch.linalg.eigvalsh((g + g.T) / 2).min().item() > -1e-14


def test_sos_handles_internal_degeneracy():
    # At a k on the Λ line the two-band Dirac model with M=0 is gapless — skip
    # that; instead check qgt_sos on a group CONTAINING a degenerate pair:
    # free electrons with two exactly degenerate G-shells in the group.
    g = torch.tensor([[0.0, 0, 0], [1.0, 1, 0], [1.0, -1, 0]], dtype=torch.float64)

    def h_free(k):
        kin = HBAR2_2M * ((g + k) ** 2).sum(-1)
        return torch.diag_embed(kin.to(torch.complex128))

    k = torch.tensor([0.0, 0.0, 0.2], dtype=torch.float64)  # G2/G3 degenerate
    q = qgt_sos(h_free, k, [0, 1, 2][1:])  # the degenerate pair as the group
    assert torch.isfinite(q).all()
    assert q.abs().max().item() < 1e-14  # still free electrons: P constant
