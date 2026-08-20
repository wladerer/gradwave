"""Orbital magnetization via the modern-theory k-space formula
(postscf.kgeometry.orbmag_tensor / orbital_magnetization).

Validation ladder:
- pointwise μ-slope identity dI_γ/dμ = 2 Ω_γ(k) against the M2 QGT
  machinery (machine precision — the same cross-gap tangents A enter both);
- mesh Chern-slope: avg I_z is exactly linear in μ with slope 2·avg(Ω)
  (machine precision) ≈ 4πC (FHS integer, up to the Riemann quadrature
  error of avg Ω — the integer itself is exact, the average is not);
- gauge/backend independence: identical I from (i) the sum-over-states
  covariant form, (ii) forward-mode AD *through* eigh with the occupied
  projector removing the arbitrary per-band gauge, (iii) a random constant
  basis conjugation of H;
- exact structural null: any traceless two-band model has ε_m + ε_n = 0
  across the gap, so I(μ=0) = 0 identically (regression for the energy
  weight), while a particle-hole-breaking identity term gives I_z ≠ 0;
- physics: time reversal ⇒ M_orb = 0 on Si (scalar and SOC-spinor, using
  the inversion-safe 20³ FFT box), in μ_B per cell via the HBAR2_2M unit
  chain documented on orbital_magnetization.
"""

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.kgeometry import (
    metric_curvature,
    orbital_magnetization,
    orbmag_tensor,
    qgt_sos,
)
from gradwave.postscf.kgeometry_topo import ModelLinkStates, chern_fhs
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc, si_upf

SX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
SY = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
SZ = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

VALENCE = [0, 1, 2, 3]


def qwz(u, t0=0.0):
    """QWZ Chern insulator, optionally with a particle-hole-breaking
    identity term t0·(cos 2πk₁ + cos 2πk₂)·I."""

    def h(k):
        a, b = 2.0 * np.pi * k[0], 2.0 * np.pi * k[1]
        m = (
            torch.sin(a).to(torch.complex128) * SX
            + torch.sin(b).to(torch.complex128) * SY
            + (u + torch.cos(a) + torch.cos(b)).to(torch.complex128) * SZ
        )
        if t0:
            m = m + (t0 * (torch.cos(a) + torch.cos(b))).to(
                torch.complex128
            ) * torch.eye(2, dtype=torch.complex128)
        return m

    return h


def _mesh_avg_i(h_fn, occ, mu_chem, n=12):
    tot = torch.zeros(3, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            k = torch.tensor([i / n, j / n, 0.0], dtype=torch.float64)
            tot += orbmag_tensor(h_fn, k, occ, mu_chem)
    return tot / n**2


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
# QGT / Chern consistency                                                     #
# --------------------------------------------------------------------------- #


def test_pointwise_mu_slope_is_berry_curvature():
    h = qwz(1.0)
    k = torch.tensor([0.31, 0.17, 0.0], dtype=torch.float64)
    i1 = orbmag_tensor(h, k, [0], 0.1)
    i2 = orbmag_tensor(h, k, [0], 0.3)
    _, omega = metric_curvature(qgt_sos(h, k, [0]))
    slope = (i2[2] - i1[2]).item() / 0.2
    assert abs(slope - 2.0 * omega[0, 1].item()) < 1e-12


def test_mesh_chern_slope():
    h = qwz(1.0)
    n = 12
    i1, i2 = _mesh_avg_i(h, [0], -0.3, n), _mesh_avg_i(h, [0], 0.4, n)
    slope = (i2[2] - i1[2]).item() / 0.7
    # machine-exact against the same-mesh average of Ω …
    avg_om = 0.0
    for i in range(n):
        for j in range(n):
            k = torch.tensor([i / n, j / n, 0.0], dtype=torch.float64)
            avg_om += metric_curvature(qgt_sos(h, k, [0]))[1][0, 1].item()
    avg_om /= n**2
    assert abs(slope - 2.0 * avg_om) < 1e-10
    # … and equal to 4πC up to the Riemann quadrature error of avg(Ω)
    c = chern_fhs(ModelLinkStates(h, [0]), n, e1=(1, 0), e2=(0, 1), origin=(0, 0)).chern
    assert c == -1
    assert abs(slope - 4.0 * np.pi * c) < 0.05
    # 2D model: in-plane components vanish identically
    assert i1[:2].abs().max().item() == 0.0


# --------------------------------------------------------------------------- #
# gauge / backend independence                                                #
# --------------------------------------------------------------------------- #


def _orbmag_fwad(h_fn, k, occ, mu_chem):
    """Independent evaluation: covariant |∂̃u⟩ = (1−P)|∂u⟩ by forward-mode AD
    THROUGH eigh (whatever per-band gauge the backend picks is projected
    out), contracted explicitly against (H + ε_n − 2μ)."""
    with torch.no_grad():
        w0, u0 = torch.linalg.eigh(h_fn(k))
    occ_t = torch.as_tensor(occ, dtype=torch.int64)
    p = u0[:, occ_t] @ u0[:, occ_t].mH
    q = torch.eye(p.shape[0], dtype=p.dtype) - p
    dus = []
    for d in range(3):
        tg = torch.zeros(3, dtype=torch.float64)
        tg[d] = 1.0
        with fwAD.dual_level():
            _, ud = torch.linalg.eigh(h_fn(fwAD.make_dual(k, tg)))
            _, du = fwAD.unpack_dual(ud)
        dus.append(q @ du[:, occ_t])
    h0 = h_fn(k)
    eye = torch.eye(h0.shape[0], dtype=h0.dtype)
    t = torch.empty((3, 3), dtype=torch.complex128)
    for a in range(3):
        for b in range(3):
            acc = torch.zeros((), dtype=torch.complex128)
            for jn, nb in enumerate(occ):
                op = h0 + (w0[nb] - 2.0 * mu_chem) * eye
                acc = acc + dus[a][:, jn].conj() @ (op @ dus[b][:, jn])
            t[a, b] = acc
    return torch.stack(
        [(t[1, 2] - t[2, 1]).imag, (t[2, 0] - t[0, 2]).imag, (t[0, 1] - t[1, 0]).imag]
    )


def test_orbmag_gauge_and_backend_independence():
    h = qwz(1.0, t0=0.3)
    k = torch.tensor([0.31, 0.17, 0.0], dtype=torch.float64)
    i_sos = orbmag_tensor(h, k, [0], 0.1)
    # (i) fwAD-through-eigh path, arbitrary backend gauge projected out
    i_fwad = _orbmag_fwad(h, k, [0], 0.1)
    assert (i_sos - i_fwad).abs().max().item() < 1e-10
    # (ii) constant random basis conjugation H → VHV† leaves I invariant
    gen = torch.Generator().manual_seed(7)
    a = torch.randn(2, 2, generator=gen, dtype=torch.float64) + 1j * torch.randn(
        2, 2, generator=gen, dtype=torch.float64
    )
    v = torch.linalg.qr(a)[0].to(torch.complex128)

    def h_rot(kk):
        return v @ h(kk) @ v.mH

    i_rot = orbmag_tensor(h_rot, k, [0], 0.1)
    assert (i_sos - i_rot).abs().max().item() < 1e-10


# --------------------------------------------------------------------------- #
# structural / physics nulls and non-nulls                                    #
# --------------------------------------------------------------------------- #


def test_traceless_two_band_zero_at_zero_mu():
    # any traceless 2-band model: ε_m + ε_n = 0 across the gap ⇒ I(μ=0) ≡ 0,
    # while at μ ≠ 0 the Chern term makes it nonzero (TR broken)
    h = qwz(1.0)
    i0 = _mesh_avg_i(h, [0], 0.0, n=6)
    assert i0.abs().max().item() < 1e-13
    i_mu = _mesh_avg_i(h, [0], 0.4, n=6)
    assert i_mu[2].abs().item() > 1.0


def test_tr_broken_nonzero_at_zero_mu():
    # particle-hole-breaking identity term: genuinely nonzero M at μ = 0
    i0 = _mesh_avg_i(qwz(1.0, t0=0.3), [0], 0.0, n=12)
    assert i0[2].abs().item() > 1.0  # measured ≈ −4.10


def test_si_orbital_magnetization_null(si_res):
    # time reversal (+ inversion-safe 20³ box) ⇒ M_orb = 0
    m = orbital_magnetization(si_res, VALENCE, float(si_res.fermi), mesh=(2, 2, 2))
    assert m.abs().max().item() < 1e-10  # μ_B per cell; measured ~8e-15


def test_si_soc_spinor_orbmag_null_pointwise():
    # orbmag_tensor is h_fn-generic: the spinor SOC H(k) plugs straight in;
    # PT symmetry zeroes the integrand pointwise
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.core.xc.spin import SpinPBE
    from gradwave.postscf.kgeometry_soc import BlochHKSpinor
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.noncollinear import scf_noncollinear
    from tests.helpers import pseudo

    torch.set_num_threads(2)
    upf = parse_upf(pseudo("Si_ONCV_PBE_fr.upf"))
    cell, pos = si_fcc()
    system = setup_system(cell, pos, [0, 0], [upf], ecut=12 * RY,
                          kmesh=(1, 1, 1), nbands=8, use_symmetry=False,
                          fft_shape=(20, 20, 20), time_reversal=False)
    xc = NoncollinearXC(SpinPBE())
    res = scf_noncollinear(system, xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                           smearing="gaussian", width=0.05,
                           etol=1e-9, rhotol=1e-8, verbose=False, max_iter=80)
    assert res.converged
    kf = np.array([0.13, 0.07, -0.21])
    hk = BlochHKSpinor.from_nc_scf(res, xc, kf)
    i_k = orbmag_tensor(hk.h, hk.k_cart(kf), list(range(8)), float(res.fermi))
    assert i_k.abs().max().item() < 1e-8  # eV·Å²; PT ⇒ pointwise zero
