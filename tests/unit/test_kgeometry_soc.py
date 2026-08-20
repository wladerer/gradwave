"""Spinor dense H(k), ℤ₂, and mirror Chern (postscf.kgeometry_soc / _topo).

- the dense spinor H(k) reproduces a converged fully-relativistic
  ``scf_noncollinear`` run's eigenvalues at a mesh k (Si FR, 12 Ry, Γ-only —
  the SO-split Γ₇/Γ₈ valence top with its exact 2/4-fold degeneracies);
- Kramers degeneracy at generic k (Si has inversion + TR);
- ℤ₂ by Soluyanov–Vanderbilt WCC crossing parity: BHZ model across its phase
  diagram (1/1/0), trivial null on real Si through the spinor plane-wave
  provider;
- mirror Chern: an 8-band mirror-symmetric toy (two QWZ pairs per sector,
  TR-related sectors) gives C_± = ∓2, C_m = −2 topological and 0 trivial;
  on Si, the glide-mirror operator on the spinor plane-wave basis commutes
  with H(k) at an invariant k, and the symmorphic-mirror sector Chern
  numbers on a mirror-invariant slice are the trivial null.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.kgeometry_soc import (
    BlochHKSpinor,
    SpinorBlochLinkStates,
    find_mirror_ops,
    mirror_matrix,
)
from gradwave.postscf.kgeometry_topo import (
    MirrorSectorStates,
    ModelLinkStates,
    chern_fhs,
    z2_invariant,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from tests.helpers import RY, pseudo, si_fcc

SX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
SY = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
SZ = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

OCC = list(range(8))  # Si: 8 occupied spinor bands


def qwz(u):
    def h(k):
        a, b = 2.0 * np.pi * k[0], 2.0 * np.pi * k[1]
        return (
            torch.sin(a).to(torch.complex128) * SX
            + torch.sin(b).to(torch.complex128) * SY
            + (u + torch.cos(a) + torch.cos(b)).to(torch.complex128) * SZ
        )

    return h


def _block2(h1, h2):
    z = torch.zeros(*h1.shape, dtype=torch.complex128)
    return torch.cat([torch.cat([h1, z], 1), torch.cat([z, h2], 1)], 0)


def bhz(u):
    """Time-reversal pair of QWZ blocks: H(k) = diag(h(k), h(−k)*)."""

    def h(k):
        return _block2(qwz(u)(k), qwz(u)(-k).conj())

    return h


def mirror_toy(u, lam=0.4):
    """8-band mirror-symmetric 3D toy: the +i sector holds two QWZ copies
    (total Chern −2 for 0<u<2), the −i sector their TR images (+2); a
    TR-allowed inter-sector coupling ∝ sin(2πk₃) vanishes on the k₃ = 0
    mirror plane. M = diag(+i·I₄, −i·I₄)."""

    def h(k):
        hp = _block2(qwz(u)(k[:2]), qwz(u)(k[:2] + 0.31))
        hm = _block2(qwz(u)(-k[:2]).conj(), qwz(u)(-k[:2] + 0.31).conj())
        c = lam * torch.sin(2.0 * np.pi * k[2]).to(torch.complex128) * torch.eye(
            4, dtype=torch.complex128
        )
        return torch.cat([torch.cat([hp, c], 1), torch.cat([c.mH, hm], 1)], 0)

    return h


MIRROR_TOY_M = torch.kron(
    torch.diag(torch.tensor([1j, -1j])), torch.eye(4, dtype=torch.complex128)
)


@pytest.fixture(scope="module")
def si_fr():
    """(res, xc) of a small fully-relativistic (SOC) Si run, nonmagnetic."""
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
    return res, xc


# --------------------------------------------------------------------------- #
# dense spinor H(k)                                                           #
# --------------------------------------------------------------------------- #


def test_spinor_h_matches_soc_scf_eigenvalues(si_fr):
    res, xc = si_fr
    hk = BlochHKSpinor.from_nc_scf(res, xc, (0.0, 0.0, 0.0))
    hmat = hk.h(torch.zeros(3, dtype=torch.float64))
    assert (hmat - hmat.mH).abs().max().item() < 1e-12
    w = torch.linalg.eigh(hmat).eigenvalues
    ref = res.eigenvalues[0]
    assert (w[: ref.shape[0]] - ref).abs().max().item() < 1e-6

    # SOC physics at Γ: valence top splits into 2-fold Γ₇ below 4-fold Γ₈
    e = w[:8].numpy()
    gamma7, gamma8 = e[2:4], e[4:8]
    assert np.ptp(gamma7) < 1e-8 and np.ptp(gamma8) < 1e-8
    assert 0.02 < gamma8.mean() - gamma7.mean() < 0.08  # Si Δ₀ ≈ 0.047 eV here


def test_spinor_h_kramers_degenerate_at_generic_k(si_fr):
    res, xc = si_fr
    kf = np.array([0.13, 0.07, -0.21])
    hk = BlochHKSpinor.from_nc_scf(res, xc, kf)
    w = torch.linalg.eigh(hk.h(hk.k_cart(kf))).eigenvalues[:16]
    # inversion + TR → every band doubly degenerate at every k
    assert (w[0::2] - w[1::2]).abs().max().item() < 1e-8


# --------------------------------------------------------------------------- #
# ℤ₂                                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("u,z2_expected", [(1.0, 1), (-1.0, 1), (3.0, 0)])
def test_bhz_z2_phase_diagram(u, z2_expected):
    prov = ModelLinkStates(bhz(u), [0, 1])
    r = z2_invariant(prov, e_loop=(1, 0), e_perp=(0, 1), origin=(0, 0),
                     n_loop=12, n_perp=8)
    assert r.z2 == z2_expected


def test_z2_si_trivial(si_fr):
    res, xc = si_fr
    prov = SpinorBlochLinkStates(res, xc, OCC)
    r = z2_invariant(prov, e_loop=(1, 0, 0), e_perp=(0, 1, 0), origin=(0, 0, 0),
                     n_loop=6, n_perp=4)
    assert r.z2 == 0
    assert r.wcc.shape == (5, len(OCC))


# --------------------------------------------------------------------------- #
# mirror Chern                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("u,cm_expected", [(1.0, -2), (3.0, 0)])
def test_mirror_toy_sector_chern(u, cm_expected):
    h = mirror_toy(u)
    # [H, M] = 0 exactly on the mirror plane, broken off it
    k_on = torch.tensor([0.13, 0.29, 0.0], dtype=torch.float64)
    k_off = torch.tensor([0.13, 0.29, 0.2], dtype=torch.float64)
    assert (h(k_on) @ MIRROR_TOY_M - MIRROR_TOY_M @ h(k_on)).abs().max().item() < 1e-14
    assert (h(k_off) @ MIRROR_TOY_M - MIRROR_TOY_M @ h(k_off)).abs().max().item() > 0.1

    cp = chern_fhs(MirrorSectorStates(h, MIRROR_TOY_M, [0, 1, 2, 3], +1), 12,
                   e1=(1, 0, 0), e2=(0, 1, 0), origin=(0, 0, 0))
    cm = chern_fhs(MirrorSectorStates(h, MIRROR_TOY_M, [0, 1, 2, 3], -1), 12,
                   e1=(1, 0, 0), e2=(0, 1, 0), origin=(0, 0, 0))
    assert cp.residual < 1e-9 and cm.residual < 1e-9
    assert cp.chern == -cm.chern  # TR-related sectors
    assert (cp.chern - cm.chern) // 2 == cm_expected


def test_si_mirror_operator_commutes(si_fr):
    res, xc = si_fr
    cell, pos = si_fcc()
    frac = pos @ np.linalg.inv(cell)
    mirrors = find_mirror_ops(cell, frac, [0, 0])
    trans = {tuple(np.round(m.t_frac, 6)) for m in mirrors}
    assert len(mirrors) == 9  # diamond: symmorphic mirrors + glides, no inversion
    assert any(max(abs(np.array(t))) > 1e-9 for t in trans)  # glides present

    op = next(m for m in mirrors if np.abs(m.t_frac).max() > 1e-9)  # a glide
    wk = np.round(np.linalg.inv(op.w_frac).T).astype(int)
    _, s, vt = np.linalg.svd(wk - np.eye(3))
    basis = vt[s < 1e-9]  # invariant plane of the k-space action
    k_inv = 0.13 * basis[0] + 0.07 * basis[1]
    hk = BlochHKSpinor.from_nc_scf(res, xc, k_inv)
    m_op = mirror_matrix(hk, op, k_inv)
    eye = torch.eye(m_op.shape[0], dtype=m_op.dtype)
    assert (m_op @ m_op.mH - eye).abs().max().item() < 1e-12  # unitary
    hmat = hk.h(hk.k_cart(k_inv))
    assert (hmat @ m_op - m_op @ hmat).abs().max().item() < 1e-6


def test_si_mirror_sector_chern_null(si_fr):
    res, xc = si_fr
    cell, pos = si_fcc()
    frac = pos @ np.linalg.inv(cell)
    op = next(m for m in find_mirror_ops(cell, frac, [0, 0])
              if np.abs(m.t_frac).max() < 1e-9)  # symmorphic
    wk = np.round(np.linalg.inv(op.w_frac).T).astype(int)
    cands = [np.array(v) for v in
             [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]]
    e1, e2 = [v for v in cands if np.array_equal(wk @ v, v)][:2]

    cp = chern_fhs(SpinorBlochLinkStates(res, xc, OCC, mirror=op, sector=+1), 4,
                   e1=e1, e2=e2, origin=(0, 0, 0))
    cm = chern_fhs(SpinorBlochLinkStates(res, xc, OCC, mirror=op, sector=-1), 4,
                   e1=e1, e2=e2, origin=(0, 0, 0))
    assert cp.chern == cm.chern == 0  # trivial null, per sector
    assert cp.residual < 1e-9 and cm.residual < 1e-9


def test_si_spinor_fhs_null(si_fr):
    res, xc = si_fr
    c = chern_fhs(SpinorBlochLinkStates(res, xc, OCC), 4,
                  e1=(1, 0, 0), e2=(0, 1, 0), origin=(0.03, 0.06, 0.17))
    assert c.chern == 0
    assert c.residual < 1e-9
