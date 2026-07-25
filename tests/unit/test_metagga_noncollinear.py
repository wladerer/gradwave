"""Non-collinear meta-GGA τ machinery (Layer A/B), SCF-free.

Synthetic two-component spinors on a small box exercise the 2×2 kinetic-energy
density matrix build (`spinor_tau_matrix_b`) and its generalized-KS operator
(`spinor_metagga_tau_operator`) against closed-form anchors, no external
reference needed:

  * COLLINEAR REDUCTION — a spinor with only its up component reproduces the
    collinear per-component τ (`tau_b`); the 2×2 operator with v_τ⃗ ∥ ẑ reduces
    to `metagga_tau_operator` applied per spin. This is the same-quantity-two-ways
    equivalence that the collinear limit of the whole spinor path rests on.
  * ROTATIONAL COVARIANCE — a global SU(2) spin rotation leaves τ_0 invariant
    and rotates τ⃗ by the matching SO(3) (the metamorphic invariant behind the
    rotational-invariance of the non-collinear meta-GGA energy).
  * HERMITICITY and the GENERALIZED-KS gate — the 2×2 operator is Hermitian and
    IS the functional derivative ∂E/∂ψ* of a τ-dependent energy, including the
    off-diagonal (spin-flip) v_τx, v_τy channels.
"""

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.batch import BatchedK
from gradwave.core.metagga import (
    metagga_tau_operator,
    spinor_metagga_tau_operator,
    spinor_tau_matrix_b,
    tau_b,
)
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_fft_grid, build_gsphere
from tests.helpers import RY


def _single_k_bk(a=6.0, ecut=12 * RY, k_frac=(0.0, 0.0, 0.0)):
    """A one-k BatchedK (no projectors) plus its grid — as in test_metagga."""
    grid = build_fft_grid(a * np.eye(3), ecut)
    s = build_gsphere(grid, ecut, k_frac=k_frac)
    m = s.npw
    bk = BatchedK(
        npw=torch.tensor([m]),
        mask=torch.ones(1, m, dtype=torch.bool),
        flat_idx=s.flat_idx[None],
        kpg=s.kpg[None],
        t=(HBAR2_2M * s.kpg2)[None],
        proj_phase_free=torch.zeros(1, 0, m, dtype=CDTYPE),
        proj_atom_index=torch.zeros(0, dtype=torch.int64),
        dij_full=torch.zeros((0, 0), dtype=RDTYPE),
    )
    return bk, grid, s


def _rand_coeffs(shape, seed):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, dtype=torch.float64, generator=gen)
            + 1j * torch.randn(shape, dtype=torch.float64, generator=gen)).to(CDTYPE)


def _spinor(m, nb, seed):
    """Random doubled spinor coefficients (1, nb, 2m)."""
    return _rand_coeffs((1, nb, 2 * m), seed)


def test_spinor_tau_collinear_reduction():
    """A spinor with a zero down component: τ_↑↑ = tau_b(c↑), τ_↓↓ = 0, τ_↑↓ = 0,
    so τ_0 = tau_b(c↑) and τ⃗ = (0, 0, tau_b(c↑)) — the collinear per-spin τ."""
    bk, grid, s = _single_k_bk()
    m = s.npw
    nb = 4
    cu = _rand_coeffs((1, nb, m), seed=1)
    c = torch.cat([cu, torch.zeros_like(cu)], dim=-1)  # down component ≡ 0
    occ = torch.tensor([[2.0, 1.5, 1.0, 0.5]])
    kw = torch.tensor([1.0])

    tau_ref = tau_b(cu, occ, kw, bk, grid.shape, grid.volume)
    tau0, tvec = spinor_tau_matrix_b(c, occ, kw, bk, grid.shape, grid.volume, m)

    assert torch.allclose(tau0, tau_ref, rtol=1e-11, atol=1e-13)
    assert torch.allclose(tvec[2], tau_ref, rtol=1e-11, atol=1e-13)  # τ_z = τ↑↑
    assert float(tvec[0].abs().max()) < 1e-12  # τ_x = 2Re τ↑↓ = 0
    assert float(tvec[1].abs().max()) < 1e-12  # τ_y = 2Im τ↑↓ = 0


def test_spinor_tau_rotational_covariance():
    """A global SU(2) rotation of the spinors leaves τ_0 invariant and rotates
    τ⃗ by the matching SO(3): here a rotation about ŷ by θ."""
    bk, grid, s = _single_k_bk()
    m = s.npw
    nb = 3
    c = _spinor(m, nb, seed=2)
    occ = torch.tensor([[2.0, 1.0, 0.7]])
    kw = torch.tensor([1.0])

    tau0, tvec = spinor_tau_matrix_b(c, occ, kw, bk, grid.shape, grid.volume, m)

    theta = 0.6
    ca, sa = np.cos(theta / 2), np.sin(theta / 2)
    cu, cd = c[..., :m], c[..., m:]
    # U = cos(θ/2)·1 − i sin(θ/2)·σ_y  ⇒  ψ↑' = ca ψ↑ − sa ψ↓, ψ↓' = sa ψ↑ + ca ψ↓
    cu2, cd2 = ca * cu - sa * cd, sa * cu + ca * cd
    c_rot = torch.cat([cu2, cd2], dim=-1)
    tau0r, tvecr = spinor_tau_matrix_b(c_rot, occ, kw, bk, grid.shape,
                                       grid.volume, m)

    # SO(3) rotation about y by θ acting on (τ_x, τ_y, τ_z)
    ct, st = np.cos(theta), np.sin(theta)
    tx = ct * tvec[0] + st * tvec[2]
    ty = tvec[1]
    tz = -st * tvec[0] + ct * tvec[2]
    assert torch.allclose(tau0r, tau0, rtol=1e-11, atol=1e-13)
    assert torch.allclose(tvecr[0], tx, rtol=1e-10, atol=1e-12)
    assert torch.allclose(tvecr[1], ty, rtol=1e-10, atol=1e-12)
    assert torch.allclose(tvecr[2], tz, rtol=1e-10, atol=1e-12)


def test_spinor_operator_collinear_reduction():
    """v_τ⃗ ∥ ẑ makes the 2×2 operator diagonal: each component equals
    `metagga_tau_operator` with v_τ↑ = v_τ0 + v_τz on ↑ and v_τ↓ = v_τ0 − v_τz
    on ↓ — the collinear per-spin generalized-KS term."""
    bk, grid, s = _single_k_bk()
    m = s.npw
    nb = 3
    c = _spinor(m, nb, seed=3)
    gen = torch.Generator().manual_seed(11)
    v0 = 0.5 + torch.rand(grid.shape, generator=gen, dtype=RDTYPE)
    vz = 0.3 * (torch.rand(grid.shape, generator=gen, dtype=RDTYPE) - 0.5)
    vvec = torch.stack([torch.zeros_like(vz), torch.zeros_like(vz), vz])

    got = spinor_metagga_tau_operator(c, v0, vvec, bk, grid.shape, m)
    up = metagga_tau_operator(c[..., :m], v0 + vz, bk, grid.shape)
    dn = metagga_tau_operator(c[..., m:], v0 - vz, bk, grid.shape)
    ref = torch.cat([up, dn], dim=-1)
    assert float((got - ref).abs().max()) < 1e-10 * float(ref.abs().max())


def test_spinor_operator_hermitian():
    """⟨φ|V_τ|ψ⟩ = ⟨V_τφ|ψ⟩ for smooth random 2×2 fields (v_τ0, v_τ⃗)."""
    bk, grid, s = _single_k_bk()
    m = s.npw
    psi = _spinor(m, 4, seed=4)
    phi = _spinor(m, 4, seed=5)
    gen = torch.Generator().manual_seed(7)
    v0 = 0.5 + torch.rand(grid.shape, generator=gen, dtype=RDTYPE)
    vvec = 0.4 * (torch.rand(3, *grid.shape, generator=gen, dtype=RDTYPE) - 0.5)
    vp = spinor_metagga_tau_operator(psi, v0, vvec, bk, grid.shape, m)
    vq = spinor_metagga_tau_operator(phi, v0, vvec, bk, grid.shape, m)
    a = torch.einsum("kbg,kcg->bc", phi.conj(), vp)
    b = torch.einsum("kbg,kcg->bc", vq.conj(), psi)
    assert float((a - b).abs().max()) < 1e-10 * float(a.abs().max())


def test_spinor_operator_is_functional_derivative():
    """Generalized-KS gate for the full 2×2 term. For the τ-linear energy
    E = ∫ (a·τ_0 + b⃗·τ⃗) dr (constant a, b⃗), the operator built with
    (v_τ0, v_τ⃗) = (a, b⃗) must satisfy

        d/dλ E[ψ+λφ]|₀ = 2 Re Σ_n f_n ⟨φ_n | V_τ | ψ_n⟩,

    exercising the off-diagonal spin-flip channels b_x, b_y as well."""
    bk, grid, s = _single_k_bk()
    m = s.npw
    nb = 3
    psi = _spinor(m, nb, seed=8)
    phi = _spinor(m, nb, seed=9)
    occ = torch.tensor([[2.0, 2.0, 1.0]])
    kw = torch.tensor([1.0])
    a, bvec = 0.37, (0.21, -0.15, 0.28)

    def energy(coeffs):
        ts, tv = spinor_tau_matrix_b(coeffs, occ, kw, bk, grid.shape,
                                     grid.volume, m)
        e_density = a * ts + bvec[0] * tv[0] + bvec[1] * tv[1] + bvec[2] * tv[2]
        return e_density.sum() * (grid.volume / grid.n_points)

    v0 = torch.full(grid.shape, a, dtype=RDTYPE)
    vvec = torch.stack([torch.full(grid.shape, bvec[i], dtype=RDTYPE)
                        for i in range(3)])
    vpsi = spinor_metagga_tau_operator(psi, v0, vvec, bk, grid.shape, m)
    fw = kw[:, None] * occ
    braket = torch.einsum("kbg,kbg->kb", phi.conj(), vpsi)
    analytic = 2.0 * float((fw * braket.real).sum())

    h = 1e-5
    fd = float((energy(psi + h * phi) - energy(psi - h * phi)) / (2 * h))
    assert abs(analytic - fd) < 1e-5 * max(abs(fd), 1.0)
