"""Meta-GGA kinetic-energy density τ and its generalized-KS operator (Layer A/B).

Synthetic orbitals on a small box exercise the τ build and the −½∇·(v_τ∇ψ)
operator without an SCF. The gates are intrinsic — they need no external
reference — because the machinery has closed-form anchors:

  * a single plane wave ψ = e^{i(k+G₀)·r} has τ = ½|k+G₀|² exactly;
  * τ is bounded below by the von Weizsäcker density τ_W = |∇ρ|²/(8ρ);
  * the operator is Hermitian;
  * for constant v_τ ≡ c the operator is exactly c·(−½∇²) = c·T̂ (the kinetic
    operator), so "add c·τ to the energy" ≡ "scale the kinetic energy by c";
  * and, the defining generalized-KS gate, the operator IS the functional
    derivative of the meta-GGA energy: dE/dλ at ψ→ψ+λφ equals 2Re Σf⟨φ|V_τ|ψ⟩.
"""

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.batch import BatchedK
from gradwave.core.metagga import metagga_tau_operator, tau_b
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_fft_grid, build_gsphere
from tests.helpers import RY


def _single_k_bk(a=6.0, ecut=12 * RY, k_frac=(0.0, 0.0, 0.0)):
    """A one-k BatchedK (no projectors) plus its grid, from a real GSphere."""
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


def _orthonormal_coeffs(m, nb, seed=0):
    gen = torch.Generator().manual_seed(seed)
    c = torch.randn(m, nb, dtype=CDTYPE, generator=gen)
    q, _ = torch.linalg.qr(c)  # columns orthonormal in G (Σ_G|c|²=1)
    return q.transpose(0, 1)[None]  # (1, nb, m)


def test_tau_single_plane_wave():
    """One filled plane wave ψ_{G₀}: τ integrates to f·½|k+G₀|² and is uniform."""
    bk, grid, s = _single_k_bk()
    m = s.npw
    # pick a mid-shell plane wave; ψ = e^{i(k+G₀)·r}/√Ω  ⇒  Σ_G|c|²=1
    j = m // 2
    c = torch.zeros(1, 1, m, dtype=CDTYPE)
    c[0, 0, j] = 1.0
    occ = torch.tensor([[2.0]])
    kw = torch.tensor([1.0])
    tau = tau_b(c, occ, kw, bk, grid.shape, grid.volume)
    kpg2 = float(s.kpg2[j])
    # τ(r) = ½·f·|k+G₀|²/Ω, uniform over the grid
    expect = 0.5 * 2.0 * kpg2 / grid.volume
    assert torch.allclose(tau, torch.full_like(tau, expect), rtol=1e-10, atol=1e-12)
    # ∫τ dr = f·½|k+G₀|²
    integ = float(tau.sum()) * grid.volume / grid.n_points
    assert abs(integ - 0.5 * 2.0 * kpg2) < 1e-9 * (0.5 * 2.0 * kpg2)


def test_tau_nonnegative_and_above_von_weizsaecker():
    """τ ≥ 0 and, for a single occupied orbital, τ ≥ τ_W = |∇ρ|²/(8ρ)."""
    bk, grid, s = _single_k_bk()
    m = s.npw
    c = _orthonormal_coeffs(m, nb=1, seed=4)
    occ = torch.tensor([[2.0]])
    kw = torch.tensor([1.0])
    tau = tau_b(c, occ, kw, bk, grid.shape, grid.volume)
    assert float(tau.min()) >= -1e-14
    # build ρ and τ_W for the single orbital
    from gradwave.core.batch import density_b
    from gradwave.core.density import sigma_from_rho

    rho = density_b(c, occ, kw, bk, grid.shape, grid.volume)
    sigma = sigma_from_rho(rho, grid.g_cart)
    tau_w = sigma / (8.0 * rho.clamp_min(1e-12))
    # τ ≥ τ_W pointwise where ρ is appreciable (the bound is exact for 1 orbital)
    mask = rho > 1e-4 * float(rho.max())
    assert float((tau - tau_w)[mask].min()) > -1e-9 * float(tau.max())


def test_operator_hermitian():
    """⟨φ|V_τ|ψ⟩ = ⟨V_τφ|ψ⟩ for a smooth random v_τ."""
    bk, grid, s = _single_k_bk()
    m = s.npw
    psi = _orthonormal_coeffs(m, nb=4, seed=1)
    phi = _orthonormal_coeffs(m, nb=4, seed=2)
    gen = torch.Generator().manual_seed(7)
    v_tau = 0.5 + torch.rand(grid.shape, generator=gen, dtype=RDTYPE)  # >0
    vp = metagga_tau_operator(psi, v_tau, bk, grid.shape)
    vq = metagga_tau_operator(phi, v_tau, bk, grid.shape)
    a = torch.einsum("kbg,kcg->bc", phi.conj(), vp)  # ⟨φ_b|V|ψ_c⟩
    b = torch.einsum("kbg,kcg->bc", vq.conj(), psi)  # ⟨Vφ_b|ψ_c⟩
    assert float((a - b).abs().max()) < 1e-10 * float(a.abs().max())


def test_operator_constant_vtau_is_scaled_kinetic():
    """v_τ ≡ c ⇒ V_τ ψ = c·(−½∇²)ψ. The τ machinery is "geometric" (ħ²/2m is
    folded into the functional's a.u. conversion, exactly as it is for ρ and σ),
    so −½∇² in this fftbox convention is the multiply by ½|k+G|². Hence
    V_τ c = c·½|k+G|²·c, and note |k+G|² = t / (ħ²/2m) from the solver diagonal."""
    bk, grid, s = _single_k_bk()
    m = s.npw
    psi = _orthonormal_coeffs(m, nb=3, seed=5)
    c_const = 0.37
    v_tau = torch.full(grid.shape, c_const, dtype=RDTYPE)
    got = metagga_tau_operator(psi, v_tau, bk, grid.shape)
    kpg2 = bk.t / HBAR2_2M  # |k+G|² [Å⁻²]
    want = c_const * (0.5 * kpg2[:, None, :] * psi)
    assert float((got - want).abs().max()) < 1e-9 * float(want.abs().max())


def test_operator_is_functional_derivative():
    """The generalized-KS gate. For a meta-GGA energy E = ∫ g(τ(r)) dr with a
    smooth g, the operator −½∇·(g'(τ)∇ψ) must equal ∂E/∂ψ*:

        d/dλ E[ψ+λφ]|₀ = 2 Re Σ_n f_n ⟨φ_n | V_τ | ψ_n⟩.

    Verified with g(τ) = ½τ² (so v_τ = g'(τ) = τ, a spatially varying field)
    against a finite difference of the energy.
    """
    bk, grid, s = _single_k_bk()
    m = s.npw
    nb = 3
    psi = _orthonormal_coeffs(m, nb=nb, seed=8)
    phi = _orthonormal_coeffs(m, nb=nb, seed=9)
    occ = torch.tensor([[2.0, 2.0, 1.0]])
    kw = torch.tensor([1.0])

    def energy(coeffs):
        tau = tau_b(coeffs, occ, kw, bk, grid.shape, grid.volume)
        e_density = 0.5 * tau ** 2  # g(τ)
        return e_density.sum() * (grid.volume / grid.n_points)

    # analytic directional derivative via the operator. The operator wants
    # v_τ = ∂e_density/∂τ POINTWISE (g'(τ) = τ here) — no n_points/volume scale;
    # that scale only undoes the (Ω/N) that `energy()` folds in, and in the SCF
    # path autograd.grad(E_xc, τ_leaf)·(N/Ω) reproduces exactly this g'(τ).
    tau0 = tau_b(psi, occ, kw, bk, grid.shape, grid.volume)
    v_tau = tau0  # g'(τ) for g = ½τ²
    vpsi = metagga_tau_operator(psi, v_tau, bk, grid.shape)
    # ∂E/∂ψ_n* contributes f_n·2Re⟨φ_n|V_τ|ψ_n⟩ to dE/dλ
    fw = (kw[:, None] * occ)
    braket = torch.einsum("kbg,kbg->kb", phi.conj(), vpsi)
    analytic = 2.0 * float((fw * braket.real).sum())

    # finite difference of E[ψ + λφ]
    h = 1e-5
    ep = energy(psi + h * phi)
    em = energy(psi - h * phi)
    fd = float((ep - em) / (2 * h))
    assert abs(analytic - fd) < 1e-5 * max(abs(fd), 1.0)
