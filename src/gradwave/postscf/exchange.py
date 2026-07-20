"""Fock exchange operator and adaptively-compressed exchange (ACE), built on the
ISDF factorization in ``isdf.py`` (Layer C, single k-point / Γ).

The exchange (Fock) operator acting on an orbital φ is

    (V_x φ)(r) = −Σ_j ψ_j(r) ∫ ψ_j*(r′) φ(r′) / |r−r′| dr′,

the sum over occupied ψ_j. Its matrix elements give the exchange energy
(E_x = ½ Σ_i ⟨ψ_i|V_x|ψ_i⟩), but a hybrid SCF needs the *operator*, applied to
trial orbitals every Davidson step. Two problems it creates, and the two tools
here that solve them:

- Building V_x directly costs O(N_occ²) Coulomb solves (a pair FFT per (j, φ)).
  ``exchange_operator_isdf`` reuses the ISDF interpolation vectors so the whole
  occupied action costs N_μ FFTs plus dense contractions — the O(N) build ISDF
  exists for.
- Even cheap, V_x is a full operator to re-apply. ``build_ace`` (Lin Lin,
  JCTC 2016) compresses it to a low-rank V_x ≈ −Σ_k |ξ_k⟩⟨ξ_k| that is *exact*
  on the occupied subspace, so each subsequent apply is a handful of inner
  products. This is the object a hybrid Hamiltonian would carry.

Conventions. Orbitals here are the *physical* ψ (normalized: the discrete inner
product ⟨a|b⟩ = (Ω/N) Σ_r a*(r) b(r) makes ⟨ψ_i|ψ_j⟩ = δ_ij), i.e. the
``g_to_r`` values divided by √Ω — see ``physical_orbitals``. The Coulomb G=0
term is excluded, matching ``isdf.py`` and ``core/energies/hartree.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gradwave.constants import E2
from gradwave.core.energies.hartree import _inv_g2_masked
from gradwave.core.fftbox import g_to_r, r_to_g
from gradwave.dtypes import CDTYPE
from gradwave.postscf.isdf import build_isdf, select_interpolation_points


def physical_orbitals(
    coeffs: torch.Tensor, flat_idx: torch.Tensor, shape, volume: float,
) -> torch.Tensor:
    """Occupied-orbital coefficients → normalized ψ on the grid, (n_orb, N_r).

    ψ_i(r) = f_i(r)/√Ω, so ⟨ψ_i|ψ_j⟩ = (Ω/N) Σ_r ψ_i* ψ_j = δ_ij for orthonormal
    plane-wave coefficients (Σ_G|c|² = δ)."""
    f = g_to_r(coeffs, flat_idx, shape).reshape(coeffs.shape[0], -1)
    return f / math.sqrt(volume)


def coulomb_potential(sigma_r: torch.Tensor, shape, g2: torch.Tensor) -> torch.Tensor:
    """Coulomb potential field v[σ](r) = ∫ σ(r′) e²/|r−r′| dr′ of a density σ(r).

    sigma_r: (..., N_r) real-space co-densities (flattened). Returns the same
    shape. v(G) = 4π e² σ(G)/G² with v(G=0) = 0, transformed back to r."""
    batch = sigma_r.shape[:-1]
    sigma_g = r_to_g(sigma_r.reshape(*batch, *shape))
    v_g = 4.0 * math.pi * E2 * sigma_g * _inv_g2_masked(g2)
    n = g2.numel()
    v_r = torch.fft.ifftn(v_g, dim=(-3, -2, -1)) * n
    return v_r.reshape(*batch, -1)


def exchange_operator_direct(
    psi_occ: torch.Tensor, psi_test: torch.Tensor, shape, g2: torch.Tensor,
) -> torch.Tensor:
    """(V_x ψ_t)(r) = −Σ_j ψ_j(r) v[ψ_j* ψ_t](r), the direct O(N_occ · N_test)
    pair-FFT Fock build. psi_occ (n_occ, N_r), psi_test (n_test, N_r), both
    physical ψ. Returns (n_test, N_r). This is the operator reference."""
    n_test = psi_test.shape[0]
    out = torch.empty_like(psi_test)
    for t in range(n_test):
        sigma = psi_occ.conj() * psi_test[t][None, :]      # (n_occ, N_r) ψ_j* ψ_t
        v = coulomb_potential(sigma, shape, g2)             # (n_occ, N_r)
        out[t] = -(psi_occ * v).sum(dim=0)
    return out


def exchange_operator_isdf(
    psi_occ: torch.Tensor, psi_test: torch.Tensor, points: torch.Tensor,
    zeta: torch.Tensor, shape, g2: torch.Tensor,
) -> torch.Tensor:
    """ISDF-accelerated exchange operator on the occupied set.

    Using ψ_j*(r) ψ_t(r) ≈ Σ_μ ζ_μ(r) ψ_j*(r_μ) ψ_t(r_μ),

        (V_x ψ_t)(r) = −Σ_μ v[ζ_μ](r) B(r, r_μ) ψ_t(r_μ),
        B(r, r_μ) = Σ_j ψ_j(r) ψ_j*(r_μ),

    so the Coulomb solve is done once per interpolation vector (N_μ FFTs) and the
    per-orbital work is a dense contraction. psi_occ/psi_test physical ψ; points,
    zeta from ``select_interpolation_points`` / ``build_isdf`` on psi_occ.
    Returns (n_test, N_r)."""
    w_zeta = coulomb_potential(zeta.transpose(0, 1).to(CDTYPE), shape, g2)  # (n_mu, N_r)
    b = psi_occ.transpose(0, 1) @ psi_occ[:, points].conj()  # (N_r, n_mu)
    g = w_zeta.transpose(0, 1) * b                            # (N_r, n_mu)
    psi_mu_test = psi_test[:, points]                        # (n_test, n_mu)
    return -(g @ psi_mu_test.transpose(0, 1)).transpose(0, 1)


def build_exchange_operator_isdf(psi_occ, shape, g2, n_mu, *, generator=None, sketch=None):
    """Convenience: pick points + fit ζ on psi_occ, return (points, zeta) for
    ``exchange_operator_isdf``."""
    points = select_interpolation_points(psi_occ, n_mu, generator=generator, sketch=sketch)
    zeta = build_isdf(psi_occ, points)
    return points, zeta


def exchange_energy_from_operator(
    psi_occ: torch.Tensor, vx_occ: torch.Tensor, volume: float,
) -> torch.Tensor:
    """E_x = ½ Σ_n ⟨ψ_n|V_x|ψ_n⟩, the (Ω/N)-weighted inner product summed over
    occupied n. vx_occ = V_x ψ_occ (n_occ, N_r). Real scalar [eV]."""
    n = psi_occ.shape[1]
    w = volume / n
    return 0.5 * w * (psi_occ.conj() * vx_occ).sum().real


@dataclass
class ACEExchange:
    """Low-rank exchange V_x ≈ −Σ_k |ξ_k⟩⟨ξ_k|, exact on the occupied subspace.

    xi: (N_r, n_occ) the ACE vectors ξ_k; volume, n_r for the (Ω/N) inner-product
    weight in ``apply``."""

    xi: torch.Tensor
    volume: float
    n_r: int

    @property
    def rank(self) -> int:
        return int(self.xi.shape[1])

    def apply(self, phi: torch.Tensor) -> torch.Tensor:
        """(V_x^ACE φ)(r) = −(Ω/N) Σ_k ξ_k(r) Σ_r′ ξ_k*(r′) φ(r′). phi (n, N_r) or
        (N_r,). Returns the same shape."""
        w = self.volume / self.n_r
        single = phi.ndim == 1
        p = phi[None, :] if single else phi
        proj = w * (p @ self.xi.conj())          # (n, n_occ)
        out = -(proj @ self.xi.transpose(0, 1))  # (n, N_r)
        return out[0] if single else out

    def energy(self, psi_occ: torch.Tensor) -> torch.Tensor:
        """E_x = ½ Σ_n ⟨ψ_n|V_x^ACE|ψ_n⟩ [eV]."""
        return exchange_energy_from_operator(psi_occ, self.apply(psi_occ), self.volume)


def build_ace(psi_occ: torch.Tensor, vx_occ: torch.Tensor, volume: float) -> ACEExchange:
    """Adaptively-compressed exchange from V_x applied to the occupied orbitals.

    psi_occ (n_occ, N_r); vx_occ = V_x ψ_occ (n_occ, N_r). With the exchange
    matrix M_mn = ⟨ψ_m|V_x|ψ_n⟩ (Hermitian, negative definite) and its Cholesky
    −M = L Lᴴ, the ACE vectors Ξ = W (Lᴴ)⁻¹ (W = vx_occ as columns) give an
    operator that reproduces V_x ψ_n exactly for every occupied n (Lin Lin 2016).
    """
    n_r = psi_occ.shape[1]
    w = volume / n_r
    m = w * (psi_occ.conj() @ vx_occ.transpose(0, 1))  # (n_occ, n_occ) M_mn=⟨ψ_m|W_n⟩
    m = 0.5 * (m + m.conj().transpose(0, 1))           # enforce Hermiticity
    neg_m = -m
    # jitter guards the Cholesky if M grazes singular (near-degenerate occupied)
    eye = torch.eye(neg_m.shape[0], dtype=neg_m.dtype, device=neg_m.device)
    jitter = 1e-14 * float(torch.diagonal(neg_m).abs().max())
    ell = torch.linalg.cholesky(neg_m + jitter * eye)  # −M = L Lᴴ
    wc = vx_occ.transpose(0, 1)                        # (N_r, n_occ) columns W_n
    # Ξ = Wc (Lᴴ)⁻¹  ⇔  Ξ Lᴴ = Wc  ⇔  L Ξᴴ = Wcᴴ (lower-triangular solve)
    xi_h = torch.linalg.solve_triangular(ell, wc.conj().transpose(0, 1), upper=False)
    xi = xi_h.conj().transpose(0, 1).contiguous()      # (N_r, n_occ)
    return ACEExchange(xi=xi, volume=volume, n_r=n_r)
