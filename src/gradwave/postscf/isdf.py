"""Interpolative separable density fitting (ISDF) for orbital pair products.

ISDF is the plane-wave-native form of tensor hypercontraction (Lu & Ying 2015;
Hu, Lin & Yang, JCTC 2017). It factorizes every orbital pair product

    ρ_ij(r) = ψ_i*(r) ψ_j(r)  ≈  Σ_μ ζ_μ(r) ψ_i*(r_μ) ψ_j(r_μ)

over a small set of interpolation points {r_μ} chosen from the real-space grid
by a pivoted QR. The O(N²) pairs collapse onto O(N) interpolation vectors
ζ_μ(r), so an object that costs O(N²) pair FFTs to Coulomb-couple (the Fock
exchange build) becomes O(N_μ) FFTs plus an O(N_μ²) contraction. It is the
substrate the exact-exchange / hybrid-functional work in docs/ideas.md is
sequenced behind, and — because the fit is a linear solve — it stays inside the
differentiable-by-construction design.

This first cut targets a single k-point (Γ), which is the regime for molecules
and large cells. Multi-k exchange (with q = k−k′ momentum transfer and the
associated phases) is a later extension.

## The fit

Following Hu–Lin–Yang, the interpolation vectors minimize, over all pairs,

    Σ_ij ‖ρ_ij(r) − Σ_μ ζ_μ(r) ρ_ij(r_μ)‖²

whose normal equations separate because the pair Gram factorizes through the
single-particle overlap B(r, r_μ) = Σ_i ψ_i(r) ψ_i*(r_μ):

    M(r, μ) = |B(r, r_μ)|²          (grid × points)
    S(ν, μ) = |B(r_ν, r_μ)|²        (points × points, = M restricted)
    ζ = M S⁻¹                        (grid × points)

## Convention

Orbitals here are the cell-periodic sums f_i(r) = Σ_G c_i(G) e^{iG·r} returned
by ``g_to_r`` (i.e. ψ_i = f_i / √Ω; Σ_G|c|² = 1). The physical pair density is
ρ_ij = f_i* f_j / Ω. The Coulomb G=0 term is EXCLUDED, matching
``core/energies/hartree.py``; both the direct and the ISDF exchange builds below
use that same convention, so their agreement is a clean test of the ISDF rank
truncation alone. A physically isolated-molecule EXX would need a truncated
Coulomb kernel or a compensating G=0 term (future work).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gradwave.constants import E2
from gradwave.core.energies.hartree import _inv_g2_masked
from gradwave.core.fftbox import g_to_r, r_to_g
from gradwave.dtypes import CDTYPE, RDTYPE

# Interpolation-point selection stops adding points once the pivoted-QR residual
# column norm falls below this fraction of the largest, i.e. the pair-product
# space is numerically exhausted and further points are redundant.
_QR_RTOL = 1e-12


def orbitals_on_grid(coeffs: torch.Tensor, flat_idx: torch.Tensor, shape) -> torch.Tensor:
    """Occupied-orbital coefficients → flattened real-space values.

    coeffs: (n_orb, npw) sphere coefficients at one k-point (e.g.
    ``res.coeffs[ik][occ]``). Returns f of shape (n_orb, N_r), N_r = ∏shape,
    with f_i(r) = Σ_G c_i(G) e^{iG·r} (the g_to_r convention, no 1/√Ω).
    """
    f = g_to_r(coeffs, flat_idx, shape)  # (n_orb, n1, n2, n3)
    return f.reshape(f.shape[0], -1)


def _pivoted_columns(a: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy column-pivoted QR: indices of the k most-independent columns of a.

    a: (m, n) complex. Selects columns by repeatedly taking the one with the
    largest residual norm after orthogonalizing against those already chosen
    (modified Gram–Schmidt with pivoting), which is the standard ISDF
    interpolation-point selector. Returns a LongTensor of ≤ k column indices;
    stops early if the residual collapses (rank deficiency).
    """
    m, n = a.shape
    r = a.clone()
    colnorm2 = (r.abs() ** 2).sum(dim=0)  # (n,)
    first = float(colnorm2.max())
    pivots: list[int] = []
    for _ in range(min(k, n)):
        p = int(torch.argmax(colnorm2))
        if float(colnorm2[p]) <= _QR_RTOL * first:
            break
        pivots.append(p)
        q = r[:, p].clone()
        q = q / q.norm()
        # project the chosen direction out of every remaining column
        r = r - torch.outer(q, q.conj() @ r)
        colnorm2 = (r.abs() ** 2).sum(dim=0)
        colnorm2[pivots] = -1.0  # never reselect
    return torch.tensor(pivots, dtype=torch.long, device=a.device)


def select_interpolation_points(
    phi_r: torch.Tensor, n_mu: int, *, generator: torch.Generator | None = None,
    sketch: int | None = None,
) -> torch.Tensor:
    """Choose ≤ n_mu interpolation points {r_μ} by pivoted QR on the pair space.

    phi_r: (n_orb, N_r) orbital values on the grid. The pair-product space is
    spanned by {φ_i*(r) φ_j(r)}. For a small orbital count the exact pair matrix
    (n_orb², N_r) is pivoted directly; when ``sketch`` is given (or n_orb is
    large) a randomized Khatri–Rao sketch of ``sketch`` random orbital
    combinations is pivoted instead (Hu–Lin–Yang randomized ISDF), which spans
    the same space at far lower row count. Returns grid indices, shape (≤n_mu,).
    """
    n_orb, _ = phi_r.shape
    if sketch is None and n_orb * n_orb > 256:
        # default to a randomized sketch once the exact pair matrix gets tall
        sketch = max(int(math.ceil(math.sqrt(4 * n_mu))), 2 * n_orb)
    if sketch is None:
        # exact pair matrix rows (i,j): φ_i*(r) φ_j(r)
        pairs = (phi_r.conj()[:, None, :] * phi_r[None, :, :]).reshape(n_orb * n_orb, -1)
    else:
        g1 = torch.randn(n_orb, sketch, generator=generator, dtype=RDTYPE).to(phi_r)
        g2 = torch.randn(n_orb, sketch, generator=generator, dtype=RDTYPE).to(phi_r)
        y1 = g1.T @ phi_r.conj()  # (sketch, N_r)
        y2 = g2.T @ phi_r          # (sketch, N_r)
        pairs = (y1[:, None, :] * y2[None, :, :]).reshape(sketch * sketch, -1)
    return _pivoted_columns(pairs, n_mu)


def build_isdf(phi_r: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Interpolation vectors ζ_μ(r) = (M S⁻¹)(r, μ) for the given points.

    phi_r: (n_orb, N_r); points: (n_mu,) grid indices. Returns ζ of shape
    (N_r, n_mu), real. Solved as a least-squares system in S for stability, since
    the point Gram S can be near-singular when n_mu approaches the pair rank.
    """
    phi_mu = phi_r[:, points]                    # (n_orb, n_mu)
    # B(r, μ) = Σ_i f_i(r) f_i*(r_μ)
    b = phi_r.transpose(0, 1) @ phi_mu.conj()    # (N_r, n_mu)
    m = (b.abs() ** 2).to(RDTYPE)                # (N_r, n_mu)
    s = m[points]                                # (n_mu, n_mu)
    # ζ solves ζ S = M, i.e. Sᵀ ζᵀ = Mᵀ; S is symmetric so lstsq(S, Mᵀ) works.
    zeta = torch.linalg.lstsq(s, m.transpose(0, 1)).solution.transpose(0, 1)
    return zeta.contiguous()


def _coulomb_coupling(zeta: torch.Tensor, shape, g2: torch.Tensor, volume: float) -> torch.Tensor:
    """W_μν = (4π e²/Ω) Σ_{G≠0} ζ_μ*(G) ζ_ν(G)/G²  [eV], the interpolation-vector
    Coulomb matrix. zeta: (N_r, n_mu) real; g2: dense box |G|². Returns (n_mu, n_mu)."""
    n_mu = zeta.shape[1]
    zg = r_to_g(zeta.transpose(0, 1).reshape(n_mu, *shape).to(CDTYPE))  # (n_mu, n1,n2,n3)
    zg = zg.reshape(n_mu, -1)
    inv_g2 = _inv_g2_masked(g2).reshape(-1)
    w = (zg.conj() * inv_g2) @ zg.transpose(0, 1)  # (n_mu, n_mu)
    return (4.0 * math.pi * E2 / volume) * w.real


@dataclass
class ISDFExchange:
    """A built ISDF factorization of the exchange for one orbital set at one k.

    points: (n_mu,) grid indices; phi_mu: (n_orb, n_mu) orbital values at the
    interpolation points; w: (n_mu, n_mu) Coulomb coupling of the interpolation
    vectors. The exchange energy is a single O(n_mu²) contraction (``energy``).
    """

    points: torch.Tensor
    phi_mu: torch.Tensor
    w: torch.Tensor

    @property
    def n_mu(self) -> int:
        return int(self.points.shape[0])

    def energy(self) -> torch.Tensor:
        """E_x = −½ Σ_ij Σ_μν [f_i*(r_μ)f_j(r_μ)]* W_μν [f_i*(r_ν)f_j(r_ν)] / Ω²
        contracted over pairs to −½ Σ_μν W_μν |D_μν|², where
        D_μν = Σ_i f_i*(r_μ) f_i(r_ν). Returns a real scalar [eV]."""
        d = self.phi_mu.conj().transpose(0, 1) @ self.phi_mu  # (n_mu, n_mu)
        s = (d.abs() ** 2).to(RDTYPE)
        return -0.5 * (self.w * s).sum()


def build_exchange(
    phi_r: torch.Tensor, shape, g2: torch.Tensor, volume: float, n_mu: int,
    *, generator: torch.Generator | None = None, sketch: int | None = None,
) -> ISDFExchange:
    """Build the ISDF exchange factorization for orbitals ``phi_r`` at one k.

    phi_r: (n_orb, N_r) occupied-orbital grid values (from ``orbitals_on_grid``).
    Selects points, fits ζ, forms the Coulomb coupling, and stores the orbital
    values at the interpolation points in ``phi_mu``.

    Normalization: ζ fits the *unnormalized* pair product f_i* f_j (not the
    physical ρ = f_i* f_j / Ω). Carrying the two 1/Ω factors of the physical
    pair density through the derivation, both cancel against the Ω in
    ⟨ρ|V|ρ⟩ = Ω·4πe²Σ|ρ(G)|²/G² and the 1/Ω already inside ``_coulomb_coupling``.
    The net result is E_x = −½ Σ_μν W_μν |D_μν|² with W = ``_coulomb_coupling``
    directly and D_μν = Σ_i f_i*(r_μ) f_i(r_ν) from the raw f values in phi_mu.
    """
    points = select_interpolation_points(phi_r, n_mu, generator=generator, sketch=sketch)
    zeta = build_isdf(phi_r, points)
    w = _coulomb_coupling(zeta, shape, g2, volume)
    return ISDFExchange(points=points, phi_mu=phi_r[:, points], w=w)


def exchange_energy_direct(
    phi_r: torch.Tensor, shape, g2: torch.Tensor, volume: float,
) -> torch.Tensor:
    """Reference plane-wave Fock exchange, O(n_orb²) pair FFTs.

    E_x = −½ Σ_ij Ω · 4π e² Σ_{G≠0} |ρ_ij(G)|² / G²,  ρ_ij = f_i* f_j / Ω.
    Uses the same G=0-excluded Coulomb convention as the ISDF build, so the two
    agree in the large-n_mu limit. phi_r: (n_orb, N_r). Returns a scalar [eV]."""
    n_orb = phi_r.shape[0]
    inv_g2 = _inv_g2_masked(g2).reshape(-1)
    prefac = 0.5 * volume * 4.0 * math.pi * E2 / (volume ** 2)
    total = phi_r.new_zeros((), dtype=RDTYPE)
    for i in range(n_orb):
        rho = phi_r[i].conj()[None, :] * phi_r  # (n_orb, N_r), pairs (i, j)
        rho_g = r_to_g(rho.reshape(n_orb, *shape)).reshape(n_orb, -1)
        total = total - prefac * ((rho_g.abs() ** 2) * inv_g2).sum()
    return total
