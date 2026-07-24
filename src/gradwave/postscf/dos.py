"""Kernel Polynomial Method DOS — stochastic trace of δ(E − H) (Layer B).

    DOS(E) = Σ_k w_k g_s Tr δ(E − H_k),   δ expanded in Chebyshev polynomials
    of H̃ = (H − b)/a (spectrum mapped into [−1, 1]), moments estimated by
    Hutchinson with complex-Rademacher random vectors, Gibbs oscillations
    damped by the Jackson kernel.

No diagonalization anywhere: cost = n_moments/2 batched H-applies on
(nk, n_random, npw) blocks (the moment-doubling identities yield two moments
per application). Resolution ≈ π·a/n_moments — plane-wave spectra span
hundreds of eV (kinetic term), so thousands of moments buy ~0.1 eV bins.

Statistical noise ~ 1/√(n_random · npw); with npw ~ 10³ even a handful of
random vectors gives sub-percent traces (self-averaging over the basis).

Accuracy note (measured on Si): the TOTAL trace is exact to 0.01%, but
cumulative counts in the sparse valence region carry ~0.5 states of
SPECTRAL LEAKAGE — the Jackson kernel's polynomial far-tails integrated
over the enormous plane-wave conduction continuum above. Leakage falls
with n_moments; for quantitative band-region DOS use ≳2× the moments the
resolution alone would suggest.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.core.batch import BatchedHamiltonian, projectors_b
from gradwave.dtypes import CDTYPE, RDTYPE


def _spectral_bounds(h, bk, n_iter: int = 40):
    """(λ_min, λ_max) across all k by power iteration (5% safety margin)."""
    device = bk.mask.device
    gen = torch.Generator(device="cpu").manual_seed(7)

    def rand_vec():
        v = torch.view_as_complex(
            torch.randn(bk.nk, 1, bk.npw_max, 2, generator=gen, dtype=torch.float64)
        ).to(device) * bk.mask[:, None, :]
        return v / torch.linalg.norm(v, dim=-1, keepdim=True)

    v = rand_vec()
    lam_max = torch.zeros(bk.nk, dtype=RDTYPE, device=device)
    for _ in range(n_iter):
        hv = h.apply(v)
        lam_max = torch.einsum("kbg,kbg->k", v.conj(), hv).real
        v = hv / torch.linalg.norm(hv, dim=-1, keepdim=True)
    lmax = float(lam_max.max())

    v = rand_vec()
    for _ in range(n_iter):
        hv = lmax * v - h.apply(v)  # power iteration on (λ_max·I − H)
        lam = torch.einsum("kbg,kbg->k", v.conj(), hv).real
        v = hv / torch.linalg.norm(hv, dim=-1, keepdim=True)
    lmin = lmax - float(lam.max())
    span = lmax - lmin
    return lmin - 0.025 * span, lmax + 0.025 * span


def _kpm_moments(h, a, b, chi, n_pairs, kw, n_random):
    """Jackson-undamped Chebyshev moments μ_m = Σ_k w_k Tr T_m(H̃) for one
    Hamiltonian, estimated by Hutchinson over the shared random block `chi`.
    Returns μ on [0, 2·n_pairs), k-summed and normalized by n_random (the spin
    degeneracy factor is applied by the caller)."""
    def h_scaled(v):
        return (h.apply(v) - b * v) / a

    mu = torch.zeros(2 * n_pairs, dtype=RDTYPE, device=chi.device)
    t_prev = chi
    t_cur = h_scaled(chi)
    mu0_k = torch.einsum("krg,krg->k", chi.conj(), chi).real
    mu1_k = torch.einsum("krg,krg->k", chi.conj(), t_cur).real
    mu[0] = (kw * mu0_k).sum()
    mu[1] = (kw * mu1_k).sum()
    for m in range(1, n_pairs):
        mu[2 * m] = (kw * (
            2.0 * torch.einsum("krg,krg->k", t_cur.conj(), t_cur).real - mu0_k
        )).sum()
        t_next = 2.0 * h_scaled(t_cur) - t_prev
        mu[2 * m + 1] = (kw * (
            2.0 * torch.einsum("krg,krg->k", t_next.conj(), t_cur).real - mu1_k
        )).sum()
        t_prev, t_cur = t_cur, t_next
    return (mu / n_random).cpu().numpy()


def _eval_dos(mu, a, b, energies, n_pairs):
    """Jackson-damped reconstruction of DOS(E) from moments μ for one channel,
    with the spectrum mapped by (a, b). Evaluated on the shared `energies`
    grid; energies outside this channel's [b−a, b+a] clip to ~0."""
    m_idx = np.arange(2 * n_pairs)
    big_n = 2 * n_pairs + 1
    jackson = ((big_n - m_idx) * np.cos(np.pi * m_idx / big_n)
               + np.sin(np.pi * m_idx / big_n) / np.tan(np.pi / big_n)) / big_n
    e_t = np.clip((np.asarray(energies) - b) / a, -0.999999, 0.999999)
    theta = np.arccos(e_t)
    cheb = np.cos(np.outer(m_idx, theta))  # T_m(Ẽ)
    weights = np.where(m_idx == 0, 1.0, 2.0) * jackson * mu
    return (weights @ cheb) / (np.pi * np.sqrt(1.0 - e_t**2)) / a


@torch.no_grad()
def kpm_dos(
    res,
    n_moments: int = 2000,
    n_random: int = 8,
    energies=None,
    n_energies: int = 800,
    seed: int = 0,
):
    """(energies [eV], DOS [states/eV/cell], info) from the converged potential.

    Collinear path (scalar-relativistic pseudos). For nspin=1 DOS is a 1-D
    array with spin factor 2; for nspin=2 it is (2, n_energies) — spin-up and
    spin-down channels, each with degeneracy 1 — on a shared energy grid, and
    info["nspin"] records which. Spinor (noncollinear) results are a separate
    2-component path and are not handled here.
    """
    nspin = getattr(res, "nspin", 1)
    system = res.system
    bk, grid = system.batch, system.grid
    device = res.v_eff.device
    kw = system.kweights.to(device)

    p_b = projectors_b(bk, system.positions)
    # collinear channels share projectors; only the local potential splits
    veff_s = res.v_eff if nspin == 2 else res.v_eff[None]
    g_spin = 2.0 / nspin  # electrons per state: 2 (nspin=1), 1 per channel (nspin=2)

    # one random block, reused across channels (same basis) for correlated noise
    gen = torch.Generator(device="cpu").manual_seed(seed)
    phases = torch.rand(bk.nk, n_random, bk.npw_max, generator=gen, dtype=torch.float64)
    chi = torch.exp(2j * math.pi * phases).to(device=device, dtype=CDTYPE)
    chi = chi * bk.mask[:, None, :]
    n_pairs = n_moments // 2  # moment doubling: M applies → 2M moments

    bounds, mus, ab = [], [], []
    for sp in range(nspin):
        h = BatchedHamiltonian(bk, grid.shape, veff_s[sp], p_b)
        lmin, lmax = _spectral_bounds(h, bk)
        a = (lmax - lmin) / 2.0
        b = (lmax + lmin) / 2.0
        mu = _kpm_moments(h, a, b, chi, n_pairs, kw, n_random) * g_spin
        bounds.append((lmin, lmax))
        mus.append(mu)
        ab.append((a, b))

    # one energy grid spanning both channels; each channel's series is
    # evaluated on it (out-of-band energies clip to ~0 DOS for that channel).
    lmin_g = min(lo for lo, _ in bounds)
    lmax_g = max(hi for _, hi in bounds)
    if energies is None:
        a_g = (lmax_g - lmin_g) / 2.0
        energies = np.linspace(lmin_g + 0.01 * a_g, lmax_g - 0.01 * a_g, n_energies)
    energies = np.asarray(energies)
    dos_s = [_eval_dos(mus[sp], ab[sp][0], ab[sp][1], energies, n_pairs)
             for sp in range(nspin)]
    dos = dos_s[0] if nspin == 1 else np.stack(dos_s, axis=0)
    info = {"nspin": nspin, "lmin": lmin_g, "lmax": lmax_g,
            "resolution_eV": math.pi * max(a for a, _ in ab) / (2 * n_pairs)}
    return energies, dos, info
