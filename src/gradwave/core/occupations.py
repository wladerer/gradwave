"""Smearing occupations, paired entropy terms, and Fermi-level search (Layer A).

Each scheme is a (occupation, entropy) PAIR kept in one class — mixing the
occupation of one scheme with the entropy of another is a classic silent
error. Definitions follow QE (wgauss/w1gauss):

  x = (ε − μ)/σ, occupations f ∈ [0, 1] (spin factor applied by callers),
  smearing contribution to the free energy: E_smear = −σ Σ_k w_k Σ_n 2·s(x_nk),
  F = E_KS − σS;  E₀ = (E + F)/2 is the σ→0 extrapolation (Gaussian case).

Every scheme derives from a smeared delta δ̃(t) = −f′(t):

    f(x) = ∫_x^∞ δ̃(t) dt          (occupation)
    s(x) = ∫_x^∞ t·δ̃(t) dt        (generalized entropy; E_smear = −σ Σ 2w s)

tests/unit/test_occupations.py verifies the (f, s) pairing for every scheme
by computing δ̃ from f via autograd and quadrature-integrating t·δ̃.

  fermi-dirac: f = 1/(1+e^x),   s = −[f ln f + (1−f)ln(1−f)]
  gaussian:    f = erfc(x)/2,   s = e^{−x²}/(2√π)
  mp1 (Methfessel–Paxton, order 1; kernel D₁ = (3/2 − t²)e^{−t²}/√π):
               f = erfc(x)/2 − x e^{−x²}/(2√π)
               s = (1 − 2x²) e^{−x²}/(4√π)          (negative for |x| > 1/√2)
  cold (Marzari–Vanderbilt; u = x + 1/√2):
               f = erfc(u)/2 + e^{−u²}/√(2π)
               s = u e^{−u²}/√(2π)

mp1/cold occupations are locally non-monotone in μ (f can slightly exceed
[0,1]); plain bisection on N(μ) still converges in practice (QE does the
same) because the wiggles are small and local.
"""

from __future__ import annotations

import math

import torch


class Smearing:
    name: str = "none"

    def occupation(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        """s(x) ≥ 0 such that E_smear = −σ · Σ 2·w·s."""
        raise NotImplementedError


class FermiDirac(Smearing):
    name = "fermi-dirac"

    def occupation(self, x):
        return torch.sigmoid(-x)

    def entropy(self, x):
        # −[f ln f + (1−f) ln(1−f)], computed stably via softplus:
        # s = softplus(−|x|) + |x|·f(|x|)  (symmetric in x)
        ax = torch.abs(x)
        return torch.nn.functional.softplus(-ax) + ax * torch.sigmoid(-ax)


class Gaussian(Smearing):
    name = "gaussian"

    def occupation(self, x):
        return 0.5 * torch.erfc(x)

    def entropy(self, x):
        return torch.exp(-x * x) / (2.0 * math.sqrt(math.pi))


class MethfesselPaxton1(Smearing):
    """MP order 1 (QE smearing='mp'). Kernel D₁(t) = (3/2 − t²)e^{−t²}/√π."""

    name = "mp1"

    def occupation(self, x):
        return 0.5 * torch.erfc(x) - x * torch.exp(-x * x) / (2.0 * math.sqrt(math.pi))

    def entropy(self, x):
        return (1.0 - 2.0 * x * x) * torch.exp(-x * x) / (4.0 * math.sqrt(math.pi))


class MarzariVanderbilt(Smearing):
    """Cold smearing (QE smearing='mv'/'cold')."""

    name = "cold"

    def occupation(self, x):
        u = x + 1.0 / math.sqrt(2.0)
        return 0.5 * torch.erfc(u) + torch.exp(-u * u) / math.sqrt(2.0 * math.pi)

    def entropy(self, x):
        u = x + 1.0 / math.sqrt(2.0)
        return u * torch.exp(-u * u) / math.sqrt(2.0 * math.pi)


SCHEMES = {
    c.name: c for c in (FermiDirac(), Gaussian(), MethfesselPaxton1(), MarzariVanderbilt())
}


def find_fermi(
    eigs: torch.Tensor,
    kweights: torch.Tensor,
    smearing: Smearing,
    width: float,
    n_electrons: float,
    tol: float = 1e-12,
    max_iter: int = 200,
    degeneracy: float = 2.0,
) -> torch.Tensor:
    """Bisection for μ such that Σ_k w_k Σ_n g·f((ε−μ)/σ) = N_e
    (g = degeneracy: 2 spin-restricted, 1 per collinear spin channel).

    eigs: (nk, nb) [eV]; kweights: (nk,) summing to 1. Monotone for FD and
    Gaussian occupations (NOT for Methfessel–Paxton — needs a bracket).
    Runs detached — μ is an implicit function handled analytically in M4.
    """
    device = eigs.device
    # bisect on a CPU copy: the eigenvalue array is tiny, and every bracket
    # comparison on a device tensor is an implicit host sync — ~45 serialized
    # GPU stalls per SCF iteration for pure latency (CPU runs: a no-op move)
    eigs = eigs.detach().cpu()
    kweights = kweights.detach().cpu()
    lo = eigs.min() - 10.0 * width - 1.0
    hi = eigs.max() + 10.0 * width + 1.0

    def count(mu):
        f = smearing.occupation((eigs - mu) / width)
        return (degeneracy * kweights[:, None] * f).sum()

    if not (count(lo) <= n_electrons <= count(hi)):
        raise RuntimeError("Fermi bisection bracket failed — not enough bands?")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if count(mid) < n_electrons:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (0.5 * (lo + hi)).to(device)


def occupations_and_entropy(
    eigs: torch.Tensor,
    mu: torch.Tensor,
    smearing: Smearing,
    width: float,
    degeneracy: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """f_nk ∈ [0, g] and s_nk per state; E_smear = −σ Σ w_k Σ_n g·s_nk."""
    x = (eigs - mu) / width
    return degeneracy * smearing.occupation(x), smearing.entropy(x)


def fixed_occupations(eigs: torch.Tensor, n_electrons: float) -> torch.Tensor:
    """Insulator: fill the lowest N_e/2 bands at every k with f = 2."""
    nk, nb = eigs.shape
    nocc = int(round(n_electrons / 2.0))
    if abs(n_electrons / 2.0 - nocc) > 1e-8:
        raise ValueError("odd electron count needs smearing (spin-restricted code)")
    if nocc > nb:
        raise ValueError(f"need at least {nocc} bands, have {nb}")
    f = torch.zeros_like(eigs)
    f[:, :nocc] = 2.0
    return f
