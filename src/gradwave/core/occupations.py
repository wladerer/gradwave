"""Smearing occupations, paired entropy terms, and Fermi-level search (Layer A).

Each scheme is a (occupation, entropy) PAIR kept in one class — mixing the
occupation of one scheme with the entropy of another is a classic silent
error. Definitions follow QE (wgauss/w1gauss):

  x = (ε − μ)/σ, occupations f ∈ [0, 1] (spin factor applied by callers),
  smearing contribution to the free energy: E_smear = −σ Σ_k w_k Σ_n 2·s(x_nk),
  F = E_KS − σS;  E₀ = (E + F)/2 is the σ→0 extrapolation (Gaussian case).

  fermi-dirac: f = 1/(1+e^x),   s = −[f ln f + (1−f)ln(1−f)]
  gaussian:    f = erfc(x)/2,   s = e^{−x²}/(2√π)

Methfessel–Paxton is DEFERRED: its occupation is non-monotone in μ (needs a
bracketed root search, not bisection) and its entropy pairing must be
validated against QE fixtures first. The SCF milestone uses gaussian/FD.
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


SCHEMES = {c.name: c for c in (FermiDirac(), Gaussian())}


def find_fermi(
    eigs: torch.Tensor,
    kweights: torch.Tensor,
    smearing: Smearing,
    width: float,
    n_electrons: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> torch.Tensor:
    """Bisection for μ such that Σ_k w_k Σ_n 2 f((ε−μ)/σ) = N_e.

    eigs: (nk, nb) [eV]; kweights: (nk,) summing to 1. Monotone for FD and
    Gaussian occupations (NOT for Methfessel–Paxton — needs a bracket).
    Runs detached — μ is an implicit function handled analytically in M4.
    """
    eigs = eigs.detach()
    lo = eigs.min() - 10.0 * width - 1.0
    hi = eigs.max() + 10.0 * width + 1.0

    def count(mu):
        f = smearing.occupation((eigs - mu) / width)
        return (2.0 * kweights[:, None] * f).sum()

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
    return 0.5 * (lo + hi)


def occupations_and_entropy(
    eigs: torch.Tensor,
    mu: torch.Tensor,
    smearing: Smearing,
    width: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """f_nk ∈ [0,2] (spin factor 2 included) and Σ_n s_nk per k (no weights).

    Returns (f (nk,nb), s (nk,nb)); E_smear = −σ Σ w_k Σ_n 2·s_nk.
    """
    x = (eigs - mu) / width
    return 2.0 * smearing.occupation(x), smearing.entropy(x)


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
