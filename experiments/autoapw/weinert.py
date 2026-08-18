"""PROD-F — Weinert's method for the FLAPW crystal Coulomb (interstitial + muffin-tin).

Weinert (J. Math. Phys. 22, 2433 (1981)) solves the periodic Poisson equation for a density split
into a smooth interstitial part (plane waves) and sharp muffin-tin parts (radial × Y_lm), which is
the crux of FLAPW self-consistency. Algorithm:

  1. interstitial density ρ_I(G) is defined over the whole cell (a smooth plane-wave continuation).
  2. for each sphere a, the multipole moments of (true ρ_MT,a − ρ_I continued inside) are matched
     by a SMOOTH pseudocharge ρ̃_a with the same moments but no sharp features at R_a.
  3. ρ̃ = ρ_I + Σ_a ρ̃_a is smooth everywhere → solve V_H via FFT (correct in the interstitial by
     the exterior-multipole theorem).
  4. inside each sphere, solve the radial Poisson for the TRUE ρ_MT,a with Dirichlet boundary
     V(R_a) = interstitial V_H(R_a).

Units: eV/Å, ρ in e/Å³, E2 = e² [eV·Å]; Poisson ∇²V = −4π E2 ρ, so V_H(G) = 4π E2 ρ(G)/|G|².

This module builds it in verifiable stages. Stage 1 (here): the interstitial FFT Poisson, checked
against the analytic potential of a periodic Gaussian charge.

    uv run python experiments/autoapw/weinert.py
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf

from gradwave.constants import E2


def g2_grid(N, L):
    """|G|² on the FFT grid (Å⁻²) and the G vectors."""
    f = np.fft.fftfreq(N, d=1.0 / N) * (2 * math.pi / L)      # (N,)
    Gx, Gy, Gz = np.meshgrid(f, f, f, indexing="ij")
    return Gx**2 + Gy**2 + Gz**2, (Gx, Gy, Gz)


def fft_poisson(rho_r, L):
    """Periodic Hartree potential (eV) of a smooth density rho_r (e/Å³) on a cubic grid.
    V_H(G) = 4π E2 ρ(G)/|G|², with the G=0 term set to 0 (neutralising background)."""
    N = rho_r.shape[0]
    G2, _ = g2_grid(N, L)
    rho_g = np.fft.fftn(rho_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        Vg = np.where(G2 > 1e-12, 4 * math.pi * E2 * rho_g / G2, 0.0)
    return np.fft.ifftn(Vg).real


def _stage1_gaussian_check():
    """A neutral density (Gaussian + uniform background) — the FFT Hartree of the Gaussian part
    must match the analytic Gaussian potential V(r) = q·E2·erf(√α r)/r away from periodic images."""
    L, N = 12.0, 96
    alpha = 4.0           # Å⁻²  (width ~0.5 Å)
    q = 1.0               # electrons
    ax = np.arange(N) * (L / N)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    c = L / 2

    def mi(a):
        return a - L * np.round(a / L)

    d = np.sqrt(mi(X - c) ** 2 + mi(Y - c) ** 2 + mi(Z - c) ** 2)
    rho = q * (alpha / math.pi) ** 1.5 * np.exp(-alpha * d**2)      # normalised to q
    rho = rho - q / L**3                                            # neutralise (background)
    V = fft_poisson(rho, L)

    # sample EXACT grid points along +x from the centre; r = k·h so num and analytic align
    ic = N // 2
    h = L / N
    ks = np.arange(2, 28, 2)                          # grid steps out (r up to ~3.4 Å)
    rs = ks * h
    Vnum = V[ic + ks, ic, ic] - V[ic, ic, ic]        # reference to the centre value
    Vana = q * E2 * erf(math.sqrt(alpha) * rs) / rs
    Vana = Vana - q * E2 * 2 * math.sqrt(alpha / math.pi)          # analytic V(0), same reference
    rel = np.abs(Vnum - Vana) / np.abs(Vana).max()
    print("  Stage 1 — interstitial FFT Poisson vs analytic periodic Gaussian:")
    print(f"  {'r (Å)':>7} | {'V_num (eV)':>11} | {'V_analytic':>11} | {'|Δ|':>8}")
    for i in range(0, len(rs), 2):
        print(f"  {rs[i]:>7.2f} | {Vnum[i]:>11.3f} | {Vana[i]:>11.3f} | "
              f"{abs(Vnum[i] - Vana[i]):>8.3f}")
    ok = float(rel[rs < 3.0].max()) < 0.02
    print(f"\n  Stage 1 VERDICT (FFT Poisson correct): {'PASS' if ok else 'FAIL'} "
          f"(max rel dev r<3Å = {float(rel[rs < 3.0].max()):.2e})")
    return ok


def main():
    print("\nPROD-F — Weinert crystal Coulomb (staged build)\n")
    _stage1_gaussian_check()


if __name__ == "__main__":
    main()
