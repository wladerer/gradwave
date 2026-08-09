"""Open-boundary electrostatics spike (S0-electrostatics / ESM core).

PR #265's key separability insight: for a surface, the electrostatics and the
wavefunctions are independent problems, and the electrostatics is the bigger,
cheaper win. A periodic Poisson solve makes a slab feel its images -> a spurious,
box-dependent field that COLLAPSES a real surface dipole (forces both vacuum sides
to one level). The open (Otani-Sugino ESM) solve fixes the FIELD only -- basis
agnostic, so it drops into the DG-ALB Hartree unchanged.

This reproduces the mechanism in pure 1D electrostatics (the G|| = 0 channel): a
NEUTRAL dipole (equal +/- Gaussians) in a box. The physical observable is the
dipole step Dv = v(right vacuum) - v(left vacuum):
  * OPEN Poisson  -> Dv is the true dipole step, box-INDEPENDENT.
  * PERIODIC Poisson -> Dv is collapsed toward 0 and drifts with box size (why a
    periodic slab needs a dipole correction).

Units are arbitrary (v'' = -rho): open vs periodic are compared on the SAME rho, so
only the relative behaviour matters. numpy only.

Run:  uv run python benchmarks/dgalb_crossover/esm_poisson_1d.py
"""

from __future__ import annotations

import numpy as np


def rho_dipole(z, box, sep=2.0, w=0.4):
    """Neutral dipole: +Gaussian and -Gaussian separated by `sep`, centred in box."""
    c = box / 2.0
    zp, zm = c - sep / 2.0, c + sep / 2.0
    g = lambda z0: np.exp(-((z - z0) / w) ** 2) / (w * np.sqrt(np.pi))  # noqa: E731
    return g(zp) - g(zm)                                  # integrates to 0 (neutral)


def poisson_periodic(rho, dz):
    """v'' = -rho with periodic BC (FFT). v_hat = rho_hat / G^2, G!=0; mean set 0."""
    n = rho.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dz)
    rhat = np.fft.fft(rho)
    vhat = np.zeros(n, dtype=complex)
    nz = k != 0
    vhat[nz] = rhat[nz] / (k[nz] ** 2)
    return np.fft.ifft(vhat).real


def poisson_open(rho, dz):
    """v'' = -rho with OPEN BC: zero field at the left boundary (no incoming field),
    integrated outward. For a neutral rho the field returns to zero on the right too,
    so both vacua are field-free and the dipole step survives."""
    e_field = -np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * dz)])  # E=-v'
    v = -np.concatenate([[0.0], np.cumsum(0.5 * (e_field[1:] + e_field[:-1]) * dz)])
    return v - v.mean()


def dipole_step(v, z, box, pad=1.5):
    """v(right vacuum) - v(left vacuum), averaged over the outer `pad` A of each side."""
    left = v[z < pad].mean()
    right = v[z > box - pad].mean()
    return right - left


def main():
    print("# Open vs periodic 1D electrostatics: the surface-dipole step vs box size")
    print("# NEUTRAL dipole; open should hold a constant step, periodic collapses/drifts\n")
    print(f"{'box':>6} {'vac/side':>9} {'Dv_open':>12} {'Dv_periodic':>13}")
    steps_open, steps_per = [], []
    slab = 4.0                                             # dipole region ~ fixed
    for vac in (3.0, 5.0, 8.0, 12.0, 20.0):
        box = slab + 2 * vac
        n = int(box / 0.05)
        z = (np.arange(n) + 0.5) * box / n
        dz = box / n
        rho = rho_dipole(z, box)
        vo = poisson_open(rho, dz)
        vp = poisson_periodic(rho, dz)
        do = dipole_step(vo, z, box)
        dp = dipole_step(vp, z, box)
        steps_open.append(do)
        steps_per.append(dp)
        print(f"{box:>6.1f} {vac:>9.1f} {do:>12.5f} {dp:>13.5f}")

    drift_open = max(steps_open) - min(steps_open)
    drift_per = max(steps_per) - min(steps_per)
    print(f"\n  open      dipole-step drift over box sweep = {drift_open:.2e}")
    print(f"  periodic  dipole-step drift over box sweep = {drift_per:.2e}")
    print(f"  open step ~ {np.mean(steps_open):.4f} (physical), "
          f"periodic step ~ {np.mean(steps_per):.4f} (collapsed toward 0)")
    ok = drift_open < 1e-3 and abs(np.mean(steps_open)) > 10 * abs(np.mean(steps_per))
    verdict = "PASS: open box-independent & resolves the dipole periodic collapses" if ok else "CHECK"
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
