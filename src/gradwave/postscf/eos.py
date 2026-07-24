"""Birch-Murnaghan equation-of-state fitting and the Lejaeghere Δ-gauge.

Pure post-processing on (volume, energy) points — no SCF dependency, so it is
trivially unit-testable. The E(V) points come from an isotropic volume scan
(see ``api.run_eos``); this module fits the 3rd-order Birch-Murnaghan form and
reports the equilibrium volume V0, bulk modulus B0, and its pressure derivative
B0'.

Base units follow the package: energy in eV, volume in Å³, so the fitted bulk
modulus is eV/Å³ internally and converted to GPa via ``EV_A3_TO_GPA``. The
Δ-value follows calcDelta 3.0 (Lejaeghere et al., Science 351, aad3000, 2016):
both curves shifted to their own minimum, RMS of the difference over
[1-w, 1+w]·V0_avg, in meV/atom.

This function set was previously copy-pasted across ``benchmarks/delta_gauge``,
``benchmarks/delta_factor`` and ``benchmarks/lejaeghere``; those now import it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 1 eV/Å³ in GPa. eV/Å³ = 1.602176634e-19 J / 1e-30 m³ = 1.602176634e11 Pa
# = 160.2176634 GPa.
EV_A3_TO_GPA = 160.2176634


def birch_murnaghan(v, e0, v0, b0, b0p):
    """3rd-order Birch-Murnaghan energy-volume curve E(V).

    ``b0`` carries the same energy/volume units as ``e0``/``v`` (eV/Å³ here);
    ``b0p`` (= dB0/dP) is dimensionless. Accepts a scalar or array ``v``.
    """
    v = np.asarray(v, dtype=float)
    x = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * (
        (x - 1.0) ** 3 * b0p + (x - 1.0) ** 2 * (6.0 - 4.0 * x))


@dataclass(frozen=True)
class BM3Fit:
    """A fitted 3rd-order Birch-Murnaghan curve. ``v0``/``e0`` inherit the
    per-atom (or per-cell) convention of the points passed to ``fit_bm3``."""

    e0: float               # eV — minimum energy
    v0: float               # Å³ — equilibrium volume
    b0: float               # eV/Å³ — bulk modulus
    b0_prime: float         # dimensionless — dB0/dP
    rms_residual_eV: float  # RMS fit residual over the input points

    @property
    def b0_GPa(self) -> float:
        return self.b0 * EV_A3_TO_GPA


def fit_bm3(volumes, energies) -> BM3Fit:
    """Least-squares 3rd-order Birch-Murnaghan fit of E(V).

    Needs at least four points (the form has four parameters). The initial
    guess seeds e0/v0 from the lowest point and B0≈0.6 eV/Å³ (~100 GPa),
    B0'≈4 — the values that make ``curve_fit`` converge across the periodic
    table in the Δ-gauge benchmarks.
    """
    from scipy.optimize import curve_fit

    v = np.asarray(volumes, dtype=float)
    e = np.asarray(energies, dtype=float)
    if v.ndim != 1 or v.shape != e.shape:
        raise ValueError("volumes and energies must be matching 1-D sequences")
    if v.size < 4:
        raise ValueError(
            f"Birch-Murnaghan needs >=4 (volume, energy) points, got {v.size}")
    i = int(np.argmin(e))
    popt, _ = curve_fit(birch_murnaghan, v, e,
                        p0=[e[i], v[i], 0.6, 4.0], maxfev=40000)
    e0, v0, b0, b0p = (float(x) for x in popt)
    resid = birch_murnaghan(v, e0, v0, b0, b0p) - e
    return BM3Fit(e0=e0, v0=v0, b0=b0, b0_prime=b0p,
                  rms_residual_eV=float(np.sqrt(np.mean(resid ** 2))))


def _as_params(fit):
    """Coerce a BM3Fit or a raw (e0, v0, b0, b0p) tuple to a params tuple."""
    if isinstance(fit, BM3Fit):
        return (fit.e0, fit.v0, fit.b0, fit.b0_prime)
    return tuple(float(x) for x in fit)


def delta_value(fit_a, fit_b, window: float = 0.06, npoints: int = 1000) -> float:
    """Lejaeghere Δ (meV) between two BM3 fits.

    RMS of the energy difference over [1-window, 1+window]·V0_avg with each
    curve shifted to its own minimum. Pass per-atom fits (v0 in Å³/atom, b0 in
    eV/Å³) to get Δ in meV/atom. Accepts ``BM3Fit`` objects or raw
    (e0, v0, b0, b0p) tuples (e.g. an all-electron reference).
    """
    a = _as_params(fit_a)
    b = _as_params(fit_b)
    v0av = 0.5 * (a[1] + b[1])
    vv = np.linspace((1.0 - window) * v0av, (1.0 + window) * v0av, npoints)
    d = birch_murnaghan(vv, 0.0, *a[1:]) - birch_murnaghan(vv, 0.0, *b[1:])
    return float(np.sqrt(np.trapezoid(d ** 2, vv) / (vv[-1] - vv[0])) * 1000.0)
