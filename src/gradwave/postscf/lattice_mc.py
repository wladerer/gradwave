"""Lattice Monte Carlo for the configurational free energy (phase-diagram phase 3).

A binary alloy's configurational thermodynamics is a lattice model in the site-
occupation variables. A cluster expansion truncated at pairs is an Ising
Hamiltonian, E = -J sum_<ij> s_i s_j - h sum_i s_i, with s = 2 occupation - 1 and
J the nearest-neighbour effective interaction. Metropolis Monte Carlo on that
model gives the configurational energy, the order parameter, the specific heat,
and the susceptibility as functions of temperature, and the specific-heat peak
locates the order-disorder transition.

Updates are checkerboard, so every site of one sublattice is proposed at once
with numpy. On a bipartite lattice a site's neighbours all lie on the other
sublattice, so a whole colour updates independently in one vectorized step.

Temperature is in energy units (k_B = 1), so for the square-lattice Ising model
with J = 1 the exact transition is at 2 / ln(1 + sqrt(2)) ~ 2.269.
"""

from __future__ import annotations

import numpy as np

_LN2 = float(np.log(2.0))  # two-state disordered entropy per site


def _energy_per_site(s, j, h):
    """Total energy divided by the number of sites. Each bond is counted once by
    rolling only in the positive direction along each axis."""
    bonds = sum((s * np.roll(s, 1, axis=ax)).sum() for ax in range(s.ndim))
    return (-j * bonds - h * s.sum()) / s.size


def _sweep(s, j, h, temp, rng):
    """One checkerboard Metropolis sweep, both sublattices."""
    coords = np.indices(s.shape).sum(axis=0)
    for parity in (0, 1):
        nn = sum(np.roll(s, 1, ax) + np.roll(s, -1, ax) for ax in range(s.ndim))
        d_e = 2.0 * s * (j * nn + h)  # energy change if this site flips
        # accept with prob min(1, exp(-dE/T)); clip keeps dE<=0 always accepted
        accept = rng.random(s.shape) < np.exp(-np.clip(d_e, 0.0, None) / temp)
        flip = (coords % 2 == parity) & accept
        s = np.where(flip, -s, s)
    return s


def simulate(shape, temp, j=1.0, h=0.0, n_equil=1500, n_sample=2500,
             seed=0) -> dict:
    """Metropolis Monte Carlo at one temperature. Returns per-site averages, the
    energy, the absolute order parameter |m|, the specific heat, and the
    susceptibility."""
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1, 1]), size=shape).astype(np.int64)
    for _ in range(n_equil):
        s = _sweep(s, j, h, temp, rng)

    n = s.size
    energies = np.empty(n_sample)
    mags = np.empty(n_sample)
    for i in range(n_sample):
        s = _sweep(s, j, h, temp, rng)
        energies[i] = _energy_per_site(s, j, h)
        mags[i] = s.mean()

    abs_m = float(np.abs(mags).mean())
    return {
        "temp": float(temp),
        "energy": float(energies.mean()),
        "abs_m": abs_m,
        "cv": float(n * energies.var() / temp ** 2),
        "chi": float(n * (np.mean(mags ** 2) - abs_m ** 2) / temp),
    }


def scan_temperature(shape, temps, **kwargs) -> dict:
    """Run `simulate` across a temperature grid. Returns arrays keyed by
    observable, so the specific-heat peak and the order parameter can be read
    off directly."""
    rows = [simulate(shape, t, **kwargs) for t in temps]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def order_disorder_temperature(temps, cv) -> float:
    """The transition temperature estimated from the specific-heat peak."""
    return float(np.asarray(temps)[int(np.argmax(cv))])


def configurational_entropy(temps, cv, s_infinity=_LN2):
    """Configurational entropy per site, integrated down from the high-
    temperature limit, S(T) = S_inf - integral_T^Tmax Cv/T' dT'. s_infinity
    defaults to the two-state disordered value ln 2, so the temperature grid must
    reach well above the transition for the reference to hold."""
    temps = np.asarray(temps, dtype=float)
    cv = np.asarray(cv, dtype=float)
    order = np.argsort(temps)
    temps, cv = temps[order], cv[order]
    integrand = cv / temps
    tail = np.zeros_like(temps)  # int_T^Tmax Cv/T' dT', by a reverse trapezoid
    for i in range(len(temps) - 2, -1, -1):
        tail[i] = tail[i + 1] + 0.5 * (integrand[i] + integrand[i + 1]) * (
            temps[i + 1] - temps[i])
    s = s_infinity - tail
    return s[np.argsort(order)]  # restore the caller's temperature order


def configurational_free_energy(temps, energies, cv, s_infinity=_LN2):
    """Configurational free energy per site F(T) = E(T) - T S(T), with S from the
    high-temperature entropy integration."""
    s = configurational_entropy(temps, cv, s_infinity)
    return np.asarray(energies, dtype=float) - np.asarray(temps, dtype=float) * s
