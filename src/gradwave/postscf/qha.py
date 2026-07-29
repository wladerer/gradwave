"""Quasi-harmonic single-phase thermodynamics (phase-diagram phase 1).

The harmonic free energy is fixed at one volume, so it carries no thermal
expansion. Instead the quasi-harmonic approximation makes the phonon frequencies
volume-dependent and minimizes the free energy over volume at each temperature.
From a static energy curve E(V) and a phonon density of states at each volume it
returns the Gibbs free energy G(T, P), the thermal-expanded volume V(T), the
isothermal bulk modulus, and the derived thermal expansion, heat capacity, and
Grueneisen parameter.

At each temperature the total free energy E(V) + F_vib(T, V) + P V is fit with
the Birch-Murnaghan form over the volume grid, and its minimum is the
equilibrium state at that temperature. The minimum shifts to larger volume as
the temperature rises, because a larger volume softens the phonons and lowers
F_vib, which is thermal expansion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gradwave.postscf.eos import fit_bm3
from gradwave.postscf.thermo import free_energy_vib, heat_capacity


@dataclass
class QHAResult:
    temperatures: np.ndarray  # (nt,) K
    volume: np.ndarray        # V(T) Å³
    gibbs: np.ndarray         # G(T, P) eV
    bulk_modulus: np.ndarray  # B(T) eV/Å³
    cv: np.ndarray            # Cv at V(T) eV/K

    def thermal_expansion(self) -> np.ndarray:
        """Volume thermal-expansion coefficient alpha(T) = d ln V / dT [1/K]."""
        return np.gradient(np.log(self.volume), self.temperatures)

    def heat_capacity_p(self) -> np.ndarray:
        """Constant-pressure heat capacity Cp = Cv + alpha^2 B V T [eV/K]."""
        a = self.thermal_expansion()
        return self.cv + a ** 2 * self.bulk_modulus * self.volume * self.temperatures

    def gruneisen(self) -> np.ndarray:
        """Thermodynamic Grueneisen parameter gamma = alpha B V / Cv."""
        a = self.thermal_expansion()
        out = np.zeros_like(a)
        nz = self.cv > 0
        out[nz] = (a[nz] * self.bulk_modulus[nz] * self.volume[nz] / self.cv[nz])
        return out


def qha(
    volumes: np.ndarray,
    energies: np.ndarray,
    dos_per_volume: list[tuple[np.ndarray, np.ndarray]],
    temperatures: np.ndarray,
    pressure: float = 0.0,
) -> QHAResult:
    """Quasi-harmonic Gibbs free energy over a temperature grid.

    volumes (nv,) Å³, energies (nv,) eV static, dos_per_volume a list of
    (freqs_cm, dos) phonon DOS at each volume, temperatures (nt,) K, pressure in
    eV/Å³ (1 GPa is 0.0062415 eV/Å³). The volume grid must bracket the
    thermal-expanded volume over the whole temperature range, otherwise the
    per-temperature fit extrapolates.
    """
    volumes = np.asarray(volumes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    order = np.argsort(volumes)
    volumes, energies = volumes[order], energies[order]
    dos_per_volume = [dos_per_volume[i] for i in order]

    v_t, g_t, b_t, cv_t = [], [], [], []
    for temp in temperatures:
        fvib = np.array([free_energy_vib(f, d, temp) for f, d in dos_per_volume])
        g_of_v = energies + fvib + pressure * volumes
        fit = fit_bm3(volumes, g_of_v)
        cvs = np.array([heat_capacity(f, d, temp) for f, d in dos_per_volume])
        v_t.append(fit.v0)
        g_t.append(fit.e0)
        b_t.append(fit.b0)
        cv_t.append(float(np.interp(fit.v0, volumes, cvs)))
    return QHAResult(temperatures, np.array(v_t), np.array(g_t),
                     np.array(b_t), np.array(cv_t))
