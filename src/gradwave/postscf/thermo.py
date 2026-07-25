"""Harmonic thermodynamics from a phonon density of states.

Each phonon mode is an independent quantum harmonic oscillator, so the cell
partition function factorizes over the DOS and every thermodynamic quantity is a
single frequency integral against g(ω). This module turns the phonon DOS from
``phonons_supercell.phonon_dos`` into zero-point energy, heat capacity, entropy,
and vibrational free energy, plus a Debye-temperature estimate. It also carries
the Sommerfeld electronic heat capacity so a metal's low-temperature Cv can be
assembled from a phonon DOS and an electronic DOS together.

Units follow the package. Frequencies enter in cm⁻¹, temperatures in K, and
every energy comes out in eV. The DOS is trusted to integrate to the mode count,
``∫ g(ω) dω = 3·N_atoms``, so no internal renormalization is applied and the
caller controls the mode budget through the DOS weights.
"""

from __future__ import annotations

import numpy as np

# Boltzmann constant in eV/K.
K_B = 8.617333262e-5
# One cm⁻¹ expressed in eV, so ħω[eV] = freq[cm⁻¹]·CM1_TO_EV.
CM1_TO_EV = 1.239841984e-4


def _positive_grid(freqs_cm, dos):
    """Frequency grid and DOS with imaginary (negative) branches removed.

    A supercell DOS can carry negative frequencies where the dynamical matrix has
    imaginary modes, and those have no thermal occupation. Zeroing the DOS there
    rather than clipping the grid keeps the trapezoid spacing intact so the
    integral stays first-moment accurate.
    """
    freqs_cm = np.asarray(freqs_cm, dtype=float)
    dos = np.asarray(dos, dtype=float)
    g = np.where(freqs_cm > 0.0, dos, 0.0)
    return freqs_cm, g


def mode_count(freqs_cm, dos) -> float:
    """Total number of modes ∫ g(ω) dω over the real branches.

    This is the mode budget every other quantity assumes, and for a clean DOS it
    equals 3·N_atoms. Report it next to any per-cell result to confirm the DOS
    normalization the caller handed in.
    """
    freqs_cm, g = _positive_grid(freqs_cm, dos)
    return float(np.trapezoid(g, freqs_cm))


def zero_point_energy(freqs_cm, dos) -> float:
    """Zero-point energy ½ ∫ ħω g(ω) dω in eV per cell.

    Each oscillator contributes ½ħω at T=0, so the cell ZPE is the first moment
    of the DOS scaled by ½. The result assumes ∫ g dω = 3·N_atoms modes, which
    ``mode_count`` exposes for checking.
    """
    freqs_cm, g = _positive_grid(freqs_cm, dos)
    hbar_omega = freqs_cm * CM1_TO_EV
    return float(0.5 * np.trapezoid(hbar_omega * g, freqs_cm))


def _cv_integrand(x):
    """Einstein specific-heat kernel x²eˣ/(eˣ−1)² evaluated as (x/2)²/sinh²(x/2).

    The sinh form is the same function with the overflow-prone eˣ moved inside a
    bounded ratio, so large x underflows smoothly to zero instead of dividing two
    infinities. Small x is replaced by its limit 1 to avoid a 0/0.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    small = x < 1e-8
    large = x > 700.0
    mid = ~small & ~large
    out[small] = 1.0
    xm = x[mid]
    s = np.sinh(0.5 * xm)
    out[mid] = (0.5 * xm / s) ** 2
    return out


def heat_capacity(freqs_cm, dos, T) -> float:
    """Vibrational heat capacity Cv in eV/K per cell at temperature T [K].

    Cv = k_B ∫ g(ω) x²eˣ/(eˣ−1)² dω with x = ħω/k_B T, the sum of Einstein terms
    over the DOS. At T→0 every kernel vanishes so Cv is exactly zero, and at high
    T every kernel saturates to 1 so Cv → 3·N_atoms·k_B, the Dulong-Petit limit.
    """
    if T <= 0.0:
        return 0.0
    freqs_cm, g = _positive_grid(freqs_cm, dos)
    x = (freqs_cm * CM1_TO_EV) / (K_B * T)
    return float(K_B * np.trapezoid(g * _cv_integrand(x), freqs_cm))


def heat_capacity_in_kB(freqs_cm, dos, T) -> float:
    """Heat capacity in units of k_B, so the Dulong-Petit limit reads 3·N_atoms.

    This is ``heat_capacity`` divided by k_B. Use it to read Cv directly against
    the mode count without carrying the eV/K scale.
    """
    return heat_capacity(freqs_cm, dos, T) / K_B


def _bose_terms(freqs_cm, dos, T):
    """Positive-branch grid, DOS, and x = ħω/k_B T for the finite-T integrals.

    Shared by entropy, internal energy, and free energy so all three trapezoid
    over the identical mask and grid. That shared discretization is what makes
    F = U − T·S hold to rounding rather than to integration error.
    """
    freqs_cm, g = _positive_grid(freqs_cm, dos)
    x = (freqs_cm * CM1_TO_EV) / (K_B * T)
    return freqs_cm, g, x


def internal_energy_vib(freqs_cm, dos, T) -> float:
    """Vibrational internal energy U in eV per cell at temperature T [K].

    U = ZPE + k_B T ∫ g(ω) x/(eˣ−1) dω, the zero-point energy plus the thermal
    occupation of each mode. At T→0 the occupation term vanishes and U reduces to
    the zero-point energy.
    """
    zpe = zero_point_energy(freqs_cm, dos)
    if T <= 0.0:
        return zpe
    freqs_cm, g, x = _bose_terms(freqs_cm, dos, T)
    # x/(eˣ−1) → 1 as x→0 and → 0 as x→∞. Clipping x into [1e-300, 700] keeps
    # expm1 finite and non-zero, so both limits come out without a 0/0 warning.
    xc = np.clip(x, 1e-300, 700.0)
    occ = xc / np.expm1(xc)
    return float(zpe + K_B * T * np.trapezoid(g * occ, freqs_cm))


def entropy(freqs_cm, dos, T) -> float:
    """Vibrational entropy S in eV/K per cell at temperature T [K].

    S = k_B ∫ g(ω) [ x/(eˣ−1) − ln(1−e⁻ˣ) ] dω, the Legendre partner of U and F.
    It is non-negative and rises monotonically with T, and at T→0 it is zero
    because every mode is frozen into its ground state.
    """
    if T <= 0.0:
        return 0.0
    freqs_cm, g, x = _bose_terms(freqs_cm, dos, T)
    xc = np.clip(x, 1e-300, 700.0)
    occ = xc / np.expm1(xc)
    # ln(1−e⁻ˣ) → 0 as x→∞ and → ln(x) as x→0; both branches stay finite here.
    log_term = np.log(-np.expm1(-xc))
    integrand = g * (occ - log_term)
    return float(K_B * np.trapezoid(integrand, freqs_cm))


def free_energy_vib(freqs_cm, dos, T) -> float:
    """Vibrational Helmholtz free energy F in eV per cell at temperature T [K].

    F = ZPE + k_B T ∫ g(ω) ln(1−e⁻ˣ) dω, which equals U − T·S by construction
    because all three integrals share one grid and mask. At T→0 the thermal term
    vanishes and F reduces to the zero-point energy.
    """
    zpe = zero_point_energy(freqs_cm, dos)
    if T <= 0.0:
        return zpe
    freqs_cm, g, x = _bose_terms(freqs_cm, dos, T)
    xc = np.clip(x, 1e-300, 700.0)
    log_term = np.log(-np.expm1(-xc))
    return float(zpe + K_B * T * np.trapezoid(g * log_term, freqs_cm))


def debye_temperature(freqs_cm, dos) -> float:
    """Debye temperature θ_D in K from the second moment of the phonon DOS.

    A Debye DOS g∝ω² on [0, ω_D] has ⟨ω²⟩ = (3/5)ω_D², so matching the actual
    second moment ⟨ω²⟩ = ∫ω²g dω / ∫g dω gives ω_D = √(5/3·⟨ω²⟩) and
    θ_D = ħω_D/k_B. This moment estimate is basis-set free and needs no fit.
    """
    freqs_cm, g = _positive_grid(freqs_cm, dos)
    norm = np.trapezoid(g, freqs_cm)
    if norm <= 0.0:
        return 0.0
    w2 = np.trapezoid((freqs_cm ** 2) * g, freqs_cm) / norm
    omega_d_cm = np.sqrt(5.0 / 3.0 * w2)
    return float(omega_d_cm * CM1_TO_EV / K_B)


def electronic_heat_capacity(dos_energy_eV, dos_states, e_fermi, T) -> float:
    """Sommerfeld electronic heat capacity in eV/K per cell at temperature T [K].

    Cel = (π²/3) k_B² T · g(E_F), the leading low-temperature term for a metal
    set by the electronic DOS at the Fermi level. g(E_F) is linearly interpolated
    from the DOS grid, and a spin-resolved (2, n) DOS is summed over channels to
    match ``postscf/dos.kpm_dos`` so both spin conventions give total states/eV.
    """
    if T <= 0.0:
        return 0.0
    energy = np.asarray(dos_energy_eV, dtype=float)
    states = np.asarray(dos_states, dtype=float)
    if states.ndim == 2:
        g_ef = sum(np.interp(e_fermi, energy, states[s]) for s in range(states.shape[0]))
    else:
        g_ef = np.interp(e_fermi, energy, states)
    return float((np.pi ** 2 / 3.0) * K_B ** 2 * T * g_ef)
