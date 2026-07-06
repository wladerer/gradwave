"""Physical constants and unit conversions — the single source of truth.

gradwave base units: eV (energy), Å (length). Everything entering the code
from UPF files (Ry, Bohr) or QE outputs (Ha in XML) is converted at the
boundary; no other module may carry its own conversion factors.

CODATA 2018 values, matching scipy.constants.
"""

HARTREE_EV = 27.211386245988
RY_EV = HARTREE_EV / 2.0  # 13.605693122994
BOHR_ANG = 0.529177210903

# ħ²/2mₑ in eV·Å² — the plane-wave kinetic prefactor: T(G) = HBAR2_2M |k+G|²
# with |k+G| in Å⁻¹. Identity: ħ²/2mₑ = Ry·a₀².
HBAR2_2M = RY_EV * BOHR_ANG**2

# e²/(4πε₀) in eV·Å — the Coulomb prefactor: V(r) = E2 · q₁q₂/r with r in Å.
# Identity: e²/(4πε₀) = Ha·a₀.
E2 = HARTREE_EV * BOHR_ANG

# Boltzmann constant in eV/K (for converting smearing widths quoted in K).
KB_EV = 8.617333262e-5


def ry_to_ev(x):
    return x * RY_EV


def ha_to_ev(x):
    return x * HARTREE_EV


def bohr_to_ang(x):
    return x * BOHR_ANG


def ang_to_bohr(x):
    return x / BOHR_ANG
