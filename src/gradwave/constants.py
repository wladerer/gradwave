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

# (−i)^l phase for real-space projector/harmonic transforms, scalar-indexed as
# MINUS_I_POW[l]. Tabulated to l = 4 (covers every caller); a longer table than
# a given caller needs is harmless since it is only ever indexed.
MINUS_I_POW = (1.0 + 0.0j, -1.0j, -1.0 + 0.0j, 1.0j, 1.0 + 0.0j)

# 1 eV/Å³ in GPa (stress/pressure conversion): 1.602176634e-19 J / 1e-30 m³
# = 1.602176634e11 Pa = 160.2176634 GPa, with the exact SI-2018 elementary
# charge. Import THESE — five modules once carried private copies and one
# (api/relax.py) drifted to the CODATA-2014 value.
EV_A3_TO_GPA = 160.2176634
EV_A3_TO_KBAR = 10.0 * EV_A3_TO_GPA
