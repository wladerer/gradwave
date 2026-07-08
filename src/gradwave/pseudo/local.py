"""Local pseudopotential form factors V_loc(|G|) with analytic long-range split.

V_loc(r) → −Z_val·e²/r as r → ∞, which has no naive Fourier transform. Split:

    V_sr(r) = V_loc(r) + Z e² erf(r/r_c)/r        (short-ranged, decays fast)
    V_loc(G) = 4π ∫ V_sr(r) j₀(Gr) r² dr  −  4π Z e² exp(−G² r_c²/4) / G²

The G=0 piece of the analytic tail is excluded (it cancels against the
Hartree G=0 and the Ewald background — see core/energies/total.py for the
ownership table). What survives at G=0 is the finite short-range moment,
the **alpha-Z term**:

    α = 4π ∫ (V_loc(r) + Z e²/r) r² dr
      = 4π ∫ V_sr(r) r² dr + π Z e² r_c²      (∫₀^∞ erfc(r/r_c) r dr = r_c²/4)

contributing E_α = N_electrons · Σ_a α_a / Ω to the total energy.

Units: form factors in eV·Å³ (energy × volume); divide by Ω when assembling.
r_c defaults to 1 Bohr, matching QE's vloc_of_g convention (erf(r) in Bohr).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

from gradwave.constants import BOHR_ANG, E2
from gradwave.pseudo.radial import sbt, simpson
from gradwave.pseudo.upf import UPFData

RC_DEFAULT = BOHR_ANG  # 1 Bohr in Å


def _msh(upf: UPFData) -> int:
    """QE's msh: local-channel integrals stop at 10 bohr (see upf.py)."""
    return upf.msh if upf.msh > 0 else len(upf.r)


def _v_short_range(upf: UPFData, rc: float) -> np.ndarray:
    """V_sr(r) = V_loc(r) + Z e² erf(r/rc)/r on r[:msh], finite r→0 limit."""
    n = _msh(upf)
    r = upf.r[:n]
    zval = upf.z_valence
    erf_over_r = np.empty_like(r)
    nonzero = r > 0
    erf_over_r[nonzero] = erf(r[nonzero] / rc) / r[nonzero]
    erf_over_r[~nonzero] = 2.0 / (np.sqrt(np.pi) * rc)
    return upf.vloc[:n] + zval * E2 * erf_over_r


def vloc_of_g(upf: UPFData, g: np.ndarray, rc: float = RC_DEFAULT) -> np.ndarray:
    """Local form factor v(|G|) in eV·Å³ for |G| > 0 (array, Å⁻¹).

    Any g ≤ a tiny threshold is rejected: the G=0 channel is handled by
    alpha_z(), never here.
    """
    g = np.asarray(g, dtype=np.float64)
    if np.any(g < 1e-10):
        raise ValueError("vloc_of_g is defined for G > 0 only; G=0 belongs to alpha_z()")
    n = _msh(upf)
    vsr = _v_short_range(upf, rc)
    short = 4.0 * np.pi * sbt(0, vsr * upf.r[:n] ** 2, upf.r[:n], upf.rab[:n], g)
    tail = -4.0 * np.pi * upf.z_valence * E2 * np.exp(-0.25 * (g * rc) ** 2) / g**2
    return short + tail


def alpha_z(upf: UPFData, rc: float = RC_DEFAULT) -> float:
    """The G=0 short-range moment α = 4π∫(V_loc + Z e²/r) r² dr, in eV·Å³."""
    n = _msh(upf)
    vsr = _v_short_range(upf, rc)
    short = 4.0 * np.pi * simpson(vsr * upf.r[:n] ** 2, upf.rab[:n])
    return float(short + np.pi * upf.z_valence * E2 * rc**2)
