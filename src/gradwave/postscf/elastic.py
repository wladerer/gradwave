"""Elastic constants by finite difference of the analytic stress tensor.

C_ij = ∂σ_i/∂ε_j (Voigt 6×6) is obtained by applying each of the six symmetric
strains ±h, re-converging the SCF, and central-differencing gradwave's analytic
stress ``postscf.stress.stress`` (norm-conserving) / ``paw_stress.stress_uspp``
(USPP/PAW). gradwave's stress is +(1/Ω)∂E/∂ε, so C comes out positive-definite
for a mechanically stable crystal with no sign flip.

This module is the pure-numerics half (Voigt bookkeeping, the FD driver over a
caller-supplied ``stress_at_strain`` closure, and the Voigt–Reuss–Hill
polycrystalline averages); the SCF/strain plumbing lives in ``api.run_elastic``.

The C returned is the CLAMPED-ION elastic tensor: the cell is strained with
fractional coordinates held fixed and only the electrons re-relaxed. This is
exact for any constant with no symmetry-allowed internal displacement — the
bulk modulus of any crystal, and every constant of rocksalt (MgO, NaCl) where
the atoms stay on their special positions under strain. It is NOT exact for
shear constants of the diamond/zincblende structure (Si, C, GaAs): a shear
strain there induces an internal sublattice shift (the Kleinman internal
parameter), so the clamped-ion C44 overestimates the fully-relaxed value
(PBE Si: clamped-ion C44 ≈ 98 GPa vs relaxed ≈ 76). C11/C12 and hence the bulk
modulus are unaffected. Adding the internal-strain correction (coupling the
strain response to the Γ displacement Hessian) is a documented follow-up.

Units: stress in eV/Å³, so C is eV/Å³ internally and reported in GPa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EV_A3_TO_GPA = 160.2176634

# Voigt index (0-based) → tensor (i, j) component, engineering-shear convention:
# ε_voigt = [ε_xx, ε_yy, ε_zz, 2ε_yz, 2ε_xz, 2ε_xy], σ_voigt = [σ_xx, …, σ_xy].
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def voigt_strain_tensor(j: int, h: float) -> np.ndarray:
    """Symmetric 3×3 strain for a unit Voigt perturbation ``h`` in direction j.

    Engineering-shear convention: a shear Voigt component ε_4 = 2·ε_yz, so the
    off-diagonal tensor entry is h/2 (setting ε_voigt exactly to h·e_j)."""
    eps = np.zeros((3, 3))
    a, b = _VOIGT[j]
    if a == b:
        eps[a, b] = h
    else:
        eps[a, b] = eps[b, a] = 0.5 * h
    return eps


def stress_to_voigt(sigma) -> np.ndarray:
    """(3,3) stress → 6-vector [xx, yy, zz, yz, xz, xy]."""
    s = np.asarray(sigma, dtype=float)
    return np.array([s[i, j] for i, j in _VOIGT])


def elastic_tensor(stress_at_strain, h: float = 0.005) -> np.ndarray:
    """Clamped-ion stiffness C (6×6) [GPa] by central FD of the stress.

    ``stress_at_strain(eps)`` takes a symmetric 3×3 strain tensor, applies it to
    the cell, re-converges the SCF and returns the analytic stress (3,3) in
    eV/Å³. Column j is C[:, j] = ∂σ/∂ε_j; the result is symmetrized
    (C = ½(C+Cᵀ)) and converted to GPa.
    """
    c = np.zeros((6, 6))
    for j in range(6):
        sp = stress_to_voigt(stress_at_strain(voigt_strain_tensor(j, +h)))
        sm = stress_to_voigt(stress_at_strain(voigt_strain_tensor(j, -h)))
        c[:, j] = (sp - sm) / (2.0 * h)
    c = 0.5 * (c + c.T)
    return c * EV_A3_TO_GPA


@dataclass(frozen=True)
class Moduli:
    """Voigt–Reuss–Hill polycrystalline averages [GPa] (Poisson dimensionless)."""

    bulk_voigt: float
    bulk_reuss: float
    bulk_hill: float
    shear_voigt: float
    shear_reuss: float
    shear_hill: float
    young: float          # Hill Young's modulus
    poisson: float        # Hill Poisson ratio


def moduli_from_cij(c) -> Moduli:
    """Voigt–Reuss–Hill averages from a 6×6 stiffness [GPa].

    Voigt (uniform strain) and Reuss (uniform stress, via the compliance
    S = C⁻¹) bounds, Hill = their mean; Young's modulus and Poisson ratio from
    the Hill K and G."""
    c = np.asarray(c, dtype=float)
    tr3 = c[0, 0] + c[1, 1] + c[2, 2]
    off3 = c[0, 1] + c[0, 2] + c[1, 2]
    sh3 = c[3, 3] + c[4, 4] + c[5, 5]
    kv = (tr3 + 2.0 * off3) / 9.0
    gv = (tr3 - off3 + 3.0 * sh3) / 15.0

    s = np.linalg.inv(c)
    str3 = s[0, 0] + s[1, 1] + s[2, 2]
    sof3 = s[0, 1] + s[0, 2] + s[1, 2]
    ssh3 = s[3, 3] + s[4, 4] + s[5, 5]
    kr = 1.0 / (str3 + 2.0 * sof3)
    gr = 15.0 / (4.0 * str3 - 4.0 * sof3 + 3.0 * ssh3)

    kh, gh = 0.5 * (kv + kr), 0.5 * (gv + gr)
    young = 9.0 * kh * gh / (3.0 * kh + gh)
    poisson = (3.0 * kh - 2.0 * gh) / (2.0 * (3.0 * kh + gh))
    return Moduli(bulk_voigt=kv, bulk_reuss=kr, bulk_hill=kh,
                  shear_voigt=gv, shear_reuss=gr, shear_hill=gh,
                  young=young, poisson=poisson)


def is_mechanically_stable(c) -> bool:
    """Born stability: the 6×6 stiffness is positive-definite (all eigenvalues
    > 0). A symmetric, weakly-negative eigenvalue from FD noise is treated as
    unstable — tighten the SCF/strain if a stable crystal trips this."""
    return bool(np.all(np.linalg.eigvalsh(np.asarray(c, dtype=float)) > 0.0))
