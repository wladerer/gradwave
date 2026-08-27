"""Infrared intensities and a broadened IR spectrum from Z* and Gamma phonons.

The zone-center (q = 0) phonon modes of a polar crystal couple to light through
the Born effective charges: a mode carries an oscillating dipole

    dP_a / dQ_m  ~  sum_{kappa, b} Z*_{kappa, a b} e_m(kappa b) / sqrt(M_kappa),

where ``e_m`` is the mass-weighted eigenvector of the Gamma dynamical matrix and
``M_kappa`` the atomic mass. The IR intensity of mode m is the squared dipole
summed over the three light-polarization directions,

    I_m  =  sum_a | sum_{kappa, b} Z*_{kappa, a b} e_m(kappa b) / sqrt(M_kappa) |^2.

This module builds ``I_m`` from the supercell force constants (via the Gamma
dynamical matrix of :mod:`gradwave.postscf.phonons_supercell`) and the Born
charges (:mod:`gradwave.postscf.born`), flags IR-active modes, and broadens the
stick spectrum with a Lorentzian.

Intensities are in units of e^2 / amu (a fixed overall prefactor is dropped);
only ratios and peak positions are physical, so a ``relative`` column normalized
to the strongest mode is also returned. The non-analytic LO--TO splitting (which
needs the electronic dielectric tensor eps_infinity) is NOT applied here — the
frequencies are the TO-like zone-center modes of the bare dynamical matrix; see
the module note in :func:`ir_intensities`.
"""

from __future__ import annotations

import numpy as np

from gradwave.postscf.phonons import _SQRT_EV_AMU_ANG2_TO_CM1
from gradwave.postscf.phonons_supercell import SupercellMap, dynamical_matrix

# Modes below this |frequency| [cm^-1] are treated as acoustic/soft and skipped
# for the IR-active flag (their Z*-weighted dipole is the acoustic translation).
_ACOUSTIC_TOL_CM1 = 1.0


def gamma_modes(
    phi_home: np.ndarray,
    scmap: SupercellMap,
    masses_amu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Zone-center (Gamma) phonon frequencies and mass-weighted eigenvectors.

    Returns ``(freqs_cm1, eigvecs)`` where ``freqs_cm1`` (3 N_prim,) are the
    signed frequencies [cm^-1] (negative = imaginary) sorted ascending and
    ``eigvecs`` (3 N_prim, 3 N_prim) has the mass-weighted eigenvector of mode m
    in column m (component ``3 kappa + b`` is ``e_m(kappa b)``).
    """
    d = dynamical_matrix(phi_home, scmap, masses_amu, [0.0, 0.0, 0.0])
    w2, vecs = np.linalg.eigh(d)  # ascending, columns = eigenvectors
    freqs = np.sign(w2) * _SQRT_EV_AMU_ANG2_TO_CM1 * np.sqrt(np.abs(w2))
    return freqs, vecs


def ir_intensities(
    phi_home: np.ndarray,
    scmap: SupercellMap,
    masses_amu: np.ndarray,
    born: np.ndarray,
    *,
    active_tol: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Per-mode IR intensities from Gamma phonons and Born charges Z*.

    ``phi_home`` (N_prim, 3, N_sc, 3) are the supercell force constants,
    ``masses_amu`` (N_prim,) the primitive-atom masses, and ``born``
    (N_prim, 3, 3) the Born effective charges ``Z*_{kappa, a b}`` (a = Cartesian
    dipole/polarization component, b = displacement axis).

    Returns a dict of arrays over the 3 N_prim modes: ``frequency_cm1`` (signed),
    ``intensity`` (e^2 / amu), ``intensity_relative`` (normalized to the strongest
    mode), and ``ir_active`` (bool, intensity above ``active_tol`` * max and not
    an acoustic mode).

    Note: these are the bare zone-center (TO-like) modes; the LO--TO non-analytic
    correction is not applied (it needs eps_infinity — deferred).
    """
    freqs, vecs = gamma_modes(phi_home, scmap, masses_amu)
    n_prim = scmap.n_prim
    z = np.asarray(born, dtype=float).reshape(n_prim, 3, 3)
    inv_sqrt_m = 1.0 / np.sqrt(np.asarray(masses_amu, dtype=float))  # (N_prim,)

    n_modes = 3 * n_prim
    intensity = np.zeros(n_modes)
    for m in range(n_modes):
        e = vecs[:, m].reshape(n_prim, 3)  # e_m(kappa, b)
        # dipole_a = sum_{kappa,b} Z*_{kappa,a,b} e_m(kappa,b) / sqrt(M_kappa)
        dipole = np.einsum("kab,kb,k->a", z, e, inv_sqrt_m)  # (3,)
        intensity[m] = float(np.abs(dipole) ** 2 @ np.ones(3))

    imax = float(np.max(intensity)) if intensity.size else 0.0
    relative = intensity / imax if imax > 0 else intensity.copy()
    ir_active = (intensity > active_tol * imax) & (np.abs(freqs) > _ACOUSTIC_TOL_CM1)
    return {
        "frequency_cm1": freqs,
        "intensity": intensity,
        "intensity_relative": relative,
        "ir_active": ir_active,
    }


def lorentzian_spectrum(
    freqs_cm1: np.ndarray,
    intensities: np.ndarray,
    *,
    width: float = 8.0,
    npoints: int = 1200,
    frange: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lorentzian-broadened IR spectrum on a frequency grid.

    ``freqs_cm1`` and ``intensities`` are the per-mode stick positions [cm^-1]
    and heights. ``width`` is the Lorentzian HWHM [cm^-1]. Only modes with
    positive frequency contribute (imaginary/acoustic modes are dropped).
    Returns ``(grid_cm1, spectrum)``.
    """
    freqs = np.asarray(freqs_cm1, dtype=float)
    inten = np.asarray(intensities, dtype=float)
    keep = freqs > _ACOUSTIC_TOL_CM1
    f, w = freqs[keep], inten[keep]
    if frange is None:
        hi = float(f.max()) * 1.15 + 5.0 * width if f.size else 100.0
        frange = (0.0, hi)
    grid = np.linspace(frange[0], frange[1], npoints)
    spectrum = np.zeros_like(grid)
    g2 = width * width
    for f0, amp in zip(f, w, strict=True):
        spectrum += amp * g2 / ((grid - f0) ** 2 + g2)
    return grid, spectrum
