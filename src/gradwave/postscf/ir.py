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
to the strongest mode is also returned.

The bare zone-center modes from :func:`ir_intensities` are the transverse-optic
(TO) frequencies. The longitudinal-optic (LO) modes are recovered by adding the
non-analytic q̂->0 term to the Gamma dynamical matrix,

    Phi^NA_{kappa a, kappa' b}(q̂) = (4 pi / Omega) E2
        (q̂ . Z*_kappa)_a (q̂ . Z*_kappa')_b / (q̂ . eps_inf . q̂),

mass-weighted and diagonalized in :func:`gamma_modes_lo_to`; the full LO--TO-split
analysis (per-direction LO frequencies + a split IR spectrum) is
:func:`lo_to_analysis`. This needs the electronic dielectric tensor ``eps_inf``
(:func:`gradwave.postscf.dielectric.dielectric_born`); with ``eps_inf`` absent
the emitted frequencies are the bare TO modes.
"""

from __future__ import annotations

import numpy as np

from gradwave.constants import E2
from gradwave.postscf.phonons import _SQRT_EV_AMU_ANG2_TO_CM1
from gradwave.postscf.phonons_supercell import SupercellMap, dynamical_matrix

# Modes below this |frequency| [cm^-1] are treated as acoustic/soft and skipped
# for the IR-active flag (their Z*-weighted dipole is the acoustic translation).
_ACOUSTIC_TOL_CM1 = 1.0

# Directions for the isotropic (powder) LO--TO average: the three Cartesian axes
# and the four body diagonals. For a cubic crystal every q̂ gives the same split,
# so the average is exact; for a lower-symmetry cell it is a coarse powder mean
# (a proper powder needs a spherical quadrature — documented on lo_to_analysis).
_ISOTROPIC_QDIRS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0), (1.0, 1.0, -1.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0),
)


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


def _mode_dipole_intensities(
    vecs: np.ndarray, born: np.ndarray, masses_amu: np.ndarray
) -> np.ndarray:
    """IR intensity ``I_m = sum_a |sum_{kappa,b} Z*_{kappa,a,b} e_m(kappa,b) /
    sqrt(M_kappa)|^2`` for each mass-weighted eigenvector (column of ``vecs``).

    ``vecs`` (3 N, n_modes) may be complex (the LO--TO dynamical matrix is complex
    Hermitian); the modulus-squared dipole handles that. Shared by the bare-TO
    :func:`ir_intensities` and the LO--TO :func:`lo_to_analysis`."""
    n_prim = int(np.asarray(masses_amu).shape[0])
    z = np.asarray(born, dtype=float).reshape(n_prim, 3, 3)
    inv_sqrt_m = 1.0 / np.sqrt(np.asarray(masses_amu, dtype=float))  # (N_prim,)
    n_modes = vecs.shape[1]
    intensity = np.zeros(n_modes)
    for m in range(n_modes):
        e = vecs[:, m].reshape(n_prim, 3)  # e_m(kappa, b)
        # dipole_a = sum_{kappa,b} Z*_{kappa,a,b} e_m(kappa,b) / sqrt(M_kappa)
        dipole = np.einsum("kab,kb,k->a", z, e, inv_sqrt_m)  # (3,)
        intensity[m] = float((np.abs(dipole) ** 2).sum())
    return intensity


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
    intensity = _mode_dipole_intensities(vecs, born, masses_amu)

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


# ---------------------------------------------------------------------------
# LO--TO non-analytic correction
# ---------------------------------------------------------------------------


def nonanalytic_force_constants(
    born: np.ndarray,
    eps_inf: np.ndarray,
    qdir: np.ndarray,
    volume: float,
) -> np.ndarray:
    """Non-analytic force-constant contribution Phi^NA (N,3,N,3) [eV/Ang^2] at q̂->0.

        Phi^NA_{kappa a, kappa' b}(q̂) = (4 pi / Omega) E2
            (q̂ . Z*_kappa)_a (q̂ . Z*_kappa')_b / (q̂ . eps_inf . q̂),

    where ``a``/``b`` are the DISPLACEMENT axes of atoms kappa/kappa', ``E2`` is the
    Coulomb prefactor e^2/(4 pi eps_0) [eV.Ang] (:data:`gradwave.constants.E2`),
    ``Omega`` = ``volume`` [Ang^3] the primitive-cell volume, and ``eps_inf`` the
    clamped-ion electronic dielectric tensor (dimensionless). ``born`` is
    ``Z*_{kappa, a, b}`` (a = polarization/Cartesian, b = displacement axis); the
    polarization index is the one contracted with q̂.

    This is a pure rank-1 (in the atom-pair block) formula; it does NOT enforce the
    acoustic sum rule on ``born`` — do that upstream (see :func:`lo_to_analysis`,
    ``enforce_asr``) so the acoustic branch stays unshifted.
    """
    z = np.asarray(born, dtype=float).reshape(-1, 3, 3)  # (N,3,3) Z*_{kappa,a,b}
    q = np.asarray(qdir, dtype=float).reshape(3)
    qn = float(np.linalg.norm(q))
    if qn == 0.0:
        raise ValueError("qdir must be a nonzero direction for the LO--TO term")
    q = q / qn
    eps = np.asarray(eps_inf, dtype=float).reshape(3, 3)
    q_eps_q = float(q @ eps @ q)
    if q_eps_q <= 0.0:
        raise ValueError(f"q̂.eps_inf.q̂ = {q_eps_q} must be positive")
    # (q̂ . Z*_kappa)_b = sum_a q_a Z*_{kappa,a,b}: the mode dipole projected on q̂,
    # resolved per displacement axis b.
    qz = np.einsum("a,kab->kb", q, z)  # (N,3)
    pref = 4.0 * np.pi * E2 / (volume * q_eps_q)
    return pref * np.einsum("kb,lc->kblc", qz, qz)  # (N,3,N,3)


def gamma_modes_lo_to(
    phi_home: np.ndarray,
    scmap: SupercellMap,
    masses_amu: np.ndarray,
    born: np.ndarray,
    eps_inf: np.ndarray,
    qdir: np.ndarray,
    volume: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gamma modes WITH the LO--TO non-analytic term for direction ``qdir``.

    Adds the mass-weighted :func:`nonanalytic_force_constants` to the analytic
    Gamma dynamical matrix before diagonalizing, so the returned frequencies carry
    the longitudinal-optic upshift along q̂. Signature/returns mirror
    :func:`gamma_modes` (``(freqs_cm1, eigvecs)``, ascending). With ``born`` = 0 the
    non-analytic term vanishes identically and this reduces to :func:`gamma_modes`.
    """
    n_prim = scmap.n_prim
    d = dynamical_matrix(phi_home, scmap, masses_amu, [0.0, 0.0, 0.0])  # (3N,3N)
    phi_na = nonanalytic_force_constants(born, eps_inf, qdir, volume)  # (N,3,N,3)
    m = np.sqrt(np.asarray(masses_amu, dtype=float))
    d_na = phi_na / (m[:, None, None, None] * m[None, None, :, None])
    d = d + d_na.reshape(3 * n_prim, 3 * n_prim)
    w2, vecs = np.linalg.eigh(d)
    freqs = np.sign(w2) * _SQRT_EV_AMU_ANG2_TO_CM1 * np.sqrt(np.abs(w2))
    return freqs, vecs


def _asr_correct(born: np.ndarray) -> np.ndarray:
    """Spread the acoustic-sum-rule residual sum_kappa Z*_kappa equally over the
    atoms so it is exactly zero — a prerequisite for the acoustic branch to stay
    unshifted by the non-analytic term (a rigid translation must carry no dipole)."""
    z = np.asarray(born, dtype=float)
    return z - z.sum(axis=0, keepdims=True) / z.shape[0]


def lo_to_analysis(
    phi_home: np.ndarray,
    scmap: SupercellMap,
    masses_amu: np.ndarray,
    born: np.ndarray,
    eps_inf: np.ndarray,
    volume: float,
    *,
    qdir: np.ndarray | None = None,
    broadening: float = 8.0,
    spectrum_points: int = 1200,
    enforce_asr: bool = True,
) -> dict[str, object]:
    """LO--TO-split zone-center analysis: LO frequencies per q̂ and a split spectrum.

    ``eps_inf`` (3,3) is the clamped-ion electronic dielectric tensor
    (:func:`gradwave.postscf.dielectric.dielectric_born`), ``volume`` [Ang^3] the
    primitive-cell volume, ``born`` (N,3,3) the Born charges Z*. If ``enforce_asr``
    the Born tensor is ASR-cleaned first so the acoustic branch stays unshifted.

    With ``qdir`` a nonzero direction the split is evaluated for that single q̂;
    with ``qdir`` None the spectrum is an isotropic average over
    :data:`_ISOTROPIC_QDIRS` (exact for a cubic crystal, a coarse powder mean
    otherwise). Returns a dict with the bare TO frequencies, per-direction LO/TO
    frequencies + intensities, the identified LO frequencies (the optic modes
    upshifted relative to the bare set), and the broadened split IR spectrum.
    """
    z = _asr_correct(born) if enforce_asr else np.asarray(born, dtype=float)
    to_freqs, _ = gamma_modes(phi_home, scmap, masses_amu)  # bare (TO) modes
    n_ac = int((np.abs(to_freqs) <= _ACOUSTIC_TOL_CM1).sum())

    if qdir is not None and float(np.linalg.norm(np.asarray(qdir, float))) > 0.0:
        dirs = [np.asarray(qdir, dtype=float)]
        isotropic = False
    else:
        dirs = [np.asarray(q, dtype=float) for q in _ISOTROPIC_QDIRS]
        isotropic = True

    per_dir: list[dict[str, object]] = []
    lo_per_dir: list[float] = []
    freqs_arr: list[np.ndarray] = []
    inten_arr: list[np.ndarray] = []
    for q in dirs:
        f, v = gamma_modes_lo_to(phi_home, scmap, masses_amu, z, eps_inf, q, volume)
        inten = _mode_dipole_intensities(v, z, masses_amu)
        # LO frequencies: the optic modes shifted relative to the bare set. Both
        # arrays are ascending; the non-analytic term only pushes the longitudinal
        # optic mode(s) up, so the per-rank shift isolates them.
        shift = f - to_freqs
        lo_modes = [float(f[i]) for i in range(len(f))
                    if i >= n_ac and shift[i] > _ACOUSTIC_TOL_CM1]
        qn = q / np.linalg.norm(q)
        freqs_arr.append(f)
        inten_arr.append(inten)
        per_dir.append({
            "qdir": qn.tolist(),
            "frequency_cm1": f.tolist(),
            "intensity": inten.tolist(),
            "lo_frequency_cm1": lo_modes,
            "lo_shift_cm1": shift.tolist(),
        })
        if lo_modes:
            lo_per_dir.append(max(lo_modes))

    # split IR spectrum: average the per-direction Lorentzian spectra on one grid
    fmax = max(float(f.max()) for f in freqs_arr)
    frange = (0.0, fmax * 1.15 + 5.0 * broadening)
    grid = np.linspace(frange[0], frange[1], spectrum_points)
    spec = np.zeros_like(grid)
    for fa, ia in zip(freqs_arr, inten_arr, strict=True):
        _, s = lorentzian_spectrum(
            fa, ia, width=broadening, npoints=spectrum_points, frange=frange)
        spec += s
    spec /= len(freqs_arr)

    return {
        "to_frequency_cm1": to_freqs.tolist(),
        "isotropic": isotropic,
        "asr_enforced": bool(enforce_asr),
        "per_direction": per_dir,
        "lo_frequency_cm1": lo_per_dir,
        "spectrum": {
            "frequency_cm1": grid.tolist(),
            "intensity": spec.tolist(),
            "broadening_cm1": broadening,
        },
    }
