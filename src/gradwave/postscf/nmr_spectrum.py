"""Solid-state NMR lineshape synthesis: NMR tensors -> simulated powder spectra.

This is the spectrum-synthesis layer above the two gradwave NMR back ends. It
consumes plain per-site scalars -- isotropic shift ``delta_iso`` (ppm), reduced
CSA anisotropy ``delta_aniso`` (ppm, Haeberlen), CSA asymmetry ``eta_csa``,
quadrupolar coupling ``C_Q`` (Hz), quadrupolar asymmetry ``eta_q``, nuclear spin
``I`` and Larmor frequency ``nu0`` (Hz) -- and returns ``(ppm_axis, intensity)``
powder lineshapes. It imports neither the PW-GIPAW shielding module nor the FLAPW
EFG module, so it composes with either (or with hand-entered literature values).

Conventions
-----------
* CSA principal values in the **Haeberlen** convention
  (Haeberlen, *Advances in Magnetic Resonance*, Suppl. 1, 1976): with
  ``|delta_zz - delta_iso| >= |delta_xx - delta_iso| >= |delta_yy - delta_iso|``,

      delta_iso  = (delta_xx + delta_yy + delta_zz) / 3
      delta_aniso = delta_zz - delta_iso                    (reduced anisotropy)
      eta_csa    = (delta_yy - delta_xx) / delta_aniso,   0 <= eta_csa <= 1

  invert to the three principal values with :func:`csa_principal_values`.
* Quadrupolar parameters ``C_Q = e^2 q Q / h`` (Hz) and the derived quadrupolar
  frequency ``nu_Q = 3 C_Q / [2 I (2I - 1)]`` (Man, "Quadrupolar Interactions",
  *Encyclopedia of NMR*, 1996).

Unit note
---------
No CODATA conversion factor enters this module: every quantity is either a
frequency (Hz) or a dimensionless ratio, and ppm is the *definitional* scale
``1e6`` between a frequency offset and the carrier ``nu0`` (Hz), not a physical
constant. ``constants.py`` is therefore intentionally not imported here (it holds
eV/Angstrom/Hartree conversions that never appear in a frequency-domain
lineshape).

Powder averaging
----------------
All lineshapes share one deterministic orientation grid: a **golden-spiral
(Fibonacci) sphere**, points ``i = 0 .. N-1`` with

    z_i     = 1 - (2 i + 1) / N          (cos of the polar angle)
    phi_i   = i * pi (3 - sqrt 5)         (the golden angle),

equal weights ``1/N``. It is deterministic (no RNG), quasi-uniform, and
convergence is controlled by the single knob ``n_orientations``. Because every
lineshape here is even under ``n -> -n`` (a bilinear form in the field
direction), the full-sphere grid double-covers each distinct orientation, which
only improves the average.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "NMRSite",
    "broaden",
    "csa_principal_values",
    "csa_static_sticks",
    "mas_sidebands",
    "powder_grid",
    "quad_first_order_static_sticks",
    "quad_iso_second_order_shift_ppm",
    "quad_second_order_ct_mas_sticks",
    "quad_second_order_ct_static_sticks",
    "spectrum",
]

# Magic angle theta_m = arccos(1/sqrt 3): cos^2 = 1/3, sin^2 = 2/3. The
# second-rank spatial average P2(cos theta_m) = 0 vanishes here, which is what
# makes MAS average the CSA (and the second-order rank-2) broadening to zero.
_COS_MAGIC = 1.0 / math.sqrt(3.0)
_SIN_MAGIC = math.sqrt(2.0 / 3.0)

_PPM = 1.0e6  # definitional Hz-offset / carrier scale, not a physical constant


# --------------------------------------------------------------------------- #
# Per-site parameter record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NMRSite:
    """Per-site NMR tensor parameters (a leaf record; no physics imports).

    All shifts in ppm, ``c_q`` in Hz. ``weight`` is the site multiplicity /
    population used by :func:`spectrum` to sum sites.
    """

    delta_iso: float = 0.0
    delta_aniso: float = 0.0
    eta_csa: float = 0.0
    c_q: float = 0.0
    eta_q: float = 0.0
    spin: float = 0.5
    weight: float = 1.0
    label: str = ""


# --------------------------------------------------------------------------- #
# Powder averaging utility
# --------------------------------------------------------------------------- #
def powder_grid(n_orientations: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Golden-spiral orientation grid ``(theta, phi, weight)`` on the sphere.

    See the module docstring for the construction. ``theta`` (polar) and ``phi``
    (azimuth) are radians; the ``n_orientations`` weights are all ``1/N``.
    """
    if n_orientations < 1:
        raise ValueError("n_orientations must be >= 1")
    i = np.arange(n_orientations, dtype=np.float64)
    z = 1.0 - (2.0 * i + 1.0) / n_orientations
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phi = np.mod(i * golden, 2.0 * math.pi)
    weight = np.full(n_orientations, 1.0 / n_orientations, dtype=np.float64)
    return theta, phi, weight


def _unit_vectors(theta: FloatArray, phi: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    st = np.sin(theta)
    return st * np.cos(phi), st * np.sin(phi), np.cos(theta)


# --------------------------------------------------------------------------- #
# Chemical-shift anisotropy (CSA)
# --------------------------------------------------------------------------- #
def csa_principal_values(
    delta_iso: float, delta_aniso: float, eta: float
) -> tuple[float, float, float]:
    """Return ``(delta_xx, delta_yy, delta_zz)`` (ppm) from Haeberlen params.

    Inverting the convention in the module docstring:

        delta_zz = delta_iso + delta_aniso
        delta_xx = delta_iso - delta_aniso (1 + eta) / 2
        delta_yy = delta_iso - delta_aniso (1 - eta) / 2

    so ``eta = (delta_yy - delta_xx) / delta_aniso`` and the three sum to
    ``3 delta_iso``.
    """
    d_zz = delta_iso + delta_aniso
    d_xx = delta_iso - delta_aniso * (1.0 + eta) / 2.0
    d_yy = delta_iso - delta_aniso * (1.0 - eta) / 2.0
    return d_xx, d_yy, d_zz


def csa_static_sticks(
    delta_iso: float,
    delta_aniso: float,
    eta: float,
    n_orientations: int = 2000,
) -> tuple[FloatArray, FloatArray]:
    """Static CSA powder pattern as ``(positions_ppm, weights)`` point masses.

    For a crystallite whose principal axis frame is oriented so the field lies
    along the unit vector ``n = (sin th cos ph, sin th sin ph, cos th)`` in the
    PAS, the resonance shift is the diagonal quadratic form

        delta(th, ph) = delta_xx n_x^2 + delta_yy n_y^2 + delta_zz n_z^2

    (Mehring, *Principles of High Resolution NMR in Solids*, 1983, Eq. 2.60).
    Histogramming these over the powder grid gives the classic CSA tent with edge
    singularities exactly at the three principal values ``delta_xx/yy/zz`` from
    :func:`csa_principal_values`.
    """
    d_xx, d_yy, d_zz = csa_principal_values(delta_iso, delta_aniso, eta)
    theta, phi, w = powder_grid(n_orientations)
    nx, ny, nz = _unit_vectors(theta, phi)
    pos = d_xx * nx * nx + d_yy * ny * ny + d_zz * nz * nz
    return pos.astype(np.float64), w


# --------------------------------------------------------------------------- #
# Rotations (ZYZ Euler) for MAS time-domain averaging
# --------------------------------------------------------------------------- #
def _rot_zyz(alpha: FloatArray, beta: FloatArray, gamma: FloatArray) -> FloatArray:
    """Batched active ZYZ rotation ``R = Rz(alpha) Ry(beta) Rz(gamma)``.

    Each of ``alpha, beta, gamma`` is shape ``(n,)``; returns ``(n, 3, 3)``.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    n = alpha.shape[0]
    r = np.empty((n, 3, 3), dtype=np.float64)
    r[:, 0, 0] = ca * cb * cg - sa * sg
    r[:, 0, 1] = -ca * cb * sg - sa * cg
    r[:, 0, 2] = ca * sb
    r[:, 1, 0] = sa * cb * cg + ca * sg
    r[:, 1, 1] = -sa * cb * sg + ca * cg
    r[:, 1, 2] = sa * sb
    r[:, 2, 0] = -sb * cg
    r[:, 2, 1] = sb * sg
    r[:, 2, 2] = cb
    return r


def _cumulative_trapezoid_periodic(y: FloatArray, dt: float) -> FloatArray:
    """Trapezoidal cumulative integral along the last axis, starting at 0.

    ``out[..., 0] = 0`` and ``out[..., m] = dt * (y[0]/2 + sum_{1..m-1} y + y[m]/2)``.
    """
    seg = 0.5 * (y[..., 1:] + y[..., :-1]) * dt
    csum = np.cumsum(seg, axis=-1)
    zero = np.zeros((*y.shape[:-1], 1), dtype=np.float64)
    return np.concatenate([zero, csum], axis=-1)


def mas_sidebands(
    delta_iso: float,
    delta_aniso: float,
    eta: float,
    nu0_hz: float,
    nu_r_hz: float,
    n_orientations: int = 500,
    n_rotor: int = 256,
) -> tuple[FloatArray, FloatArray]:
    """MAS spinning-sideband manifold as ``(positions_ppm, intensities)``.

    Magic-angle spinning modulates the anisotropic shift periodically at the
    rotor rate ``nu_r``. In the rotor frame the field direction traces the
    magic-angle cone ``n_R(t) = (sin thm cos w_r t, sin thm sin w_r t, cos thm)``
    with ``w_r = 2 pi nu_r``, and the instantaneous (traceless, anisotropic)
    shift of a crystallite is ``delta(t) = n_R(t)^T A_R n_R(t)`` where ``A_R`` is
    the (iso-subtracted) CSA tensor rotated into the rotor frame.

    Sidebands are the Fourier coefficients of the single-rotor-period phase
    factor ``exp[i Phi(t)]``, ``Phi(t) = 2 pi integral_0^t (delta - delta_iso)
    nu0 1e-6 dt'``. The intensity of order ``N`` is the powder average of the
    squared modulus ``<|A_N|^2>`` of the one-period FFT -- the gamma-averaged
    integral (Bak & Nielsen, *J. Magn. Reson.* 125, 132 (1997); the "gamma-
    COMPUTE" identity, in which averaging ``|A_N|^2`` over the two remaining Euler
    angles reproduces the full three-angle powder-plus-rotor-phase average). This
    approximates the exact MAS spectrum by a single rotor period sampled at
    ``n_rotor`` points; it is exact in the limit ``n_rotor -> infinity``.

    Placed at ``delta_iso + N nu_r / nu0 * 1e6`` ppm, the manifold reduces to a
    single centerband stick at ``delta_iso`` as ``nu_r -> infinity`` (the phase
    accumulated over a vanishing period tends to zero, so ``A_N -> delta_{N,0}``),
    and per crystallite ``sum_N |A_N|^2 = 1`` while ``sum_N N |A_N|^2 = 0`` (the
    rotor mean of the anisotropy vanishes at the magic angle), so the manifold
    conserves total intensity and its center of mass is exactly ``delta_iso``.
    """
    if nu0_hz <= 0.0 or nu_r_hz <= 0.0:
        raise ValueError("nu0_hz and nu_r_hz must be positive")
    # Traceless (anisotropic) CSA tensor in ppm: subtract the isotropic part so
    # the accumulated phase carries only the sideband-generating modulation.
    d_xx, d_yy, d_zz = csa_principal_values(0.0, delta_aniso, eta)
    a_pas = np.diag([d_xx, d_yy, d_zz]).astype(np.float64)

    # Sample two Euler angles over the sphere; the third (rotor phase) is folded
    # in by the |A_N|^2 average.
    beta, gamma, w = powder_grid(n_orientations)
    alpha = np.zeros_like(beta)
    rot = _rot_zyz(alpha, beta, gamma)  # (n, 3, 3)
    a_r = rot @ a_pas @ rot.transpose(0, 2, 1)  # (n, 3, 3)

    tau = 1.0 / nu_r_hz
    dt = tau / n_rotor
    tm = np.arange(n_rotor, dtype=np.float64) * dt
    psi = 2.0 * math.pi * nu_r_hz * tm
    n_r = np.stack(
        [_SIN_MAGIC * np.cos(psi), _SIN_MAGIC * np.sin(psi), _COS_MAGIC * np.ones_like(psi)]
    )  # (3, M)

    # delta(t) per crystallite (ppm): n_R^T A_R n_R.
    delta_t = np.einsum("nij,im,jm->nm", a_r, n_r, n_r)  # (n, M)
    dnu_hz = delta_t * nu0_hz / _PPM
    phase = 2.0 * math.pi * _cumulative_trapezoid_periodic(dnu_hz, dt)  # (n, M)
    signal = np.exp(1j * phase)
    amp = np.fft.fft(signal, axis=-1) / n_rotor  # (n, M)
    inten = np.einsum("n,nm->m", w, np.abs(amp) ** 2)  # (M,)

    orders = np.rint(np.fft.fftfreq(n_rotor) * n_rotor).astype(np.int64)
    pos = delta_iso + orders.astype(np.float64) * nu_r_hz / nu0_hz * _PPM
    return pos, inten


# --------------------------------------------------------------------------- #
# Quadrupolar interaction
# --------------------------------------------------------------------------- #
def _nu_q(c_q: float, spin: float) -> float:
    """Quadrupolar frequency ``nu_Q = 3 C_Q / [2 I (2I - 1)]`` (Hz)."""
    if spin < 1.0:
        raise ValueError("quadrupolar lineshapes require spin I >= 1")
    return 3.0 * c_q / (2.0 * spin * (2.0 * spin - 1.0))


def quad_first_order_static_sticks(
    c_q: float,
    eta_q: float,
    spin: float,
    nu0_hz: float,
    n_orientations: int = 2000,
) -> tuple[FloatArray, FloatArray]:
    """First-order quadrupolar powder pattern (all single-quantum transitions).

    Each ``m <-> m-1`` transition is shifted to first order by

        dnu_m(th, ph) = -nu_Q (m - 1/2) * 1/2 (3 cos^2 th - 1 - eta sin^2 th cos 2ph)

    (Man, *Encyclopedia of NMR*, 1996). The central transition ``m = 1/2`` has
    ``(m - 1/2) = 0`` and is unshifted to first order; the satellites form the
    classic mirrored pattern. Each transition carries the single-quantum weight
    ``I(I+1) - m(m-1)``. Returns ``(positions_ppm, weights)``.
    """
    if nu0_hz <= 0.0:
        raise ValueError("nu0_hz must be positive")
    nu_q = _nu_q(c_q, spin)
    theta, phi, w = powder_grid(n_orientations)
    ct2 = np.cos(theta) ** 2
    st2 = np.sin(theta) ** 2
    ang = 0.5 * (3.0 * ct2 - 1.0 - eta_q * st2 * np.cos(2.0 * phi))  # (n,)

    # Half-integer and integer spins: m runs over the upper level of each
    # single-quantum transition, m = I, I-1, ..., -I+1.
    ms = np.arange(spin, -spin, -1.0)
    pos_all: list[FloatArray] = []
    wt_all: list[FloatArray] = []
    for m in ms:
        dnu_hz = -nu_q * (m - 0.5) * ang  # (n,)
        prob = spin * (spin + 1.0) - m * (m - 1.0)
        pos_all.append(dnu_hz / nu0_hz * _PPM)
        wt_all.append(w * prob)
    pos = np.concatenate(pos_all)
    wt = np.concatenate(wt_all)
    return pos, wt


def _second_order_ct_F(
    nx: FloatArray, ny: FloatArray, nz: FloatArray, eta: float
) -> FloatArray:
    """Angular factor ``F = A cos^4 th + B cos^2 th + C`` of the 2nd-order CT shift.

    With ``cos 2ph = (n_x^2 - n_y^2)/(n_x^2 + n_y^2)`` and the standard
    coefficients (Man 1996; equivalently the fourth-rank expansion of Amoureux):

        A = -27/8 + 9/4 eta cos2ph - 3/8 eta^2 cos^2 2ph
        B = 30/8 - eta^2/2 - 2 eta cos2ph + 3/4 eta^2 cos^2 2ph
        C = -3/8 + eta^2/3 - 1/4 eta cos2ph - 3/8 eta^2 cos^2 2ph

    whose full-sphere average is ``1/5 (1 + eta^2/3)`` -- the source of the
    isotropic second-order shift (verified in the tests).
    """
    c2 = nz * nz
    c4 = c2 * c2
    s2 = nx * nx + ny * ny
    cos2p = np.where(s2 > 1e-30, (nx * nx - ny * ny) / np.where(s2 > 1e-30, s2, 1.0), 0.0)
    cos2p2 = cos2p * cos2p
    a = -27.0 / 8.0 + (9.0 / 4.0) * eta * cos2p - (3.0 / 8.0) * eta**2 * cos2p2
    b = 30.0 / 8.0 - eta**2 / 2.0 - 2.0 * eta * cos2p + (3.0 / 4.0) * eta**2 * cos2p2
    c = -3.0 / 8.0 + eta**2 / 3.0 - (1.0 / 4.0) * eta * cos2p - (3.0 / 8.0) * eta**2 * cos2p2
    return a * c4 + b * c2 + c


def _second_order_ct_prefactor_hz(c_q: float, spin: float, nu0_hz: float) -> float:
    """``-(nu_Q^2 / nu0) (1/6) [I(I+1) - 3/4]`` (Hz), the CT 2nd-order scale."""
    if spin < 1.5 or not math.isclose(spin % 1.0, 0.5, abs_tol=1e-9):
        raise ValueError("second-order central-transition needs half-integer I >= 3/2")
    nu_q = _nu_q(c_q, spin)
    return -(nu_q**2 / nu0_hz) * (1.0 / 6.0) * (spin * (spin + 1.0) - 0.75)


def quad_iso_second_order_shift_ppm(
    c_q: float, eta_q: float, spin: float, nu0_hz: float
) -> float:
    """Isotropic second-order quadrupolar shift of the CT centroid (ppm).

        delta_iso^(2) = -(3/40) (C_Q/nu0)^2 [I(I+1) - 3/4] / [I^2 (2I-1)^2]
                        (1 + eta^2/3) * 1e6

    (Samoson, Kundla & Lippmaa; see Man 1996). It is the ``1/5 (1 + eta^2/3)``
    full-sphere average of :func:`_second_order_ct_F` times the CT prefactor, and
    survives infinite-speed MAS. Equivalently ``-(1/30)(nu_Q^2/nu0^2)[I(I+1)-3/4]
    (1 + eta^2/3) * 1e6``.
    """
    if nu0_hz <= 0.0:
        raise ValueError("nu0_hz must be positive")
    pref_hz = _second_order_ct_prefactor_hz(c_q, spin, nu0_hz)
    iso_hz = pref_hz * (1.0 / 5.0) * (1.0 + eta_q**2 / 3.0)
    return iso_hz / nu0_hz * _PPM


def quad_second_order_ct_static_sticks(
    c_q: float,
    eta_q: float,
    spin: float,
    nu0_hz: float,
    n_orientations: int = 4000,
) -> tuple[FloatArray, FloatArray]:
    """Static second-order central-transition powder pattern (half-integer I).

    Shift ``dnu^(2)(th, ph) = prefactor * F(th, ph)`` with ``F`` from
    :func:`_second_order_ct_F` and the prefactor from
    :func:`_second_order_ct_prefactor_hz`. The powder mean equals
    :func:`quad_iso_second_order_shift_ppm`. Returns ``(positions_ppm, weights)``.
    """
    if nu0_hz <= 0.0:
        raise ValueError("nu0_hz must be positive")
    pref_hz = _second_order_ct_prefactor_hz(c_q, spin, nu0_hz)
    theta, phi, w = powder_grid(n_orientations)
    nx, ny, nz = _unit_vectors(theta, phi)
    shift_hz = pref_hz * _second_order_ct_F(nx, ny, nz, eta_q)
    pos = shift_hz / nu0_hz * _PPM
    return pos, w


def quad_second_order_ct_mas_sticks(
    c_q: float,
    eta_q: float,
    spin: float,
    nu0_hz: float,
    n_orientations: int = 4000,
    n_rotor: int = 200,
) -> tuple[FloatArray, FloatArray]:
    """Infinite-speed-MAS second-order CT powder pattern (half-integer I).

    Fast MAS averages the second-order shift over the rotor: for each crystallite
    the field direction traces the magic-angle cone in the PAS, and the observed
    shift is the rotor average of the static second-order shift. This removes the
    second-rank (P2) broadening but retains the fourth-rank (P4) part, scaled by
    ``P4(cos theta_m) = -7/18``, plus the isotropic shift -- reproducing the
    standard fourth-rank MAS central-transition lineshape.

    For ``eta = 0`` the rotor-averaged shift is the parabola
    ``nu(w) = K [21/2 w^2 - 9 w + 5/2]`` in ``w = cos^2 beta`` (crystallite tilt),
    with ``K = -(nu_Q^2/nu0)(1/48)[I(I+1) - 3/4]``. Its two characteristic
    singularities sit at ``nu = 4/7 K`` (the ``w = 3/7`` van Hove turning point)
    and ``nu = 5/2 K`` (the ``w = 0`` edge divergence), with the pattern
    terminating at ``nu = 4 K`` (``w = 1``). Returns ``(positions_ppm, weights)``.
    """
    if nu0_hz <= 0.0:
        raise ValueError("nu0_hz must be positive")
    pref_hz = _second_order_ct_prefactor_hz(c_q, spin, nu0_hz)
    beta, gamma, w = powder_grid(n_orientations)
    alpha = np.zeros_like(beta)
    rot = _rot_zyz(alpha, beta, gamma)  # (n, 3, 3): field(PAS) = R^T field(rotor)

    psi = 2.0 * math.pi * np.arange(n_rotor, dtype=np.float64) / n_rotor
    n_r = np.stack(
        [_SIN_MAGIC * np.cos(psi), _SIN_MAGIC * np.sin(psi), _COS_MAGIC * np.ones_like(psi)]
    )  # (3, M): field direction on the magic cone in the rotor frame
    # Field in the PAS for each crystallite and rotor phase: n_PAS = R^T n_R.
    n_pas = np.einsum("nji,jm->nim", rot, n_r)  # (n, 3, M)
    fac = _second_order_ct_F(n_pas[:, 0, :], n_pas[:, 1, :], n_pas[:, 2, :], eta_q)  # (n, M)
    shift_hz = pref_hz * fac.mean(axis=1)  # rotor average -> (n,)
    pos = shift_hz / nu0_hz * _PPM
    return pos, w


# --------------------------------------------------------------------------- #
# Broadening and assembly
# --------------------------------------------------------------------------- #
def _kernel(
    axis_step: float, fwhm_gauss: float, fwhm_lorentz: float, max_half: int
) -> FloatArray | None:
    """Area-normalized Gaussian (x) Lorentzian broadening kernel on the grid.

    ``max_half`` caps the kernel half-width so it never exceeds the target grid
    (``np.convolve(..., mode="same")`` returns ``max(len(density), len(kernel))``,
    so an over-long kernel would silently grow the output). Returns ``None`` when
    both widths are zero (no convolution needed).
    """
    fwhm = max(fwhm_gauss, fwhm_lorentz)
    if fwhm <= 0.0:
        return None
    half = min(int(math.ceil(5.0 * fwhm / axis_step)), max(max_half, 1))
    x = np.arange(-half, half + 1, dtype=np.float64) * axis_step
    ker = np.ones_like(x)
    if fwhm_gauss > 0.0:
        sigma = fwhm_gauss / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        ker = ker * np.exp(-0.5 * (x / sigma) ** 2)
    if fwhm_lorentz > 0.0:
        hwhm = fwhm_lorentz / 2.0
        lor = hwhm**2 / (x**2 + hwhm**2)
        # Voigt = Gaussian convolved with Lorentzian (else pure Lorentzian).
        ker = np.convolve(ker, lor, mode="same") if fwhm_gauss > 0.0 else lor
    area = np.trapezoid(ker, dx=axis_step)
    return ker / area if area > 0.0 else None


def broaden(
    positions_ppm: FloatArray,
    weights: FloatArray,
    axis_ppm: FloatArray,
    fwhm_gauss: float = 0.0,
    fwhm_lorentz: float = 0.0,
) -> FloatArray:
    """Bin sticks onto ``axis_ppm`` and apply Gaussian/Lorentzian broadening.

    The sticks are deposited by histogram (total weight conserved), converted to
    a density, and convolved with the area-normalized :func:`_kernel`. The result
    is **not** renormalized here (so callers can compose sites); use
    :func:`spectrum` for a unit-area spectrum.
    """
    if axis_ppm.ndim != 1 or axis_ppm.shape[0] < 2:
        raise ValueError("axis_ppm must be a 1-D grid with >= 2 points")
    diffs = np.diff(axis_ppm)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError("axis_ppm must be strictly monotonic (ascending or descending)")
    # the histogram/kernel math needs an ascending grid; NMR axes are conventionally
    # plotted descending, so flip to ascending here and flip the density back to match.
    descending = diffs[0] < 0
    axis_asc = axis_ppm[::-1] if descending else axis_ppm
    step = float(axis_asc[1] - axis_asc[0])
    edges = np.empty(axis_asc.shape[0] + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (axis_asc[1:] + axis_asc[:-1])
    edges[0] = axis_asc[0] - 0.5 * step
    edges[-1] = axis_asc[-1] + 0.5 * step
    counts, _ = np.histogram(positions_ppm, bins=edges, weights=weights)
    density = counts.astype(np.float64) / step
    ker = _kernel(step, fwhm_gauss, fwhm_lorentz, (density.shape[0] - 1) // 2)
    if ker is not None:
        density = np.convolve(density, ker, mode="same") * step
    return density[::-1] if descending else density


def _auto_axis(
    positions_ppm: FloatArray, fwhm: float, n_points: int
) -> FloatArray:
    lo = float(np.min(positions_ppm))
    hi = float(np.max(positions_ppm))
    span = hi - lo
    pad = max(3.0 * fwhm, 0.05 * span, 1.0)
    return np.linspace(lo - pad, hi + pad, n_points)


_STICK_KINDS = frozenset(
    {
        "csa_static",
        "mas",
        "quad1_static",
        "quad2_ct_static",
        "quad2_ct_mas",
    }
)


def _site_sticks(
    site: NMRSite,
    kind: str,
    nu0_hz: float,
    nu_r_hz: float,
    n_orientations: int,
) -> tuple[FloatArray, FloatArray]:
    if kind == "csa_static":
        pos, wt = csa_static_sticks(
            site.delta_iso, site.delta_aniso, site.eta_csa, n_orientations
        )
    elif kind == "mas":
        pos, wt = mas_sidebands(
            site.delta_iso, site.delta_aniso, site.eta_csa, nu0_hz, nu_r_hz, n_orientations
        )
    elif kind == "quad1_static":
        pos, wt = quad_first_order_static_sticks(
            site.c_q, site.eta_q, site.spin, nu0_hz, n_orientations
        )
        pos = pos + site.delta_iso
    elif kind == "quad2_ct_static":
        pos, wt = quad_second_order_ct_static_sticks(
            site.c_q, site.eta_q, site.spin, nu0_hz, n_orientations
        )
        pos = pos + site.delta_iso
    elif kind == "quad2_ct_mas":
        pos, wt = quad_second_order_ct_mas_sticks(
            site.c_q, site.eta_q, site.spin, nu0_hz, n_orientations
        )
        pos = pos + site.delta_iso
    else:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(_STICK_KINDS)}")
    return pos, wt * site.weight


def spectrum(
    sites: list[NMRSite] | tuple[NMRSite, ...],
    kind: str = "csa_static",
    *,
    nu0_hz: float = 0.0,
    nu_r_hz: float = 0.0,
    n_orientations: int = 2000,
    axis_ppm: FloatArray | None = None,
    n_points: int = 2048,
    fwhm_gauss: float = 0.0,
    fwhm_lorentz: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Multiplicity-weighted powder spectrum over ``sites`` of the chosen ``kind``.

    ``kind`` selects the per-site lineshape: ``"csa_static"``, ``"mas"``
    (needs ``nu0_hz`` and ``nu_r_hz``), ``"quad1_static"``, ``"quad2_ct_static"``,
    or ``"quad2_ct_mas"`` (the last three need ``nu0_hz``). Each site's sticks are
    scaled by ``site.weight``, accumulated, binned onto ``axis_ppm`` (auto-ranged
    if ``None``), broadened by Gaussian and/or Lorentzian widths (ppm), and the
    total is normalized to unit area. Returns ``(ppm_axis, intensity)``.
    """
    if kind not in _STICK_KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(_STICK_KINDS)}")
    if not sites:
        raise ValueError("sites must be non-empty")
    all_pos: list[FloatArray] = []
    all_wt: list[FloatArray] = []
    for site in sites:
        pos, wt = _site_sticks(site, kind, nu0_hz, nu_r_hz, n_orientations)
        all_pos.append(pos)
        all_wt.append(wt)
    positions = np.concatenate(all_pos)
    weights = np.concatenate(all_wt)

    if axis_ppm is None:
        axis = _auto_axis(positions, max(fwhm_gauss, fwhm_lorentz), n_points)
    else:
        axis = np.asarray(axis_ppm, dtype=np.float64)
    inten = broaden(positions, weights, axis, fwhm_gauss, fwhm_lorentz)
    area = float(np.trapezoid(inten, axis))
    if area > 0.0:
        inten = inten / area
    return axis, inten
