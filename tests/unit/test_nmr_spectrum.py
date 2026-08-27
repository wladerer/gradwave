"""Analytic-limit tests for the ssNMR lineshape-synthesis layer.

Every test pins a closed-form physical limit (peak position, support edge,
sideband spacing, sum rule, scaling law, singularity position) rather than a
golden lineshape. Grids are kept tiny so the whole file runs on a laptop.
"""

from __future__ import annotations

import math

import numpy as np

from gradwave.postscf.nmr_spectrum import (
    NMRSite,
    broaden,
    csa_principal_values,
    csa_static_sticks,
    mas_sidebands,
    powder_grid,
    quad_first_order_static_sticks,
    quad_iso_second_order_shift_ppm,
    quad_second_order_ct_mas_sticks,
    quad_second_order_ct_static_sticks,
    spectrum,
)


# --------------------------------------------------------------------------- #
# Powder grid
# --------------------------------------------------------------------------- #
def test_powder_grid_normalized_and_deterministic() -> None:
    theta, phi, w = powder_grid(500)
    assert w.shape == (500,)
    assert math.isclose(float(w.sum()), 1.0, rel_tol=0, abs_tol=1e-12)
    # Deterministic: same call, same points.
    theta2, _, _ = powder_grid(500)
    assert np.array_equal(theta, theta2)
    # Mean of cos(theta) over a quasi-uniform sphere is ~0.
    assert abs(float(np.cos(theta).mean())) < 0.02
    assert theta.min() >= 0.0 and theta.max() <= math.pi


# --------------------------------------------------------------------------- #
# CSA: isotropic limit and support edges at the principal values
# --------------------------------------------------------------------------- #
def test_csa_zero_aniso_single_peak() -> None:
    pos, w = csa_static_sticks(delta_iso=100.0, delta_aniso=0.0, eta=0.0, n_orientations=500)
    assert np.allclose(pos, 100.0, atol=1e-9)
    site = NMRSite(delta_iso=100.0, delta_aniso=0.0)
    axis, inten = spectrum([site], "csa_static", n_orientations=500, fwhm_gauss=2.0)
    assert math.isclose(float(axis[int(inten.argmax())]), 100.0, abs_tol=0.5)


def test_csa_support_edges_eta0_and_eta1() -> None:
    diso, dan = 0.0, 60.0
    # eta = 0: axial. Support spans [delta_iso - dan/2, delta_iso + dan].
    d_xx, d_yy, d_zz = csa_principal_values(diso, dan, 0.0)
    pos, _ = csa_static_sticks(diso, dan, 0.0, n_orientations=4000)
    assert math.isclose(float(pos.max()), d_zz, abs_tol=0.2)
    assert math.isclose(float(pos.min()), d_xx, abs_tol=0.2)
    assert math.isclose(d_xx, diso - dan / 2.0, abs_tol=1e-9)
    # eta = 1: support spans [delta_iso - dan, delta_iso + dan], middle at delta_iso.
    e_xx, e_yy, e_zz = csa_principal_values(diso, dan, 1.0)
    pos1, _ = csa_static_sticks(diso, dan, 1.0, n_orientations=4000)
    assert math.isclose(float(pos1.max()), e_zz, abs_tol=0.2)
    assert math.isclose(float(pos1.min()), e_xx, abs_tol=0.2)
    assert math.isclose(e_xx, diso - dan, abs_tol=1e-9)
    assert math.isclose(e_yy, diso, abs_tol=1e-9)


def test_csa_principal_values_sum_and_eta() -> None:
    d_xx, d_yy, d_zz = csa_principal_values(30.0, 40.0, 0.4)
    assert math.isclose(d_xx + d_yy + d_zz, 3.0 * 30.0, abs_tol=1e-9)
    assert math.isclose((d_yy - d_xx) / 40.0, 0.4, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# MAS: sideband spacing, infinite-speed limit, center-of-mass sum rule
# --------------------------------------------------------------------------- #
def test_mas_sideband_spacing_equals_rotor_rate() -> None:
    nu0, nu_r = 100.0e6, 5000.0
    pos, inten = mas_sidebands(0.0, 80.0, 0.3, nu0, nu_r, n_orientations=200, n_rotor=128)
    # ppm spacing between consecutive orders is exactly nu_r / nu0 * 1e6.
    order = np.argsort(pos)
    dppm = np.diff(pos[order])
    assert np.allclose(dppm, nu_r / nu0 * 1e6, atol=1e-6)


def test_mas_infinite_speed_collapses_to_centerband() -> None:
    nu0 = 100.0e6
    # Very fast spinning relative to the (nu0 * dan_ppm) anisotropy width.
    pos, inten = mas_sidebands(0.0, 60.0, 0.5, nu0, 5.0e6, n_orientations=200, n_rotor=128)
    inten = inten / inten.sum()
    center = int(np.argmin(np.abs(pos)))
    assert inten[center] > 0.999
    # Side intensity vanishes.
    side = inten.sum() - inten[center]
    assert side < 1e-3


def test_mas_total_intensity_conserved_and_com_is_iso() -> None:
    nu0, nu_r = 100.0e6, 4000.0
    diso, dan = 25.0, 90.0
    pos, inten = mas_sidebands(diso, dan, 0.4, nu0, nu_r, n_orientations=300, n_rotor=256)
    # sum_N |A_N|^2 = 1 per crystallite -> conserved after powder average.
    assert math.isclose(float(inten.sum()), 1.0, rel_tol=0, abs_tol=1e-9)
    # Center of mass of the whole manifold equals the isotropic shift.
    com = float((pos * inten).sum() / inten.sum())
    assert math.isclose(com, diso, abs_tol=1e-3)


def test_mas_zero_aniso_is_single_stick() -> None:
    pos, inten = mas_sidebands(10.0, 0.0, 0.0, 100.0e6, 3000.0, n_orientations=100, n_rotor=64)
    inten = inten / inten.sum()
    center = int(np.argmin(np.abs(pos - 10.0)))
    assert inten[center] > 0.999999


# --------------------------------------------------------------------------- #
# First-order quadrupolar
# --------------------------------------------------------------------------- #
def test_quad_first_order_ct_unshifted_satellites_symmetric() -> None:
    # spin 3/2: CT at 0, two satellites mirror-symmetric about 0.
    pos, w = quad_first_order_static_sticks(
        c_q=2.0e6, eta_q=0.0, spin=1.5, nu0_hz=100.0e6, n_orientations=1000
    )
    # The pattern is symmetric about the isotropic (delta_iso = 0) position.
    assert math.isclose(float(pos.max()), -float(pos.min()), rel_tol=1e-6)
    # Weighted mean sits at 0 (satellites cancel, CT at 0).
    assert abs(float((pos * w).sum() / w.sum())) < 1e-6


# --------------------------------------------------------------------------- #
# Second-order central transition
# --------------------------------------------------------------------------- #
def test_quad_iso_shift_formula_and_eta_dependence() -> None:
    c_q, spin, nu0 = 3.0e6, 2.5, 100.0e6
    # Closed form: -(3/40)(C_Q/nu0)^2 [I(I+1)-3/4]/[I^2 (2I-1)^2] (1+eta^2/3) 1e6.
    def analytic(eta: float) -> float:
        pref = -(3.0 / 40.0) * (c_q / nu0) ** 2
        pref *= (spin * (spin + 1.0) - 0.75) / (spin**2 * (2.0 * spin - 1.0) ** 2)
        return pref * (1.0 + eta**2 / 3.0) * 1e6

    for eta in (0.0, 0.5, 1.0):
        got = quad_iso_second_order_shift_ppm(c_q, eta, spin, nu0)
        assert math.isclose(got, analytic(eta), rel_tol=1e-12)

    # The powder mean of the static CT pattern equals the isotropic shift.
    pos, w = quad_second_order_ct_static_sticks(c_q, 0.6, spin, nu0, n_orientations=6000)
    mean = float((pos * w).sum() / w.sum())
    assert math.isclose(mean, quad_iso_second_order_shift_ppm(c_q, 0.6, spin, nu0), rel_tol=2e-3)

    # eta-dependence factor (1 + eta^2/3): ratio eta=1 vs eta=0 is exactly 4/3.
    r = quad_iso_second_order_shift_ppm(c_q, 1.0, spin, nu0) / quad_iso_second_order_shift_ppm(
        c_q, 0.0, spin, nu0
    )
    assert math.isclose(r, 4.0 / 3.0, rel_tol=1e-12)


def test_quad_second_order_width_scales_as_cq2_over_nu0() -> None:
    c_q, spin = 3.0e6, 2.5
    nu0_a, nu0_b = 100.0e6, 200.0e6
    pa, _ = quad_second_order_ct_static_sticks(c_q, 0.3, spin, nu0_a, n_orientations=4000)
    pb, _ = quad_second_order_ct_static_sticks(c_q, 0.3, spin, nu0_b, n_orientations=4000)
    width_a = float(pa.max() - pa.min())
    width_b = float(pb.max() - pb.min())
    # ppm width scales as C_Q^2 / nu0^2 at fixed C_Q -> ratio = (nu0_b/nu0_a)^2 = 4.
    assert math.isclose(width_a / width_b, (nu0_b / nu0_a) ** 2, rel_tol=1e-3)


def test_quad_second_order_mas_eta0_singularity_positions() -> None:
    c_q, spin, nu0 = 4.0e6, 2.5, 130.0e6
    pos, w = quad_second_order_ct_mas_sticks(c_q, 0.0, spin, nu0, n_orientations=8000, n_rotor=240)
    # Analytic K (ppm): -(nu_Q^2/nu0)(1/48)[I(I+1)-3/4] / nu0 * 1e6.
    nu_q = 3.0 * c_q / (2.0 * spin * (2.0 * spin - 1.0))
    k_ppm = -(nu_q**2 / nu0) * (1.0 / 48.0) * (spin * (spin + 1.0) - 0.75) / nu0 * 1e6
    sing_turn = (4.0 / 7.0) * k_ppm  # w = 3/7 van Hove divergence
    sing_edge = (5.0 / 2.0) * k_ppm  # w = 0 edge divergence
    end = 4.0 * k_ppm  # w = 1 termination
    # Pattern support runs between the extreme of {turn, edge, end}.
    assert math.isclose(float(pos.min()), min(sing_turn, sing_edge, end), abs_tol=0.05)
    assert math.isclose(float(pos.max()), max(sing_turn, sing_edge, end), abs_tol=0.05)

    # Histogram density diverges at both singularities: build a fine histogram
    # and confirm the two tallest interior bins sit at the analytic positions.
    axis = np.linspace(float(pos.min()) - 0.5, float(pos.max()) + 0.5, 400)
    dens = broaden(pos, w, axis, fwhm_gauss=0.15)
    # Two singularities are the two dominant peaks; locate local maxima.
    peaks = _local_maxima(axis, dens)
    assert _has_peak_near(peaks, sing_turn, tol=0.6)
    assert _has_peak_near(peaks, sing_edge, tol=0.6)


def _local_maxima(axis: np.ndarray, dens: np.ndarray) -> list[float]:
    out: list[float] = []
    for i in range(1, len(dens) - 1):
        if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1] and dens[i] > 0.2 * dens.max():
            out.append(float(axis[i]))
    return out


def _has_peak_near(peaks: list[float], target: float, tol: float) -> bool:
    return any(abs(p - target) < tol for p in peaks)


# --------------------------------------------------------------------------- #
# Broadening + assembly: area conservation, normalization
# --------------------------------------------------------------------------- #
def test_spectrum_unit_area_and_broadening_conserves_area() -> None:
    site = NMRSite(delta_iso=50.0, delta_aniso=40.0, eta_csa=0.5, weight=2.0)
    axis, inten = spectrum([site], "csa_static", n_orientations=2000, fwhm_gauss=1.0)
    area = float(np.trapezoid(inten, axis))
    assert math.isclose(area, 1.0, rel_tol=1e-6)
    # More broadening: still unit area.
    _, inten2 = spectrum(
        [site], "csa_static", n_orientations=2000, axis_ppm=axis, fwhm_gauss=5.0
    )
    assert math.isclose(float(np.trapezoid(inten2, axis)), 1.0, rel_tol=1e-6)


def test_spectrum_multiplicity_weighted_two_sites() -> None:
    # Two isotropic sites of unequal weight -> two peaks, taller one at the
    # heavier site.
    a = NMRSite(delta_iso=0.0, delta_aniso=0.0, weight=1.0)
    b = NMRSite(delta_iso=100.0, delta_aniso=0.0, weight=3.0)
    axis, inten = spectrum([a, b], "csa_static", n_orientations=400, fwhm_gauss=2.0)
    ia = float(inten[int(np.argmin(np.abs(axis - 0.0)))])
    ib = float(inten[int(np.argmin(np.abs(axis - 100.0)))])
    assert ib > ia
    assert math.isclose(ib / ia, 3.0, rel_tol=0.1)


# --------------------------------------------------------------------------- #
# Powder-grid convergence
# --------------------------------------------------------------------------- #
def test_csa_powder_convergence_under_doubling() -> None:
    diso, dan, eta = 0.0, 50.0, 0.5
    axis = np.linspace(-40.0, 60.0, 600)
    p1, w1 = csa_static_sticks(diso, dan, eta, n_orientations=2000)
    p2, w2 = csa_static_sticks(diso, dan, eta, n_orientations=4000)
    d1 = broaden(p1, w1, axis, fwhm_gauss=1.5)
    d2 = broaden(p2, w2, axis, fwhm_gauss=1.5)
    d1 = d1 / np.trapezoid(d1, axis)
    d2 = d2 / np.trapezoid(d2, axis)
    l1 = float(np.trapezoid(np.abs(d1 - d2), axis))
    assert l1 < 0.05
