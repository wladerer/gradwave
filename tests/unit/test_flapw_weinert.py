"""Gates for the exact G-space Weinert chain (analytic own-field subtraction + high-L
pseudocharges).

The boundary (lattice) term of the full-potential FLAPW chain is only correct if each
sphere's surface projection of the interstitial grid, minus the ANALYTIC own-sphere
multipole field 4πE2 q^MT/((2L+1)R^{L+1}), contains the external field alone. The old
real-space-sampled pseudocharge left a ~20% own-field aliasing residue and retained the
fictitious in-sphere ρ_I continuation entirely (forensics D2) — feeding the EFG magnitude
deficit and the |ρ|≈1.02 unstable interstitial SCF mode. These tests pin the exact chain:

- Bessel in-sphere moments == dense quadrature (exact for band-limited densities);
- lone-sphere boundary null: external field ≈ 0 for all matched (L,M) (the D2 gate;
  residual = Fourier truncation + tiny image-lattice fields, measured in
  experiments/autoapw/weinert_gates.py);
- retained-ρ_I identity: an EMPTY sphere immersed in a plane-wave interstitial density has
  external field == [analytic plane-wave projection] − [multipole field of the plane wave's
  in-sphere moments] — the second term is exactly what the old chain retained;
- near-touching cross delivery (the D4 gate): the rutile 0.024 Å O|Ti gap geometry, receiver
  surface field vs the analytic image-summed multipole potential.
"""

import math

import numpy as np
import pytest
from scipy.special import sph_harm_y

from gradwave.constants import E2
from gradwave.flapw.coulomb import (
    cell_matrix,
    gvec_ylm_tables,
    reciprocal,
    sphere_interstitial_moments,
)
from gradwave.flapw.efg import _angular_grid, interstitial_boundary_multi
from gradwave.flapw.radial import log_mesh
from gradwave.flapw.scf import _weinert_multi

CELL = 10.0  # Å cubic box for all gates


def _sphere(R, tau, moments, r_np, dx, z=0.0):
    """A synthetic _weinert_multi sphere: Gaussian l=0 density normalized to ``z`` electrons
    (neutral against Z=z) + prescribed aspherical multipole moments."""
    rr = r_np[r_np <= R]
    rho_sph = np.exp(-((rr / 0.25) ** 2))
    drw = rr * dx
    q = float(np.sum(4 * math.pi * rho_sph * rr**2 * drw))
    rho_sph *= z / q if z else 0.0
    rho_2m = {(0, 0): np.zeros_like(rr, dtype=complex)}
    for (lang, m), qlm in moments.items():
        shape = rr**lang * np.exp(-((rr / (0.35 * R)) ** 2))
        shape /= np.sum(shape * rr ** (lang + 2) * drw)
        rho_2m[(lang, m)] = (qlm * shape).astype(complex)
    return {"tau": np.asarray(tau, float), "rr": rr, "dx": dx, "rho_sph": rho_sph,
            "Z": z, "R": R, "rho_2m": rho_2m}


def _own_field(qlm, R):
    """The analytic own-sphere surface field 4πE2 q_LM/((2L+1)R^{L+1})."""
    return {lm: (4 * math.pi * E2 / (2 * lm[0] + 1)) * q / R ** (lm[0] + 1)
            for lm, q in qlm.items()}


def _hermitian_moments(lmax):
    """Moments of a REAL density: q_{L,-M} = (-1)^M conj(q_{L,M})."""
    out = {}
    for lang in range(1, lmax + 1):
        m = min(lang, 2)
        q = 1.0 + (0.3j if m else 0.0)
        out[(lang, m)] = q
        if m:
            out[(lang, -m)] = (-1) ** m * np.conj(q)
    return out


def test_interstitial_moments_match_dense_quadrature():
    """Bessel-projection in-sphere moments are exact for a band-limited density (here a
    single plane wave): they must match a dense radial × angular quadrature."""
    nfft, lmax, R = 32, 4, 0.9
    a = cell_matrix(CELL)
    tau = np.array([4.1, 5.3, 5.9])
    g1 = reciprocal(a)[0] + 2 * reciprocal(a)[1] + reciprocal(a)[2]
    ax = np.arange(nfft) * (CELL / nfft)
    xg, yg, zg = np.meshgrid(ax, ax, ax, indexing="ij")
    rho = 0.05 * np.cos(np.stack([xg, yg, zg], -1) @ g1)
    gvec, gnorm, ylm = gvec_ylm_tables(a, nfft, lmax)
    rho_g = (np.fft.fftn(rho) / nfft**3).reshape(-1)
    q_bessel = sphere_interstitial_moments(rho_g, R, tau, gvec, gnorm, ylm,
                                           list(range(lmax + 1)))
    th, ph, wgt = _angular_grid(20, 30)
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    xg_r, wg_r = np.polynomial.legendre.leggauss(120)          # radial GL on [0,R]
    rq = 0.5 * R * (xg_r + 1.0)
    wr = 0.5 * R * wg_r
    pts = tau[None, None] + rq[:, None, None] * dirs.reshape(1, -1, 3)
    rho_pts = 0.05 * np.cos(pts @ g1)
    for lang in range(lmax + 1):
        for m in range(-lang, lang + 1):
            yl = sph_harm_y(lang, m, th, ph).reshape(-1)
            q_dense = np.sum(rho_pts * np.conj(yl)[None, :] * wgt.reshape(-1)[None, :]
                             * (wr * rq ** (lang + 2))[:, None])
            assert abs(q_bessel[(lang, m)] - q_dense) < 1e-9, (lang, m)


def test_isolated_sphere_boundary_null():
    """D2 gate: a lone sphere's external boundary field (v_bc − analytic own multipole
    field) vanishes for every matched (L,M). The old numerically-subtracted chain left ~20%
    at l=2; the exact chain's residual is Fourier truncation + image-lattice fields
    (~1e-3 relative at this resolution, measured in weinert_gates.py stage B/N)."""
    nfft, lmax, R = 48, 4, 0.824
    r, dx = log_mesh(1e-5, 28.0, 2500)
    sp = _sphere(R, [CELL / 2] * 3, _hermitian_moments(lmax), r.numpy(), dx, z=6.0)
    rho_i = np.zeros((nfft, nfft, nfft))
    _, _, _, v_hart, qmt = _weinert_multi(rho_i, [sp], CELL, nfft)
    lset = [(lang, m) for lang in range(lmax + 1) for m in range(-lang, lang + 1)]
    v_bc = interstitial_boundary_multi(v_hart, sp["tau"], R, cell_matrix(CELL), lset)
    ownf = _own_field(qmt[0], R)
    gates = {0: 5e-3, 1: 8e-3, 2: 8e-3, 3: 1.5e-2, 4: 5e-2}
    for lang in range(1, lmax + 1):
        m = min(lang, 2)
        own = ownf[(lang, m)]
        rel = abs(v_bc[(lang, m)] - own) / abs(own)
        assert rel < gates[lang], f"L={lang}: rel={rel:.2e} (old chain: ~0.2 at l=2)"
    # l=0: net-neutral sphere, external monopole field ~ 0 (absolute, eV)
    assert abs(v_bc[(0, 0)]) < 5e-3


@pytest.mark.slow
def test_isolated_sphere_boundary_null_l6():
    """The fullpot_lmax=6 regime of the D2 gate, at the grid the production rule would pick
    (R·Gmax ≈ 18.5): all L ≤ 6 external residuals small — the configuration whose
    unresolvable L=5,6 pseudocharges previously DIVERGED the fp6 SCF."""
    nfft, lmax, R = 72, 6, 0.824
    r, dx = log_mesh(1e-5, 28.0, 2500)
    sp = _sphere(R, [CELL / 2] * 3, _hermitian_moments(lmax), r.numpy(), dx, z=6.0)
    rho_i = np.zeros((nfft, nfft, nfft))
    _, _, _, v_hart, qmt = _weinert_multi(rho_i, [sp], CELL, nfft)
    lset = [(lang, min(lang, 2)) for lang in range(1, lmax + 1)]
    v_bc = interstitial_boundary_multi(v_hart, sp["tau"], R, cell_matrix(CELL), lset)
    ownf = _own_field(qmt[0], R)
    for lang, m in lset:
        own = ownf[(lang, m)]
        rel = abs(v_bc[(lang, m)] - own) / abs(own)
        assert rel < 4e-2, f"L={lang}: rel={rel:.2e}"


def test_retained_rho_i_term_removed():
    """The D2 retained-ρ_I identity: an EMPTY sphere immersed in a plane-wave interstitial
    density must see external field = [analytic plane-wave projection] − [multipole field of
    the wave's in-sphere moments q^I]. The old chain retained the own(q^I) term ENTIRELY
    (its pseudocharge only carried q^MT − q^I while ρ_I inside R stayed in the grid sum
    uncorrected); the exact chain must remove ≥ 85% of it."""
    nfft, lmax, R = 48, 4, 0.824
    a = cell_matrix(CELL)
    r, dx = log_mesh(1e-5, 28.0, 2500)
    sp = _sphere(R, [CELL / 2] * 3, {(lang, 0): 0.0 for lang in range(1, lmax + 1)},
                 r.numpy(), dx, z=0.0)
    g1 = reciprocal(a)[0] + 2 * reciprocal(a)[1] + reciprocal(a)[2]
    ax = np.arange(nfft) * (CELL / nfft)
    xg, yg, zg = np.meshgrid(ax, ax, ax, indexing="ij")
    rho_i = 0.05 * np.cos(np.stack([xg, yg, zg], -1) @ g1)
    _, _, _, v_hart, _qmt = _weinert_multi(rho_i, [sp], CELL, nfft)
    lset = [(2, m) for m in range(-2, 3)]
    v_bc = interstitial_boundary_multi(v_hart, sp["tau"], R, a, lset)
    # analytic plane-wave surface projection
    th, ph, wgt = _angular_grid(20, 30)
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    pts = sp["tau"] + R * dirs.reshape(-1, 3)
    vpw = (4 * math.pi * E2 * 0.05 / (g1 @ g1)) * np.cos(pts @ g1)
    # exact in-sphere moments of the wave (Bessel = exact band-limited)
    gvec, gnorm, ylm = gvec_ylm_tables(a, nfft, lmax)
    rho_g = (np.fft.fftn(rho_i) / nfft**3).reshape(-1)
    q_i = sphere_interstitial_moments(rho_g, R, sp["tau"], gvec, gnorm, ylm, [2])
    own_i = _own_field(q_i, R)
    for lm in lset:
        ref = complex(np.sum(vpw.reshape(th.shape) * wgt
                             * np.conj(sph_harm_y(2, lm[1], th, ph)))) - own_i[lm]
        err = abs(v_bc[lm] - ref)
        assert err < 0.15 * abs(own_i[lm]), (
            f"M={lm[1]}: err={err:.2e} vs retained-term size {abs(own_i[lm]):.2e}")


def test_near_touching_cross_field():
    """D4 gate: rutile O|Ti geometry (R 1.098 + 0.824, centers 1.946 Å → 0.024 Å gap). The
    receiver sphere's surface projections of the source's multipole field must match the
    analytic image-summed multipole potential — the near-field delivery whose truncation
    (matched L ≤ aug-lmax, unresolved orders) was the leading suspect for the O EFG
    magnitude deficit."""
    nfft, lmax = 48, 4
    r, dx = log_mesh(1e-5, 28.0, 2500)
    r_np = r.numpy()
    d = 1.946
    tau1 = np.array([CELL / 2 - d / 2, CELL / 2, CELL / 2])
    tau2 = np.array([CELL / 2 + d / 2, CELL / 2, CELL / 2])
    moments = {}
    for lang in range(2, lmax + 1):
        m = min(lang, 2)
        moments[(lang, m)] = 0.5 + 0.0j
        moments[(lang, -m)] = (-1) ** m * 0.5
    sp1 = _sphere(1.098, tau1, moments, r_np, dx, z=0.0)
    sp2 = _sphere(0.824, tau2, {}, r_np, dx, z=0.0)
    rho_i = np.zeros((nfft, nfft, nfft))
    _, _, _, v_hart, _qmt = _weinert_multi(rho_i, [sp1, sp2], CELL, nfft)
    lset = [(lang, m) for lang in range(lmax + 1) for m in range(-lang, lang + 1)]
    v_bc2 = interstitial_boundary_multi(v_hart, tau2, sp2["R"], cell_matrix(CELL), lset)
    th, ph, wgt = _angular_grid(24, 36)
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    pts = tau2 + sp2["R"] * dirs.reshape(-1, 3)
    v_ref = np.zeros(pts.shape[0])
    for sh in np.ndindex(5, 5, 5):
        u = pts - tau1 - CELL * (np.array(sh) - 2)
        un = np.linalg.norm(u, axis=1)
        thu = np.arccos(np.clip(u[:, 2] / un, -1, 1))
        phu = np.arctan2(u[:, 1], u[:, 0])
        for (lang, m), q in moments.items():
            v_ref += (4 * math.pi * E2 / (2 * lang + 1)) * np.real(
                q * sph_harm_y(lang, m, thu, phu)) / un ** (lang + 1)
    scale = float(np.abs(v_ref).max())
    worst = 0.0
    for lm in lset:
        ref = complex(np.sum(v_ref.reshape(th.shape) * wgt
                             * np.conj(sph_harm_y(lm[0], lm[1], th, ph))))
        if abs(ref) > 1e-2 * scale:
            worst = max(worst, abs(v_bc2[lm] - ref) / abs(ref))
    assert worst < 8e-2, f"worst significant cross-field rel err {worst:.2e}"


def test_full_state_restore_across_nfft():
    """A ``__full_state__`` warm start saved at one FFT grid must restore onto a run whose
    pseudocharge-resolvability rule chose a different nfft (fp_lmax=6 bumps 28³→32³): the saved
    v_grid is band-limited-resampled to the current grid, not boolean-masked at the wrong shape
    (the measured fp4-state → fp6-run IndexError). Upsampling a band-limited field is exact."""
    from gradwave.flapw.scf import _fft_resample_cube, _restore_full_state
    n_old, n_new = 8, 12
    ax = np.arange(n_old) * (2 * np.pi / n_old)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    v = 1.5 + np.cos(x) * np.sin(2 * y) + np.cos(z)       # band-limited well below Nyquist
    fs = {"v_grid": v, "v_i0": 0.25, "v_nsph": None}
    theta = np.ones((n_new,) * 3, dtype=bool)
    theta[: n_new // 2] = False
    _, v_rs, warp, v_i0 = _restore_full_state(fs, theta, n_new)
    assert v_rs.shape == (n_new,) * 3 and warp is not None and v_i0 == 0.25
    ax2 = np.arange(n_new) * (2 * np.pi / n_new)
    x2, y2, z2 = np.meshgrid(ax2, ax2, ax2, indexing="ij")
    v_exact = 1.5 + np.cos(x2) * np.sin(2 * y2) + np.cos(z2)
    assert np.abs(v_rs - v_exact).max() < 1e-12           # exact band-limited upsample
    rt = _fft_resample_cube(_fft_resample_cube(v, n_new), n_old)
    assert np.abs(rt - v).max() < 1e-12                   # up-down round trip is the identity
