"""Absolute-sign (and magnitude) calibration of the orbital magnetization
against Peierls-flux thermodynamics (milestone 6).

The ground truth needs no external code: put a TR-broken but CHERN-TRIVIAL
two-band lattice model (C = 0 makes M μ-independent, so the fixed-filling
energy slope is exactly −M·dB) on a q×1 magnetic supercell with honest
electron Peierls phases — H = (p − qA)²/2m with q = −e and Landau gauge
A = (0, Bx, 0) gives the +y hop from column x the phase e^{−iφx},
φ = eBa²/ħ the flux per plaquette in radians. Then thermodynamics fixes

    dE/dφ = −M_z·(ħ/e a²)  and the claim under test:  dE/dφ = −avg_k I_z / 2

with I the CTVR integrand of kgeometry.orbmag_tensor in RADIAN-k units
(I_frac/(2π)² for a model parameterized by fractional k). The measured
slopes converge O(φ²) onto −I/2 (q = 32 → 64 errors shrink 4:1; Richardson
extrapolation agrees to ~1e-5 relative), pinning the ELECTRON sign:

    M_z = −dE/dB = +(e/2ħ)·avg_k I_z   ⇒   m/μ_B = +avg_k I / (2·HBAR2_2M)

i.e. the convention implemented in kgeometry.orbital_magnetization is the
physically correct one — no flip. This test is the record of that fact.
"""

import numpy as np
import torch

from gradwave.postscf.kgeometry import orbmag_tensor

SX = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
SY = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
SZ = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
I2 = torch.eye(2, dtype=torch.complex128)

U, T0 = 3.0, 0.4  # trivial (C = 0), TR- and particle-hole-broken


def h_rad(k):
    """The model at k in radians."""
    return (
        torch.sin(k[0]).to(torch.complex128) * SX
        + torch.sin(k[1]).to(torch.complex128) * SY
        + (U + torch.cos(k[0]) + torch.cos(k[1])).to(torch.complex128) * SZ
        + (T0 * (torch.cos(k[0]) + torch.cos(k[1]))).to(torch.complex128) * I2
    )


def h_frac(k):
    return h_rad(2.0 * np.pi * k[:2])


# real-space hop matrices from H(k) = Σ_δ T_δ e^{ik·δ} (single-site model,
# so the TB and CTVR position conventions coincide)
TX = SX / 2j + 0.5 * SZ + 0.5 * T0 * I2  # hop +x
TY = SY / 2j + 0.5 * SZ + 0.5 * T0 * I2  # hop +y
ONSITE = U * SZ


def _avg_iz(mu_c, n=24):
    tot = 0.0
    for i in range(n):
        for j in range(n):
            k = torch.tensor([i / n, j / n, 0.0], dtype=torch.float64)
            tot += orbmag_tensor(h_frac, k, [0], mu_c)[2].item()
    return tot / n**2


def _energy_per_cell(qcells, phi, nkx=8, nky=8):
    """Filled-band (lowest half) energy per original cell at flux φ."""
    e_sum = 0.0
    for ikx in range(nkx):
        kx = 2.0 * np.pi * ikx / nkx  # supercell Bloch phase (x boundary)
        for iky in range(nky):
            ky = 2.0 * np.pi * iky / nky
            hmat = torch.zeros(2 * qcells, 2 * qcells, dtype=torch.complex128)
            for x in range(qcells):
                s = slice(2 * x, 2 * x + 2)
                ph = np.exp(1j * (ky - phi * x))  # electron Peierls: e^{−iφx}
                hmat[s, s] = ONSITE + TY * ph + TY.mH * np.conj(ph)
                x2 = (x + 1) % qcells
                s2 = slice(2 * x2, 2 * x2 + 2)
                bloch = np.exp(1j * kx) if x2 == 0 else 1.0
                hmat[s2, s] += TX * bloch
                hmat[s, s2] += TX.mH * np.conj(bloch)
            w = torch.linalg.eigvalsh(hmat)
            e_sum += w[:qcells].sum().item()
    return e_sum / (nkx * nky * qcells)


def test_orbmag_sign_calibrated_by_peierls_flux():
    torch.set_num_threads(2)
    # μ-independence sanity (C = 0): the anchor is well-posed at fixed N.
    # (dI/dμ = 2·avg Ω vanishes only as the 24² Riemann average of Ω → 0,
    # so the gate is the quadrature scale, not machine precision.)
    i1, i2 = _avg_iz(2.0), _avg_iz(2.6)
    assert abs(i1 - i2) < 1e-5
    i_rad = i1 / (2.0 * np.pi) ** 2  # fractional-k I → radian-k I (a = 1)
    predicted_slope = -i_rad / 2.0
    assert abs(i_rad) > 1e-3  # genuinely nonzero moment (TR broken)

    slopes = {}
    for q in (32, 64):
        phi0 = 2.0 * np.pi / q
        slopes[q] = (
            _energy_per_cell(q, phi0) - _energy_per_cell(q, -phi0)
        ) / (2.0 * phi0)
    # O(φ²) convergence toward −I/2 (measured 4:1 between q = 32 and 64) …
    err32 = abs(slopes[32] - predicted_slope)
    err64 = abs(slopes[64] - predicted_slope)
    assert err64 < err32 / 3.0
    # … and the Richardson extrapolation nails it (measured ~1e-4 relative)
    extrap = slopes[64] + (slopes[64] - slopes[32]) / 3.0
    assert abs(extrap - predicted_slope) < 1e-3 * abs(predicted_slope)
    # the SIGN statement itself, unmistakably:
    assert np.sign(slopes[64]) == -np.sign(i_rad)
