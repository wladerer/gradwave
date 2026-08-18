"""Aspherical muffin-tin density multipoles and the valence electric-field-gradient precursor.

The electric field gradient (EFG) at a nucleus — the observable behind the quadrupolar coupling
``C_Q = eQ V_zz/h`` in solid-state NMR — is the l=2 component of the Coulomb potential at the
origin, and it is driven by the l=2 asphericity of the charge density inside the muffin-tin sphere.
The spherical (muffin-tin) SCF keeps only l=0; this module adds the l>0 density components as a
post-processing step, by direct angular projection of ``|ψ|²`` onto spherical harmonics.

Inside a sphere the augmented wavefunction is ``ψ(r,Ω) = Σ_l [u_l(r)·SA_l(Ω) + u̇_l(r)·SB_l(Ω)]``
with ``SA_l(Ω) = Σ_m a_lm Y_lm(Ω)`` (and ``SB_l`` from the ``u̇`` amplitudes ``b_lm``) — radius and
angle factorize, so ``|ψ|²`` is cheap to build on an angular grid and project.

Null test: a cubic site has no l=2 invariant (the lowest cubic anisotropy is l=4), so the l=2
density and hence the EFG vanish by symmetry — ``Q2 ≈ 0``. The full V_zz (with the l=2 sphere
Poisson + interstitial matching) is the next step; the r^-3 valence moment here is its leading part.
"""

from __future__ import annotations

import math

import numpy as np

from gradwave.constants import E2


def _angular_grid(nx: int, nphi: int):
    """A product Gauss-Legendre(cosθ) × uniform(φ) angular grid; weights sum to 4π."""
    xg, wx = np.polynomial.legendre.leggauss(nx)
    theta = np.arccos(xg)
    phi = 2 * np.pi * np.arange(nphi) / nphi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    wgt = wx[:, None] * (2 * np.pi / nphi) * np.ones((1, nphi))
    return th, ph, wgt


def sphere_density_multipoles(amps, us, lmax, lset, nx: int = 16, nphi: int = 24):
    """Aspherical density components ``ρ_LM(r)`` inside a sphere by angular projection of ``|ψ|²``.

    ``amps`` = list over occupied states of ``(f, a, b)`` with occupation ``f`` and amplitude dicts
    ``a[l]``, ``b[l]`` each ``(2l+1,)`` complex (the ``u_l`` and ``u̇_l`` coefficients, m=-l..l).
    ``us`` = ``{l: (u_l, u̇_l)}`` radial functions on the in-sphere mesh (each ``(nr,)``).
    Returns ``{(L,M): ρ_LM(r)}`` (complex ``(nr,)``) for every ``(L,M)`` in ``lset``.
    """
    from scipy.special import sph_harm_y
    th, ph, wgt = _angular_grid(nx, nphi)
    ylm = {(l, m): sph_harm_y(l, m, th, ph)
           for l in range(lmax + 1) for m in range(-l, l + 1)}
    nr = len(us[0][0])
    rho_ang = np.zeros((nr,) + th.shape)
    for f, a, b in amps:
        if f == 0:
            continue
        psi = np.zeros((nr,) + th.shape, dtype=complex)
        for l in range(lmax + 1):
            u, ud = us[l]
            sa = sum(a[l][m + l] * ylm[(l, m)] for m in range(-l, l + 1))
            sb = sum(b[l][m + l] * ylm[(l, m)] for m in range(-l, l + 1))
            psi += u[:, None, None] * sa[None] + ud[:, None, None] * sb[None]
        rho_ang += f * np.abs(psi) ** 2
    out = {}
    for (lang, m) in lset:
        proj = (rho_ang * (wgt * np.conj(sph_harm_y(lang, m, th, ph)))[None]).sum(axis=(1, 2))
        out[(lang, m)] = proj
    return out


def valence_efg_moments(multipoles, rr, drw):
    """The r⁻³ valence moments ``q_M = ∫ ρ_2M(r)/r dr`` (M=-2..2) and their magnitude
    ``Q2 = √Σ_M|q_M|²`` — the leading (valence-asphericity) contribution to the EFG. ``drw`` is the
    per-point radial ``dr`` weight (``r·dx`` on a log mesh). ``Q2 ≈ 0`` at a cubic site."""
    q = {m: complex(np.sum(multipoles[(2, m)] * drw / rr)) for m in range(-2, 3)}
    q2 = float(np.sqrt(sum(abs(v) ** 2 for v in q.values())))
    return q, q2


def l2_sphere_poisson(rho2m, rr, drw):
    """The l=2 radial Poisson inside the sphere: ``V_2M(r)`` (eV) from the on-site l=2 density
    (the particular solution, density contained in [0,R]):

        V_2M(r) = (4π E2 / 5) [ r⁻³ ∫_0^r ρ_2M r'⁴ dr' + r² ∫_r^R ρ_2M / r' dr' ].

    Near the origin ``V_2M(r) → v_M r²`` with ``v_M = (4π E2/5) ∫_0^R ρ_2M/r' dr'`` — the r²
    coefficient that becomes the EFG. ``drw`` is the radial ``dr`` weight."""
    inner = np.cumsum(rho2m * rr**4 * drw)
    outer = np.cumsum((rho2m / rr * drw)[::-1])[::-1]
    return (4 * math.pi * E2 / 5.0) * (inner / rr**3 + rr**2 * outer)


def efg_tensor(multipoles, rr, drw):
    """The valence electric field gradient from the l=2 density multipoles, via the l=2 sphere
    Poisson r² coefficients ``v_M = (4π E2/5) ∫ ρ_2M/r dr``. Returns ``(V, V_zz, eta)``: the 3×3
    Cartesian tensor (eV/Å²), the principal component ``|V_zz| = max|eigenvalue|``, and the
    asymmetry ``η = |V_xx−V_yy|/|V_zz|`` (``|V_zz|≥|V_yy|≥|V_xx|``). Vanishes at a cubic site."""
    c0, c = math.sqrt(5.0 / math.pi), math.sqrt(15.0 / (2.0 * math.pi))
    v = {m: (4 * math.pi * E2 / 5.0) * complex(np.sum(multipoles[(2, m)] / rr * drw))
         for m in range(-2, 3)}
    v0, v1, v2 = v[0].real, v[1], v[2]
    vxx, vyy, vzz = -0.5 * c0 * v0 + c * v2.real, -0.5 * c0 * v0 - c * v2.real, c0 * v0
    vxy, vxz, vyz = -c * v2.imag, -c * v1.real, c * v1.imag
    tensor = np.array([[vxx, vxy, vxz], [vxy, vyy, vyz], [vxz, vyz, vzz]])
    w = np.linalg.eigvalsh(tensor)
    order = np.argsort(np.abs(w))                    # |V_xx| ≤ |V_yy| ≤ |V_zz|
    v_zz, v_yy, v_xx = w[order[2]], w[order[1]], w[order[0]]
    eta = abs((v_xx - v_yy) / v_zz) if abs(v_zz) > 1e-30 else 0.0
    return tensor, float(v_zz), float(eta)
