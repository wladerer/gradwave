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
from collections import OrderedDict
from typing import Any

import numpy as np

from gradwave.constants import E2
from gradwave.flapw.coulomb import cell_matrix, grid_gvectors


def _angular_grid(nx: int, nphi: int):
    """A product Gauss-Legendre(cosθ) × uniform(φ) angular grid; weights sum to 4π."""
    xg, wx = np.polynomial.legendre.leggauss(nx)
    theta = np.arccos(xg)
    phi = 2 * np.pi * np.arange(nphi) / nphi
    th, ph = np.meshgrid(theta, phi, indexing="ij")
    wgt = wx[:, None] * (2 * np.pi / nphi) * np.ones((1, nphi))
    return th, ph, wgt


def nonspherical_potential(rho_sph, rho_2m, rr, drw, lset=None, nx: int = 16, nphi: int = 24):
    """The non-spherical sphere potential ``V_LM(r)`` = on-site Hartree + XC, for the full-potential
    SCF, for every ``(L,M)`` in ``lset`` (default: the l=2 set, the historical behaviour).
    ``rho_sph`` is the spherical density (val+core, e/Å³); ``rho_2m = {(L,M): ρ_LM}`` the aspherical
    valence density (must cover ``lset``). Hartree from ``lx_sphere_poisson``; XC by evaluating
    ``V_xc[ρ(r,Ω)]`` on the angular grid (ρ = ρ_sph + Σ ρ_LM Y_LM) and projecting onto each Y_LM."""
    import torch
    from scipy.special import sph_harm_y

    from gradwave.flapw.functionals import vxc_lda
    if lset is None:
        lset = [(2, m) for m in range(-2, 3)]
    th, ph, wgt = _angular_grid(nx, nphi)
    ylm = {(lang, m): sph_harm_y(lang, m, th, ph) for (lang, m) in lset}
    rho_ang = np.broadcast_to(np.asarray(rho_sph)[:, None, None], (len(rr), *th.shape)).copy()
    for lm in lset:
        rho_ang += (rho_2m[lm][:, None, None] * ylm[lm][None]).real
    vxc_ang = vxc_lda(torch.tensor(np.clip(rho_ang, 1e-10, None))).numpy()
    out = {}
    for (lang, m) in lset:
        v_h = lx_sphere_poisson(rho_2m[(lang, m)], rr, drw, lang)
        v_xc = (vxc_ang * (wgt * np.conj(ylm[(lang, m)]))[None]).sum(axis=(1, 2))
        out[(lang, m)] = v_h + v_xc
    return out


def ylm_rotations_complex(sg, cell, big_ls):
    """Per-op complex-harmonic rotation matrices ``{L: D^L}`` with
    ``Y_LM(S⁻¹ r̂) = Σ_M' D^L_{M'M} Y_LM'(r̂)``, ``S = Aᵀ W A⁻ᵀ`` the Cartesian rotation of each
    space-group op. Built by projecting on the Gauss-Legendre × uniform-φ grid (exact for
    band-limited integrands) — the same convention-proof construction as
    ``scf.paw_symmetry.ylm_rotation_matrices``, but in the complex basis the FLAPW multipoles use.
    A sphere density ρ_a = Σ_M c_M Y_LM rotated by op ``g: a → b`` lands on b as ``c_b = D^L c_a``.
    Improper operations are handled naturally (the quadrature just evaluates at S⁻¹r̂)."""
    from scipy.special import sph_harm_y
    a_t = np.asarray(cell, dtype=float).T
    lmax = max(big_ls)
    th, ph, wgt = _angular_grid(lmax + 2, 2 * lmax + 3)
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)
    out = []
    for w_mat in sg.rotations:
        s = a_t @ w_mat @ np.linalg.inv(a_t)
        rd = dirs @ np.linalg.inv(s).T                     # S⁻¹ r̂ (rows)
        rn = np.linalg.norm(rd, axis=-1)
        th_r = np.arccos(np.clip(rd[..., 2] / rn, -1.0, 1.0))
        ph_r = np.arctan2(rd[..., 1], rd[..., 0])
        blocks = {}
        for bl in big_ls:
            y0 = np.stack([sph_harm_y(bl, m, th, ph) for m in range(-bl, bl + 1)], axis=-1)
            yr = np.stack([sph_harm_y(bl, m, th_r, ph_r) for m in range(-bl, bl + 1)], axis=-1)
            blocks[bl] = np.einsum("xyi,xyj,xy->ij", np.conj(y0), yr, wgt)
        out.append(blocks)
    return out


_GAUNT_CACHE: dict[Any, Any] = {}


def gaunt_matrix(l, big_l, big_m, lp, nx: int = 16, nphi: int = 24):
    """The Gaunt coupling block ``G[m,m'] = ∫ Y*_lm Y_LM Y_l'm' dΩ`` (complex harmonics), shape
    ``(2l+1, 2l'+1)``, computed by Gauss-Legendre angular quadrature. This is the angular factor of
    the non-spherical potential matrix element ``⟨u_l Y_lm | V_LM Y_LM | u_l' Y_l'm'⟩``.

    Pure constants — memoized (the fullpot secular build asks for the same blocks at every k-point
    of every iteration). The cached array is returned read-only."""
    key = (l, big_l, big_m, lp, nx, nphi)
    hit = _GAUNT_CACHE.get(key)
    if hit is not None:
        return hit
    from scipy.special import sph_harm_y
    th, ph, wgt = _angular_grid(nx, nphi)
    ylm = [sph_harm_y(l, m, th, ph) for m in range(-l, l + 1)]
    ylpm = [sph_harm_y(lp, mp, th, ph) for mp in range(-lp, lp + 1)]
    y_big = sph_harm_y(big_l, big_m, th, ph)
    g = np.zeros((2 * l + 1, 2 * lp + 1), dtype=complex)
    for i in range(2 * l + 1):
        for j in range(2 * lp + 1):
            g[i, j] = np.sum(np.conj(ylm[i]) * y_big * ylpm[j] * wgt)
    g.setflags(write=False)
    _GAUNT_CACHE[key] = g
    return g


def sphere_density_multipoles(amps, us, rr, lmax, lset, nx: int | None = None,
                              nphi: int | None = None):
    """Aspherical density components ``ρ_LM(r)`` inside a sphere by angular projection of ``|ψ|²``.

    ``amps`` = list over occupied states of ``(f, a, b)`` with occupation ``f`` and amplitude dicts
    ``a[l]``, ``b[l]`` each ``(2l+1,)`` complex (the ``u_l`` and ``u̇_l`` coefficients, m=-l..l).
    ``us`` = ``{l: (u_l, u̇_l)}`` **reduced** radial functions ``u = r·R`` on the in-sphere mesh
    (each ``(nr,)``); ``rr`` = the radial mesh, used to recover the true radial ``R = u/r`` before
    squaring (see ``sphere_density_multipoles_multi``). Returns ``{(L,M): ρ_LM(r)}`` (complex
    ``(nr,)``) for every ``(L,M)`` in ``lset``.

    ``nx``/``nphi`` default (``None``) to the smallest grid that integrates the projector exactly:
    ``|ψ|²`` has angular degree ``2·lmax``, so ``|ψ|²·Y*_LM`` is a polynomial of degree
    ``2·lmax + max(L)`` — Gauss-Legendre ``nx`` is exact to ``2nx-1``, and the uniform ``φ`` rule is
    exact for azimuthal orders below ``nphi``. This is bit-exact vs an over-resolved grid.
    """
    norm_amps = []
    for f, a, b in amps:
        norm_amps.append((f, {l: [a[l], b[l]] for l in range(lmax + 1)}))
    norm_us = {l: list(us[l]) for l in range(lmax + 1)}
    return sphere_density_multipoles_multi(norm_amps, norm_us, rr, lmax, lset, nx=nx, nphi=nphi)


def sphere_density_multipoles_multi(amps, us, rr, lmax, lset, nx: int | None = None,
                                    nphi: int | None = None):
    """General-radial-set form of ``sphere_density_multipoles``: each l-channel carries an arbitrary
    list of radial functions (u, u̇, plus any local orbitals' second-energy radials).

    ``us[l]`` = list of **reduced** radial arrays ``u = r·R`` (each ``(nr,)``); ``rr`` = the radial
    mesh. The density is ``|ψ|²`` with ``ψ = Σ R_l(r) Y_lm``, so each radial is divided by ``r`` to
    recover the true radial ``R = u/r`` **before** it enters ``ψ`` (mirrors
    ``scf._sphere_valence_density``'s ``rads[i]·rads[j]/r²`` — without the ``/r`` the moments carry
    an extra ``r²`` and every downstream consumer — the EFG Poisson and the non-spherical
    potential — is mis-scaled). ``amps`` = list over occupied states of ``(f, coeffs)`` with
    ``coeffs[l]`` a list of ``(2l+1,)`` complex amplitude vectors aligned with ``us[l]``. Returns
    ``{(L,M): ρ_LM(r)}`` for every ``(L,M)`` in ``lset``."""
    from scipy.special import sph_harm_y
    rr = np.asarray(rr, dtype=float)
    us = {l: [np.asarray(rad) / rr for rad in us[l]] for l in range(lmax + 1)}
    max_l = max((lang for (lang, _) in lset), default=0)
    deg = 2 * lmax + max_l
    if nx is None:
        nx = deg // 2 + 1                                 # GL exact to 2nx-1 >= deg
    if nphi is None:
        nphi = deg + 1                                    # uniform φ exact for orders < nphi
    th, ph, wgt = _angular_grid(nx, nphi)
    ylm = {(l, m): sph_harm_y(l, m, th, ph)
           for l in range(lmax + 1) for m in range(-l, l + 1)}
    nr = len(us[0][0])
    rho_ang = np.zeros((nr, *th.shape))
    for f, coeffs in amps:
        if f == 0:
            continue
        psi = np.zeros((nr, *th.shape), dtype=complex)
        for l in range(lmax + 1):
            for rad, vec in zip(us[l], coeffs[l], strict=True):
                s = sum(vec[m + l] * ylm[(l, m)] for m in range(-l, l + 1))
                psi += rad[:, None, None] * s[None]
        rho_ang += f * np.abs(psi) ** 2
    out = {}
    for (lang, m) in lset:
        proj = (rho_ang * (wgt * np.conj(sph_harm_y(lang, m, th, ph)))[None]).sum(axis=(1, 2))
        out[(lang, m)] = proj
    return out


def sphere_density_multipoles_bands(amps_all, occ, us, rr, lmax, lset, nx: int | None = None,
                                    nphi: int | None = None):
    """All-band aspherical density components ``{(L,M): ρ_LM(r)}`` by a **Gaunt contraction** over
    the ``(l,m)`` channel products of ``|ψ|²`` — no angular grid. ``amps_all`` = ``{l: [(nb, 2l+1)
    per radial]}`` (from ``scf._bands_amps``), ``occ`` the per-band occupations, ``us`` = ``{l:
    [**reduced** radial arrays ``u = r·R``]}`` aligned with ``amps_all``, ``rr`` = the radial mesh
    (each radial is divided by ``r`` to the true radial ``R = u/r`` before the density product; see
    ``sphere_density_multipoles_multi``).

    Writing the in-sphere wavefunction ``ψ_n = Σ_{l,p} u^p_l(r) Σ_m A^{l,p}_{n,m} Y_lm`` gives, for
    the band-summed density,

        ρ_LM(r) = Σ_{l,p,l',q} u^p_l u^q_{l'} Σ_{m,m'} D^{l,p;l',q}_{m,m'} conj(G^{LM}_{lm,l'm'})

    with the per-channel density matrix ``D^{l,p;l',q}_{m,m'} = Σ_n occ_n A^{l,p}_{n,m}
    conj(A^{l',q}_{n,m'})`` and the Gaunt block ``G^{LM}_{lm,l'm'} = ∫ conj(Y_lm) Y_LM Y_l'm' dΩ``
    (``gaunt_matrix``; ``∫ Y_lm conj(Y_l'm') conj(Y_LM) = conj(G)``). Selection rules
    ``|l−l'|≤L≤l+l'`` and ``l+l'+L`` even skip the null blocks. This replaces the
    ``(nb, nr, nθ, nφ)`` ``|ψ|²`` accumulation (the ~40% fullpot-iteration hotspot) with small
    ``(2l+1)×(2l'+1)`` contractions plus radial products.

    Both the old angular-grid projection and this contraction compute the *same* exact integral of a
    band-limited integrand by exact quadrature (the grid's default ``nx``/``nphi`` and
    ``gaunt_matrix``'s are each exact to the polynomial degree ``2lmax+max L``), so the two agree to
    ulp; the floating summation order differs. ``nx``/``nphi`` (if given) set the ``gaunt_matrix``
    quadrature; ``None`` picks the exactness-guaranteeing grid (mirrors the old default). The
    angular-grid reference lives at ``_sphere_density_multipoles_bands_grid`` (parity gate)."""
    rr = np.asarray(rr, dtype=float)
    us = {l: [np.asarray(rad) / rr for rad in us[l]] for l in range(lmax + 1)}
    max_l = max((lang for (lang, _) in lset), default=0)
    deg = 2 * lmax + max_l
    ng_x = deg // 2 + 1 if nx is None else nx
    ng_p = deg + 1 if nphi is None else nphi
    occ_arr = np.asarray(occ, dtype=float)
    nr = len(us[0][0])
    out = {lm: np.zeros(nr, dtype=complex) for lm in lset}
    lm_by_l = {}                                          # L -> [(M, (L,M) key)] present in lset
    for (big_l, big_m) in lset:
        lm_by_l.setdefault(big_l, []).append((big_m, (big_l, big_m)))
    for l in range(lmax + 1):
        amps_l = amps_all[l]
        for lp in range(lmax + 1):
            amps_lp = amps_all[lp]
            big_ls = [big_l for big_l in lm_by_l
                      if abs(l - lp) <= big_l <= l + lp and (l + lp + big_l) % 2 == 0]
            if not big_ls:
                continue
            for p, rad_p in enumerate(us[l]):
                cp = occ_arr[:, None] * amps_l[p]         # (nb, 2l+1)
                for q, rad_q in enumerate(us[lp]):
                    dmat = cp.T @ np.conj(amps_lp[q])     # (2l+1, 2lp+1)
                    radprod = rad_p * rad_q               # (nr,)
                    for big_l in big_ls:
                        for (big_m, key) in lm_by_l[big_l]:
                            g = gaunt_matrix(l, big_l, big_m, lp, ng_x, ng_p)
                            s = complex(np.sum(dmat * np.conj(g)))
                            if s != 0.0:
                                out[key] += s * radprod
    return out


def _sphere_density_multipoles_bands_grid(amps_all, occ, us, rr, lmax, lset, nx: int | None = None,
                                          nphi: int | None = None):
    """Angular-grid reference for ``sphere_density_multipoles_bands`` (the pre-Gaunt path). Builds
    every band's in-sphere ψ on the Gauss-Legendre×φ grid in stacked tensor ops and projects |ψ|²
    onto each ``Y*_LM``. ``us[l]`` are the **reduced** radials ``u = r·R``; ``rr`` recovers the true
    ``R = u/r`` before ψ (see ``sphere_density_multipoles_multi``). Retained only as the parity
    oracle for the Gaunt contraction; the SCF uses the Gaunt path."""
    from scipy.special import sph_harm_y
    rr = np.asarray(rr, dtype=float)
    us = {l: [np.asarray(rad) / rr for rad in us[l]] for l in range(lmax + 1)}
    max_l = max((lang for (lang, _) in lset), default=0)
    deg = 2 * lmax + max_l
    if nx is None:
        nx = deg // 2 + 1
    if nphi is None:
        nphi = deg + 1
    th, ph, wgt = _angular_grid(nx, nphi)
    occ_arr = np.asarray(occ, dtype=float)
    nb = len(occ_arr)
    nr = len(us[0][0])
    psi = np.zeros((nb, nr, *th.shape), dtype=complex)
    for l in range(lmax + 1):
        yst = np.stack([sph_harm_y(l, m, th, ph) for m in range(-l, l + 1)], axis=0)
        for rad, amp in zip(us[l], amps_all[l], strict=True):
            ang = np.tensordot(amp, yst, axes=(1, 0))         # (nb, nx, nphi)
            psi += rad[None, :, None, None] * ang[:, None, :, :]
    rho_ang = np.einsum("n,nrxy->rxy", occ_arr, np.abs(psi) ** 2)
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


def lx_sphere_poisson(rho_lm, rr, drw, big_l):
    """The general-L radial Poisson inside the sphere: ``V_LM(r)`` (eV) from the on-site L-component
    density (the particular solution, density contained in [0,R]):

        V_LM(r) = (4π E2 / (2L+1)) [ r^{-(L+1)} ∫_0^r ρ_LM r'^{L+2} dr'
                                     + r^L ∫_r^R ρ_LM r'^{1-L} dr' ].

    For L=2, near the origin ``V_2M(r) → v_M r²`` with ``v_M = (4πE2/5)∫ρ_2M/r' dr'`` — the r²
    coefficient that becomes the EFG. ``drw`` is the radial ``dr`` weight."""
    inner = np.cumsum(rho_lm * rr ** (big_l + 2) * drw)
    # exclude the diagonal point from the outer integral so each mesh point falls in
    # exactly one of ∫_0^r / ∫_r^R (otherwise point i is double-counted). Mirrors the
    # sibling radial_poisson_to_R in coulomb.py.
    outer_w = rho_lm * rr ** (1 - big_l) * drw
    outer = np.cumsum(outer_w[::-1])[::-1] - outer_w
    return (4 * math.pi * E2 / (2 * big_l + 1)) * (inner / rr ** (big_l + 1)
                                                  + rr**big_l * outer)


def l2_sphere_poisson(rho2m, rr, drw):
    """The l=2 case of ``lx_sphere_poisson`` (kept as the historical entry point)."""
    return lx_sphere_poisson(rho2m, rr, drw, 2)


def _tensor_from_v(v):
    """Cartesian EFG tensor + ``(V_zz, η)`` from the five r² potential coefficients ``v_M`` (real V,
    complex harmonics): V_zz=√(5/π)v0; V_xx/yy=-½√(5/π)v0±√(15/2π)Re v2; V_xy=-√(15/2π)Im v2; …"""
    c0, c = math.sqrt(5.0 / math.pi), math.sqrt(15.0 / (2.0 * math.pi))
    v0, v1, v2 = v[0].real, v[1], v[2]
    vxx, vyy, vzz = -0.5 * c0 * v0 + c * v2.real, -0.5 * c0 * v0 - c * v2.real, c0 * v0
    vxy, vxz, vyz = -c * v2.imag, -c * v1.real, c * v1.imag
    tensor = np.array([[vxx, vxy, vxz], [vxy, vyy, vyz], [vxz, vyz, vzz]])
    w = np.linalg.eigvalsh(tensor)
    order = np.argsort(np.abs(w))                    # |V_xx| ≤ |V_yy| ≤ |V_zz|
    v_zz, v_yy, v_xx = w[order[2]], w[order[1]], w[order[0]]
    eta = abs((v_xx - v_yy) / v_zz) if abs(v_zz) > 1e-30 else 0.0
    return tensor, float(v_zz), float(eta)


def _valence_v(multipoles, rr, drw):
    """The valence r² coefficients ``v_M = (4π E2/5) ∫ ρ_2M/r dr`` from the l=2 sphere Poisson."""
    return {m: (4 * math.pi * E2 / 5.0) * complex(np.sum(multipoles[(2, m)] / rr * drw))
            for m in range(-2, 3)}


def efg_tensor(multipoles, rr, drw):
    """The valence electric field gradient from the l=2 density multipoles (on-site sphere Poisson).
    Returns ``(V, V_zz, eta)``: the 3×3 Cartesian tensor (eV/Å²), the principal component
    ``|V_zz| = max|eigenvalue|``, and the asymmetry ``η = |V_xx−V_yy|/|V_zz|``. Cubic → 0."""
    return _tensor_from_v(_valence_v(multipoles, rr, drw))


_PHASE_CACHE: OrderedDict[Any, Any] = OrderedDict()      # geometry key -> (E0[p, G], gvec[G, 3])
_PHASE_CACHE_BUDGET = 2**32                    # 4 GiB cap on the cached phase matrices


def _surface_phases(a, nfft, R, nx: int, nphi: int):
    """The center-independent surface phase matrix ``E0[p,G] = exp(iR G·Ω_p)`` (and its G-vector
    table) of ``interstitial_boundary_multi`` — the dominant cost of every boundary projection (a
    complex exp over ``npts × nfft³``), depending only on cell, grid size, sphere RADIUS and the
    angular grid. The atom position enters as the rank-1 phase ``exp(iG·c)``, applied by the caller
    as a cheap G-vector scaling — so the cache holds one matrix per SPECIES radius, not per atom.
    (The first, per-center version of this memo thrashed at production scale: ~1 GiB per entry ×
    one entry per atom cycling every iteration blew the byte budget, giving zero hits, ~1 GiB of
    allocation churn per call, and a 6x wall regression on a 14 GiB box — caught by the A/B
    campaign's control variant.) Factoring the center phase out changes the floating-point
    association ``(E0 ⊙ c) @ v -> E0 @ (c ⊙ v)`` — mathematically identical, rounding differs at
    the ulp level (validated ≤1e-13 relative against the inline expression). A byte-budget LRU
    bounds the memory; an over-budget matrix is computed inline as before, not retained."""
    key = (a.tobytes(), nfft, float(R), nx, nphi)
    hit = _PHASE_CACHE.get(key)
    if hit is not None:
        _PHASE_CACHE.move_to_end(key)
        return hit
    gvec = grid_gvectors(a, nfft)
    th, ph, _wgt = _angular_grid(nx, nphi)
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=-1)
    e0 = np.exp(1j * (R * dirs.reshape(-1, 3) @ gvec.T))           # (npts, nfft^3)
    if e0.nbytes <= _PHASE_CACHE_BUDGET:
        e0.setflags(write=False)
        gvec.setflags(write=False)
        _PHASE_CACHE[key] = (e0, gvec)
        total = sum(v[0].nbytes for v in _PHASE_CACHE.values())
        while total > _PHASE_CACHE_BUDGET and len(_PHASE_CACHE) > 1:
            _, (old, _g) = _PHASE_CACHE.popitem(last=False)
            total -= old.nbytes
    return e0, gvec


def interstitial_boundary_multi(v_grid, center_cart, R, cell, lset, nx: int = 16, nphi: int = 24):
    """The ``(L,M)`` components ``v_bc_LM = ∮ V_int(center + R Ω) Y*_LM(Ω) dΩ`` of the interstitial
    FFT potential on the sphere surface, for every ``(L,M)`` in ``lset``. The potential is evaluated
    on the sphere by its exact band-limited Fourier series ``V(r) = Σ_G v_G e^{iG·r}`` (no
    interpolation error, so a cubic-symmetric grid projects to machine-zero l=2). The surface values
    are computed once and projected onto each harmonic. ``center_cart`` is the atom in Å. The
    surface phase matrix is geometry-only and memoized (``_surface_phases``)."""
    from scipy.special import sph_harm_y
    a = cell_matrix(cell)
    nfft = v_grid.shape[0]
    vg = (np.fft.fftn(v_grid) / nfft**3).reshape(-1)               # V(r) = Σ_G v_G e^{iG·r}
    th, ph, wgt = _angular_grid(nx, nphi)
    e0, gvec = _surface_phases(a, nfft, R, nx, nphi)
    cphase = np.exp(1j * (gvec @ np.asarray(center_cart, dtype=float)))
    vals = (e0 @ (cphase * vg)).reshape(th.shape).real
    return {(lang, m): complex(np.sum(vals * wgt * np.conj(sph_harm_y(lang, m, th, ph))))
            for (lang, m) in lset}


def interstitial_l2_boundary(v_grid, center_cart, R, cell, nx: int = 16, nphi: int = 24):
    """The l=2 case of ``interstitial_boundary_multi`` (historical entry); returns ``{M: v}``."""
    full = interstitial_boundary_multi(v_grid, center_cart, R, cell,
                                       [(2, m) for m in range(-2, 3)], nx=nx, nphi=nphi)
    return {m: full[(2, m)] for m in range(-2, 3)}


def efg_tensor_full(multipoles, rr, drw, v_bc_2m, R):
    """The full EFG = valence (on-site sphere Poisson) + lattice term. ``v_bc_2m`` is the EXTERNAL
    l=2 surface field — the caller removes the own sphere's entire field (pseudocharge + the
    fictitious in-sphere ρ_I continuation, total moment q^MT) analytically via
    ``v_bc − 4πE2 q^MT_2M/(5R³)`` (Elk's homogeneous-solution construction):
    ``v_M^full = v_M^valence + v_bc_2M/R²``."""
    v_val = _valence_v(multipoles, rr, drw)
    v_full = {}
    for m in range(-2, 3):
        v_full[m] = v_val[m] + v_bc_2m[m] / R**2
    return _tensor_from_v(v_full)
