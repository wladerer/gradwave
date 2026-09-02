"""FLAPW-DFPT (Phase 2): the self-consistent density response in the augmented LAPW basis.

Phase 1 (:mod:`gradwave.flapw.efg_torch`) delivered the EXPLICIT frozen-density partial
``dV_zz/dtau|_rho`` — the moving muffin-tin Theta(G) boundary term. This module builds the
other, dominant half: the density response ``drho_LM/du`` that flows through the l=2
muffin-tin multipoles into ``V_zz``. Together

    dV_zz/du = dV_zz/dtau|_rho (Phase 1, explicit)  +  (dV_zz/drho_LM)(drho_LM/du)  (here).

The density response is assembled from the first-order wavefunction response of the
converged augmented eigenstates. For the generalized secular problem ``H c = eps S c`` with
``c^H S c = 1``, first-order perturbation theory gives the eigenvector derivative

    dc_n = sum_{m != n, non-degenerate} c_m [c_m^H(dH - eps_n dS)c_n]/(eps_n - eps_m)
           - (1/2)(c_n^H dS c_n) c_n                                    (metric/normalisation term)

(derivation: differentiate ``H c = eps S c`` and ``c^H S c = 1``; the diagonal term keeps the
S-normalisation to first order and is the piece the plane-wave DFPT never sees because its
basis is fixed — here ``dS != 0`` is the moving-muffin-tin overlap derivative). The l=2
sphere-density multipoles are a Gaunt contraction of the per-channel density matrix
``D^{lp,l'q}_{mm'} = sum_n occ_n A^{lp}_{nm} conj(A^{l'q}_{nm'})`` (see
:func:`gradwave.flapw.efg.sphere_density_multipoles_bands`); its response is
``dD = sum_n occ_n (dA A* + A dA*)`` with ``dA`` the amplitude map applied to ``dc_n`` (plus,
for a structural perturbation, the explicit geometry derivative of the amplitude map — the
moving structure phase). This module provides the pure response calculus; the driver
(``experiments/autoapw/dfpt_validate.py``) supplies ``dH``/``dS``/``dA`` (analytic or by
finite difference of the frozen-potential secular assembly) and validates the assembled
``drho_LM/du`` against a one-step frozen-``v_eff`` finite difference (the bare chi_0), before
the K_Hxc Dyson screening closes the self-consistency.

The whole file is numpy (the secular problem and its eigenbasis are numpy in
``gradwave.flapw.scf``); the small nonlinear ``rho_LM -> V_zz`` map is delegated to the torch
path in :mod:`gradwave.flapw.efg_torch` so the ``dV_zz/drho_LM`` Jacobian is autograd-exact.
"""

from __future__ import annotations

import numpy as np

from gradwave.flapw.efg import gaunt_matrix
from gradwave.flapw.efg_torch import _tensor_from_v_torch


def eig_response_occupied(ev, C, dH, dS, occ, degen_tol: float = 1e-4):
    """First-order eigenvector response ``dC[:, n]`` for the occupied bands of one k-point.

    ``ev`` (rank,) real eigenvalues, ``C`` (dim, rank) the S-orthonormal eigenvectors
    (``C^H S C = I``), ``dH``/``dS`` (dim, dim) the perturbation of the secular matrices, ``occ``
    (nb,) the occupations (only ``occ_n > 0`` bands are returned). ``degen_tol`` (eV) skips the
    singular denominators of a degenerate manifold (their in-manifold response leaves the
    occupied-projector density invariant). Returns ``(dC_occ, occ_idx)`` with ``dC_occ`` shape
    ``(dim, n_occ)`` the coefficient-space response of each occupied band and ``occ_idx`` their
    indices into ``ev``/``C``.

    Implements ``dc_n = sum_m c_m (Pmn - eps_n Qmn)/(eps_n - eps_m) - (1/2) Q_nn c_n`` with
    ``P = C^H dH C``, ``Q = C^H dS C`` (the metric/normalisation diagonal is the ``dS != 0``
    moving-boundary term)."""
    ev = np.asarray(ev, dtype=float)
    rank = C.shape[1]
    occ = np.asarray(occ, dtype=float)
    occ_idx = np.nonzero(occ[:rank] > 1e-12)[0]
    Cc = C.conj().T
    P = Cc @ dH @ C                                    # (rank, rank)
    Q = Cc @ dS @ C
    dC = np.zeros((C.shape[0], len(occ_idx)), dtype=complex)
    for j, n in enumerate(occ_idx):
        num = P[:, n] - ev[n] * Q[:, n]               # (rank,)
        den = ev[n] - ev                              # (rank,)
        gamma = np.zeros(rank, dtype=complex)
        mask = np.abs(den) > degen_tol                # skip n itself and any degeneracy
        gamma[mask] = num[mask] / den[mask]
        gamma[n] = -0.5 * Q[n, n]                     # normalisation (real part; gauge Im=0)
        dC[:, j] = C @ gamma
    return dC, occ_idx


def dmats_from_amps(amps_all, occ, lmax):
    """Per-channel density matrices ``D[(l,p,lp,q)] = (occ*A^lp).T @ conj(A^l'q)``, (2l+1,2lp+1).

    ``amps_all`` = ``{l: [A per radial]}`` with ``A`` shape ``(nb, 2l+1)`` (from
    :func:`gradwave.flapw.scf._bands_amps`); ``occ`` the per-band occupations. This is the density
    matrix :func:`gradwave.flapw.efg.sphere_density_multipoles_bands` builds inline; factored out so
    the response reuses the identical Gaunt+radial contraction."""
    occ_arr = np.asarray(occ, dtype=float)
    out = {}
    for l in range(lmax + 1):
        for lp in range(lmax + 1):
            for p, ap in enumerate(amps_all[l]):
                cp = occ_arr[:, None] * ap
                for q, aq in enumerate(amps_all[lp]):
                    out[(l, p, lp, q)] = cp.T @ np.conj(aq)
    return out


def dmats_response(amps_all, damps_all, occ, lmax):
    """Response of :func:`dmats_from_amps`: ``dD = sum_n occ_n (dA A* + A dA*)`` per channel.

    ``damps_all`` = ``{l: [dA per radial]}`` the first-order amplitude response (same layout as
    ``amps_all``). Linear in ``dA``, so it feeds the same contraction as the base density matrix."""
    occ_arr = np.asarray(occ, dtype=float)
    out = {}
    for l in range(lmax + 1):
        for lp in range(lmax + 1):
            for p in range(len(amps_all[l])):
                ap, dap = amps_all[l][p], damps_all[l][p]
                for q in range(len(amps_all[lp])):
                    aq, daq = amps_all[lp][q], damps_all[lp][q]
                    cp = occ_arr[:, None]
                    out[(l, p, lp, q)] = (cp * dap).T @ np.conj(aq) + (cp * ap).T @ np.conj(daq)
    return out


def multipoles_from_dmats(dmats, us, rr, lmax, lset, nx: int | None = None,
                          nphi: int | None = None):
    """Sphere multipoles ``{(L,M): rho_LM(r)}`` from precomputed per-channel density matrices.

    The Gaunt+radial contraction of :func:`gradwave.flapw.efg.sphere_density_multipoles_bands`,
    taking ``dmats`` (from :func:`dmats_from_amps` or :func:`dmats_response`) instead of rebuilding
    it from amplitudes — so the base density and its response share one code path. ``us[l]`` are the
    **reduced** radials ``u = r*R`` (divided by ``r`` here to the true radial), ``rr`` the mesh."""
    rr = np.asarray(rr, dtype=float)
    us = {l: [np.asarray(rad) / rr for rad in us[l]] for l in range(lmax + 1)}
    max_l = max((lang for (lang, _) in lset), default=0)
    deg = 2 * lmax + max_l
    ng_x = deg // 2 + 1 if nx is None else nx
    ng_p = deg + 1 if nphi is None else nphi
    nr = len(us[0][0])
    out = {lm: np.zeros(nr, dtype=complex) for lm in lset}
    lm_by_l: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for (big_l, big_m) in lset:
        lm_by_l.setdefault(big_l, []).append((big_m, (big_l, big_m)))
    for l in range(lmax + 1):
        for lp in range(lmax + 1):
            big_ls = [big_l for big_l in lm_by_l
                      if abs(l - lp) <= big_l <= l + lp and (l + lp + big_l) % 2 == 0]
            if not big_ls:
                continue
            for p, rad_p in enumerate(us[l]):
                for q, rad_q in enumerate(us[lp]):
                    dmat = dmats[(l, p, lp, q)]
                    radprod = rad_p * rad_q
                    for big_l in big_ls:
                        for (big_m, key) in lm_by_l[big_l]:
                            g = gaunt_matrix(l, big_l, big_m, lp, ng_x, ng_p)
                            s = complex(np.sum(dmat * np.conj(g)))
                            if s != 0.0:
                                out[key] += s * radprod
    return out


def sphere_amp_response_k(C, dC_occ, occ_idx, ks, abl, lmax, vol, tau, dtau=None):
    """Base and response sphere amplitudes for one atom at one k (no local orbitals).

    ``C`` (npw, rank) eigenvectors, ``dC_occ`` (npw, n_occ) their occupied response, ``occ_idx`` the
    occupied indices, ``ks`` (npw, 3) the (k+G) vectors, ``abl`` this atom's ``{l: (npw, 2)}`` a/b
    matching coefficients, ``tau`` (3,) the atom position (Angstrom). Returns
    ``(amps_all, damps_all)`` each ``{l: [A_a, A_b]}`` of shape ``(n_occ, 2l+1)`` — the
    occupied-band amplitudes and their first-order response.

    The only explicit ``tau`` dependence of the sphere amplitude is the structure phase
    ``exp(i (k+G).tau)``; when ``dtau`` (3,) is given the response adds the analytic geometry term
    ``i (k+G).dtau * exp(i (k+G).tau)`` (the moving-boundary contribution to ``dA``)."""
    from gradwave.flapw.scf import _bands_amps, _ylm_star
    ylm_by_l = {lang: _ylm_star(lang, ks) for lang in range(lmax + 1)}
    phase = np.exp(1j * (ks @ np.asarray(tau, dtype=float)))     # (npw,)
    c_occ = C[:, occ_idx] * phase[:, None]                       # (npw, n_occ)
    amps = _bands_amps(c_occ, ks, abl, lmax, vol, ylm_by_l=ylm_by_l)
    dc = dC_occ * phase[:, None]
    if dtau is not None:
        dphase = 1j * (ks @ np.asarray(dtau, dtype=float))       # (npw,)
        dc = dc + C[:, occ_idx] * (dphase * phase)[:, None]      # geometry term at fixed C
    damps = _bands_amps(dc, ks, abl, lmax, vol, ylm_by_l=ylm_by_l)
    return amps, damps


def dvzz_du_jvp(v_base, dv_du):
    """Total ``dV_zz/du`` from the base l=2 r^2 coefficients ``v_base = {0,1,2}`` and their
    directional response ``dv_du = {0,1,2}`` (each a complex scalar).

    ``V_zz`` is the largest-magnitude eigenvalue of the Cartesian EFG tensor built from
    ``(v0, v1, v2)`` — nonlinear, so the Jacobian is taken by autograd through the exact
    :func:`gradwave.flapw.efg_torch._tensor_from_v_torch` at the base ``v``. A scalar parameter
    ``s`` carries the tangent (``v(s) = v_base + s * dv_du``) so ``dV_zz/du = dV_zz/ds|_0`` is an
    unambiguous forward JVP regardless of the complex intermediates. Both the chi_0 (valence) and
    the Phase-1 (boundary) contributions to ``dv_du`` are summed here at the SAME base ``v`` — the
    correct place, since ``V_zz`` is one nonlinear function of the combined coefficients."""
    import torch
    s = torch.zeros((), dtype=torch.float64, requires_grad=True)
    v = []
    for m in range(3):
        base = torch.tensor(complex(v_base[m]), dtype=torch.complex128)
        tang = torch.tensor(complex(dv_du[m]), dtype=torch.complex128)
        v.append(base + s.to(torch.complex128) * tang)
    _, v_zz, _ = _tensor_from_v_torch(v[0], v[1], v[2])
    (grad,) = torch.autograd.grad(v_zz, s)
    return float(grad), float(v_zz.detach())


def khxc_response(drho, rho_sph, rho_2m, rr, drw, lset, nx: int = 16, nphi: int = 24):
    """The ``K_Hxc`` Dyson kernel applied to a density response: the aspherical potential change
    ``dV_LM(r)`` induced by an aspherical density change ``drho_LM(r)`` at the base density
    ``(rho_sph, rho_2m)``. The linearisation (JVP) of
    :func:`gradwave.flapw.efg.nonspherical_potential`, so the screening iterate reuses the exact
    forward density→potential map:

      * Hartree (Weinert on-site sphere Poisson :func:`gradwave.flapw.efg.lx_sphere_poisson`) is
        LINEAR in ``rho_LM``, so its response is itself applied to ``drho_LM`` (diagonal in L,M).
      * XC is nonlinear: ``dV_xc_LM = ∮ f_xc[ρ(r,Ω)] · dρ(r,Ω) Y*_LM dΩ`` with the base angular
        density ``ρ(r,Ω) = rho_sph + Σ_{lset} rho_LM Y_LM``, the tangent
        ``dρ(r,Ω) = Σ_{lset} drho_LM Y_LM``, and the elementwise kernel
        ``f_xc = dV_xc/dρ`` (:func:`gradwave.flapw.functionals.fxc_lda`) — the SAME angular
        quadrature + Y_LM projection ``nonspherical_potential`` uses for the forward XC.

    ``drho``/``rho_2m`` are ``{(L,M): radial}`` (``drho`` must cover ``lset``; missing keys are 0),
    ``rho_sph`` the spherical (val+core) density, ``lset`` the aspherical channels (L≥1). Returns
    ``{(L,M): dV_LM(r)}`` over ``lset``. Cross-``L`` XC coupling (an anisotropic ``f_xc`` mixing
    channels) is retained exactly by the shared angular grid; the Hartree stays diagonal."""
    import torch
    from scipy.special import sph_harm_y

    from gradwave.flapw.efg import _angular_grid, lx_sphere_poisson
    from gradwave.flapw.functionals import fxc_lda
    th, ph, wgt = _angular_grid(nx, nphi)
    ylm = {(lang, m): sph_harm_y(lang, m, th, ph) for (lang, m) in lset}
    rho_ang = np.broadcast_to(np.asarray(rho_sph)[:, None, None], (len(rr), *th.shape)).copy()
    drho_ang = np.zeros((len(rr), *th.shape))
    for lm in lset:
        rho_ang += (rho_2m[lm][:, None, None] * ylm[lm][None]).real
        drho_ang += (np.asarray(drho[lm])[:, None, None] * ylm[lm][None]).real
    fxc = fxc_lda(torch.tensor(np.clip(rho_ang, 1e-10, None))).numpy()
    dvxc_ang = fxc * drho_ang
    out = {}
    for (lang, m) in lset:
        v_h = lx_sphere_poisson(np.asarray(drho[(lang, m)]), rr, drw, lang)
        v_xc = (dvxc_ang * (wgt * np.conj(ylm[(lang, m)]))[None]).sum(axis=(1, 2))
        out[(lang, m)] = v_h + v_xc
    return out


def valence_v_coeffs(multipoles, rr, drw):
    """The l=2 valence r^2 coefficients ``v_m = (4 pi E2/5) sum(rho_2m/r) dr`` (numpy). Delegates
    to :func:`gradwave.flapw.efg._valence_v` (the canonical on-site l=2 sphere-Poisson coefficient)
    to turn a multipole response ``drho_2m/du`` into the valence part of ``dv_m/du``."""
    from gradwave.flapw.efg import _valence_v

    return _valence_v(multipoles, rr, drw)
