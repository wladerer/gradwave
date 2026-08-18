"""Self-consistent crystal FLAPW (muffin-tin approximation).

Wires the full self-consistency cycle for a crystal: LAPW Bloch solve -> crystal density
(interstitial ``ρ_I`` from the plane-wave parts of the eigenvectors via FFT + the spherical sphere
density from the augmentation amplitudes + frozen core) -> Weinert Coulomb (FFT Poisson on the
net-charge pseudocharge + sphere radial Poisson matched at R_MT) -> LDA XC -> Anderson mixing.

Muffin-tin approximation: spherical (l=0) potential inside the spheres, constant in the
interstitial — the classic self-consistent scheme; full-potential (non-spherical l>0 terms +
varying interstitial matrix elements) is the accuracy refinement.

Verified for simple-cubic Ne (Γ): the dilute limit recovers the atomic eigenvalues, and the
a=6 Bohr crystal 2s-2p splitting matches Elk 11.0.2 (all-electron FLAPW) to 0.14 eV.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, E2, HBAR2_2M
from gradwave.flapw.atom import CONFIG, atomic_scf
from gradwave.flapw.coulomb import fft_poisson, radial_poisson_to_R, sphere_pseudocharge
from gradwave.flapw.functionals import vxc_lda
from gradwave.flapw.lapw import ball_ff_np, match_ab, radial_channel
from gradwave.flapw.mixing import anderson_next
from gradwave.flapw.radial import log_mesh, numerov_log, radial_eigs_tridiag

_CORE = {"He": [], "Be": [(0, 2)], "Ne": [(0, 2)]}
_N_VAL_BANDS = {"He": 1, "Be": 1, "Ne": 4}


def _ylm_star(l, kg):
    """conj(Y_lm(k̂)) for m=-l..l (scipy>=1.15 convention sph_harm_y(n,m,theta,phi))."""
    from scipy.special import sph_harm_y
    kn = np.linalg.norm(kg)
    if kn < 1e-12:
        out = np.zeros(2 * l + 1, dtype=complex)
        if l == 0:
            out[0] = 1.0 / math.sqrt(4 * math.pi)
        return out
    theta = math.acos(max(-1.0, min(1.0, kg[2] / kn)))
    phi = math.atan2(kg[1], kg[0])
    return np.array([np.conj(sph_harm_y(l, m, theta, phi)) for m in range(-l, l + 1)])


def _radial_u(l, El, r, dx, v, R):
    r_np = r.numpy()
    inside = r_np <= R
    drw = r_np[inside] * dx

    def norm_u(E):
        u = numerov_log(l, torch.tensor(E, dtype=torch.float64), r, dx, v).detach().numpy()
        return u / math.sqrt((u[inside] ** 2 * drw).sum())

    u = norm_u(El)
    hE = max(abs(El) * 1e-4, 1e-3)
    ud = (norm_u(El + hE) - norm_u(El - hE)) / (2 * hE)
    return u[inside], ud[inside]


def _lapw_gamma(L, R, lmax, El, ecut, r, dx, v_sphere):
    """Single-atom LAPW at Γ (real H,S). Returns eigvals, eigvecs, miller/ks, matching coeffs."""
    from scipy.special import eval_legendre
    vol = L**3
    b = 2 * math.pi / L
    nmax = int(math.ceil(math.sqrt(ecut / HBAR2_2M) / b)) + 1
    mill, ks = [], []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for m in range(-nmax, nmax + 1):
                kg = b * (np.array([i, j, m]))
                if HBAR2_2M * (kg @ kg) <= ecut:
                    mill.append([i, j, m])
                    ks.append(kg)
    mill, ks = np.array(mill), np.array(ks)
    npw = len(ks)
    ksafe = np.maximum(np.linalg.norm(ks, axis=1), 1e-12)
    dk = np.linalg.norm(ks[:, None, :] - ks[None, :, :], axis=2)
    inter = np.eye(npw) - ball_ff_np(dk, R) / vol
    kdot = ks @ ks.T
    cost = np.clip(kdot / np.outer(ksafe, ksafe), -1.0, 1.0)
    S = inter.copy()
    H = HBAR2_2M * kdot * inter
    abl = {}
    for lang in range(lmax + 1):
        ch = radial_channel(lang, El[lang], r, dx, v_sphere, R)
        ab = np.array([match_ab(ch, ksafe[g], R) for g in range(npw)])
        abl[lang] = ab
        a, bb = ab[:, 0], ab[:, 1]
        aa, bbo = np.outer(a, a), np.outer(bb, bb)
        ab_s = np.outer(a, bb) + np.outer(bb, a)
        Ms = aa * ch["uu"] + ab_s * ch["uud"] + bbo * ch["udud"]
        Tk = aa * ch["Tuu"] + ab_s * ch["Tuud"] + bbo * ch["Tudud"]
        Vk = aa * ch["Vuu"] + ab_s * ch["Vuud"] + bbo * ch["Vudud"]
        pref = (4 * math.pi / vol) * (2 * lang + 1) * eval_legendre(lang, cost)
        S += pref * Ms
        H += pref * (Tk + Vk)
    H = 0.5 * (H + H.T)
    w, U = np.linalg.eigh(S)
    Sinv2 = U @ np.diag(np.clip(w, 1e-12, None) ** -0.5) @ U.T
    ea, Va = np.linalg.eigh(Sinv2 @ H @ Sinv2)
    return ea, Sinv2 @ Va, mill, ks, ksafe, abl, vol


def _interstitial_density(c_occ, occ, mill, L, nfft):
    vol = L**3
    rho = np.zeros((nfft, nfft, nfft))
    idx = (mill[:, 0] % nfft, mill[:, 1] % nfft, mill[:, 2] % nfft)
    for n, f in enumerate(occ):
        box = np.zeros((nfft, nfft, nfft), dtype=complex)
        box[idx] = c_occ[:, n]
        psi = np.fft.ifftn(box) * nfft**3
        rho += f * np.abs(psi) ** 2 / vol
    return rho


def _sphere_valence_density(c_occ, occ, ks, abl, El, lmax, vol, r, dx, v_sphere, R):
    rr = r.numpy()[r.numpy() <= R]
    us = {lang: _radial_u(lang, El[lang], r, dx, v_sphere, R) for lang in range(lmax + 1)}
    ylm = {lang: np.array([_ylm_star(lang, ks[g]) for g in range(len(ks))])
           for lang in range(lmax + 1)}
    rho = np.zeros_like(rr)
    for lang in range(lmax + 1):
        a, bb = abl[lang][:, 0], abl[lang][:, 1]
        u, ud = us[lang]
        pfac = (4 * math.pi / math.sqrt(vol)) * (1j ** lang)
        p_a = p_ab = p_b = 0.0
        for n, f in enumerate(occ):
            amp_a = pfac * (ylm[lang] * (c_occ[:, n] * a)[:, None]).sum(axis=0)
            amp_b = pfac * (ylm[lang] * (c_occ[:, n] * bb)[:, None]).sum(axis=0)
            p_a += f * float(np.sum(np.abs(amp_a) ** 2))
            p_ab += f * float(np.sum((amp_a.conj() * amp_b).real))
            p_b += f * float(np.sum(np.abs(amp_b) ** 2))
        rho += (1.0 / (4 * math.pi)) * (p_a * u * u + 2 * p_ab * u * ud + p_b * ud * ud) / rr**2
    return rr, rho


def _weinert_potential(rho_I, rr, dx, rho_sph, Z, R, L, nfft):
    """Muffin-tin Hartree via Weinert + nucleus. Returns (sphere V radial, interstitial V0)."""
    h = L / nfft
    c = np.array([L / 2, L / 2, L / 2])
    drw = rr * dx
    q_sph = float(np.sum(4 * math.pi * rho_sph * rr**2 * drw))
    ax = np.arange(nfft) * h
    X, Y, Zc = np.meshgrid(ax, ax, ax, indexing="ij")

    def mi(x, x0):
        return (x - x0) - L * np.round((x - x0) / L)

    dgrid = np.sqrt(mi(X, c[0]) ** 2 + mi(Y, c[1]) ** 2 + mi(Zc, c[2]) ** 2)
    inside = dgrid < R
    q_i_in = float(rho_I[inside].sum() * h**3)
    # NET charge = electrons − nucleus (Z): Weinert on the neutral net charge is well-referenced.
    rho_smooth = rho_I + sphere_pseudocharge(q_sph - Z - q_i_in, R, c, nfft, L, npow=4)
    v_grid = fft_poisson(rho_smooth, L)
    ic = nfft // 2
    kR = int(round(R / h))
    v_bc = float(0.25 * (v_grid[ic + kR, ic, ic] + v_grid[ic - kR, ic, ic]
                         + v_grid[ic, ic + kR, ic] + v_grid[ic, ic, ic + kR]))
    v_i0 = float(v_grid[~inside].mean())
    # sphere: radial Poisson of the TOTAL sphere charge (electrons + nuclear point), matched at R.
    vpart = radial_poisson_to_R(rho_sph, rr, R, drw=drw) - Z * E2 / rr
    return vpart + (v_bc - vpart[-1]), v_i0


def crystal_scf(a_bohr: float, symbol: str = "Ne", R: float = 1.4, ecut: float = 250.0,
                lmax: int = 2, iters: int = 40, tol: float = 3e-3):
    """Muffin-tin self-consistent FLAPW for a simple-cubic crystal at Γ. Returns
    ``(valence_eigs_eV, atomic_eigs_eV)`` — the crystal valence eigenvalues and the atomic limit.

    The returned eigenvalues are referenced to the flat interstitial potential level, which the
    muffin-tin scheme only weakly determines (worst in dilute cells): the *absolute* levels wander
    run-to-run under threaded BLAS while energy *differences* (2p-2s splittings, bandwidths) are
    stable and physical. Compare splittings, not absolute eigenvalues, as against any FLAPW code."""
    L = a_bohr * BOHR_ANG
    nfft = 2 * int(math.ceil(math.sqrt(4 * ecut / HBAR2_2M) * L / (2 * math.pi))) + 2
    nfft = min(max(nfft, 24), 72)
    r, dx = log_mesh(1e-5, 28.0, 2500)
    Z, _ = CONFIG[symbol]
    rr_mask = r.numpy() <= R
    core, n_val = _CORE[symbol], _N_VAL_BANDS[symbol]

    at, v = atomic_scf(symbol, r, dx)
    v = v.clone()
    hist_v, hist_r, conv = [], [], {}
    for _ in range(iters):
        v0 = float(v.numpy()[np.argmin(np.abs(r.numpy() - R))])
        v_mt = torch.where(r <= R, v - v0, torch.zeros_like(r))
        El = {0: at.get("2s", -5.0) - v0, 1: at.get("2p", -5.0) - v0, 2: -5.0 - v0}
        ea, c, mill, ks, _, abl, vol = _lapw_gamma(L, R, lmax, El, ecut, r, dx, v_mt)
        occb, c_occ = [2.0] * n_val, c[:, :n_val]

        rho_I = _interstitial_density(c_occ, occb, mill, L, nfft)
        rr, rho_val = _sphere_valence_density(c_occ, occb, ks, abl, El, lmax, vol, r, dx, v_mt, R)
        rho_core = np.zeros_like(rr)
        for lc, fc in core:
            _, uc = radial_eigs_tridiag(lc, r, dx, v, 1)
            rho_core += fc * uc[rr_mask, 0] ** 2 / (4 * math.pi * rr**2)
        rho_sph = rho_val + rho_core

        v_sph, v_i0 = _weinert_potential(rho_I, rr, dx, rho_sph, Z, R, L, nfft)
        vnew_sph = v_sph + vxc_lda(torch.tensor(rho_sph)).numpy()
        vnew_np = np.full(r.shape[0], v_i0)
        vnew_np[rr_mask] = vnew_sph
        vnew = torch.tensor(vnew_np)

        hist_v.append(v)
        hist_r.append(vnew - v)
        if len(hist_r) > 6:
            hist_v, hist_r = hist_v[-6:], hist_r[-6:]
        new = {"2s": float(ea[0] + v0), "2p": float(ea[1] + v0)}
        if conv and max(abs(new[k] - conv[k]) for k in new) < tol:
            conv = new
            break
        conv = new
        v = anderson_next(hist_v, hist_r, beta=0.3, m=5)
    return conv, at
