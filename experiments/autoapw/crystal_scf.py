"""PROD-F — the full crystal self-consistent FLAPW loop (muffin-tin approximation).

Grinds out the self-consistent cycle for a real crystal, wiring together every verified piece:
  LAPW solve (Bloch states) -> crystal density (interstitial ρ_I from the plane-wave parts +
  spherical sphere density from the augmentation + frozen core) -> Weinert Coulomb (interstitial
  FFT Poisson on the pseudocharge + sphere radial Poisson matched at R_MT) -> LDA XC -> Anderson
  mixing -> repeat.

Muffin-tin approximation: spherical (l=0) potential inside the spheres, constant in the
interstitial. For weakly-bound, nearly-spherical Ne this is close to full-potential; the exact
full-potential (non-spherical sphere terms + varying interstitial potential matrix elements) is
the remaining refinement.

Verified two ways:
  1. dilute limit (large L): recovers the isolated-atom eigenvalues (self-consistent, vs PROD-C).
  2. real crystal (simple-cubic Ne, a=6 Bohr): compared to Elk 11.0.2 (all-electron FLAPW).

    uv run python experiments/autoapw/crystal_scf.py

Simple-cubic Ne, Γ-point. eV/Å units throughout.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from atomic_scf import CONFIG, anderson_next, vxc_lda
from prod_lapw import ball_ff_np, match_ab, radial_channel
from prod_scf import _radial_u, _ylm_star
from radial_eigen import radial_eigs_tridiag
from radial_log import log_mesh
from weinert import fft_poisson, radial_poisson_to_R, sphere_pseudocharge

from gradwave.constants import BOHR_ANG, E2, HBAR2_2M


def enumerate_g(L, kfrac, ecut):
    b = 2 * math.pi / L
    nmax = int(math.ceil(math.sqrt(ecut / HBAR2_2M) / b)) + 1
    mill, ks = [], []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for m in range(-nmax, nmax + 1):
                kg = b * (np.array([i, j, m]) + np.asarray(kfrac))
                if HBAR2_2M * (kg @ kg) <= ecut:
                    mill.append([i, j, m])
                    ks.append(kg)
    return np.array(mill), np.array(ks)


def lapw_gamma(L, R, lmax, El, ecut, r, dx, v_sphere):
    """Single-atom LAPW at Γ (real H,S). Returns eigvals, S-normalized eigvecs, G miller/ks, ab."""
    from scipy.special import eval_legendre
    vol = L**3
    mill, ks = enumerate_g(L, [0, 0, 0], ecut)
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
    c = Sinv2 @ Va
    return ea, c, mill, ks, ksafe, abl, vol


def interstitial_density(c_occ, occ, mill, L, Nfft):
    """ρ_I(r) on an Nfft³ grid from the plane-wave parts of the occupied bands (e/Å³)."""
    vol = L**3
    rho = np.zeros((Nfft, Nfft, Nfft))
    for n, f in enumerate(occ):
        box = np.zeros((Nfft, Nfft, Nfft), dtype=complex)
        idx = (mill[:, 0] % Nfft, mill[:, 1] % Nfft, mill[:, 2] % Nfft)
        box[idx] = c_occ[:, n]
        psi = np.fft.ifftn(box) * Nfft**3            # Σ_G c_G e^{iG·r}
        rho += f * np.abs(psi) ** 2 / vol
    return rho


def sphere_valence_density(c_occ, occ, ks, ksafe, abl, El, lmax, vol, r, dx, v_sphere, R):
    """Spherically-averaged valence density (e/Å³) inside R from the augmentation amplitudes."""
    rr = r.numpy()[r.numpy() <= R]
    us = {lang: _radial_u(lang, El[lang], r, dx, v_sphere, R) for lang in range(lmax + 1)}
    ylm = {lang: np.array([_ylm_star(lang, ks[g]) for g in range(len(ks))])
           for lang in range(lmax + 1)}
    rho = np.zeros_like(rr)
    for lang in range(lmax + 1):
        a, bb = abl[lang][:, 0], abl[lang][:, 1]
        u, ud = us[lang]
        pfac = (4 * math.pi / math.sqrt(vol)) * (1j ** lang)
        PA = PAB = PB = 0.0
        for n, f in enumerate(occ):
            A = pfac * (ylm[lang] * (c_occ[:, n] * a)[:, None]).sum(axis=0)
            B = pfac * (ylm[lang] * (c_occ[:, n] * bb)[:, None]).sum(axis=0)
            PA += f * float(np.sum(np.abs(A) ** 2))
            PAB += f * float(np.sum((A.conj() * B).real))
            PB += f * float(np.sum(np.abs(B) ** 2))
        rho += (1.0 / (4 * math.pi)) * (PA * u * u + 2 * PAB * u * ud + PB * ud * ud) / rr**2
    return rr, rho


def weinert_potential(rho_I, rr, rho_sph, Z, R, L, Nfft, r_full, dx):
    """Muffin-tin Hartree via Weinert + nucleus. Returns (sphere V_H+nuc radial on r_full<R,
    interstitial constant V_I0). rho_sph is the total electron sphere density on rr (r<R)."""
    h = L / Nfft
    c = np.array([L / 2, L / 2, L / 2])
    # sphere is on the LOG mesh: ∫dr weight is r·dx (NOT a constant spacing)
    drw = rr * dx
    q_sph = float(np.sum(4 * math.pi * rho_sph * rr**2 * drw))
    ax = np.arange(Nfft) * h
    X, Y, Z_ = np.meshgrid(ax, ax, ax, indexing="ij")

    def mi(x, x0):
        return (x - x0) - L * np.round((x - x0) / L)

    dgrid = np.sqrt(mi(X, c[0]) ** 2 + mi(Y, c[1]) ** 2 + mi(Z_, c[2]) ** 2)
    inside = dgrid < R
    q_I_in = float(rho_I[inside].sum() * h**3)
    # NET charge = electrons − nucleus (Z). Weinert on the neutral net charge gives a uniquely-
    # referenced electrostatic potential (no G=0 ambiguity). The pseudocharge carries the sphere's
    # NET monopole (q_sph − Z) minus the interstitial-in-sphere; for a neutral atom this is ~0.
    rho_smooth = rho_I + sphere_pseudocharge(q_sph - Z - q_I_in, R, c, Nfft, L, npow=4)
    V_I = fft_poisson(rho_smooth, L)                 # electrostatic potential, cell-average zero
    ic = Nfft // 2
    kR = int(round(R / h))
    V_bc = float(0.25 * (V_I[ic + kR, ic, ic] + V_I[ic - kR, ic, ic]
                         + V_I[ic, ic + kR, ic] + V_I[ic, ic, ic + kR]))
    V_I0 = float(V_I[~inside].mean())                # interstitial average (muffin-tin zero)
    # sphere: radial Poisson of the TOTAL sphere charge (electron density + nuclear point charge),
    # THEN match to the interstitial V(R_MT). The nucleus must be inside the matched quantity so its
    # −Z·E2/R boundary value is referenced consistently (adding −Z/r after matching is a huge bug).
    Vpart = radial_poisson_to_R(rho_sph, rr, R, drw=drw) - Z * E2 / rr
    V_sphere = Vpart + (V_bc - Vpart[-1])
    return rr, V_sphere, V_I0


def crystal_scf(a_bohr, R=1.4, ecut=250.0, lmax=2, iters=40, symbol="Ne"):
    """Muffin-tin self-consistent FLAPW for simple-cubic `symbol`, Γ-point. Returns valence eigs."""
    from atomic_scf import atomic_scf
    L = a_bohr * BOHR_ANG
    Nfft = 2 * int(math.ceil(math.sqrt(4 * ecut / HBAR2_2M) * L / (2 * math.pi))) + 2
    Nfft = min(max(Nfft, 24), 72)
    r, dx = log_mesh(1e-5, 28.0, 2500)
    Z, occ = CONFIG[symbol]
    rr_mask = r.numpy() <= R
    core = {"Ne": [(0, 2)], "Be": [(0, 2)]}[symbol]
    n_val_bands = {"Ne": 4, "Be": 1}[symbol]

    at, v = atomic_scf(symbol, r, dx)          # start from the atomic potential + eigenvalues
    v = v.clone()
    hist_v, hist_r, conv = [], [], {}
    for _ in range(iters):
        v0 = float(v.numpy()[np.argmin(np.abs(r.numpy() - R))])
        v_mt = torch.where(r <= R, v - v0, torch.zeros_like(r))
        El = {0: at.get("2s", -5.0) - v0, 1: at.get("2p", -5.0) - v0, 2: -5.0 - v0}
        ea, c, mill, ks, ksafe, abl, vol = lapw_gamma(L, R, lmax, El, ecut, r, dx, v_mt)
        occb = [2.0] * n_val_bands
        c_occ = c[:, :n_val_bands]

        rho_I = interstitial_density(c_occ, occb, mill, L, Nfft)
        rr, rho_val = sphere_valence_density(c_occ, occb, ks, ksafe, abl, El, lmax, vol,
                                             r, dx, v_mt, R)
        # frozen core density (l=0), from the current potential
        rho_core = np.zeros_like(rr)
        for lc, fc in core:
            _, uc = radial_eigs_tridiag(lc, r, dx, v, 1)
            rho_core += fc * uc[rr_mask, 0] ** 2 / (4 * math.pi * rr**2)
        rho_sph = rho_val + rho_core

        rr2, V_sph, V_I0 = weinert_potential(rho_I, rr, rho_sph, Z, R, L, Nfft, r, dx)
        # XC on the sphere (radial) + full potential inside the sphere
        vxc_sph = vxc_lda(torch.tensor(rho_sph)).numpy()
        vnew_sph = V_sph + vxc_sph
        # build the full-mesh potential: sphere inside, interstitial constant outside
        vnew = torch.full_like(r, V_I0)
        vnew_np = vnew.numpy().copy()
        vnew_np[rr_mask] = vnew_sph
        vnew = torch.tensor(vnew_np)

        hist_v.append(v)
        hist_r.append(vnew - v)
        if len(hist_r) > 6:
            hist_v, hist_r = hist_v[-6:], hist_r[-6:]
        new = {"2s": float(ea[0] + v0), "2p": float(ea[1] + v0)}
        if conv and max(abs(new[k] - conv[k]) for k in new) < 3e-3:
            conv = new
            break
        conv = new
        v = anderson_next(hist_v, hist_r, beta=0.3, m=5)
    return conv, at


def main():
    print("\nPROD-F — full crystal self-consistent FLAPW loop (muffin-tin), simple-cubic Ne\n")

    print("  [1] dilute limit (a=10 Bohr) must recover the isolated-atom eigenvalues:")
    conv, at = crystal_scf(10.0, R=2.0, ecut=120.0, iters=25)
    ok1 = True
    for lvl in ("2s", "2p"):
        d = abs(conv[lvl] - at[lvl])
        ok1 = ok1 and d < 0.5
        print(f"      {lvl}: SCF {conv[lvl]:.3f}  atomic {at[lvl]:.3f}  |Δ| {d:.3f}")

    print("\n  [2] real crystal a=6 Bohr — 2s-2p splitting vs Elk 11 (22.77 eV):")
    conv2, _ = crystal_scf(6.0, R=1.4, ecut=250.0, iters=30)
    split = conv2["2p"] - conv2["2s"]
    ok2 = abs(split - 22.77) < 0.5
    print(f"      2s {conv2['2s']:.3f}  2p {conv2['2p']:.3f}  splitting {split:.2f} eV "
          f"(Elk 22.77)  |Δ| {abs(split - 22.77):.2f}")

    print("\n  VERDICT:")
    print(f"    [1] dilute limit -> atomic eigenvalues : {'PASS' if ok1 else 'FAIL'}")
    print(f"    [2] crystal splitting -> Elk           : {'PASS' if ok2 else 'FAIL'}")


if __name__ == "__main__":
    main()
