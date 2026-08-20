"""PROD-F (self-consistent loop) — an isolated-atom self-consistent LAPW calculation.

Closes the LAPW SCF cycle with no external reference: build the valence charge density from the
LAPW Bloch eigenstates (the FLAPW augmentation amplitudes A_lm, B_lm), add the frozen core
density, form the Hartree (radial Poisson) + LDA-XC potential, mix, and re-solve — iterating to
self-consistency. Verified against the PROD-C all-electron atomic eigenvalues.

The isolated atom keeps the Coulomb spherical (radial Poisson), so this exercises the whole SCF
machinery — density-from-eigenvectors, potential generation, mixing — before the crystal Coulomb
(Weinert's method) is added. Single atom at Γ, so H, S are real.

    uv run python experiments/autoapw/prod_scf.py

Inside the muffin tin, band n expands as ψ_n = Σ_lm [A^n_lm u_l(r) + B^n_lm u̇_l(r)] Y_lm(r̂), with
    A^n_lm = (4π/√Ω) i^l Σ_G c^n_G Y*_lm(k̂_G) a_l(|k_G|),   B^n_lm the same with b_l,
so the spherically-averaged valence density is
    ρ_val(r) = (1/4π) Σ_l [ P^A_l u_l² + 2 P^AB_l u_l u̇_l + P^B_l u̇_l² ] / r²,
    P^A_l = Σ_{n∈occ,m} f_n |A^n_lm|², etc.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from atomic_scf import CONFIG, anderson_next, hartree, vxc_lda
from prod_lapw import match_ab, radial_channel
from radial_eigen import radial_eigs_tridiag
from radial_log import log_mesh

from gradwave.constants import E2, HBAR2_2M


def _ylm_star(l, kg):
    """conj(Y_lm(k̂)) for m=-l..l as a (2l+1,) complex array; kg a (3,) vector.

    scipy ≥1.15: sph_harm_y(n, m, theta_polar, phi_azimuthal) (physics convention)."""
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


def lapw_single(L, R, lmax, El, ecut, r, dx, v):
    """Single-atom LAPW at Γ: return eigvals, eigvecs, the G-vectors, and per-l matching coeffs."""
    from scipy.special import eval_legendre
    vol = L**3
    b = 2 * math.pi / L
    nmax = int(math.ceil(math.sqrt(ecut / HBAR2_2M) / b)) + 1
    ks = np.array([b * np.array([i, j, m]) for i in range(-nmax, nmax + 1)
                   for j in range(-nmax, nmax + 1) for m in range(-nmax, nmax + 1)
                   if HBAR2_2M * b * b * (i * i + j * j + m * m) <= ecut])
    npw = len(ks)
    ksafe = np.maximum(np.linalg.norm(ks, axis=1), 1e-12)
    dk = np.linalg.norm(ks[:, None, :] - ks[None, :, :], axis=2)
    from prod_lapw import ball_ff_np
    inter = np.eye(npw) - ball_ff_np(dk, R) / vol
    kdot = ks @ ks.T
    cost = np.clip(kdot / np.outer(ksafe, ksafe), -1.0, 1.0)
    S = inter.copy()
    H = HBAR2_2M * kdot * inter
    abl = {}
    chans = {}
    for lang in range(lmax + 1):
        ch = radial_channel(lang, El[lang], r, dx, v, R)
        chans[lang] = ch
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
    return ea, c, ks, ksafe, abl, chans, vol


def valence_density(occ_bands, c, ks, ksafe, abl, chans, lmax, vol, rr):
    """Spherically-averaged valence density on the sphere mesh rr from occupied LAPW bands."""
    npw = len(ks)
    ylm = {lang: np.array([_ylm_star(lang, ks[g]) for g in range(npw)])
           for lang in range(lmax + 1)}
    rho = np.zeros_like(rr)
    for lang in range(lmax + 1):
        a, bb = abl[lang][:, 0], abl[lang][:, 1]
        u, ud = chans[lang]["_u"], chans[lang]["_udot"]
        pfac = (4 * math.pi / math.sqrt(vol)) * (1j ** lang)
        PA = PAB = PB = 0.0
        for n, f in occ_bands:
            cn = c[:, n]
            # A_lm = pfac Σ_G c_G Y*_lm(k̂_G) a_l(G) ; B_lm same with b_l
            wa = cn * a
            wb = cn * bb
            A = pfac * (ylm[lang] * wa[:, None]).sum(axis=0)      # (2l+1,)
            B = pfac * (ylm[lang] * wb[:, None]).sum(axis=0)
            PA += f * float(np.sum(np.abs(A) ** 2))
            PAB += f * float(np.sum((A.conj() * B).real))
            PB += f * float(np.sum(np.abs(B) ** 2))
        rho += (1.0 / (4 * math.pi)) * (PA * u * u + 2 * PAB * u * ud + PB * ud * ud) / rr**2
    return rho


def run_atom_scf(symbol="Ne", R=2.2, L=8.0, ecut=180.0, lmax=2, iters=40):
    r, dx = log_mesh(1e-5, 28.0, 2500)
    Z, occ = CONFIG[symbol]
    inside = r.numpy() <= R
    rr = r.numpy()[inside]
    # core states (frozen): the l=0 states below valence (1s for Ne/Be)
    core = {"He": [], "Be": [(0, 2)], "Ne": [(0, 2)]}[symbol]     # (l, occ) core shells
    val_bands_target = {"Ne": 4, "Be": 1}[symbol]                # spatial valence bands (2s+2p)

    # reference all-electron eigenvalues + a starting potential from PROD-C
    from atomic_scf import atomic_scf
    at_eigs, v = atomic_scf(symbol, r, dx)
    v = v.clone()
    hist_v, hist_r = [], []
    El = {0: at_eigs.get("2s", -5.0), 1: at_eigs.get("2p", -5.0), 2: -5.0}

    print(f"\n  isolated-atom self-consistent LAPW ({symbol}, R_MT={R} Å, ecut={ecut} eV):")
    conv = {}
    for _ in range(iters):
        v0 = float(v.numpy()[np.argmin(np.abs(r.numpy() - R))])
        v_mt = torch.where(r <= R, v - v0, torch.zeros_like(r))
        for lg in El:
            El[lg] = at_eigs.get({0: "2s", 1: "2p"}.get(lg, ""), -5.0) - v0
        # attach the radial functions for density build
        chans_ref = {}
        ea, c, ks, ksafe, abl, chans, vol = lapw_single(L, R, lmax, El, ecut, r, dx, v_mt)
        for lg in range(lmax + 1):
            uu = _radial_u(lg, El[lg], r, dx, v_mt, R)
            chans[lg]["_u"], chans[lg]["_udot"] = uu
            chans_ref[lg] = chans[lg]
        occ_bands = [(n, 2.0) for n in range(val_bands_target)]
        rho_val = valence_density(occ_bands, c, ks, ksafe, abl, chans_ref, lmax, vol, rr)

        # core density (frozen) from the current potential
        rho_core_full = np.zeros_like(r.numpy())
        for lc, fc in core:
            Ec, uc = radial_eigs_tridiag(lc, r, dx, v, 1)
            uu = uc[:, 0]
            rho_core_full += fc * uu * uu / (4 * math.pi * r.numpy() ** 2)
        rho_full = rho_core_full.copy()
        rho_full[inside] += rho_val

        rho_t = torch.tensor(rho_full, dtype=torch.float64)
        vnew = -Z * E2 / r + hartree(rho_t, r, dx) + vxc_lda(rho_t)
        hist_v.append(v)
        hist_r.append(vnew - v)
        if len(hist_r) > 6:
            hist_v, hist_r = hist_v[-6:], hist_r[-6:]
        new = {"2s": float(ea[0] + v0), "2p": float(ea[1] + v0)}
        if conv and max(abs(new[k] - conv[k]) for k in new) < 5e-3:
            conv = new
            break
        conv = new
        v = anderson_next(hist_v, hist_r, beta=0.4, m=5)

    print(f"  {'level':>6} | {'LAPW-SCF (eV)':>14} | {'PROD-C AE (eV)':>15} | {'|Δ| (eV)':>9}")
    ok = True
    for lvl in ("2s", "2p"):
        d = abs(conv[lvl] - at_eigs[lvl])
        ok = ok and d < 0.5
        print(f"  {lvl:>6} | {conv[lvl]:>14.3f} | {at_eigs[lvl]:>15.3f} | {d:>9.3f}")
    print(f"\n  VERDICT: self-consistent LAPW loop converges to the all-electron atom: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def _radial_u(l, El, r, dx, v, R):
    from radial_log import numerov_log
    rr_np = r.numpy()
    inside = rr_np <= R
    drw = rr_np[inside] * dx

    def norm_u(E):
        u = numerov_log(l, torch.tensor(E, dtype=torch.float64), r, dx, v).detach().numpy()
        return u / math.sqrt((u[inside] ** 2 * drw).sum())

    u = norm_u(El)
    hE = max(abs(El) * 1e-4, 1e-3)
    ud = (norm_u(El + hE) - norm_u(El - hE)) / (2 * hE)
    return u[inside], ud[inside]


def main():
    print("\nPROD-F (self-consistent loop) — isolated-atom self-consistent LAPW\n")
    run_atom_scf("Ne")


if __name__ == "__main__":
    main()
