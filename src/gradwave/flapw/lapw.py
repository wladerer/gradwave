"""The (L)APW secular equation in production units (eV/Å) on a log mesh.

A basis function for reciprocal vector G (wavevector ``k_G = k+G``), cell volume Ω, muffin-tin
radius R, is a plane wave ``e^{i k_G·r}/√Ω`` in the interstitial, augmented inside the sphere by
``Σ_lm [a_l u_l(r) + b_l u̇_l(r)] Y_lm`` with ``(a_l, b_l)`` the value+slope match of the radial
functions to ``r·j_l(|k_G| r)`` at R. The resulting overlap S and Hamiltonian H are

    S_GG'  = δ_GG' - W(|Δk|)/Ω          + (4π/Ω) Σ_l (2l+1) P_l(k̂_G·k̂_G') M^S_l
    H_GG'  = ℏ²2m k_G·k_G' (δ_GG' - W/Ω) + (4π/Ω) Σ_l (2l+1) P_l(k̂_G·k̂_G') (T^S_l + V^S_l)

where W is the ball form factor (the interstitial step ``Θ(G)``), the muffin-tin kinetic uses the
weak form, and the radial integrals come from ``radial_channel``. ``build_matrices_multi`` adds the
per-atom structure phases ``e^{i(k_G'-k_G)·τ_a}`` for a crystal (complex Hermitian S/H).
"""

from __future__ import annotations

import math

import numpy as np

from gradwave.constants import HBAR2_2M
from gradwave.flapw.coulomb import ball_ff_np
from gradwave.flapw.radial import numerov_log_np


def sph_jn(l, x):
    from scipy.special import spherical_jn
    return spherical_jn(l, np.asarray(x))


def radial_channel(l, El, r, dx, v, R):
    """``u_l``, ``u̇_l`` on the log mesh + value/slope at R, overlaps, weak-form kinetic (ℏ²2m), and
    potential integrals. ``d/dr = (1/r) d/dx`` on the log mesh; ``dr = r·dx``."""
    r_np = r.detach().numpy()
    inside = r_np <= R
    rr = r_np[inside]
    drw = rr * dx

    hE = max(abs(El) * 1e-4, 1e-3)
    n_cut = int(np.searchsorted(r_np, 2.0 * R)) + 5           # only integrate a little past R_MT
    uraw = numerov_log_np(l, np.array([El, El + hE, El - hE]), r, dx, v, n_cut=n_cut)
    un = uraw / np.sqrt((uraw[:, inside] ** 2 * drw).sum(axis=1))[:, None]
    u = un[0]
    udot = (un[1] - un[2]) / (2 * hE)

    def val_slope(f):
        idx = np.sort(np.argsort(np.abs(r_np - R))[:7])
        c = np.polyfit(r_np[idx] - R, f[idx], 3)
        return float(c[-1]), float(c[-2])

    uR, upR = val_slope(u)
    udR, udpR = val_slope(udot)
    ui, udi, v_in = u[inside], udot[inside], v.detach().numpy()[inside]
    ov = {"uu": (ui * ui * drw).sum(), "uud": (ui * udi * drw).sum(),
          "udud": (udi * udi * drw).sum()}
    ll = l * (l + 1)
    rf = {"u": ui / rr, "ud": udi / rr}
    drf = {"u": (np.gradient(ui, dx) - ui) / rr**2, "ud": (np.gradient(udi, dx) - udi) / rr**2}

    def kin(i, j):   # weak-form muffin-tin kinetic: ℏ²2m ∫[R_i'R_j' r² + l(l+1)R_iR_j] dr
        return HBAR2_2M * ((drf[i] * drf[j] * rr**2 + ll * rf[i] * rf[j]) * drw).sum()

    def pot(i, j):
        f = {"u": ui, "ud": udi}
        return (f[i] * v_in * f[j] * drw).sum()

    return {"uR": uR, "upR": upR, "udR": udR, "udpR": udpR,
            "uu": float(ov["uu"]), "uud": float(ov["uud"]), "udud": float(ov["udud"]),
            "Tuu": float(kin("u", "u")), "Tuud": float(kin("u", "ud")),
            "Tudud": float(kin("ud", "ud")),
            "Vuu": float(pot("u", "u")), "Vuud": float(pot("u", "ud")),
            "Vudud": float(pot("ud", "ud")), "El": El, "l": l}


def match_ab(ch, q, R):
    """Value+slope match of ``(u_l, u̇_l)`` to ``r·j_l(qr)`` at R (the ``u = r·R`` factor)."""
    x = q * R
    jl = float(sph_jn(ch["l"], x))
    djl = -float(sph_jn(1, x)) if ch["l"] == 0 else (
        float(sph_jn(ch["l"] - 1, x)) - (ch["l"] + 1) / x * float(sph_jn(ch["l"], x)))
    tval, tslope = R * jl, jl + R * q * djl
    w = ch["uR"] * ch["udpR"] - ch["upR"] * ch["udR"]
    a = (tval * ch["udpR"] - tslope * ch["udR"]) / w
    b = (ch["uR"] * tslope - ch["upR"] * tval) / w
    return a, b


def _accumulate(H, S, ch, a, bb, aa, ab_s, bbo, pref):
    Ms = aa * ch["uu"] + ab_s * ch["uud"] + bbo * ch["udud"]
    Tk = aa * ch["Tuu"] + ab_s * ch["Tuud"] + bbo * ch["Tudud"]
    Vk = aa * ch["Vuu"] + ab_s * ch["Vuud"] + bbo * ch["Vudud"]
    return S + pref * Ms, H + pref * (Tk + Vk), pref * Ms


def build_matrices(kfrac, L, R, lmax, El_by_l, ecut, r, dx, v):
    """Single-atom LAPW S, H (eV/Å) over ``ℏ²2m|k+G|² < ecut`` for one atom at the origin."""
    from scipy.special import eval_legendre
    vol = L**3
    b = 2 * math.pi / L
    nmax = int(math.ceil(math.sqrt(ecut / HBAR2_2M) / b)) + 1
    ks = [b * (np.array([i, j, m]) + np.asarray(kfrac))
          for i in range(-nmax, nmax + 1) for j in range(-nmax, nmax + 1)
          for m in range(-nmax, nmax + 1)]
    ks = np.array([k for k in ks if HBAR2_2M * (k @ k) <= ecut])
    npw = len(ks)
    knorm = np.linalg.norm(ks, axis=1)
    ksafe = np.maximum(knorm, 1e-12)
    dk = np.linalg.norm(ks[:, None, :] - ks[None, :, :], axis=2)
    inter = np.eye(npw) - ball_ff_np(dk, R) / vol
    kdot = ks @ ks.T
    cost = np.clip(kdot / np.outer(ksafe, ksafe), -1.0, 1.0)
    S = inter.copy()
    H = HBAR2_2M * kdot * inter
    for lang in range(lmax + 1):
        ch = radial_channel(lang, El_by_l[lang], r, dx, v, R)
        ab = np.array([match_ab(ch, ksafe[g], R) for g in range(npw)])
        a, bb = ab[:, 0], ab[:, 1]
        aa, bbo = np.outer(a, a), np.outer(bb, bb)
        ab_s = np.outer(a, bb) + np.outer(bb, a)
        pref = (4 * math.pi / vol) * (2 * lang + 1) * eval_legendre(lang, cost)
        S, H, _ = _accumulate(H, S, ch, a, bb, aa, ab_s, bbo, pref)
    return H, S, knorm


def build_matrices_multi(kfrac, L, atoms, lmax, ecut, r, dx, species):
    """Multi-atom LAPW S, H (complex Hermitian). ``atoms = [(τ (3,), species_key), ...]``;
    ``species[key] = {'R': R_MT, 'v': v_tensor, 'El': {l: E_l}}``. Each atom contributes an
    interstitial sphere-removal and augmentation weighted by ``e^{i(k_G'-k_G)·τ}``. Returns
    ``(H, S, comps)`` with ``comps = {'inter': S_interstitial, 'aug': [S_aug per atom]}`` for
    population analysis. Spheres must not overlap."""
    from scipy.special import eval_legendre
    vol = L**3
    b = 2 * math.pi / L
    nmax = int(math.ceil(math.sqrt(ecut / HBAR2_2M) / b)) + 1
    ks = [b * (np.array([i, j, m]) + np.asarray(kfrac))
          for i in range(-nmax, nmax + 1) for j in range(-nmax, nmax + 1)
          for m in range(-nmax, nmax + 1)]
    ks = np.array([k for k in ks if HBAR2_2M * (k @ k) <= ecut])
    npw = len(ks)
    ksafe = np.maximum(np.linalg.norm(ks, axis=1), 1e-12)
    dkvec = ks[None, :, :] - ks[:, None, :]
    dk_norm = np.linalg.norm(dkvec, axis=2)
    kdot = ks @ ks.T
    cost = np.clip(kdot / np.outer(ksafe, ksafe), -1.0, 1.0)
    chan = {key: {l: radial_channel(l, sp["El"][l], r, dx, sp["v"], sp["R"])
                  for l in range(lmax + 1)} for key, sp in species.items()}
    inter = np.eye(npw, dtype=complex)
    Saug = np.zeros((npw, npw), dtype=complex)
    Haug = np.zeros((npw, npw), dtype=complex)
    aug_by_atom = []
    for tau, key in atoms:
        R = species[key]["R"]
        phase = np.exp(1j * (dkvec @ np.asarray(tau, dtype=float)))
        inter -= (ball_ff_np(dk_norm, R) / vol) * phase
        Sa = np.zeros((npw, npw), dtype=complex)
        for lang in range(lmax + 1):
            ch = chan[key][lang]
            ab = np.array([match_ab(ch, ksafe[g], R) for g in range(npw)])
            a, bb = ab[:, 0], ab[:, 1]
            aa, bbo = np.outer(a, a), np.outer(bb, bb)
            ab_s = np.outer(a, bb) + np.outer(bb, a)
            Ms = aa * ch["uu"] + ab_s * ch["uud"] + bbo * ch["udud"]
            Tk = aa * ch["Tuu"] + ab_s * ch["Tuud"] + bbo * ch["Tudud"]
            Vk = aa * ch["Vuu"] + ab_s * ch["Vuud"] + bbo * ch["Vudud"]
            pref = (4 * math.pi / vol) * (2 * lang + 1) * eval_legendre(lang, cost)
            Sa += phase * (pref * Ms)
            Haug += phase * (pref * (Tk + Vk))
        Saug += Sa
        aug_by_atom.append(0.5 * (Sa + Sa.conj().T))
    S = inter + Saug
    H = HBAR2_2M * kdot * inter + Haug
    comps = {"inter": 0.5 * (inter + inter.conj().T), "aug": aug_by_atom}
    return 0.5 * (H + H.conj().T), 0.5 * (S + S.conj().T), comps


def solve_geneig(H, S, nbands, with_vecs=False, tol=1e-8):
    """Generalized eigensolve ``H c = ε S c`` (real or complex Hermitian) by canonical
    orthogonalization: diagonalize S and keep only the subspace with eigenvalue ``> tol·max``,
    dropping the near-linearly-dependent directions of the augmented LAPW basis. Clipping those to a
    floor (instead of dropping) turns a tiny S-eigenvalue into a huge ``w^{-1/2}`` amplification and
    spurious "ghost" eigenvalues — catastrophic for multi-atom cells with large muffin tins. For a
    well-conditioned S this is identical to Löwdin ``S^{-1/2}``. Returns sorted eigenvalues (eV);
    with ``with_vecs`` also the S-normalized eigenvectors."""
    w, u = np.linalg.eigh(S)
    keep = w > tol * w[-1]                                # w ascending; w[-1] = largest
    x = u[:, keep] * (w[keep] ** -0.5)                   # npw × nkeep, S-orthonormal columns
    ea, va = np.linalg.eigh(x.conj().T @ H @ x)
    order = np.argsort(ea.real)[:nbands]
    evals = ea.real[order]
    if len(evals) < nbands:                              # rank-deficient: pad empty (high) states
        evals = np.concatenate([evals, np.full(nbands - len(evals), 1e10)])
    if with_vecs:
        vecs = x @ va[:, order]
        if vecs.shape[1] < nbands:
            vecs = np.concatenate([vecs, np.zeros((x.shape[0], nbands - vecs.shape[1]),
                                                  dtype=vecs.dtype)], axis=1)
        return evals, vecs
    return evals
