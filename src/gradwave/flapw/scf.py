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
from gradwave.flapw.lapw import ball_ff_np, match_ab, radial_channel, solve_geneig
from gradwave.flapw.mixing import anderson_next
from gradwave.flapw.radial import log_mesh, numerov_log, radial_eigs_tridiag
from gradwave.kpoints import monkhorst_pack

_CORE = {"He": [], "Be": [(0, 2)], "Ne": [(0, 2)]}
_N_VAL_BANDS = {"He": 1, "Be": 1, "Ne": 4}
_VAL_E = {"He": 2, "Be": 2, "Ne": 8}       # valence electron count (Z − frozen core)


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


def _lapw_k(kfrac, L, R, lmax, El, ecut, r, dx, v_sphere):
    """Single-atom LAPW at wavevector k (fractional ``kfrac``). One atom at the origin carries no
    structure phase, so H,S are real symmetric at any k. Returns eigvals, eigvecs, miller/ks,
    matching coeffs. ``ks`` are the Cartesian k+G; the ecut sphere is centred at -k."""
    from scipy.special import eval_legendre
    vol = L**3
    b = 2 * math.pi / L
    kf = np.asarray(kfrac, dtype=float)
    nmax = int(math.ceil(math.sqrt(ecut / HBAR2_2M) / b)) + 2
    mill, ks = [], []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for m in range(-nmax, nmax + 1):
                kg = b * (np.array([i, j, m]) + kf)
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
    vol = float(np.prod(np.broadcast_to(np.asarray(L, dtype=float), (3,))))
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
                lmax: int = 2, iters: int = 40, tol: float = 3e-3, kmesh=(1, 1, 1)):
    """Muffin-tin self-consistent FLAPW for a simple-cubic crystal. Returns
    ``(valence_eigs_eV, atomic_eigs_eV)`` — the crystal Γ valence eigenvalues and the atomic limit.

    ``kmesh`` is a Monkhorst-Pack division for BZ integration of the density (default Γ-only). The
    reported eigenvalues are always at Γ; the density is summed over the mesh with symmetry weights.

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
    rr = r.numpy()[rr_mask]
    core, n_val = _CORE[symbol], _N_VAL_BANDS[symbol]
    kfracs, kw = monkhorst_pack(tuple(kmesh))
    kw = kw / kw.sum()

    at, v = atomic_scf(symbol, r, dx)
    v = v.clone()
    hist_v, hist_r, conv = [], [], {}
    for _ in range(iters):
        v0 = float(v.numpy()[np.argmin(np.abs(r.numpy() - R))])
        v_mt = torch.where(r <= R, v - v0, torch.zeros_like(r))
        El = {0: at.get("2s", -5.0) - v0, 1: at.get("2p", -5.0) - v0, 2: -5.0 - v0}
        occb = [2.0] * n_val
        # BZ-integrate the valence density over the k-mesh; keep Γ eigenvalues for reporting.
        rho_I = np.zeros((nfft, nfft, nfft))
        rho_val = np.zeros_like(rr)
        ea_gamma = None
        for kf, w in zip(kfracs, kw, strict=True):
            ea, c, mill, ks, _, abl, vol = _lapw_k(kf, L, R, lmax, El, ecut, r, dx, v_mt)
            c_occ = c[:, :n_val]
            rho_I += w * _interstitial_density(c_occ, occb, mill, L, nfft)
            _, rk = _sphere_valence_density(c_occ, occb, ks, abl, El, lmax, vol, r, dx, v_mt, R)
            rho_val += w * rk
            if np.all(np.abs(kf) < 1e-9):
                ea_gamma = ea
        if ea_gamma is None:                       # mesh excludes Γ; one extra solve for reporting
            ea_gamma = _lapw_k((0.0, 0.0, 0.0), L, R, lmax, El, ecut, r, dx, v_mt)[0]
        ea = ea_gamma
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


def _lapw_multi_k(kf, L, atoms_cart, species, lmax, ecut, r, dx, nbands):
    """Multi-atom LAPW at wavevector k, exposing the density internals the SCF needs. Returns
    ``(eigvals, eigvecs, miller, ks, [abl per atom], vol)``. H,S are complex Hermitian (structure
    phases ``e^{i(k_G'-k_G)·τ_a}``). ``atoms_cart`` = ``[(τ_cart, species_key), ...]``."""
    from scipy.special import eval_legendre
    L3 = np.broadcast_to(np.asarray(L, dtype=float), (3,))
    vol = float(np.prod(L3))
    bvec = 2 * math.pi / L3
    kf = np.asarray(kf, dtype=float)
    nmax = int(math.ceil(math.sqrt(ecut / HBAR2_2M) / bvec.min())) + 2
    mill, ks = [], []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for m in range(-nmax, nmax + 1):
                kg = bvec * (np.array([i, j, m]) + kf)
                if HBAR2_2M * (kg @ kg) <= ecut:
                    mill.append([i, j, m])
                    ks.append(kg)
    mill, ks = np.array(mill), np.array(ks)
    npw = len(ks)
    ksafe = np.maximum(np.linalg.norm(ks, axis=1), 1e-12)
    dkvec = ks[None, :, :] - ks[:, None, :]
    dk_norm = np.linalg.norm(dkvec, axis=2)
    kdot = ks @ ks.T
    cost = np.clip(kdot / np.outer(ksafe, ksafe), -1.0, 1.0)
    chan = {key: {lang: radial_channel(lang, sp["El"][lang], r, dx, sp["v"], sp["R"])
                  for lang in range(lmax + 1)} for key, sp in species.items()}
    inter = np.eye(npw, dtype=complex)
    Saug = np.zeros((npw, npw), dtype=complex)
    Haug = np.zeros((npw, npw), dtype=complex)
    abl_by_atom = []
    for tau, key in atoms_cart:
        R = species[key]["R"]
        phase = np.exp(1j * (dkvec @ np.asarray(tau, dtype=float)))
        inter -= (ball_ff_np(dk_norm, R) / vol) * phase
        abl = {}
        for lang in range(lmax + 1):
            ch = chan[key][lang]
            ab = np.array([match_ab(ch, ksafe[g], R) for g in range(npw)])
            abl[lang] = ab
            a, bb = ab[:, 0], ab[:, 1]
            aa, bbo = np.outer(a, a), np.outer(bb, bb)
            ab_s = np.outer(a, bb) + np.outer(bb, a)
            Ms = aa * ch["uu"] + ab_s * ch["uud"] + bbo * ch["udud"]
            Tk = aa * ch["Tuu"] + ab_s * ch["Tuud"] + bbo * ch["Tudud"]
            Vk = aa * ch["Vuu"] + ab_s * ch["Vuud"] + bbo * ch["Vudud"]
            pref = (4 * math.pi / vol) * (2 * lang + 1) * eval_legendre(lang, cost)
            Saug += phase * (pref * Ms)
            Haug += phase * (pref * (Tk + Vk))
        abl_by_atom.append(abl)
    S = 0.5 * (inter + Saug + (inter + Saug).conj().T)
    Hm = HBAR2_2M * kdot * inter + Haug
    Hm = 0.5 * (Hm + Hm.conj().T)
    ev, c = solve_geneig(Hm, S, nbands, with_vecs=True)
    return ev, c, mill, ks, abl_by_atom, vol


def _weinert_multi(rho_I, spheres, L, nfft):
    """Weinert Hartree for several muffin tins. ``spheres`` = list of
    ``{tau (cart), rr, dx, rho_sph, Z, R}``. ``L`` cubic side or length-3 orthorhombic edges.
    Returns ``([v_sph radial per sphere], v_i0)``."""
    L3 = np.broadcast_to(np.asarray(L, dtype=float), (3,))
    h3 = L3 / nfft
    X, Y, Zc = np.meshgrid(np.arange(nfft) * h3[0], np.arange(nfft) * h3[1],
                           np.arange(nfft) * h3[2], indexing="ij")

    def mi(x, x0, Le):
        return (x - x0) - Le * np.round((x - x0) / Le)

    rho_smooth = rho_I.astype(float).copy()
    inside_any = np.zeros((nfft, nfft, nfft), dtype=bool)
    dvol = float(np.prod(h3))
    for sp in spheres:
        c = np.asarray(sp["tau"])
        dgrid = np.sqrt(mi(X, c[0], L3[0]) ** 2 + mi(Y, c[1], L3[1]) ** 2
                        + mi(Zc, c[2], L3[2]) ** 2)
        inside = dgrid < sp["R"]
        inside_any |= inside
        drw = sp["rr"] * sp["dx"]
        q_sph = float(np.sum(4 * math.pi * sp["rho_sph"] * sp["rr"] ** 2 * drw))
        q_i_in = float(rho_I[inside].sum() * dvol)
        net = q_sph - sp["Z"] - q_i_in           # net charge: electrons − nucleus − interstitial-in
        rho_smooth += sphere_pseudocharge(net, sp["R"], c, nfft, L3, npow=4)
    v_grid = fft_poisson(rho_smooth, L3)
    v_i0 = float(v_grid[~inside_any].mean())
    v_sph_list = []
    for sp in spheres:
        c = np.asarray(sp["tau"])
        R = sp["R"]
        rr = sp["rr"]
        drw = rr * sp["dx"]
        pts = []
        for axis in range(3):
            for sgn in (+1, -1):
                p = c.copy()
                p[axis] += sgn * R
                idx = tuple(int(round(p[d] / h3[d])) % nfft for d in range(3))
                pts.append(v_grid[idx])
        v_bc = float(np.mean(pts))
        vpart = radial_poisson_to_R(sp["rho_sph"], rr, R, drw=drw) - sp["Z"] * E2 / rr
        v_sph_list.append(vpart + (v_bc - vpart[-1]))
    return v_sph_list, v_i0


def crystal_scf_multi(a_bohr, atoms, radii, ecut: float = 200.0, lmax: int = 2,
                      iters: int = 40, tol: float = 3e-3, kmesh=(1, 1, 1)):
    """Multi-sphere self-consistent muffin-tin FLAPW, cubic or orthorhombic cell (insulator).

    ``a_bohr`` is the cubic edge, or a length-3 vector of orthorhombic edge lengths (Bohr).
    ``atoms`` = ``[(frac (3,), symbol), ...]``; ``radii`` = ``{symbol: R_MT}``. Each atom gets its
    own sphere potential (so inequivalent same-species atoms are handled). Returns
    ``(bands, info)`` with ``bands['ev']`` the Γ valence eigenvalues (eV, referenced to the
    interstitial zero — compare splittings, not absolute levels; see ``crystal_scf``).

    Cubic/orthorhombic cell (Step 3; general triclinic lattices are still to do); insulators only,
    valence bands filled two-per-state (Step 4 adds Fermi smearing for metals)."""
    L = np.broadcast_to(np.asarray(a_bohr, dtype=float) * BOHR_ANG, (3,))
    nfft = 2 * int(math.ceil(math.sqrt(4 * ecut / HBAR2_2M) * L.max() / (2 * math.pi))) + 2
    nfft = min(max(nfft, 24), 72)
    r, dx = log_mesh(1e-5, 28.0, 2500)
    r_np = r.numpy()
    atoms_cart = [(np.asarray(f, dtype=float) * L, sym) for f, sym in atoms]
    keys = [f"a{i}" for i in range(len(atoms))]
    syms = [sym for _, sym in atoms]
    nbands = sum(_VAL_E[s] for s in syms) // 2
    kfracs, kw = monkhorst_pack(tuple(kmesh))
    kw = kw / kw.sum()

    at_by_sym, vat_by_sym = {}, {}
    for s in set(syms):
        at_by_sym[s], vat_by_sym[s] = atomic_scf(s, r, dx)
    R_by_key = {k: radii[s] for k, s in zip(keys, syms, strict=True)}
    rr_by_key = {k: r_np[r_np <= R_by_key[k]] for k in keys}
    mask_by_key = {k: r_np <= R_by_key[k] for k in keys}
    acart = [(tau, key) for (tau, _), key in zip(atoms_cart, keys, strict=True)]
    v_by_key = {k: vat_by_sym[s].clone() for k, s in zip(keys, syms, strict=True)}

    conv = None
    hist = {k: ([], []) for k in keys}
    for _ in range(iters):
        species, El_by_key, vmt_by_key = {}, {}, {}
        for k, s in zip(keys, syms, strict=True):
            R = R_by_key[k]
            v0 = float(v_by_key[k].numpy()[np.argmin(np.abs(r_np - R))])
            vmt = torch.where(r <= R, v_by_key[k] - v0, torch.zeros_like(r))
            El = {0: at_by_sym[s].get("2s", -5.0) - v0, 1: at_by_sym[s].get("2p", -5.0) - v0,
                  2: -5.0 - v0}
            species[k] = {"R": R, "v": vmt, "El": El}
            El_by_key[k], vmt_by_key[k] = El, vmt

        rho_I = np.zeros((nfft, nfft, nfft))
        rho_val = {k: np.zeros_like(rr_by_key[k]) for k in keys}
        ev_gamma = None
        for kf, w in zip(kfracs, kw, strict=True):
            ev, c, mill, ks, abl_all, vol = _lapw_multi_k(kf, L, acart, species, lmax, ecut,
                                                          r, dx, nbands)
            occ = [2.0] * nbands
            rho_I += w * _interstitial_density(c[:, :nbands], occ, mill, L, nfft)
            for ai, k in enumerate(keys):
                phase = np.exp(1j * (ks @ np.asarray(acart[ai][0])))
                cp = c[:, :nbands] * phase[:, None]
                _, rk = _sphere_valence_density(cp, occ, ks, abl_all[ai], El_by_key[k], lmax,
                                                vol, r, dx, vmt_by_key[k], R_by_key[k])
                rho_val[k] += w * rk
            if np.all(np.abs(kf) < 1e-9):
                ev_gamma = ev
        if ev_gamma is None:
            ev_gamma = _lapw_multi_k((0, 0, 0), L, acart, species, lmax, ecut, r, dx, nbands)[0]

        spheres, rho_sph_by_key = [], {}
        for ai, (k, s) in enumerate(zip(keys, syms, strict=True)):
            rr, mask = rr_by_key[k], mask_by_key[k]
            rho_core = np.zeros_like(rr)
            for lc, fc in _CORE[s]:
                _, uc = radial_eigs_tridiag(lc, r, dx, v_by_key[k], 1)
                rho_core += fc * uc[mask, 0] ** 2 / (4 * math.pi * rr**2)
            rho_sph_by_key[k] = rho_val[k] + rho_core
            spheres.append({"tau": acart[ai][0], "rr": rr, "dx": dx,
                            "rho_sph": rho_sph_by_key[k], "Z": CONFIG[s][0], "R": R_by_key[k]})
        v_sph_list, v_i0 = _weinert_multi(rho_I, spheres, L, nfft)

        for ai, k in enumerate(keys):
            mask = mask_by_key[k]
            vnew_sph = v_sph_list[ai] + vxc_lda(torch.tensor(rho_sph_by_key[k])).numpy()
            vnew_np = np.full(r.shape[0], v_i0)
            vnew_np[mask] = vnew_sph
            vnew = torch.tensor(vnew_np)
            hist[k][0].append(v_by_key[k])
            hist[k][1].append(vnew - v_by_key[k])
            if len(hist[k][1]) > 6:
                hist[k] = (hist[k][0][-6:], hist[k][1][-6:])
        for k in keys:
            v_by_key[k] = anderson_next(hist[k][0], hist[k][1], beta=0.3, m=5)

        span = float(ev_gamma[nbands - 1] - ev_gamma[0])
        new = {"span": span, "ev": ev_gamma.tolist()}
        if conv is not None and abs(new["span"] - conv["span"]) < tol:
            conv = new
            break
        conv = new
    return conv, {"nbands": nbands, "symbols": syms}
