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

Units at the public entry points (``crystal_scf`` / ``crystal_scf_multi``):

    cell        Å    (``cell=`` keyword; scalar cubic edge, length-3 edges, or a 3×3 matrix)
    a_bohr      Bohr (legacy positional cell; pass exactly one of ``a_bohr`` / ``cell``)
    radii, R    Å    (muffin-tin radii — NOT Bohr; mixing these up caused a real bug)
    ecut        eV
    energies    eV   (eigenvalues referenced to the interstitial zero — compare splittings)
    smearing    eV
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, E2, HBAR2_2M
from gradwave.flapw.atom import CONFIG, atomic_scf
from gradwave.flapw.coulomb import (
    _min_image_dist,
    _min_image_vec,
    cell_matrix,
    fft_poisson,
    radial_poisson_to_R,
    reciprocal,
    sphere_pseudocharge,
    sphere_pseudocharge_lm,
)
from gradwave.flapw.functionals import vxc_lda
from gradwave.flapw.lapw import (
    ball_ff_np,
    enumerate_kg,
    match_ab_vec,
    radial_channel,
    solve_geneig,
    solve_geneig_subspace,
)
from gradwave.flapw.mixing import anderson_next
from gradwave.flapw.radial import log_mesh, numerov_log_np, radial_eigs_tridiag
from gradwave.kpoints import monkhorst_pack

# frozen-core states as (l, n_radial_index, occupation): n_radial_index=1 is the lowest l-state
# (1s/2p/3d…), 2 the next (2s/3p…), so an element with several core states of the same l (Ti's
# [Ar] core: 1s2s3s, 2p3p) lists one entry each.
_CORE = {"He": [], "Be": [(0, 1, 2)], "O": [(0, 1, 2)], "Ne": [(0, 1, 2)],
         # Ti: freeze 1s2s3s2p; 3p is SEMICORE and stays in the valence (its EFG contribution is
         # comparable to the valence one — frozen 3p gives a badly wrong Ti EFG). Only the l=1
         # energy parameter linearizes at 3p (Ti 4p is empty), which captures the 3p response.
         "Ti": [(0, 1, 2), (0, 2, 2), (0, 3, 2), (1, 1, 6)]}
_N_VAL_BANDS = {"He": 1, "Be": 1, "O": 3, "Ne": 4, "Ti": 9}
_VAL_E = {"He": 2, "Be": 2, "O": 6, "Ne": 8, "Ti": 10}  # valence electron count (Z − frozen core)
# LAPW energy-parameter orbital per angular momentum: the crystal valence linearization point for
# each l (the atomic KS eigenvalue used to build u_l/u̇_l). Second-row atoms use 2s/2p; Ti uses
# 4s/3p/3d (l=1 at the 3p semicore, since 4p is empty). A missing l falls back to a default.
_VALENCE_NL = {"He": {0: "1s"}, "Be": {0: "2s"}, "O": {0: "2s", 1: "2p"},
               "Ne": {0: "2s", 1: "2p"}, "Ti": {0: "4s", 1: "3p", 2: "3d"}}


def _ylm_star(l, ks):
    """conj(Y_lm(k̂)) for m=-l..l over all plane waves ``ks`` (npw,3) → ``(npw, 2l+1)``. Vectorized:
    2l+1 array-valued scipy calls, not one per plane wave. G=0 maps to the l=0 constant."""
    from scipy.special import sph_harm_y
    ks = np.asarray(ks, dtype=float)
    kn = np.linalg.norm(ks, axis=1)
    small = kn < 1e-12
    kns = np.where(small, 1.0, kn)
    theta = np.arccos(np.clip(ks[:, 2] / kns, -1.0, 1.0))
    phi = np.arctan2(ks[:, 1], ks[:, 0])
    out = np.stack([np.conj(sph_harm_y(l, m, theta, phi)) for m in range(-l, l + 1)], axis=1)
    if small.any():
        out[small, :] = 0.0
        if l == 0:
            out[small, 0] = 1.0 / math.sqrt(4 * math.pi)
    return out


def _radial_u(l, El, r, dx, v, R):
    r_np = r.numpy()
    inside = r_np <= R
    drw = r_np[inside] * dx
    hE = max(abs(El) * 1e-4, 1e-3)
    n_cut = int(np.searchsorted(r_np, 2.0 * R)) + 5           # only integrate a little past R_MT
    uraw = numerov_log_np(l, np.array([El, El + hE, El - hE]), r, dx, v, n_cut=n_cut)
    un = uraw / np.sqrt((uraw[:, inside] ** 2 * drw).sum(axis=1))[:, None]
    u, ud = un[0], (un[1] - un[2]) / (2 * hE)
    return u[inside], ud[inside]


def _pair_integrals(l, f, g, rr, drw, dx, v_in):
    """``(S, T, V)`` between two in-sphere radial u-functions on the log mesh: overlap ``∫fg dr``,
    the weak-form kinetic (same discretization as ``radial_channel``), and ``∫f v g dr``."""
    ll = l * (l + 1)

    def drf(u):
        return (np.gradient(u, dx) - u) / rr**2               # R' where R = u/r

    t = HBAR2_2M * ((drf(f) * drf(g) * rr**2 + ll * (f / rr) * (g / rr)) * drw).sum()
    return float((f * g * drw).sum()), float(t), float((f * v_in * g * drw).sum())


def _build_lo(l, e2, ch, u, ud, r, dx, v, R):
    """One confined local orbital ``φ = a·u(E_l) + b·u̇(E_l) + c·u(E₂)`` with ``φ(R)=φ'(R)=0``,
    normalized to ``∫φ²dr=1``. ``ch`` is the l-channel ``radial_channel`` dict (supplies the u/u̇
    boundary values); ``u, ud`` the matching ``_radial_u`` arrays. Returns the radial data + the
    S/H integrals against (u, u̇) and itself that the LAPW+LO matrix blocks need."""
    r_np = r.numpy()
    inside = r_np <= R
    rr = r_np[inside]
    drw = rr * dx
    n_cut = int(np.searchsorted(r_np, 2.0 * R)) + 5
    u2raw = numerov_log_np(l, np.array([e2]), r, dx, v, n_cut=n_cut)[0]
    u2n = u2raw / math.sqrt(float((u2raw[inside] ** 2 * drw).sum()))
    idx = np.sort(np.argsort(np.abs(r_np - R))[:7])           # value+slope at R (cubic fit,
    c3 = np.polyfit(r_np[idx] - R, u2n[idx], 3)               #  same scheme as radial_channel)
    u2R, u2pR = float(c3[-1]), float(c3[-2])
    mat = np.array([[ch["uR"], ch["udR"]], [ch["upR"], ch["udpR"]]])
    a, b = np.linalg.solve(mat, -np.array([u2R, u2pR]))       # φ(R)=0, φ'(R)=0 with c=1
    u2 = u2n[inside]
    phi = a * u + b * ud + u2
    nrm = math.sqrt(float((phi**2 * drw).sum()))
    a, b, cn = float(a / nrm), float(b / nrm), float(1.0 / nrm)
    phi = phi / nrm
    v_in = v.numpy()[inside]
    s_pu, t_pu, v_pu = _pair_integrals(l, phi, u, rr, drw, dx, v_in)
    s_pud, t_pud, v_pud = _pair_integrals(l, phi, ud, rr, drw, dx, v_in)
    _, t_pp, v_pp = _pair_integrals(l, phi, phi, rr, drw, dx, v_in)
    return {"l": l, "a": a, "b": b, "cn": cn, "u2": u2, "phi": phi,
            "S_pu": s_pu, "S_pud": s_pud, "H_pu": t_pu + v_pu, "H_pud": t_pud + v_pud,
            "S_pp": 1.0, "H_pp": t_pp + v_pp}


def _build_lodat(los_by_key, chan, El_by_key, vmt_by_key, R_by_key, at_by_sym, key_sym, v0_by_key,
                 r, dx):
    """Per-iteration local-orbital data: for each atom key, build every requested LO against the
    current sphere potential. ``los_by_key[key]`` = list of ``(l, spec)`` with ``spec`` an atomic
    orbital label ("3p") or an absolute atomic energy (eV); either is shifted by the current
    muffin-tin zero like the energy parameters. One LO per (key, l)."""
    lodat = {}
    for key, los in los_by_key.items():
        seen = set()
        out = []
        for l, spec in los:
            if l in seen:
                raise ValueError(f"one local orbital per l per atom (duplicate l={l} on {key})")
            seen.add(l)
            e2 = (at_by_sym[key_sym[key]][spec] if isinstance(spec, str) else float(spec))
            e2 = e2 - v0_by_key[key]
            u, ud = _radial_u(l, El_by_key[key][l], r, dx, vmt_by_key[key], R_by_key[key])
            out.append(_build_lo(l, e2, chan[key][l], u, ud, r, dx, vmt_by_key[key], R_by_key[key]))
        lodat[key] = out
    return lodat


def _lapw_k(kfrac, L, R, lmax, El, ecut, r, dx, v_sphere):
    """Single-atom LAPW at wavevector k (fractional ``kfrac``). One atom at the origin carries no
    structure phase, so H,S are real symmetric at any k. Returns eigvals, eigvecs, miller/ks,
    matching coeffs. ``ks`` are the Cartesian k+G; the ecut sphere is centred at -k."""
    from scipy.special import eval_legendre
    vol = L**3
    b = 2 * math.pi / L
    mill, ks = enumerate_kg(kfrac, b * np.eye(3), ecut)
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
        ab = np.stack(match_ab_vec(ch, ksafe, R), axis=1)
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


def _interstitial_density(c_occ, occ, mill, vol, nfft):
    rho = np.zeros((nfft, nfft, nfft))
    idx = (mill[:, 0] % nfft, mill[:, 1] % nfft, mill[:, 2] % nfft)
    for n, f in enumerate(occ):
        box = np.zeros((nfft, nfft, nfft), dtype=complex)
        box[idx] = c_occ[:, n]
        psi = np.fft.ifftn(box) * nfft**3
        rho += f * np.abs(psi) ** 2 / vol
    return rho


def _lo_row_slices(los):
    """Row layout of one atom's LO block: ``{l: (start, stop, lo)}`` in the deterministic order the
    matrix build used (defs in list order, m=-l..l within each)."""
    out, off = {}, 0
    for lo in los or []:
        out[lo["l"]] = (off, off + 2 * lo["l"] + 1, lo)
        off += 2 * lo["l"] + 1
    return out


def _bands_amps(c_pw, ks, abl, lmax, vol, lo_block=None, los=None, ylm_by_l=None):
    """All-band sphere amplitudes ``{l: [(nb, 2l+1) per radial]}`` — one GEMM per (l, radial)
    over every band at once instead of a Python loop of per-band reductions. ``c_pw`` = phased PW
    coefficients ``(npw, nb)``; ``lo_block`` = this atom's LO rows ``(nlo_atom, nb)``."""
    slices = _lo_row_slices(los)
    out = {}
    for lang in range(lmax + 1):
        ylm = ylm_by_l[lang] if ylm_by_l is not None else _ylm_star(lang, ks)
        a, bb = abl[lang][:, 0], abl[lang][:, 1]
        pfac = (4 * math.pi / math.sqrt(vol)) * (1j ** lang)
        amp_a = pfac * ((c_pw * a[:, None]).T @ ylm)          # (nb, npw) @ (npw, 2l+1)
        amp_b = pfac * ((c_pw * bb[:, None]).T @ ylm)
        amps = [amp_a, amp_b]
        if lang in slices:
            lo0, lo1, lo = slices[lang]
            cv = lo_block[lo0:lo1, :].T                       # (nb, 2l+1)
            amps[0] = amps[0] + cv * lo["a"]
            amps[1] = amps[1] + cv * lo["b"]
            amps.append(cv * lo["cn"])
        out[lang] = amps
    return out


def _us_ext(El, lmax, r, dx, v_sphere, R, los=None):
    """The per-l radial-function lists ``[u, u̇ (, u₂)]`` matching ``_bands_amps``' amplitudes."""
    slices = _lo_row_slices(los)
    us = {}
    for lang in range(lmax + 1):
        u, ud = _radial_u(lang, El[lang], r, dx, v_sphere, R)
        us[lang] = [u, ud] + ([slices[lang][2]["u2"]] if lang in slices else [])
    return us


def _sphere_valence_density(c_occ, occ, ks, abl, El, lmax, vol, r, dx, v_sphere, R,
                            lo_block=None, los=None, ylm_by_l=None):
    """The spherical (l=0-projected) valence density inside one sphere, LO-aware: each l-channel
    carries ``[u, u̇]`` plus an LO's second radial when present, and the density sums every radial
    pair ``ρ += (1/4π) Σ_ij P_ij f_i f_j / r²`` with ``P_ij = Σ_{n,m} occ·Re(A*_i A_j)``."""
    rr = r.numpy()[r.numpy() <= R]
    us = _us_ext(El, lmax, r, dx, v_sphere, R, los)
    rho = np.zeros_like(rr)
    if ylm_by_l is None:
        ylm_by_l = {lang: _ylm_star(lang, ks) for lang in range(lmax + 1)}
    amps_all = _bands_amps(c_occ, ks, abl, lmax, vol, lo_block=lo_block, los=los,
                           ylm_by_l=ylm_by_l)
    occ_arr = np.asarray(occ, dtype=float)
    for lang in range(lmax + 1):
        rads = us[lang]
        al = amps_all[lang]
        nrf = len(rads)
        for i in range(nrf):
            for j in range(nrf):
                pij = float(np.sum(occ_arr[:, None] * (np.conj(al[i]) * al[j]).real))
                rho += (1.0 / (4 * math.pi)) * pij * rads[i] * rads[j] / rr**2
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


def crystal_scf(a_bohr: float | None = None, symbol: str = "Ne", R: float = 1.4,
                ecut: float = 250.0, lmax: int = 2, iters: int = 40, tol: float = 3e-3,
                kmesh=(1, 1, 1), cell: float | None = None):
    """Muffin-tin self-consistent FLAPW for a simple-cubic crystal. Returns
    ``(valence_eigs_eV, atomic_eigs_eV)`` — the crystal Γ valence eigenvalues and the atomic limit.

    ``kmesh`` is a Monkhorst-Pack division for BZ integration of the density (default Γ-only). The
    reported eigenvalues are always at Γ; the density is summed over the mesh with symmetry weights.

    The returned eigenvalues are referenced to the flat interstitial potential level, which the
    muffin-tin scheme only weakly determines (worst in dilute cells): the *absolute* levels wander
    run-to-run under threaded BLAS while energy *differences* (2p-2s splittings, bandwidths) are
    stable and physical. Compare splittings, not absolute eigenvalues, as against any FLAPW code."""
    if (a_bohr is None) == (cell is None):
        raise ValueError("pass exactly one of a_bohr (Bohr, legacy) or cell (Å cubic edge)")
    L = float(cell) if cell is not None else a_bohr * BOHR_ANG
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
        nl = _VALENCE_NL[symbol]
        El = {lang: at.get(nl.get(lang, ""), -5.0) - v0 for lang in range(max(lmax, 2) + 1)}
        occb = [2.0] * n_val
        # BZ-integrate the valence density over the k-mesh; keep Γ eigenvalues for reporting.
        rho_I = np.zeros((nfft, nfft, nfft))
        rho_val = np.zeros_like(rr)
        ea_gamma = None
        for kf, w in zip(kfracs, kw, strict=True):
            ea, c, mill, ks, _, abl, vol = _lapw_k(kf, L, R, lmax, El, ecut, r, dx, v_mt)
            c_occ = c[:, :n_val]
            rho_I += w * _interstitial_density(c_occ, occb, mill, L**3, nfft)
            _, rk = _sphere_valence_density(c_occ, occb, ks, abl, El, lmax, vol, r, dx, v_mt, R)
            rho_val += w * rk
            if np.all(np.abs(kf) < 1e-9):
                ea_gamma = ea
        if ea_gamma is None:                       # mesh excludes Γ; one extra solve for reporting
            ea_gamma = _lapw_k((0.0, 0.0, 0.0), L, R, lmax, El, ecut, r, dx, v_mt)[0]
        ea = ea_gamma
        rho_core = np.zeros_like(rr)
        for lc, nidx, fc in core:
            _, uc = radial_eigs_tridiag(lc, r, dx, v, nidx)
            rho_core += fc * uc[rr_mask, nidx - 1] ** 2 / (4 * math.pi * rr**2)
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


def _lapw_multi_k(kf, L, atoms_cart, species, lmax, ecut, r, dx, nbands, v_nsph=None, chan=None,
                  lodat=None, nsph_int=None, c_prev=None, subspace_tol=1e-5, warp=None):
    """Multi-atom LAPW at wavevector k, exposing the density internals the SCF needs. Returns
    ``(eigvals, eigvecs, miller, ks, [abl per atom], vol)``. H,S are complex Hermitian (structure
    phases ``e^{i(k_G'-k_G)·τ_a}``). ``atoms_cart`` = ``[(τ_cart, species_key), ...]``.

    ``v_nsph`` (optional) = ``{species_key: {(L,M): V_LM(r) in-sphere}}`` — the non-spherical
    (full-potential) sphere potential components (harmonic coefficients, ``V(r,Ω)=Σ V_LM Y_LM``).
    Their L>0 parts add the l-channel coupling ``⟨u_l Y_lm|V_LM Y_LM|u_l' Y_l'm'⟩`` to H (the L=0
    spherical part is already in the muffin tin). ``None`` → muffin-tin (backward compatible).

    ``chan`` (optional) = precomputed ``{species_key: {l: radial_channel}}``. The radial channels
    are k-independent, so a caller sweeping a k-mesh builds them once per SCF iteration and passes
    them in, instead of paying the Numerov + radial-integral cost again at every k. ``None`` →
    build locally (backward compatible)."""
    from scipy.special import eval_legendre
    A = cell_matrix(L)
    B = reciprocal(A)                                     # rows = reciprocal vectors
    vol = float(abs(np.linalg.det(A)))
    mill, ks = enumerate_kg(kf, B, ecut)
    npw = len(ks)
    ksafe = np.maximum(np.linalg.norm(ks, axis=1), 1e-12)
    kdot = ks @ ks.T
    kk = np.einsum("ij,ij->i", ks, ks)
    # |k_G - k_G'| from kdot (no (npw,npw,3) displacement tensor): |dk|² = k² + k'² - 2 k·k'
    dk_norm = np.sqrt(np.maximum(kk[:, None] + kk[None, :] - 2.0 * kdot, 0.0))
    cost = np.clip(kdot / np.outer(ksafe, ksafe), -1.0, 1.0)
    if chan is None:
        chan = {key: {lang: radial_channel(lang, sp["El"][lang], r, dx, sp["v"], sp["R"])
                      for lang in range(lmax + 1)} for key, sp in species.items()}
    inter = np.eye(npw, dtype=complex)
    Saug = np.zeros((npw, npw), dtype=complex)
    Haug = np.zeros((npw, npw), dtype=complex)
    # atom-independent geometry: Legendre prefactors per l, ball form factor per unique R
    pref_l = {lang: (4 * math.pi / vol) * (2 * lang + 1) * eval_legendre(lang, cost)
              for lang in range(lmax + 1)}
    ballf = {}
    abl_by_atom = []
    for tau, key in atoms_cart:
        R = species[key]["R"]
        # e^{i(k_G'-k_G)·τ} factorizes exactly: outer(conj(p), p) with p = e^{i k_G·τ} — npw
        # exponentials instead of npw² (the npw² exp was a top build cost).
        pvec = np.exp(1j * (ks @ np.asarray(tau, dtype=float)))
        phase = np.conj(pvec)[:, None] * pvec[None, :]
        if R not in ballf:
            ballf[R] = ball_ff_np(dk_norm, R) / vol
        inter -= ballf[R] * phase
        abl = {}
        for lang in range(lmax + 1):
            ch = chan[key][lang]
            ab = np.stack(match_ab_vec(ch, ksafe, R), axis=1)
            abl[lang] = ab
            a, bb = ab[:, 0], ab[:, 1]
            aa, bbo = np.outer(a, a), np.outer(bb, bb)
            ab_s = np.outer(a, bb) + np.outer(bb, a)
            Ms = aa * ch["uu"] + ab_s * ch["uud"] + bbo * ch["udud"]
            Tk = aa * ch["Tuu"] + ab_s * ch["Tuud"] + bbo * ch["Tudud"]
            Vk = aa * ch["Vuu"] + ab_s * ch["Vuud"] + bbo * ch["Vudud"]
            Saug += phase * (pref_l[lang] * Ms)
            Haug += phase * (pref_l[lang] * (Tk + Vk))
        abl_by_atom.append(abl)
    if v_nsph:
        if nsph_int is None:
            nsph_int = _nsph_radial_integrals(v_nsph, species, lmax, r, dx, lodat=lodat)
        ylm_nsph = {lang: _ylm_star(lang, ks) for lang in range(lmax + 1)}
        for ai, (tau, key) in enumerate(atoms_cart):
            ints = nsph_int.get(key)
            if not ints or not ints["uu"]:
                continue
            ph_a = np.exp(1j * (ks @ np.asarray(tau, dtype=float)))
            cstk = _channel_stack(abl_by_atom[ai], ks, lmax, vol, ph_a, ylm_by_l=ylm_nsph)
            # ONE triple product per atom replaces the per-(L,M,l,l') chained GEMMs (109x
            # measured, equal to 2.5e-14; _nonspherical_augment kept as the reference impl)
            Haug = Haug + cstk.conj() @ (ints["m_uu"] @ cstk.T)
    S = 0.5 * (inter + Saug + (inter + Saug).conj().T)
    Hm = HBAR2_2M * kdot * inter + Haug
    if warp is not None:
        # warped interstitial: <G|(V_I - v_i0)·Θ_I|G'> = FT[U](G-G') — the standard full-potential
        # LAPW term this code lacked (the interstitial was Hamiltonian-flat while the spheres got
        # l>0 refinement; TiO2's covalent bonding lives exactly in the interstitial). U is the
        # FFT of (v_grid - v_i0)·Θ_I; Miller-difference indexed (Toeplitz in G).
        u_fft, nfft_w = warp
        dm0 = (mill[:, None, 0] - mill[None, :, 0]) % nfft_w
        dm1 = (mill[:, None, 1] - mill[None, :, 1]) % nfft_w
        dm2 = (mill[:, None, 2] - mill[None, :, 2]) % nfft_w
        Hm = Hm + u_fft[dm0, dm1, dm2]
    Hm = 0.5 * (Hm + Hm.conj().T)
    if lodat:
        # LAPW+LO: extend the secular problem with the confined local orbitals. Row order (the
        # density pass replicates it): atoms outer, this atom's LO defs inner, m innermost. The
        # LO couples only to its own sphere's augmentation — S/H[LO,G] = amp_lm(G)·(a_G·⟨φ|u⟩ +
        # b_G·⟨φ|u̇⟩) with amp_lm(G) = (4π/√Ω) iˡ Y*_lm(k̂_G) e^{i k_G·τ}; LO-LO is diagonal
        # (φ(R)=φ'(R)=0 confines it, one LO per l, spherical V keeps m diagonal). The aspherical
        # v_nsph coupling of LO rows is neglected (deep semicore, second-order small).
        rows_s, rows_h, diag_s, diag_h = [], [], [], []
        for ai, (tau, key) in enumerate(atoms_cart):
            ph = np.exp(1j * (ks @ np.asarray(tau, dtype=float)))
            for lo in lodat.get(key, []):
                lang = lo["l"]
                ylm = _ylm_star(lang, ks)
                pf = (4 * math.pi / math.sqrt(vol)) * (1j ** lang)
                a_g, b_g = abl_by_atom[ai][lang][:, 0], abl_by_atom[ai][lang][:, 1]
                base_s = pf * (a_g * lo["S_pu"] + b_g * lo["S_pud"]) * ph
                base_h = pf * (a_g * lo["H_pu"] + b_g * lo["H_pud"]) * ph
                for mi in range(2 * lang + 1):
                    rows_s.append(base_s * ylm[:, mi])
                    rows_h.append(base_h * ylm[:, mi])
                    diag_s.append(lo["S_pp"])
                    diag_h.append(lo["H_pp"])
        nlo = len(rows_s)
        if nlo:
            rows_s_arr, rows_h_arr = np.array(rows_s), np.array(rows_h)
            lo_lo_h = np.diag(diag_h).astype(complex)
            if v_nsph:
                # the LOs' aspherical coupling — the semicore's response to the crystal field
                dh_pw, dh_lo = _nonspherical_augment_lo(v_nsph, atoms_cart, abl_by_atom, ks,
                                                        species, lmax, vol, r, dx, lodat,
                                                        nsph_int=nsph_int)
                rows_h_arr = rows_h_arr + dh_pw
                lo_lo_h = lo_lo_h + dh_lo
            s_ext = np.zeros((npw + nlo, npw + nlo), dtype=complex)
            h_ext = np.zeros((npw + nlo, npw + nlo), dtype=complex)
            s_ext[:npw, :npw], h_ext[:npw, :npw] = S, Hm
            s_ext[npw:, :npw], h_ext[npw:, :npw] = rows_s_arr, rows_h_arr
            s_ext[:npw, npw:] = rows_s_arr.conj().T
            h_ext[:npw, npw:] = rows_h_arr.conj().T
            s_ext[npw:, npw:] = np.diag(diag_s)
            h_ext[npw:, npw:] = lo_lo_h
            S, Hm = s_ext, h_ext
    if c_prev is not None and c_prev.shape[0] == Hm.shape[0]:
        # Rayleigh-Ritz in last iteration's subspace; the residual gate falls back to the exact
        # dense solve whenever the subspace has drifted (band crossings, early SCF, mixing jumps).
        ev, c, resid = solve_geneig_subspace(Hm, S, c_prev, nbands)
        if resid < subspace_tol:
            return ev, c, mill, ks, abl_by_atom, vol
    ev, c = solve_geneig(Hm, S, nbands, with_vecs=True)
    return ev, c, mill, ks, abl_by_atom, vol


def _nsph_radial_integrals(v_nsph, species, lmax, r, dx, lodat=None):
    """The k-independent radial integrals of the aspherical potential — ``∫u_l V_LM u_l' dr`` (and
    the u̇/LO variants) per atom key. These were being recomputed at every k-point; a k-mesh sweep
    builds them once per SCF iteration and passes them into the secular build (same pattern as the
    ``chan`` hoist). Returns ``{key: {"uu": {(L,M,l,l'): (i_aa,i_ab,i_ba,i_bb)}, "lo_u": {...},
    "lo_lo": {...}}}``."""
    r_np = r.numpy()
    out = {}
    for key, comps in v_nsph.items():
        if not comps:
            continue
        rr = r_np[r_np <= species[key]["R"]]
        drw = rr * dx
        el, vsph = species[key]["El"], species[key]["v"]
        us = {lang: _radial_u(lang, el[lang], r, dx, vsph, species[key]["R"])
              for lang in range(lmax + 1)}
        los = (lodat or {}).get(key, [])
        uu, lo_u, lo_lo = {}, {}, {}
        for (big_l, big_m), vlm in comps.items():
            if big_l == 0:
                continue                                   # spherical part is in the muffin tin
            vlm = np.asarray(vlm)
            for lang in range(lmax + 1):
                for lp in range(lmax + 1):
                    if big_l < abs(lang - lp) or big_l > lang + lp or (lang + lp + big_l) % 2:
                        continue
                    ul, udl = us[lang]
                    ulp, udlp = us[lp]
                    uu[(big_l, big_m, lang, lp)] = (
                        np.sum(ul * vlm * ulp * drw), np.sum(ul * vlm * udlp * drw),
                        np.sum(udl * vlm * ulp * drw), np.sum(udl * vlm * udlp * drw))
            for i, lo in enumerate(los):
                li = lo["l"]
                for lp in range(lmax + 1):
                    if big_l < abs(li - lp) or big_l > li + lp or (li + lp + big_l) % 2:
                        continue
                    ulp, udlp = us[lp]
                    lo_u[(big_l, big_m, i, lp)] = (np.sum(lo["phi"] * vlm * ulp * drw),
                                                   np.sum(lo["phi"] * vlm * udlp * drw))
                for j, lo2 in enumerate(los):
                    lj = lo2["l"]
                    if big_l < abs(li - lj) or big_l > li + lj or (li + lj + big_l) % 2:
                        continue
                    lo_lo[(big_l, big_m, i, j)] = np.sum(lo["phi"] * vlm * lo2["phi"] * drw)
        out[key] = {"uu": uu, "lo_u": lo_u, "lo_lo": lo_lo,
                    "m_uu": _nsph_coupling_matrix(uu, lmax)}
    return out


def _nsph_coupling_matrix(ints_uu, lmax):
    """The k-independent aspherical coupling matrix ``M`` over the stacked augmentation
    channels ``[(l, m, u|udot)]`` (size ``2(lmax+1)^2``):
    ``M[(l,m,X),(l',m',Y)] = Sum_LM  i_XY(L,M,l,l') * G^{LM}_{lm,l'm'}``. With the per-atom
    channel amplitudes ``C`` (npw x n_ch), the whole non-spherical augmentation collapses to ONE
    triple product ``dH = conj(C) @ M @ C.T`` per atom per k — replacing ~hundreds of chained
    small GEMMs (one per (L,M,l,l',term)), which were the dominant fullpot glue cost. Built once
    per iteration from the hoisted radial integrals."""
    from gradwave.flapw.efg import gaunt_matrix
    nlm = (lmax + 1) ** 2
    n_ch = 2 * nlm
    off = {}
    o = 0
    for lang in range(lmax + 1):
        off[lang] = o
        o += 2 * lang + 1
    m = np.zeros((n_ch, n_ch), dtype=complex)
    for (big_l, big_m, lang, lp), (i_aa, i_ab, i_ba, i_bb) in ints_uu.items():
        g = gaunt_matrix(lang, big_l, big_m, lp)
        r0, c0 = off[lang], off[lp]
        m[r0:r0 + 2 * lang + 1, c0:c0 + 2 * lp + 1] += i_aa * g
        m[r0:r0 + 2 * lang + 1, nlm + c0:nlm + c0 + 2 * lp + 1] += i_ab * g
        m[nlm + r0:nlm + r0 + 2 * lang + 1, c0:c0 + 2 * lp + 1] += i_ba * g
        m[nlm + r0:nlm + r0 + 2 * lang + 1, nlm + c0:nlm + c0 + 2 * lp + 1] += i_bb * g
    return m


def _channel_stack(abl_atom, ks, lmax, vol, ph, ylm_by_l=None):
    """The stacked augmentation amplitudes ``C`` (npw x 2(lmax+1)^2): columns are the
    ``B_lm(g)`` coefficients of ``u_l Y_lm`` (first (lmax+1)^2 columns) then ``udot_l Y_lm``,
    matching ``_nsph_coupling_matrix``'s channel layout."""
    nlm = (lmax + 1) ** 2
    npw = len(ks)
    c = np.empty((npw, 2 * nlm), dtype=complex)
    o = 0
    for lang in range(lmax + 1):
        ylm = ylm_by_l[lang] if ylm_by_l is not None else _ylm_star(lang, ks)
        pf = (4 * math.pi / math.sqrt(vol)) * (1j ** lang)
        c[:, o:o + 2 * lang + 1] = pf * ylm * (abl_atom[lang][:, 0] * ph)[:, None]
        c[:, nlm + o:nlm + o + 2 * lang + 1] = pf * ylm * (abl_atom[lang][:, 1] * ph)[:, None]
        o += 2 * lang + 1
    return c


def _nonspherical_augment(v_nsph, atoms_cart, abl_by_atom, ks, species, lmax, vol, r, dx,
                          nsph_int=None):
    """The full-potential non-spherical augmentation ``ΔH[g,g'] = Σ_{lm,l'm',LM>0}
    B_lm(g)* [∫u_l V_LM u_l' dr] G^{LM}_{lm,l'm'} B_l'm'(g')`` (and the u̇ cross terms), where
    ``B_lm(g) = (4π/√Ω) i^l Y*_lm(k+G) [a_l or b_l](g) e^{i(k+G)·τ}`` is the per-plane-wave
    augmentation amplitude. Per-(L,M,l,l') radial integrals × Gaunt blocks contract to npw×npw.
    ``nsph_int`` = the precomputed ``_nsph_radial_integrals`` (built locally when ``None``)."""
    from gradwave.flapw.efg import gaunt_matrix
    if nsph_int is None:
        nsph_int = _nsph_radial_integrals(v_nsph, species, lmax, r, dx)
    npw = len(ks)
    ha = np.zeros((npw, npw), dtype=complex)
    ylm_by_l = {lang: _ylm_star(lang, ks) for lang in range(lmax + 1)}
    for ai, (tau, key) in enumerate(atoms_cart):
        ints = nsph_int.get(key)
        if not ints:
            continue
        ph = np.exp(1j * (ks @ np.asarray(tau, dtype=float)))
        b_a, b_b = {}, {}
        for lang in range(lmax + 1):
            pf = (4 * math.pi / math.sqrt(vol)) * (1j ** lang)
            b_a[lang] = pf * ylm_by_l[lang] * (abl_by_atom[ai][lang][:, 0] * ph)[:, None]
            b_b[lang] = pf * ylm_by_l[lang] * (abl_by_atom[ai][lang][:, 1] * ph)[:, None]
        for (big_l, big_m, lang, lp), (i_aa, i_ab, i_ba, i_bb) in ints["uu"].items():
            g = gaunt_matrix(lang, big_l, big_m, lp)
            ca, cb = b_a[lang].conj(), b_b[lang].conj()
            ha += (i_aa * (ca @ g @ b_a[lp].T) + i_ab * (ca @ g @ b_b[lp].T)
                   + i_ba * (cb @ g @ b_a[lp].T) + i_bb * (cb @ g @ b_b[lp].T))
    return ha


def _nonspherical_augment_lo(v_nsph, atoms_cart, abl_by_atom, ks, species, lmax, vol, r, dx,
                             lodat, nsph_int=None):
    """The local orbitals' coupling to the non-spherical potential — ``ΔH[LO_lm, G] = Σ_{LM,l'm'}
    [∫φ V_LM u_l' dr] G^{LM}_{lm,l'm'} B_l'm'(G)`` (+ u̇) and the same-atom LO-LO block
    ``[∫φ_i V_LM φ_j dr] G``. This is the channel by which a semicore LO polarizes in the crystal
    field (the Sternheimer antishielding of a d⁰ cation) — without it, an LO feels only the
    spherical potential and cannot respond aspherically. Returns ``(ΔH_lo_pw (nlo,npw),
    ΔH_lo_lo (nlo,nlo))`` in the matrix build's LO row order. ``nsph_int`` = the precomputed
    ``_nsph_radial_integrals`` (built locally when ``None``)."""
    from gradwave.flapw.efg import gaunt_matrix
    if nsph_int is None:
        nsph_int = _nsph_radial_integrals(v_nsph, species, lmax, r, dx, lodat=lodat)
    npw = len(ks)
    nlo = sum(2 * lo["l"] + 1 for (_t, key) in atoms_cart for lo in (lodat or {}).get(key, []))
    dh_pw = np.zeros((nlo, npw), dtype=complex)
    dh_lo = np.zeros((nlo, nlo), dtype=complex)
    ylm_by_l = {lang: _ylm_star(lang, ks) for lang in range(lmax + 1)}
    row0 = 0
    for ai, (tau, key) in enumerate(atoms_cart):
        los = (lodat or {}).get(key, [])
        n_at = sum(2 * lo["l"] + 1 for lo in los)
        ints = nsph_int.get(key)
        if not ints or not n_at:
            row0 += n_at
            continue
        ph = np.exp(1j * (ks @ np.asarray(tau, dtype=float)))
        b_a, b_b = {}, {}
        for lang in range(lmax + 1):
            pf = (4 * math.pi / math.sqrt(vol)) * (1j ** lang)
            b_a[lang] = pf * ylm_by_l[lang] * (abl_by_atom[ai][lang][:, 0] * ph)[:, None]
            b_b[lang] = pf * ylm_by_l[lang] * (abl_by_atom[ai][lang][:, 1] * ph)[:, None]
        offs = _lo_row_slices(los)
        for (big_l, big_m, i, lp), (i_pu, i_pud) in ints["lo_u"].items():
            li = los[i]["l"]
            sl0, _sl1, _ = offs[li]
            g = gaunt_matrix(li, big_l, big_m, lp)
            dh_pw[row0 + sl0:row0 + sl0 + 2 * li + 1] += (
                i_pu * (g @ b_a[lp].T) + i_pud * (g @ b_b[lp].T))
        for (big_l, big_m, i, j), i_pp in ints["lo_lo"].items():
            li, lj = los[i]["l"], los[j]["l"]
            sl0, _s1, _ = offs[li]
            sl0j, _s2, _ = offs[lj]
            g = gaunt_matrix(li, big_l, big_m, lj)
            dh_lo[row0 + sl0:row0 + sl0 + 2 * li + 1,
                  row0 + sl0j:row0 + sl0j + 2 * lj + 1] += i_pp * g
        row0 += n_at
    return dh_pw, 0.5 * (dh_lo + dh_lo.conj().T)


def _weinert_multi(rho_I, spheres, L, nfft):
    """Weinert Hartree for several muffin tins. ``spheres`` = list of
    ``{tau (cart), rr, dx, rho_sph, Z, R}``. ``L`` is any cell (cubic side, orthorhombic edges, or a
    3×3 triclinic matrix). Returns ``([v_sph radial per sphere], v_i0)``."""
    A = cell_matrix(L)
    ainv = np.linalg.inv(A)
    dvol = float(abs(np.linalg.det(A))) / nfft**3
    rho_smooth = rho_I.astype(float).copy()
    own_ps_by_sphere = []                       # per sphere: (own L>0 pseudocharge grid, lset)
    inside_any = np.zeros((nfft, nfft, nfft), dtype=bool)
    for sp in spheres:
        cfrac = np.asarray(sp["tau"]) @ ainv
        inside = _min_image_dist(cfrac, nfft, A) < sp["R"]
        inside_any |= inside
        drw = sp["rr"] * sp["dx"]
        q_sph = float(np.sum(4 * math.pi * sp["rho_sph"] * sp["rr"] ** 2 * drw))
        q_i_in = float(rho_I[inside].sum() * dvol)
        net = q_sph - sp["Z"] - q_i_in           # net charge: electrons − nucleus − interstitial-in
        rho_smooth += sphere_pseudocharge(net, sp["R"], sp["tau"], nfft, A, npow=4)
        if sp.get("rho_2m") is not None:
            # Weinert L>0: match every sphere multipole moment present in rho_2m so the interstitial
            # FFT potential carries each sphere's full exterior multipole field (the inter-atomic
            # lattice terms of the non-spherical potential and the EFG).
            from scipy.special import sph_harm_y
            big_ls = sorted({lm[0] for lm in sp["rho_2m"] if lm[0] > 0})
            disp = _min_image_vec(cfrac, nfft, A)
            dv = np.linalg.norm(disp, axis=-1)
            dsafe = np.where(dv < 1e-12, 1.0, dv)
            th = np.arccos(np.clip(disp[..., 2] / dsafe, -1.0, 1.0))
            ph = np.arctan2(disp[..., 1], disp[..., 0])
            own_ps = np.zeros_like(rho_smooth)
            for bl in big_ls:
                rlw = sp["rr"] ** (bl + 2) * drw
                ql = {mm: complex(np.sum(sp["rho_2m"][(bl, mm)] * rlw))
                      for mm in range(-bl, bl + 1)}
                for mm in range(-bl, bl + 1):    # subtract the interstitial density's part inside R
                    yc = np.conj(sph_harm_y(bl, mm, th[inside], ph[inside]))
                    ql[mm] -= complex(np.sum(rho_I[inside] * dv[inside] ** bl * yc) * dvol)
                own_ps += sphere_pseudocharge_lm(ql, sp["R"], sp["tau"], nfft, A, bl,
                                                 geom=(dv, th, ph))
            rho_smooth += own_ps
            own_ps_by_sphere.append((own_ps, sorted({lm for lm in sp["rho_2m"] if lm[0] > 0})))
    v_hart = fft_poisson(rho_smooth, A)
    # Interstitial XC: the spheres carry vxc(rho_sph) but the interstitial previously had NONE —
    # a potential discontinuity at every muffin-tin boundary and an O(eV) hole in the bonding
    # region. The TOTAL grid (Hartree+XC) feeds v_i0, the lattice boundary projections and the
    # warped-interstitial term; the SPHERE MATCHING below stays Hartree-only because the caller
    # adds vxc(rho_sph) inside the sphere itself — using the total there would double-count XC at
    # the boundary (a reintroduced ~eV discontinuity that kept the aspherical loop oscillating).
    v_grid = v_hart + vxc_lda(torch.tensor(np.clip(rho_I, 1e-12, None))).numpy()
    # The own-sphere L>0 field must be excluded from each sphere's lattice term. The analytic
    # cancellation (v_bc - V_part(R)) fails at the ~20% level because the ~5-grid-point pseudo-
    # charge ALIASES: its band-limited near-field at R differs from the analytic multipole value
    # even with exact grid moments (verified: a single atom kept a 20%-of-V_zz "lattice" term,
    # identical under two moment normalizations). Subtracting the own pseudocharge's ACTUAL
    # band-limited surface field makes C_LM the others' field by construction.
    vbc_own = []
    if own_ps_by_sphere:
        from gradwave.flapw.efg import interstitial_boundary_multi
        for sp, (own_ps, lset) in zip(spheres, own_ps_by_sphere, strict=True):
            v_own = fft_poisson(own_ps, A)
            vbc_own.append(interstitial_boundary_multi(v_own, sp["tau"], sp["R"], A, lset)
                           if lset else {})
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
                pf = (c + sgn * R * np.eye(3)[axis]) @ ainv       # a Cartesian surface point → frac
                idx = tuple(int(round(pf[d] * nfft)) % nfft for d in range(3))
                pts.append(v_hart[idx])
        v_bc = float(np.mean(pts))
        vpart = radial_poisson_to_R(sp["rho_sph"], rr, R, drw=drw) - sp["Z"] * E2 / rr
        v_sph_list.append(vpart + (v_bc - vpart[-1]))
    return v_sph_list, v_i0, v_grid, vbc_own


def _occ_degenerate_aware(ev, nbands, tol=1e-3):
    """T→0 occupations that respect degeneracies: filled bands get 2; if the filling boundary
    cuts a degenerate group (levels within ``tol`` eV), the remaining electrons are spread EQUALLY
    over the group — the symmetry-invariant subspace-trace density. Winner-take-all filling of a
    partially-occupied degenerate manifold lets the diagonalizer's arbitrary in-subspace basis
    choose where charge goes; SCF feedback then locks in a spuriously symmetry-broken, basin-
    dependent state (observed: translation-equivalent atoms disagreeing by 40% in the EFG). For a
    clean-gap insulator this reduces exactly to the old fill-lowest-nbands behaviour."""
    ev = np.asarray(ev)
    occ = np.zeros(len(ev))
    electrons = 2.0 * nbands
    i = 0
    while i < len(ev) and electrons > 1e-12:
        j = i + 1
        while j < len(ev) and ev[j] - ev[j - 1] < tol:    # extend the degenerate group
            j += 1
        fill = min(2.0 * (j - i), electrons)
        occ[i:j] = fill / (j - i)
        electrons -= fill
        i = j
    return occ


def _fermi_level(ev_all, w_all, nelec, kT):
    """Bisect the Fermi energy: ``Σ_k w_k Σ_n 2·f_FD((E_F-ε)/kT) = nelec`` (weights sum to 1)."""
    ev_all = np.asarray(ev_all)
    w_all = np.asarray(w_all)

    def count(ef):
        f = 1.0 / (1.0 + np.exp(np.clip((ev_all - ef) / kT, -60, 60)))
        return 2.0 * float((w_all[:, None] * f).sum())

    lo, hi = float(ev_all.min()) - 5.0, float(ev_all.max()) + 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if count(mid) < nelec:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def crystal_scf_multi(a_bohr=None, atoms=None, radii=None, ecut: float = 200.0, lmax: int = 2,
                      iters: int = 40, tol: float = 3e-3, kmesh=(1, 1, 1), smearing: float = 0.0,
                      efg: bool = False, fullpot: bool = False, use_symmetry: bool = True,
                      fullpot_lmax: int = 2, los=None, val_e=None, core=None, el_override=None,
                      v_start=None, kworkers: int = 1, subspace_reuse: bool = False,
                      subspace_tol: float = 1e-5, cell=None, verbose: bool = False):
    """Multi-sphere self-consistent muffin-tin FLAPW, cubic or orthorhombic cell.

    ``a_bohr`` is the cubic edge, or a length-3 vector of orthorhombic edge lengths (Bohr).
    ``atoms`` = ``[(frac (3,), symbol), ...]``; ``radii`` = ``{symbol: R_MT}``. Each atom gets its
    own sphere potential (so inequivalent same-species atoms are handled). Returns
    ``(bands, info)`` with ``bands['ev']`` the Γ valence eigenvalues (eV, referenced to the
    interstitial zero — compare splittings, not absolute levels; see ``crystal_scf``).

    ``smearing`` (eV) > 0 enables Fermi-Dirac occupations for metals: extra conduction bands are
    solved, the Fermi level is found by charge conservation over the k-mesh, and the density uses
    fractional occupations (``info['e_fermi']`` is returned). ``smearing=0`` fills exactly
    ``N_val/2`` bands two-per-state (insulators). ``a_bohr`` may be a scalar (cubic edge), a
    length-3 vector (orthorhombic edges), or a 3×3 matrix (triclinic cell, rows = lattice
    vectors), all in Bohr.

    ``fullpot=True`` runs the self-consistent full-potential loop: each iteration computes the
    non-spherical sphere potential (on-site Hartree + XC + the inter-atomic lattice field) from the
    aspherical density for every ``(L,M)`` with ``1 ≤ L ≤ fullpot_lmax``, and feeds it back into the
    Hamiltonian, so the wavefunctions respond to the non-spherical field. ``fullpot_lmax`` should be
    checked for convergence (odd L are included — a site without inversion symmetry has L=1,3
    components that polarize the valence). ``fullpot=False`` is the muffin-tin scheme.

    ``use_symmetry`` (default True) reduces the k-mesh to the irreducible wedge and resymmetrizes
    the density each iteration to restore the full-BZ symmetry the reduced sum drops — the
    interstitial density in G-space (``RhoSymmetrizer``), the muffin-tin density by averaging
    symmetry-equivalent atoms, and the aspherical multipoles by the Wigner-D group average
    (``_symmetrize_rho_lm`` — the l>0 star-unfolding, so ``fullpot`` and the EFG run on the wedge
    too). Falls back to the full mesh if the space group can't be built.

    ``los`` = ``{symbol: [(l, spec), ...]}`` adds LAPW+LO local orbitals per species — confined
    ``φ = a·u(E_l)+b·u̇(E_l)+c·u(E₂)`` with ``φ(R)=φ'(R)=0`` — for semicore states (Ti 3s/3p) or
    extra radial freedom. ``spec`` is an atomic orbital label ("3s") or an absolute energy (eV).
    When an LO carries semicore electrons, raise that species' valence count with
    ``val_e = {symbol: n}`` and drop the state from the frozen core with ``core = {symbol: [...]}``
    (both override the module defaults) — the LO does not change electron bookkeeping by itself.
    ``el_override = {symbol: {l: spec}}`` moves an energy parameter (e.g. Ti l=1 into the valence
    region once a 3p LO carries the semicore).

    ``v_start`` = ``{key: radial potential array}`` warm-starts the SCF from a previous run's
    converged spherical potentials (returned as ``info["v_by_key"]``) instead of the atomic ones —
    a convergence-sweep point then starts nearly converged (~2-3x fewer iterations).

    ``kworkers > 1`` solves the k-points of each iteration in a process pool (each k's secular
    build+solve is independent). Results are accumulated in mesh order, so the density is identical
    to the serial run. Divide the BLAS threads accordingly (e.g. ``OMP_NUM_THREADS=3`` with
    ``kworkers=6`` on ~20 cores) — the pool multiplies whatever thread count each worker uses.
    The workers are spawned (fork inherits live BLAS state and computes wrong numbers), so the
    CALLING script must be import-safe: guard its executable body with ``if __name__ ==
    "__main__":`` or the workers re-run it on import.

    ``subspace_reuse`` (default True) solves most iterations by Rayleigh-Ritz in the previous
    iteration's eigenvector subspace (residual-gated, exact-solve fallback); every 5th iteration
    is a forced exact solve and convergence is only accepted on exact-solve iterations, so the
    reported state never rests on a projected solve."""
    if (a_bohr is None) == (cell is None):
        raise ValueError("pass exactly one of a_bohr (Bohr, legacy) or cell (Å)")
    if atoms is None or radii is None:
        raise ValueError("atoms and radii are required")
    cell_ang = (np.asarray(cell, dtype=float) if cell is not None
                else np.asarray(a_bohr, dtype=float) * BOHR_ANG)
    A = cell_matrix(cell_ang)                                       # 3×3 cell in Å
    vol = float(abs(np.linalg.det(A)))
    amax = float(np.linalg.norm(A, axis=1).max())
    nfft = 2 * int(math.ceil(math.sqrt(4 * ecut / HBAR2_2M) * amax / (2 * math.pi))) + 2
    nfft = min(max(nfft, 24), 72)
    r, dx = log_mesh(1e-5, 28.0, 2500)
    r_np = r.numpy()
    atoms_cart = [(np.asarray(f, dtype=float) @ A, sym) for f, sym in atoms]
    keys = [f"a{i}" for i in range(len(atoms))]
    syms = [sym for _, sym in atoms]
    val_e_map = dict(_VAL_E)
    val_e_map.update(val_e or {})
    core_map = dict(_CORE)
    core_map.update(core or {})
    nbands = sum(val_e_map[s] for s in syms) // 2
    los_by_key = {k: los[s] for k, s in zip(keys, syms, strict=True) if s in (los or {})}
    key_sym = dict(zip(keys, syms, strict=True))
    kfracs, kw = monkhorst_pack(tuple(kmesh))
    kw = kw / kw.sum()
    sg, rho_symm, atom_orbits, d_ops = None, None, None, None
    if use_symmetry:
        try:                                            # IBZ reduction + resymmetrization
            from gradwave.flapw.efg import ylm_rotations_complex
            from gradwave.symmetry import RhoSymmetrizer, find_spacegroup, reduce_mesh
            uniq = {s: i for i, s in enumerate(dict.fromkeys(syms))}
            sg = find_spacegroup(A, np.array([f for f, _ in atoms], dtype=float),
                                 [uniq[s] for s in syms])
            kfracs, kw = reduce_mesh(tuple(kmesh), (0, 0, 0), sg)
            kw = kw / kw.sum()
            rho_symm = RhoSymmetrizer((nfft, nfft, nfft), sg)
            atom_orbits = _atom_orbits(sg.atom_map)
            # rotation blocks for the aspherical multipoles (fullpot / EFG star-unfolding)
            big_ls = sorted({2} | set(range(1, fullpot_lmax + 1)))
            d_ops = ylm_rotations_complex(sg, A, big_ls)
        except Exception:                               # any failure → full mesh (set above)
            kfracs, kw = monkhorst_pack(tuple(kmesh))
            kw = kw / kw.sum()
            sg, rho_symm, atom_orbits, d_ops = None, None, None, None

    at_by_sym, vat_by_sym = {}, {}
    for s in set(syms):
        at_by_sym[s], vat_by_sym[s] = atomic_scf(s, r, dx)
    R_by_key = {k: radii[s] for k, s in zip(keys, syms, strict=True)}
    min_edge = float(np.linalg.norm(A, axis=1).min())
    r_max, r_min = max(R_by_key.values()), min(R_by_key.values())
    if r_max > 0.45 * min_edge:                # Å cell passed as Bohr / Bohr radius passed as Å
        raise ValueError(f"muffin-tin radius {r_max:.2f} Å is >45% of the shortest cell edge "
                         f"({min_edge:.2f} Å) — check the cell/radii units (cell in Å via cell=, "
                         "Bohr via a_bohr; radii always Å)")
    if r_min < 0.02 * min_edge:
        import warnings
        warnings.warn(f"muffin-tin radius {r_min:.2f} Å is <2% of the shortest cell edge "
                      f"({min_edge:.2f} Å) — an Å radius mistakenly converted to Bohr?",
                      stacklevel=2)
    rr_by_key = {k: r_np[r_np <= R_by_key[k]] for k in keys}
    mask_by_key = {k: r_np <= R_by_key[k] for k in keys}
    acart = [(tau, key) for (tau, _), key in zip(atoms_cart, keys, strict=True)]
    for i in range(len(acart)):                # muffin tins must not overlap (R_MT is Å): overlap
        for j in range(i, len(acart)):         # makes the interstitial overlap matrix S indefinite
            sep = min(float(np.linalg.norm(acart[j][0] + np.asarray(sh) @ A - acart[i][0]))
                      for sh in itertools.product((-1, 0, 1), repeat=3)
                      if not (i == j and sh == (0, 0, 0)))
            if R_by_key[keys[i]] + R_by_key[keys[j]] > sep + 1e-9:
                raise ValueError(
                    f"muffin-tin spheres overlap ({keys[i]},{keys[j]}): "
                    f"R={R_by_key[keys[i]]:.3f}+{R_by_key[keys[j]]:.3f} Å > separation {sep:.3f} Å "
                    "(radii are in ångström)")
    # static interstitial indicator for the warped-interstitial term (geometry only)
    _ainv = np.linalg.inv(A)
    theta_i = np.ones((nfft, nfft, nfft), dtype=bool)
    for (tau_c, _sym2), kk2 in zip(atoms_cart, keys, strict=True):
        theta_i &= ~(_min_image_dist(np.asarray(tau_c[0] if isinstance(tau_c, tuple) else tau_c)
                                     @ _ainv, nfft, A) < R_by_key[kk2])
    warp_state = None                                     # (FFT of (v_grid-v_i0)·Θ_I, nfft)
    v_grid_prev = None                                    # mixed interstitial grid state
    v_by_key = {}
    for k, s in zip(keys, syms, strict=True):
        if v_start is not None and k in v_start:
            v_by_key[k] = torch.as_tensor(np.asarray(v_start[k]), dtype=torch.float64).clone()
        else:
            v_by_key[k] = vat_by_sym[s].clone()

    conv = None
    v_nsph = None                                          # non-spherical potential (fullpot)
    r_nsph = float("inf")                                  # aspherical residual (convergence gate)
    vns_new = None
    # staged start: a cold fullpot SCF couples the spherical and aspherical loops from iteration 0
    # and can diverge violently (observed: span bouncing 20 eV, r_v ~100). Converge the muffin-tin
    # problem first, then switch the aspherical potential on from that state (what every
    # well-behaved fullpot run implicitly did via warm-starting).
    mt_phase = bool(fullpot)
    from gradwave.flapw.recorder import FLAPWRecorder
    recorder = FLAPWRecorder()
    hist = ([], [])                                        # joint Anderson history
    c_prev_by_k = [None] * len(kfracs)
    for it in range(iters):
        # subspace reuse is safe in the muffin-tin phase but blind to states entering the
        # window while the aspherical potential ramps (observed blow-ups at [subsp] iterations);
        # fullpot iterations always solve exactly.
        full_iter = (not subspace_reuse) or (it % 5 == 0) or (fullpot and not mt_phase)
        t_it = time.time()
        r_sph = 0.0
        vnew_by_key = {}
        species, El_by_key, vmt_by_key, v0_by_key = {}, {}, {}, {}
        for k, s in zip(keys, syms, strict=True):
            R = R_by_key[k]
            v0 = float(v_by_key[k].numpy()[np.argmin(np.abs(r_np - R))])
            vmt = torch.where(r <= R, v_by_key[k] - v0, torch.zeros_like(r))
            nl = _VALENCE_NL[s]
            El = {lang: at_by_sym[s].get(nl.get(lang, ""), -5.0) - v0
                  for lang in range(max(lmax, 2) + 1)}
            for lang, spec in (el_override or {}).get(s, {}).items():
                El[lang] = (at_by_sym[s][spec] if isinstance(spec, str) else float(spec)) - v0
            species[k] = {"R": R, "v": vmt, "El": El}
            El_by_key[k], vmt_by_key[k], v0_by_key[k] = El, vmt, v0

        # pass 1 — solve every k, keep the results; pass 2 — occupy and accumulate the density.
        npad = max(2, nbands // 2) if smearing > 0 else 4
        nb_solve = nbands + npad
        # radial channels are k-independent — build once per iteration, reuse across the k-mesh.
        chan = {key: {lang: radial_channel(lang, sp["El"][lang], r, dx, sp["v"], sp["R"])
                      for lang in range(lmax + 1)} for key, sp in species.items()}
        lodat = (_build_lodat(los_by_key, chan, El_by_key, vmt_by_key, R_by_key, at_by_sym,
                              key_sym, v0_by_key, r, dx) if los_by_key else None)
        # aspherical radial integrals are k-independent — build once per iteration, like chan.
        nsph_int = (_nsph_radial_integrals(v_nsph, species, lmax, r, dx, lodat=lodat)
                    if v_nsph else None)
        kdata, ev_all, ev_gamma = [], [], None
        cprevs = [None if full_iter else c_prev_by_k[ik] for ik in range(len(kfracs))]
        if kworkers > 1 and len(kfracs) > 1:
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor
            argl = [(kf, A, acart, species, lmax, ecut, r, dx, nb_solve, v_nsph, chan, lodat,
                     nsph_int, cprevs[ik], subspace_tol, warp_state)
                    for ik, kf in enumerate(kfracs)]
            # spawn, not fork: forked children inherit live OpenMP/BLAS state and can compute
            # silently wrong numbers (observed: a 6 eV span error). Spawned workers re-import
            # (~seconds per iteration) — negligible on the multi-minute fullpot iterations the
            # pool exists for.
            with ProcessPoolExecutor(max_workers=min(kworkers, len(argl)),
                                     mp_context=_mp.get_context("spawn")) as ex:
                res_list = list(ex.map(_solve_k_args, argl))
        else:
            res_list = [_lapw_multi_k(kf, A, acart, species, lmax, ecut, r, dx, nb_solve,
                                      v_nsph=v_nsph, chan=chan, lodat=lodat, nsph_int=nsph_int,
                                      c_prev=cprevs[ik], subspace_tol=subspace_tol,
                                      warp=warp_state)
                        for ik, kf in enumerate(kfracs)]
        for ik, (kf, w, res) in enumerate(zip(kfracs, kw, res_list, strict=True)):
            kdata.append((kf, w, res))
            ev_all.append(res[0])
            c_prev_by_k[ik] = res[1]
            if np.all(np.abs(kf) < 1e-9):
                ev_gamma = res[0]
        if ev_gamma is None:
            ev_gamma = _lapw_multi_k((0, 0, 0), A, acart, species, lmax, ecut, r, dx, nb_solve,
                                     v_nsph=v_nsph, chan=chan, lodat=lodat,
                                     nsph_int=nsph_int, warp=warp_state)[0]
        ev_all = np.array(ev_all)

        e_fermi = None
        if smearing > 0:
            e_fermi = _fermi_level(ev_all, kw, 2 * nbands, smearing)
            occ_by_k = [2.0 / (1.0 + np.exp(np.clip((ev - e_fermi) / smearing, -60, 60)))
                        for ev in ev_all]
        else:
            occ_by_k = [_occ_degenerate_aware(ev, nbands) for ev in ev_all]

        rho_I = np.zeros((nfft, nfft, nfft))
        rho_val = {k: np.zeros_like(rr_by_key[k]) for k in keys}
        # The aspherical (l=2) density feeds the SCF only in the full-potential loop; for a plain
        # muffin-tin EFG run it is a pure diagnostic (it does not change the spherical potential,
        # and the l=2 Weinert pseudocharge has zero monopole so it leaves v_sph untouched). So
        # accumulate it once after convergence (_efg_density_pass, below) instead of paying the
        # per-k sphere_density_multipoles at every SCF iteration.
        if fullpot and not mt_phase:
            from gradwave.flapw.efg import sphere_density_multipoles_bands
            us_by_key = {k: _us_ext(El_by_key[k], lmax, r, dx, vmt_by_key[k], R_by_key[k],
                                    (lodat or {}).get(k)) for k in keys}
            lset_pot = [(lg, m) for lg in range(1, fullpot_lmax + 1) for m in range(-lg, lg + 1)]
            if (2, 0) not in lset_pot:                      # the EFG observable always needs l=2
                lset_pot += [(2, m) for m in range(-2, 3)]
            lset2 = [(0, 0)] + lset_pot
            rho_2m = {k: {lm: np.zeros(rr_by_key[k].shape, dtype=complex) for lm in lset2}
                      for k in keys}
        lo_off = {}                                         # each atom's LO-row offset past npw
        off = 0
        for _tau, k in acart:
            lo_off[k] = off
            off += sum(2 * lo["l"] + 1 for lo in (lodat or {}).get(k, []))
        for (_kf, w, res), occ in zip(kdata, occ_by_k, strict=True):
            _, c, mill, ks, abl_all, vol = res
            occl = list(occ)
            npw_k = len(mill)
            rho_I += w * _interstitial_density(c[:npw_k, :nb_solve], occl, mill, vol, nfft)
            ylm_by_l = {lang: _ylm_star(lang, ks) for lang in range(lmax + 1)}
            for ai, k in enumerate(keys):
                phase = np.exp(1j * (ks @ np.asarray(acart[ai][0])))
                cp = c[:npw_k, :nb_solve] * phase[:, None]
                los_k = (lodat or {}).get(k)
                nlo_k = sum(2 * lo["l"] + 1 for lo in los_k or [])
                lo_block = (c[npw_k + lo_off[k]:npw_k + lo_off[k] + nlo_k, :nb_solve]
                            if nlo_k else None)
                _, rk = _sphere_valence_density(cp, occl, ks, abl_all[ai], El_by_key[k], lmax,
                                                vol, r, dx, vmt_by_key[k], R_by_key[k],
                                                lo_block=lo_block, los=los_k, ylm_by_l=ylm_by_l)
                rho_val[k] += w * rk
                if fullpot and not mt_phase:
                    amps_all = _bands_amps(cp, ks, abl_all[ai], lmax, vol, lo_block=lo_block,
                                           los=los_k, ylm_by_l=ylm_by_l)
                    rlm = sphere_density_multipoles_bands(amps_all, occl, us_by_key[k], lmax,
                                                          lset2)
                    for lm in lset2:
                        rho_2m[k][lm] += w * rlm[lm]

        if rho_symm is not None:                        # restore full-BZ symmetry lost to IBZ:
            rho_I = np.fft.ifftn(                        #   interstitial ρ symmetrized in G-space,
                rho_symm.apply(torch.tensor(np.fft.fftn(rho_I))).numpy()).real
        if atom_orbits is not None:                     #   muffin-tin ρ averaged over equiv. atoms
            sym_dev = 0.0
            for orbit in atom_orbits:
                ok = [keys[i] for i in orbit]
                avg = sum(rho_val[k] for k in ok) / len(ok)
                scale = max(float(np.abs(avg).max()), 1e-30)
                for k in ok:
                    sym_dev = max(sym_dev, float(np.abs(rho_val[k] - avg).max()) / scale)
                    rho_val[k] = avg.copy()
        if fullpot and not mt_phase and d_ops is not None:   # ρ_LM star-unfolded (Wigner-D)
            raw = rho_2m
            rho_2m = _symmetrize_rho_lm(rho_2m, keys, sg, d_ops)
            for k in keys:                              # residual of the projector = asymmetry,
                scale = max(max(float(np.abs(v).max())  # scaled per ATOM (a numerically-empty
                                for v in rho_2m[k].values()), 1e-30)   # (L,M) is not a violation)
                for lm, v in rho_2m[k].items():
                    sym_dev = max(sym_dev, float(np.abs(raw[k][lm] - v).max()) / scale)

        spheres, rho_sph_by_key = [], {}
        for ai, (k, s) in enumerate(zip(keys, syms, strict=True)):
            rr, mask = rr_by_key[k], mask_by_key[k]
            rho_core = np.zeros_like(rr)
            for lc, nidx, fc in core_map[s]:
                _, uc = radial_eigs_tridiag(lc, r, dx, v_by_key[k], nidx)
                rho_core += fc * uc[mask, nidx - 1] ** 2 / (4 * math.pi * rr**2)
            rho_sph_by_key[k] = rho_val[k] + rho_core
            spheres.append({"tau": acart[ai][0], "rr": rr, "dx": dx,
                            "rho_sph": rho_sph_by_key[k], "Z": CONFIG[s][0], "R": R_by_key[k],
                            "rho_2m": rho_2m[k] if fullpot and not mt_phase else None})
        v_sph_list, v_i0, v_grid, vbc_own = _weinert_multi(rho_I, spheres, A, nfft)

        if fullpot and not mt_phase:
            from gradwave.flapw.efg import (
                interstitial_boundary_multi,
                nonspherical_potential,
            )
            vns_new = {}
            for ai, k in enumerate(keys):
                rr, drw, R = rr_by_key[k], rr_by_key[k] * dx, R_by_key[k]
                vns = nonspherical_potential(rho_sph_by_key[k], rho_2m[k], rr, drw, lset=lset_pot)
                # Add the lattice field *inside* the sphere for each (L,M): the source-free r^L
                # harmonic whose value at R matches the interstitial boundary,
                # (v_bc − V_part(R))/R^L. Without it the muffin-tin electrons never feel the
                # crystal field, so the valence/semicore cannot polarize in it (the Sternheimer
                # antishielding). The pseudocharges are moment-matched for every L in the lset, so
                # v_bc's own-sphere part cancels V_part(R) exactly and C_LM is the others' field.
                v_bc = interstitial_boundary_multi(v_grid, acart[ai][0], R, A, lset_pot)
                own = vbc_own[ai] if vbc_own else {}
                for (bl, m) in lset_pot:
                    c_lm = (v_bc[(bl, m)] - own.get((bl, m), 0.0)) / R**bl
                    vns[(bl, m)] = vns[(bl, m)] + c_lm * rr**bl
                vns_new[k] = vns
            if v_nsph is None:
                v_nsph = vns_new
                vns_new = None                     # first fullpot iteration: seed, nothing to mix
                r_nsph = float("inf")
            else:
                # relative aspherical residual (feeds the convergence gate); the mixing itself is
                # the JOINT Anderson below — one history over (all v_sph ⊕ v_nsph), so the coupled
                # spherical/aspherical fixed point is accelerated as one system instead of three
                # loops fighting at different rates.
                scale = max(max(float(np.abs(v).max()) for v in v_nsph[k2].values())
                            for k2 in keys) or 1.0
                r_nsph = max(float(np.abs(vns_new[k2][lm] - v_nsph[k2][lm]).max())
                             for k2 in keys for lm in vns_new[k2]) / scale

        for ai, k in enumerate(keys):
            mask = mask_by_key[k]
            vnew_sph = v_sph_list[ai] + vxc_lda(torch.tensor(rho_sph_by_key[k])).numpy()
            vnew_np = np.full(r.shape[0], v_i0)
            vnew_np[mask] = vnew_sph
            vnew = torch.tensor(vnew_np)
            vnew_by_key[k] = vnew
            r_sph = max(r_sph, float((vnew - v_by_key[k]).abs().max()))
        # Unified metric-weighted Anderson: EVERY component of the state — the spherical sphere
        # potentials, the aspherical components, and the INTERSTITIAL grid (previously unmixed:
        # it swung freely each iteration and drove the sphere boundary conditions at full
        # amplitude) — in one history, in the physical L2 metric (√(r²dr) radial, √dvol grid).
        # The metric is the principled form of "scale-weighting": without it the near-nucleus
        # points of v_sph (|v|~1e5 at r~1e-5, measure→0) dominate the least squares meaninglessly.
        nsph_now = fullpot and not mt_phase and v_nsph is not None and vns_new is not None
        sqdx = math.sqrt(dx)
        w_segs = [torch.from_numpy(rr_by_key[k] ** 1.5 * sqdx) for k in keys]
        segs = [v_by_key[k][torch.from_numpy(mask_by_key[k])] for k in keys]
        news = [vnew_by_key[k][torch.from_numpy(mask_by_key[k])] for k in keys]
        if nsph_now:
            for k in keys:
                wr = torch.from_numpy(rr_by_key[k] ** 1.5 * sqdx)
                for lm in sorted(v_nsph[k]):
                    for fn in (np.real, np.imag):
                        segs.append(torch.from_numpy(np.ascontiguousarray(fn(v_nsph[k][lm]))))
                        news.append(torch.from_numpy(np.ascontiguousarray(fn(vns_new[k][lm]))))
                        w_segs.append(wr)
        sqdv = math.sqrt(vol / nfft**3)
        ti_flat = theta_i.reshape(-1)
        w_segs.append(torch.full((int(ti_flat.sum()),), sqdv, dtype=torch.float64))
        segs.append(torch.from_numpy((v_grid_prev if v_grid_prev is not None
                                      else v_grid).reshape(-1)[ti_flat]))
        news.append(torch.from_numpy(v_grid.reshape(-1)[ti_flat]))
        w_vec = torch.cat(w_segs)
        x_old = torch.cat([t.reshape(-1) for t in segs]) * w_vec
        x_new = torch.cat([t.reshape(-1) for t in news]) * w_vec
        if hist and hist[0] and hist[0][-1].numel() != x_old.numel():
            hist = ([], [])                        # phase switch changed the vector: fresh history
        if not hist:
            hist = ([], [])
        hist[0].append(x_old)
        hist[1].append(x_new - x_old)
        if len(hist[1]) > 6:
            hist = (hist[0][-6:], hist[1][-6:])
        x_next = anderson_next(hist[0], hist[1], beta=0.3, m=5) / w_vec
        off = 0
        for k in keys:
            n = int(mask_by_key[k].sum())
            full_v = torch.full((r.shape[0],), v_i0, dtype=torch.float64)
            full_v[torch.from_numpy(mask_by_key[k])] = x_next[off:off + n]
            v_by_key[k] = full_v
            off += n
        if nsph_now:
            for k in keys:
                for lm in sorted(v_nsph[k]):
                    n = v_nsph[k][lm].size
                    re = x_next[off:off + n].numpy()
                    im = x_next[off + n:off + 2 * n].numpy()
                    v_nsph[k][lm] = re + 1j * im
                    off += 2 * n
        v_mix_i = x_next[off:].numpy()
        v_grid_prev = v_grid.copy()
        v_grid_prev.reshape(-1)[ti_flat] = v_mix_i
        # warped interstitial for the NEXT iteration, from the MIXED grid, zero-mean over Θ_I so
        # the interstitial-zero eigenvalue referencing is preserved
        u = np.where(theta_i, v_grid_prev - float(v_mix_i.mean()), 0.0)
        warp_state = (np.fft.fftn(u) / nfft**3, nfft)

        span = float(ev_gamma[nbands - 1] - ev_gamma[0])
        new = {"span": span, "ev": ev_gamma[:nbands].tolist(), "e_fermi": e_fermi}
        d_span = abs(new["span"] - conv["span"]) if conv is not None else float("inf")
        if verbose:
            sd = f"{sym_dev:.1e}" if atom_orbits is not None else "-"
            rn = f"{r_nsph:.1e}" if fullpot else "-"
            print(f"  flapw it={it:3d} span={span:9.4f} d={d_span:8.1e} r_v={r_sph:8.1e} "
                  f"r_nsph={rn} symdev={sd} "
                  f"[{'exact' if full_iter else 'subsp'}] {time.time() - t_it:5.1f}s",
                  flush=True)
        recorder.record(it=it, span=span, d_span=(None if conv is None else d_span),
                        r_v=r_sph, r_nsph=(r_nsph if fullpot else None),
                        symmetry_dev=(sym_dev if atom_orbits is not None else None),
                        e_fermi=e_fermi, mt_phase=mt_phase, exact_solve=full_iter,
                        t_s=time.time() - t_it)
        if mt_phase and ((full_iter and d_span < 3 * tol) or it >= max(20, iters // 2)):
            mt_phase = False           # spherical loop settled (or capped) -> enable fullpot
            if verbose:
                print("  flapw: muffin-tin phase converged -> fullpot on", flush=True)
        nsph_ok = (not fullpot) or (r_nsph < 0.05)
        if conv is not None and full_iter and nsph_ok and d_span < tol and r_sph < 0.1:
            conv = new
            break
        conv = new

    info = {"nbands": nbands, "symbols": syms, "e_fermi": conv.get("e_fermi"),
            "v_by_key": {k: v_by_key[k].numpy().copy() for k in keys}}
    info["recorder"] = recorder
    if atom_orbits is not None:
        # relative asymmetry of the RAW final-iteration density across symmetry-equivalent atoms
        # (and, for fullpot, the star-unfolding projector residual). Near convergence this should
        # be small; an order-unity value means the wedge sum sits in a symmetry-broken state —
        # multistable SCF basin or a symmetry bug. Surfaced so it can never fail silently.
        info["symmetry_dev"] = sym_dev
    if efg:
        if not fullpot or mt_phase:
            # Build the aspherical density once from the converged wavefunctions. On an IBZ run the
            # reduced k-sum misses the star of each k, so the l=2 multipoles are star-unfolded with
            # the Wigner-D group average (_symmetrize_rho_lm) — no full-mesh re-solve needed. Then
            # recompute the interstitial potential with the l=2 Weinert pseudocharge (lattice term).
            rho_2m = _efg_density_pass(kdata, occ_by_k, keys, acart, El_by_key, vmt_by_key,
                                       R_by_key, rr_by_key, lmax, r, dx, nb_solve, lodat=lodat)
            if d_ops is not None:
                rho_2m = _symmetrize_rho_lm(rho_2m, keys, sg, d_ops)
            for ai, k in enumerate(keys):
                spheres[ai]["rho_2m"] = rho_2m[k]
            _, _, v_grid, vbc_own = _weinert_multi(rho_I, spheres, A, nfft)
        info["efg"] = _efg_from_multipoles(rho_2m, v_grid, acart, keys, R_by_key, rr_by_key,
                                           dx, A, vbc_own=vbc_own)
    return conv, info


def _symmetrize_rho_lm(rho_2m, keys, sg, d_ops):
    """Group-average the aspherical sphere multipoles — the l>0 piece of the IBZ star-unfolding.
    An IBZ-weighted k-sum misses the star of each k; averaging over the space group restores it:
    op ``g: a → b`` contributes ``ρ_b[L] += D^L(S_g) ρ_a[L]`` (``d_ops`` from
    ``efg.ylm_rotations_complex``; atom permutation from ``sg.atom_map``). Time reversal needs no
    extra term: each k's density contribution is real and ρ_{-k} = ρ_k exactly. Equivalent-atom
    spheres share one radial mesh (same species), so the radial arrays add directly."""
    n_ops = len(d_ops)
    big_ls = sorted({lm[0] for lm in next(iter(rho_2m.values()))})
    acc = {k: {lm: np.zeros_like(v) for lm, v in rho_2m[k].items()} for k in keys}
    for iop in range(n_ops):
        amap = sg.atom_map[iop]
        for ai, k in enumerate(keys):
            kb = keys[int(amap[ai])]                     # op sends atom ai onto amap[ai]
            for bl in big_ls:
                if bl == 0:
                    acc[kb][(0, 0)] += rho_2m[k][(0, 0)]
                    continue
                mat = np.stack([rho_2m[k][(bl, m)] for m in range(-bl, bl + 1)])
                rot = d_ops[iop][bl] @ mat
                for mi, m in enumerate(range(-bl, bl + 1)):
                    acc[kb][(bl, m)] += rot[mi]
    return {k: {lm: v / n_ops for lm, v in acc[k].items()} for k in keys}


def _atom_orbits(atom_map):
    """Group atoms into symmetry orbits from a space group's ``atom_map`` ((nops, na);
    ``atom_map[op, a]`` = image of atom a under op). The orbit of atom a is column a — the set of
    atoms it is carried onto. Symmetry-equivalent atoms share one orbit; averaging the spherical
    density over each orbit restores the muffin-tin symmetry that IBZ k-reduction would break."""
    na = atom_map.shape[1]
    seen: set[int] = set()
    orbits: list[list[int]] = []
    for i in range(na):
        if i in seen:
            continue
        orbit = sorted({int(x) for x in atom_map[:, i]})
        seen.update(orbit)
        orbits.append(orbit)
    return orbits


def _solve_k_args(args):
    """Picklable per-k secular solve for the k-point process pool (``kworkers``)."""
    (kf, cell, acart, species, lmax, ecut, r, dx, nb_solve, v_nsph, chan, lodat, nsph_int,
     c_prev, subspace_tol, warp) = args
    return _lapw_multi_k(kf, cell, acart, species, lmax, ecut, r, dx, nb_solve,
                         v_nsph=v_nsph, chan=chan, lodat=lodat, nsph_int=nsph_int,
                         c_prev=c_prev, subspace_tol=subspace_tol, warp=warp)


def _efg_density_pass(kdata, occ_by_k, keys, acart, El_by_key, vmt_by_key, R_by_key, rr_by_key,
                      lmax, r, dx, nb_solve, lodat=None):
    """One BZ pass building the l=2 (and l=0) aspherical sphere-density multipoles from the
    converged wavefunctions — the EFG diagnostic. Factored out of the SCF loop so a muffin-tin EFG
    run pays the per-k ``sphere_density_multipoles`` cost once at convergence, not every iteration
    (the aspherical density does not enter the muffin-tin self-consistency). Mirrors the in-loop
    fullpot accumulation (LO-aware), so it matches an every-iteration accumulation."""
    from gradwave.flapw.efg import sphere_density_multipoles_bands
    lset2 = [(0, 0)] + [(2, m) for m in range(-2, 3)]
    us_by_key = {k: _us_ext(El_by_key[k], lmax, r, dx, vmt_by_key[k], R_by_key[k],
                            (lodat or {}).get(k)) for k in keys}
    rho_2m = {k: {lm: np.zeros(rr_by_key[k].shape, dtype=complex) for lm in lset2} for k in keys}
    lo_off = {}
    off = 0
    for _tau, k in acart:
        lo_off[k] = off
        off += sum(2 * lo["l"] + 1 for lo in (lodat or {}).get(k, []))
    for (_kf, w, res), occ in zip(kdata, occ_by_k, strict=True):
        _, c, mill, ks, abl_all, vol = res
        occl = list(occ)
        npw_k = len(mill)
        ylm_by_l = {lang: _ylm_star(lang, ks) for lang in range(lmax + 1)}
        for ai, k in enumerate(keys):
            phase = np.exp(1j * (ks @ np.asarray(acart[ai][0])))
            cp = c[:npw_k, :nb_solve] * phase[:, None]
            los_k = (lodat or {}).get(k)
            nlo_k = sum(2 * lo["l"] + 1 for lo in los_k or [])
            lo_block = (c[npw_k + lo_off[k]:npw_k + lo_off[k] + nlo_k, :nb_solve]
                        if nlo_k else None)
            amps_all = _bands_amps(cp, ks, abl_all[ai], lmax, vol, lo_block=lo_block,
                                   los=los_k, ylm_by_l=ylm_by_l)
            rlm = sphere_density_multipoles_bands(amps_all, occl, us_by_key[k], lmax, lset2)
            for lm in lset2:
                rho_2m[k][lm] += w * rlm[lm]
    return rho_2m


def _efg_from_multipoles(rho_by_key, v_grid, acart, keys, R_by_key, rr_by_key, dx, A,
                         vbc_own=None):
    """Per-atom EFG from the converged BZ-averaged aspherical density (``rho_by_key`` = the l=2 (and
    l=0) sphere-density multipoles accumulated over the k-mesh with the SCF occupations, i.e. the
    same density that produced the potential — not a fresh Γ solve at occ=2). Returns the valence
    V_zz (on-site l=2 sphere Poisson), the full V_zz adding the interstitial l=2 boundary (lattice)
    term, the asymmetry η, the tensor, and the in-sphere valence charge."""
    from gradwave.flapw.efg import (
        efg_tensor,
        efg_tensor_full,
        interstitial_l2_boundary,
        valence_efg_moments,
    )
    out = {}
    for ai, k in enumerate(keys):
        rho, rr, drw = rho_by_key[k], rr_by_key[k], rr_by_key[k] * dx
        q, q2 = valence_efg_moments(rho, rr, drw)
        _, v_zz_val, eta_val = efg_tensor(rho, rr, drw)
        v_bc = interstitial_l2_boundary(v_grid, acart[ai][0], R_by_key[k], A)
        own = vbc_own[ai] if vbc_own else {}
        v_bc = {m: v_bc[m] - own.get((2, m), 0.0) for m in range(-2, 3)}
        tensor, v_zz, eta = efg_tensor_full(rho, rr, drw, v_bc, R_by_key[k])
        charge = float(math.sqrt(4 * math.pi) * np.sum(rho[(0, 0)].real * rr**2 * drw))
        out[k] = {"Q2": q2, "V_zz": v_zz, "eta": eta, "V_zz_valence": v_zz_val,
                  "eta_valence": eta_val, "tensor": tensor, "sphere_charge": charge,
                  "q_moments": {m: complex(v) for m, v in q.items()}}
    return out
