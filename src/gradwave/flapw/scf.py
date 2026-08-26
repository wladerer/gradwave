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
from typing import Any, NamedTuple, cast

import numpy as np
import torch

from gradwave.constants import BOHR_ANG, E2, HBAR2_2M
from gradwave.flapw.atom import CONFIG, atomic_scf
from gradwave.flapw.coulomb import (
    _min_image_dist,
    cell_matrix,
    fft_poisson,
    gvec_ylm_tables,
    radial_poisson_to_R,
    reciprocal,
    sphere_interstitial_moments,
    sphere_pseudocharge,
    sphere_pseudocharge_ft,
)
from gradwave.flapw.functionals import vxc_lda
from gradwave.flapw.lapw import (
    ball_ff_np,
    enumerate_kg,
    match_ab_vec,
    radial_channel,
    radial_channels_all,
    solve_geneig,
    solve_geneig_shift_invert,
    solve_geneig_subspace_aug,
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

# Local-orbital overlap conditioning: valence-orthogonalize an LO when the fraction of its norm
# orthogonal to span{u, u̇} falls below this (its confined form is then near-linearly-dependent on
# the valence APW and the extended overlap S goes singular — see _build_lo). Chosen to trip the
# shallow O 2s semicore LO while leaving the deep, well-separated Ti 3s/3p LOs untouched: the
# measured rutile-TiO2 iter-0 residual fractions are O 2s 0.13 (E₂ pinned at the valence 2s,
# S_pu −0.99), Ti 3p 0.21, Ti 3s 0.53, so 0.18 trips only the O LO (which alone NaNs the pencil).
_LO_COND_TOL = 0.18


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


def _build_lo(l, e2, ch, u, ud, r, dx, v, R, cond_tol=_LO_COND_TOL, confine=True):
    """One local orbital built from a second-energy radial ``u(E₂)``, normalized to ``∫φ²dr=1``.
    ``ch`` is the l-channel ``radial_channel`` dict (supplies the u/u̇ boundary values); ``u, ud``
    the matching ``_radial_u`` arrays. Returns the radial data + the S/H integrals against (u, u̇)
    and itself that the LAPW+LO matrix blocks need.

    Two constructions, selected by ``confine``:

    * ``confine=True`` (default, the semicore LO): ``φ = a·u(E_l) + b·u̇(E_l) + c·u(E₂)`` with the
      two boundary conditions ``φ(R)=φ'(R)=0``. Standard for a deep semicore radial (Ti 3s/3p, O
      2s) that sits well below the valence window — there its ``u(E₂)`` is distinct from
      ``span{u, u̇}`` and the confined ``φ`` is a genuine extra degree of freedom.

    * ``confine=False`` (the HELO, high-energy local orbital): ``φ = a·u(E_l) + c·u(E₂)`` with the
      single boundary condition ``φ(R)=0`` (the ``u̇`` term is dropped, ``b=0``; the slope ``φ'(R)``
      is free). This is the fix for a channel whose confined LO is near-redundant with the valence
      p-span: forcing ``φ'(R)=0`` as well pulls in a large ``u̇`` component that collapses ``φ`` back
      into ``span{u, u̇}`` (measured resid_frac ~0.15 for the O 2p sphere at every E₂ tried,
      confined). Dropping that constraint and taking ``E₂`` high — a scattering-like ``u(E₂)`` with
      an extra radial node — keeps ``φ`` genuinely distinct (resid_frac → ~1), which is exactly the
      in-sphere aspherical (l=2) radial freedom the on-site EFG density needs. ``φ(R)=0`` alone is
      sufficient for the LO to stay strictly inside the muffin tin (no interstitial matching); the
      in-sphere kinetic element uses the gradient-square weak form (``_pair_integrals``), finite for
      a free ``φ'(R)``.

    Conditioning (``cond_tol`` > 0): a semicore ``E₂`` close to the valence linearization energy
    makes the confined ``φ`` near-linearly-dependent on the valence ``span{u, u̇}`` inside the
    sphere — its augmentation overlap row is then a near-copy of the plane-wave augmentation, so
    the extended LAPW+LO overlap ``S`` goes near-singular and the canonical ``S^{-1/2}`` NaNs
    (observed for the O 2s LO on rutile TiO₂; Ti 3s/3p sit deep enough to stay well-conditioned).
    ``resid_frac`` = the fraction of ``‖φ‖`` orthogonal to ``span{u, u̇}`` measures exactly this
    near-dependence (→0 degenerate, →1 orthogonal). When ``resid_frac < cond_tol`` the LO is
    Löwdin-projected onto the valence complement (``⟨φ|u⟩=⟨φ|u̇⟩=0`` by construction, so its
    ``S`` rows vanish and it re-enters the pencil as an orthonormal direction coupling only
    through ``H``) and renormalized — it stays a full variational basis function (the semicore
    radial's component beyond valence), it is not dropped. Well-conditioned LOs are left as the
    boundary-confined ``φ(R)=φ'(R)=0`` form (``cond_tol=0`` disables the check entirely)."""
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
    if confine:
        mat = np.array([[ch["uR"], ch["udR"]], [ch["upR"], ch["udpR"]]])
        a, b = np.linalg.solve(mat, -np.array([u2R, u2pR]))   # φ(R)=0, φ'(R)=0 with c=1
    else:
        a, b = -u2R / ch["uR"], 0.0                           # HELO: φ(R)=0 only (φ'(R) free)
    u2 = u2n[inside]
    phi = a * u + b * ud + u2
    nrm = math.sqrt(float((phi**2 * drw).sum()))
    if nrm < 1e-9 * math.sqrt(float((u2**2 * drw).sum())):
        # confined LO collapsed to ~zero norm: E₂ coincides with the channel's linearization
        # energy E_l, so u₂ ≈ u and the φ(R)=φ'(R)=0 fit yields φ ≈ u₂ − u ≈ 0. Conditioning
        # cannot rescue a null basis function; the l channel must be linearized away from E₂
        # (el_override) so the LO carries a radial distinct from the valence u.
        raise ValueError(
            f"local orbital (l={l}, E₂={e2:.3f} eV) is degenerate with the valence linearization "
            f"(E_l≈E₂): its confined form has ~zero norm. Move the l={l} energy parameter off E₂ "
            f"with el_override so the LO adds a distinct radial (semicore-LO setup).")
    # near-linear-dependence probe: project the confined φ onto span{u, u̇} (Gram of the valence
    # subspace = the radial-channel overlaps) and measure the orthogonal residual fraction.
    s_pu_raw = float((phi * u * drw).sum())
    s_pud_raw = float((phi * ud * drw).sum())
    guu = np.array([[ch["uu"], ch["uud"]], [ch["uud"], ch["udud"]]])
    p = np.linalg.solve(guu, np.array([s_pu_raw, s_pud_raw]))
    phi_orth = phi - p[0] * u - p[1] * ud
    nrm_orth = math.sqrt(float((phi_orth**2 * drw).sum()))
    resid_frac = nrm_orth / nrm
    orthogonalized = bool(cond_tol > 0.0 and resid_frac < cond_tol)
    if orthogonalized:                                        # valence-orthogonalize the LO
        a, b = a - p[0], b - p[1]                             # (u₂ coefficient unchanged)
        phi, nrm = phi_orth, nrm_orth
    a, b, cn = float(a / nrm), float(b / nrm), float(1.0 / nrm)
    phi = phi / nrm
    v_in = v.numpy()[inside]
    s_pu, t_pu, v_pu = _pair_integrals(l, phi, u, rr, drw, dx, v_in)
    s_pud, t_pud, v_pud = _pair_integrals(l, phi, ud, rr, drw, dx, v_in)
    _, t_pp, v_pp = _pair_integrals(l, phi, phi, rr, drw, dx, v_in)
    return {"l": l, "a": a, "b": b, "cn": cn, "u2": u2, "phi": phi,
            "S_pu": s_pu, "S_pud": s_pud, "H_pu": t_pu + v_pu, "H_pud": t_pud + v_pud,
            "S_pp": 1.0, "H_pp": t_pp + v_pp,
            "resid_frac": resid_frac, "orthogonalized": orthogonalized}


def _build_lodat(los_by_key, chan, El_by_key, vmt_by_key, R_by_key, at_by_sym, key_sym, v0_by_key,
                 r, dx, cond_tol=_LO_COND_TOL):
    """Per-iteration local-orbital data: for each atom key, build every requested LO against the
    current sphere potential. ``los_by_key[key]`` = list of ``(l, spec)`` with ``spec`` an atomic
    orbital label ("3p") or an absolute atomic energy (eV); either is shifted by the current
    muffin-tin zero like the energy parameters. One LO per (key, l). ``spec`` may also be a
    ``{"e": <label-or-eV>, "confine": False}`` mapping to request an unconfined HELO
    (``_build_lo(confine=False)``, φ(R)=0 only, high E₂) instead of the default confined semicore
    LO. ``cond_tol`` sets the overlap-conditioning threshold passed to ``_build_lo`` (see there)."""
    lodat = {}
    for key, los in los_by_key.items():
        seen = set()
        out = []
        for l, spec in los:
            if l in seen:
                raise ValueError(f"one local orbital per l per atom (duplicate l={l} on {key})")
            seen.add(l)
            confine = True
            if isinstance(spec, dict):
                confine = bool(spec.get("confine", True))
                spec = spec["e"]
            e2 = (at_by_sym[key_sym[key]][spec] if isinstance(spec, str) else float(spec))
            e2 = e2 - v0_by_key[key]
            u, ud = chan[key][l]["u_in"], chan[key][l]["ud_in"]   # reuse the channel's Numerov
            out.append(_build_lo(l, e2, chan[key][l], u, ud, r, dx, vmt_by_key[key],
                                 R_by_key[key], cond_tol=cond_tol, confine=confine))
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
            cv = cast("torch.Tensor", lo_block)[lo0:lo1, :].T  # (nb, 2l+1)
            amps[0] = amps[0] + cv * lo["a"]
            amps[1] = amps[1] + cv * lo["b"]
            amps.append(cv * lo["cn"])
        out[lang] = amps
    return out


def _us_ext(El, lmax, r, dx, v_sphere, R, los=None, chan_sp=None):
    """The per-l radial-function lists ``[u, u̇ (, u₂)]`` matching ``_bands_amps``' amplitudes.
    ``chan_sp`` = this sphere's ``{l: radial_channel}`` — when given, the channel's stored
    ``u_in``/``ud_in`` arrays are reused instead of re-running Numerov per l."""
    slices = _lo_row_slices(los)
    us = {}
    for lang in range(lmax + 1):
        if chan_sp is not None:
            u, ud = chan_sp[lang]["u_in"], chan_sp[lang]["ud_in"]
        else:
            u, ud = _radial_u(lang, El[lang], r, dx, v_sphere, R)
        us[lang] = [u, ud] + ([slices[lang][2]["u2"]] if lang in slices else [])
    return us


def _sphere_valence_density(c_occ, occ, ks, abl, El, lmax, vol, r, dx, v_sphere, R,
                            lo_block=None, los=None, ylm_by_l=None, us=None):
    """The spherical (l=0-projected) valence density inside one sphere, LO-aware: each l-channel
    carries ``[u, u̇]`` plus an LO's second radial when present, and the density sums every radial
    pair ``ρ += (1/4π) Σ_ij P_ij f_i f_j / r²`` with ``P_ij = Σ_{n,m} occ·Re(A*_i A_j)``.
    ``us`` = a precomputed ``_us_ext`` result (k-independent — a k-mesh sweep hoists it)."""
    rr = r.numpy()[r.numpy() <= R]
    if us is None:
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
    L = float(cell) if cell is not None else cast("float", a_bohr) * BOHR_ANG
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


def _k_geometry(kf, L, atoms_cart, radii, lmax, ecut):
    """The potential-INDEPENDENT per-k pieces of the secular build: the k+G enumeration, the
    pairwise |Δk| / cos matrices, the Legendre prefactors, the ball form factors per unique R,
    and the per-atom structure-phase vectors. All of it is pure geometry — fixed for the whole
    SCF — so the serial driver caches one of these per k-point and stops rebuilding them every
    iteration (they were a top per-iteration cost after the radial hoists). ``radii`` = the
    muffin-tin radius per atom of ``atoms_cart`` (order-aligned). Ylm / warp Miller-difference
    blocks are filled lazily (``_geom_ylm`` / ``_geom_dm``) so a muffin-tin-phase call computes
    exactly what the un-cached build did."""
    from scipy.special import eval_legendre
    A = cell_matrix(L)
    B = reciprocal(A)                                     # rows = reciprocal vectors
    vol = float(abs(np.linalg.det(A)))
    mill, ks = enumerate_kg(kf, B, ecut)
    ksafe = np.maximum(np.linalg.norm(ks, axis=1), 1e-12)
    kdot = ks @ ks.T
    kk = np.einsum("ij,ij->i", ks, ks)
    # |k_G - k_G'| from kdot (no (npw,npw,3) displacement tensor): |dk|² = k² + k'² - 2 k·k'
    dk_norm = np.sqrt(np.maximum(kk[:, None] + kk[None, :] - 2.0 * kdot, 0.0))
    cost = np.clip(kdot / np.outer(ksafe, ksafe), -1.0, 1.0)
    pref_l = {lang: (4 * math.pi / vol) * (2 * lang + 1) * eval_legendre(lang, cost)
              for lang in range(lmax + 1)}
    ballf = {}
    for R in radii:
        if R not in ballf:
            ballf[R] = ball_ff_np(dk_norm, R) / vol
    # e^{i(k_G'-k_G)·τ} factorizes exactly: outer(conj(p), p) with p = e^{i k_G·τ} — npw
    # exponentials instead of npw² (the npw² exp was a top build cost).
    pvec = [np.exp(1j * (ks @ np.asarray(tau, dtype=float))) for tau, _key in atoms_cart]
    return {"A": A, "vol": vol, "mill": mill, "ks": ks, "ksafe": ksafe, "kdot": kdot,
            "pref_l": pref_l, "ballf": ballf, "pvec": pvec}


def _geom_ylm(geom, l):
    """Lazily-cached ``conj(Y_lm)`` block of a ``_k_geometry`` dict (pure function of its ks)."""
    y = geom.setdefault("ylm", {})
    if l not in y:
        y[l] = _ylm_star(l, geom["ks"])
    return y[l]


def _geom_dm(geom, nfft_w):
    """Lazily-cached Miller-difference index triple for the warped-interstitial term."""
    if geom.get("dm_nfft") != nfft_w:
        mill = geom["mill"]
        geom["dm"] = tuple(((mill[:, None, i] - mill[None, :, i]) % nfft_w).astype(np.int32)
                           for i in range(3))
        geom["dm_nfft"] = nfft_w
    return geom["dm"]


# Measured shift-invert crossover (experiments/autoapw/si_crossover.py, asus, production rutile
# TiO2, OMP=2): below this pencil dimension the dense generalized eigensolve (eigh(S) + MRRR
# subset) is faster than ARPACK shift-invert + the two-shift inertia certificate; above it
# shift-invert wins and the margin grows with the basis. Measured per-solve speedup vs dim:
# 265→0.79× (loss), 439→1.75×, 705→1.73×, 1105→2.78× — break-even is ~dim 300. The threshold is
# set with margin above break-even so "auto" engages only where the win is solidly ≥1.5×.
# The "auto" policy (the crystal_scf_multi default) engages shift-invert only above this dim; below
# it, and for the two determinism-required contexts (Newton's F, the pool bit-equality path), the
# exact dense solve stays in force.
_SHIFT_INVERT_CROSSOVER_DIM = 400


def _resolve_shift_invert(mode, dim: int) -> bool:
    """Resolve the ``shift_invert`` knob (``True`` / ``False`` / ``"auto"``) to an engage decision
    for a secular pencil of size ``dim``. ``True`` = always attempt (the completeness certificate
    still guards the loud dense fallback); ``False`` / ``None`` = never (exact dense only, the
    determinism-preserving choice); ``"auto"`` = attempt only above the measured crossover dim
    ``_SHIFT_INVERT_CROSSOVER_DIM``, where the win is real. The per-solve gate for a valid σ and
    headroom (``sigma is not None and dim > nbands + 1``) stays at the call site."""
    if mode is True:
        return True
    if not mode:                                    # False / None
        return False
    return dim > _SHIFT_INVERT_CROSSOVER_DIM        # "auto" (or any truthy non-True)


def _lapw_multi_k(kf, L, atoms_cart, species, lmax, ecut, r, dx, nbands, v_nsph=None, chan=None,
                  lodat=None, nsph_int=None, c_prev=None, subspace_tol=1e-4, warp=None,
                  geom=None, shift_invert=False, sigma=None, return_HS=False):
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
    build locally (backward compatible).

    ``geom`` (optional) = this k's ``_k_geometry`` (potential-independent — a caller iterating an
    SCF builds it once per k per RUN and passes it back in every iteration). ``None`` → build
    locally (backward compatible; the k-worker process pool ships no geometry)."""
    if geom is None:
        geom = _k_geometry(kf, L, atoms_cart, [species[key]["R"] for _t, key in atoms_cart],
                           lmax, ecut)
    vol, mill, ks = geom["vol"], geom["mill"], geom["ks"]
    ksafe, kdot, pref_l = geom["ksafe"], geom["kdot"], geom["pref_l"]
    npw = len(ks)
    if chan is None:
        chan = {key: radial_channels_all(lmax, sp["El"], r, dx, sp["v"], sp["R"])
                for key, sp in species.items()}
    inter = np.eye(npw, dtype=complex)
    Saug = np.zeros((npw, npw), dtype=complex)
    Haug = np.zeros((npw, npw), dtype=complex)
    abl_by_atom = []
    for ai, (_tau, key) in enumerate(atoms_cart):
        R = species[key]["R"]
        v0off = float(species[key].get("v0off", 0.0))
        pvec = geom["pvec"][ai]
        phase = np.conj(pvec)[:, None] * pvec[None, :]
        inter -= geom["ballf"][R] * phase
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
            # + the per-sphere constant (v0 - v̄_I): the l-channels carry v - v0(R) while the
            # interstitial is referenced to the Θ-mean; once the interstitial warps and carries
            # XC these references differ by ~eV per sphere, and omitting the constant skews the
            # inter-sphere level alignment (charge transfer, hence the EFG).
            Haug += phase * (pref_l[lang] * (Tk + Vk + v0off * Ms))
        abl_by_atom.append(abl)
    if v_nsph:
        if nsph_int is None:
            nsph_int = _nsph_radial_integrals(v_nsph, species, lmax, r, dx, lodat=lodat,
                                              chan=chan)
        ylm_nsph = {lang: _geom_ylm(geom, lang) for lang in range(lmax + 1)}
        for ai, (_tau, key) in enumerate(atoms_cart):
            ints = nsph_int.get(key)
            if not ints or not ints["uu"]:
                continue
            ph_a = geom["pvec"][ai]
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
        dm0, dm1, dm2 = _geom_dm(geom, nfft_w)
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
        for ai, (_tau, key) in enumerate(atoms_cart):
            ph = geom["pvec"][ai]
            for lo in lodat.get(key, []):
                lang = lo["l"]
                ylm = _geom_ylm(geom, lang)
                pf = (4 * math.pi / math.sqrt(vol)) * (1j ** lang)
                a_g, b_g = abl_by_atom[ai][lang][:, 0], abl_by_atom[ai][lang][:, 1]
                v0off_lo = float(species[key].get("v0off", 0.0))
                base_s = pf * (a_g * lo["S_pu"] + b_g * lo["S_pud"]) * ph
                base_h = pf * (a_g * (lo["H_pu"] + v0off_lo * lo["S_pu"])
                               + b_g * (lo["H_pud"] + v0off_lo * lo["S_pud"])) * ph
                for mi in range(2 * lang + 1):
                    rows_s.append(base_s * ylm[:, mi])
                    rows_h.append(base_h * ylm[:, mi])
                    diag_s.append(lo["S_pp"])
                    diag_h.append(lo["H_pp"] + v0off_lo * lo["S_pp"])
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
            import os
            if os.environ.get("GRADWAVE_FLAPW_LO_DIAG"):
                # In-context LO conditioning: which LO is redundant given the FULL APW basis at
                # this k/potential (the per-LO cond_tol only sees each LO's own {u,u̇}). Exact,
                # opt-in (off the hot path). See flapw.lo_schur.
                from gradwave.flapw.lo_schur import lo_conditioning_report, lo_labels
                labels = lo_labels(atoms_cart, lodat)
                redundant, report = lo_conditioning_report(S, npw, labels)
                print(f"[LO-DIAG] npw={npw} nlo={nlo} in-context resid_frac per LO direction:\n"
                      f"{report}", flush=True)
                if redundant:
                    worst = ", ".join(f"{lbl} (resid_frac={f:.2e})" for f, lbl in redundant)
                    print(f"[LO-DIAG] REDUNDANT LO directions (spanned by the APW basis): {worst}",
                          flush=True)
    # ``shift_invert`` is True / False / "auto"; resolve it against the final pencil dim (after any
    # LO extension) — "auto" engages only above the measured crossover, True always attempts (the
    # certificate still guards), False/None never. subspace-aug and shift-invert are alternative
    # warm paths that both reuse ``c_prev``: when shift_invert is truthy ``c_prev`` is the SI start
    # seed (NOT a subspace-aug seed), so the aug branch is gated on the shift_invert MODE, not the
    # per-dim engage decision — a below-crossover "auto" solve then falls straight to exact dense
    # (the SI seed is simply unused), identical to the old shift_invert=False default.
    if return_HS:
        # FLAPW-DFPT (Phase 2): the linear-response machinery needs the assembled secular
        # matrices (and the full eigenbasis) at the frozen fixed-point potentials, not just the
        # occupied window. Force the exact dense solve over the WHOLE pencil so the sum-over-states
        # chi_0 has the complete unoccupied manifold, and hand back (Hm, S). Warm/SI paths are
        # bypassed (they return a truncated window and no matrices).
        ev, c = solve_geneig(Hm, S, Hm.shape[0], with_vecs=True)
        return ev, c, mill, ks, abl_by_atom, vol, Hm, S
    si_engage = _resolve_shift_invert(shift_invert, Hm.shape[0])
    if c_prev is not None and c_prev.shape[0] == Hm.shape[0] and not shift_invert:
        # Augmented Rayleigh-Ritz in span[last eigenvectors, their current residuals]; the
        # residual gate falls back to the exact dense solve whenever the subspace has drifted
        # (band crossings, early SCF, mixing jumps).
        ev, c, resid = solve_geneig_subspace_aug(Hm, S, c_prev, nbands)
        if resid < subspace_tol:
            return ev, c, mill, ks, abl_by_atom, vol
    if si_engage and sigma is not None and Hm.shape[0] > nbands + 1:
        # shift-invert Lanczos on the (H,S) pencil; the internal two-shift inertia certificate
        # returns None (→ LOUD dense fallback) on any incompleteness, so the exact spectrum is
        # never silently corrupted. Determinism-required contexts (Newton's F, the pool
        # bit-equality path) pass shift_invert=False and never reach here.
        cp = c_prev if (c_prev is not None and c_prev.shape[0] == Hm.shape[0]) else None
        si = solve_geneig_shift_invert(Hm, S, nbands, sigma, c_prev=cp)
        if si is not None:
            return si[0], si[1], mill, ks, abl_by_atom, vol
        import warnings
        warnings.warn("FLAPW shift-invert secular certificate failed at this k/iteration; "
                      "falling back to the exact dense eigensolve", RuntimeWarning, stacklevel=2)
    ev, c = solve_geneig(Hm, S, nbands, with_vecs=True)
    return ev, c, mill, ks, abl_by_atom, vol


def _nsph_radial_integrals(v_nsph, species, lmax, r, dx, lodat=None, chan=None):
    """The k-independent radial integrals of the aspherical potential — ``∫u_l V_LM u_l' dr`` (and
    the u̇/LO variants) per atom key. These were being recomputed at every k-point; a k-mesh sweep
    builds them once per SCF iteration and passes them into the secular build (same pattern as the
    ``chan`` hoist). ``chan`` = the per-key radial channels — when given, their stored
    ``u_in``/``ud_in`` arrays are reused instead of re-running Numerov per l. Returns
    ``{key: {"uu": {(L,M,l,l'): (i_aa,i_ab,i_ba,i_bb)}, "lo_u": {...}, "lo_lo": {...}}}``."""
    r_np = r.numpy()
    out = {}
    for key, comps in v_nsph.items():
        if not comps:
            continue
        rr = r_np[r_np <= species[key]["R"]]
        drw = rr * dx
        el, vsph = species[key]["El"], species[key]["v"]
        if chan is not None and key in chan:
            us = {lang: (chan[key][lang]["u_in"], chan[key][lang]["ud_in"])
                  for lang in range(lmax + 1)}
        else:
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
    ``{tau (cart), rr, dx, rho_sph, Z, R}`` (+ optional ``rho_2m`` aspherical multipole densities).
    ``L`` is any cell (cubic side, orthorhombic edges, or a 3×3 triclinic matrix). Returns
    ``([v_sph radial per sphere], v_i0, v_grid, v_hart, qmt_list)`` with ``v_hart`` the
    Coulomb-only interstitial grid (the boundary/lattice projections must use it, not the
    XC-contaminated ``v_grid``) and ``qmt_list[i] = {(L,M): q^MT_LM}`` each sphere's analytic
    multipole moments (electrons − nucleus at L=0) — the caller subtracts the own-sphere field
    ``4πE2 q^MT_LM/((2L+1)R^{L+1})`` from its surface projections analytically and exactly
    (Elk ``zpotcoul``'s ``z1 = (zlm − zvclmt(R))/R^l``).

    The whole construction is in G-space (``sphere_interstitial_moments`` /
    ``sphere_pseudocharge_ft``): analytic Bessel moments of ρ_I inside each sphere, and analytic
    Fourier synthesis of moment-matched pseudocharges (order policy below), so every matched L —
    including the high L of a fullpot_lmax=6 run — is exact to Fourier truncation. The earlier
    real-space-sampled pseudocharge left a ~20% own-field aliasing residue
    (retaining, worse, the fictitious ρ_I continuation inside the own sphere in the boundary term),
    which anti-fed the aspherical SCF (the measured runaway fixed point / |ρ|≈1.02 mode)."""
    A = cell_matrix(L)
    ainv = np.linalg.inv(A)
    vol = float(abs(np.linalg.det(A)))
    lmax_match = max((lm[0] for sp in spheres if sp.get("rho_2m") is not None
                      for lm in sp["rho_2m"]), default=0)
    gvec, gnorm, ylm = gvec_ylm_tables(A, nfft, lmax_match)
    rho_g = (np.fft.fftn(rho_I) / nfft**3).reshape(-1)
    gmax = math.pi * nfft / float(np.linalg.norm(A, axis=1).max())   # min per-axis Nyquist
    ps_g = np.zeros(nfft**3, dtype=complex)
    qmt_list = []
    inside_any = np.zeros((nfft, nfft, nfft), dtype=bool)
    for sp in spheres:
        cfrac = np.asarray(sp["tau"]) @ ainv
        inside_any |= _min_image_dist(cfrac, nfft, A) < sp["R"]
        drw = sp["rr"] * sp["dx"]
        q_sph = float(np.sum(4 * math.pi * sp["rho_sph"] * sp["rr"] ** 2 * drw))
        qmt = {(0, 0): complex((q_sph - sp["Z"]) / math.sqrt(4 * math.pi))}
        big_ls = [0]
        if sp.get("rho_2m") is not None:
            for bl in sorted({lm[0] for lm in sp["rho_2m"] if lm[0] > 0}):
                rlw = sp["rr"] ** (bl + 2) * drw
                for mm in range(-bl, bl + 1):
                    rho_lm = sp["rho_2m"].get((bl, mm))
                    qmt[(bl, mm)] = (complex(np.sum(rho_lm * rlw))
                                     if rho_lm is not None else 0.0 + 0.0j)
                big_ls.append(bl)
        q_i = sphere_interstitial_moments(rho_g, sp["R"], sp["tau"], gvec, gnorm, ylm, big_ls)
        deficit = {lm: qmt[lm] - q_i[lm] for lm in qmt}
        # Order policy: Weinert's npow ≈ R·Gmax/4 assumes the asymptotic (GR)^-(npow+2) decay
        # regime; at this code's bandwidths (R·Gmax ~ 12-24) the measured own-surface residual
        # (weinert_gates.py stage N) is minimized by LOWER orders — a high npow concentrates
        # the shape and pushes spectral weight past Nyquist before the asymptotic decay pays.
        npow = int(np.clip(round(sp["R"] * gmax / 6.0), 2, 8))
        ps_g += sphere_pseudocharge_ft(deficit, sp["R"], sp["tau"], vol, npow,
                                       gvec, gnorm, ylm)
        qmt_list.append(qmt)
    g2 = gnorm**2
    v_g = np.where(g2 > 1e-12, 4 * math.pi * E2 * (rho_g + ps_g) / np.where(g2 > 1e-12, g2, 1.0),
                   0.0)
    v_hart = np.fft.ifftn((v_g * nfft**3).reshape(nfft, nfft, nfft)).real
    # Interstitial XC: the spheres carry vxc(rho_sph) but the interstitial previously had NONE —
    # a potential discontinuity at every muffin-tin boundary and an O(eV) hole in the bonding
    # region. The TOTAL grid (Hartree+XC) feeds v_i0 and the warped-interstitial term; the SPHERE
    # MATCHING and the boundary (lattice) projections stay Hartree-only — the sphere adds its own
    # vxc(rho_sph) internally (XC is local: the interstitial XC neither continues into the sphere
    # as r^L nor belongs in the EFG, which is the second derivative of the COULOMB potential).
    v_grid = v_hart + vxc_lda(torch.tensor(np.clip(rho_I, 1e-12, None))).numpy()
    v_i0 = float(v_grid[~inside_any].mean())
    v_sph_list = []
    for sp in spheres:
        c = np.asarray(sp["tau"])
        R = sp["R"]
        rr = sp["rr"]
        drw = rr * sp["dx"]
        # proper l=0 surface average (exact Fourier evaluation on the sphere) — the old 6-axis-
        # point sample was harmless over a flat interstitial but samples a warped one
        # anisotropically
        from gradwave.flapw.efg import interstitial_boundary_multi
        v_bc = float(interstitial_boundary_multi(v_hart, c, R, A, [(0, 0)])[(0, 0)].real
                     / math.sqrt(4 * math.pi))
        vpart = radial_poisson_to_R(sp["rho_sph"], rr, R, drw=drw) - sp["Z"] * E2 / rr
        v_sph_list.append(vpart + (v_bc - vpart[-1]))
    return v_sph_list, v_i0, v_grid, v_hart, qmt_list


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


class _MultiCtx(NamedTuple):
    """Immutable per-run context of ``crystal_scf_multi`` — everything computed before the
    iteration loop that does not depend on the evolving potential state: geometry (cell, FFT
    grid, sphere meshes/masks, the interstitial Θ_I indicator), the symmetry machinery
    (space group, ρ symmetrizer, atom orbits, Wigner-D blocks), the atomic references
    (``atomic_scf`` per species) and the static knobs. Built once by ``_multi_setup`` and
    shared read-only across ``_multi_iterate`` calls — the Newton fixed-point map evaluates
    the iteration ~30 times per polish and must not re-pay this setup on every F evaluation."""

    A: np.ndarray
    vol: float
    nfft: int
    r: torch.Tensor
    dx: float
    r_np: np.ndarray
    keys: list[str]
    syms: list[str]
    key_sym: dict[str, str]
    los_by_key: dict[str, Any]
    nbands: int
    kfracs: np.ndarray
    kw: np.ndarray
    sg: Any
    rho_symm: Any
    atom_orbits: Any
    d_ops: Any
    at_by_sym: dict[str, Any]
    vat_by_sym: dict[str, Any]
    R_by_key: dict[str, float]
    rr_by_key: dict[str, np.ndarray]
    mask_by_key: dict[str, np.ndarray]
    acart: list[tuple[np.ndarray, str]]
    theta_i: np.ndarray
    kerker_K: np.ndarray | None
    core_map: dict[str, Any]
    lmax: int
    ecut: float
    smearing: float
    fullpot: bool
    fullpot_lmax: int
    el_override: dict[str, Any] | None
    kworkers: int
    subspace_reuse: bool
    subspace_tol: float
    shift_invert: bool | str
    lo_cond_tol: float
    verbose: bool
    # mutable per-run caches of potential-INDEPENDENT data (the NamedTuple itself stays
    # frozen): "geom" = per-k _k_geometry for the serial solve path (keyed by ik / "gamma"),
    # "ylm" = per-k conj(Y_lm) blocks for the density pass. Reused across _multi_iterate
    # calls — and across Newton F evaluations, which share one ctx.
    caches: dict[str, Any]


class _MultiState:
    """Mutable state of the ``crystal_scf_multi`` fixed-point iteration: the evolving
    potentials (spherical, aspherical, interstitial grid + reference + warp), the mixing
    history, the staging flag, and the last iteration's outputs that ``_multi_finalize``
    consumes (density, k-solve results, EFG inputs). One instance per SCF run / per Newton
    F evaluation; created by ``_multi_init_state``."""

    def __init__(self):
        self.v_by_key: dict[str, torch.Tensor] = {}
        self.conv: dict[str, Any] | None = None
        self.v_nsph: dict[Any, Any] | None = None   # non-spherical potential (fullpot)
        self.r_nsph: float = float("inf")      # aspherical residual (convergence gate)
        self.vns_new: dict[Any, Any] | None = None
        self.mt_phase: bool = False
        self.warp_state: tuple[Any, int] | None = None  # (FFT of (v_grid-v_i0)·Θ_I, nfft)
        self.v_grid_prev: np.ndarray | None = None      # mixed interstitial grid state
        self.v_i0_prev: float | None = None    # interstitial reference of last iter
        self.recorder: Any = None
        self.hist: tuple[list[Any], list[Any]] = ([], [])  # joint Anderson history
        self.c_prev_by_k: list[Any] = []
        self.c_prev_gamma: Any = None          # warm start for the Γ reporting solve
        self.ev_prev_by_k: list[Any] = []      # last iter's eigenvalues per k (shift-invert σ)
        self.ev_prev_gamma: Any = None
        # last-iteration outputs consumed by _multi_finalize
        self.kdata: Any = None
        self.occ_by_k: Any = None
        self.lodat: dict[Any, Any] | None = None
        self.El_by_key: Any = None
        self.vmt_by_key: Any = None
        self.rho_I: np.ndarray | None = None
        self.spheres: list[dict[str, Any]] | None = None
        self.nb_solve: Any = None
        self.sym_dev: float | None = None
        self.rho_2m: dict[Any, Any] | None = None
        self.v_grid: np.ndarray | None = None
        self.v_hart: Any = None                # Coulomb-only grid (boundary/EFG projections)
        self.qmt: Any = None                   # per-sphere analytic multipole moments q^MT_LM


def _multi_setup(a_bohr=None, atoms=None, radii=None, ecut: float = 200.0, lmax: int = 2,
                 kmesh=(1, 1, 1), smearing: float = 0.0, fullpot: bool = False,
                 use_symmetry: bool = True, fullpot_lmax: int = 2, los=None, val_e=None,
                 core=None, el_override=None, kworkers: int = 1, subspace_reuse: bool = False,
                 subspace_tol: float = 1e-4, cell=None, kerker: float | None = None,
                 shift_invert: bool | str = "auto", lo_cond_tol: float = _LO_COND_TOL,
                 verbose: bool = False) -> _MultiCtx:
    """The state-independent setup phase of ``crystal_scf_multi`` (see ``_MultiCtx``).
    Argument semantics and validation are exactly the public entry point's."""
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
    # Weinert pseudocharge resolvability (the D4 near-field gate): an order-L pseudocharge of
    # order npow ≈ R·Gmax/4 needs its Bessel factor j_{L+npow+1}(R·Gmax) past the first peak at
    # the grid's (min-axis) Nyquist, i.e. R·Gmax ≳ (4/3)(L+7) for the SMALLEST sphere — otherwise
    # the highest matched moments are not band-limited-representable and their near field is
    # garbage (the measured fullpot_lmax=6 divergence). The ecut-based grid already satisfies
    # this for L ≤ 4 at production settings; L = 5,6 raise nfft here (Elk's counterpart is its
    # ecut-independent Gmax = 12 bohr⁻¹ density grid).
    lmax_match = max(2, fullpot_lmax) if fullpot else 2
    r_mt_min = min(radii[s] for s in {s for _, s in atoms})
    gmax_req = (4.0 / 3.0) * (lmax_match + 7) / r_mt_min
    nfft_psd = 2 * int(math.ceil(gmax_req * amax / (2 * math.pi)))
    nfft = min(max(nfft, nfft_psd, 24), 72)
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
    los_by_key = {k: cast("dict[str, Any]", los)[s]
                  for k, s in zip(keys, syms, strict=True) if s in (los or {})}
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
    kerker_K = None if kerker is None else _kerker_screen(nfft, A, kerker)
    return _MultiCtx(A=A, vol=vol, nfft=nfft, r=r, dx=dx, r_np=r_np, keys=keys, syms=syms,
                     key_sym=key_sym, los_by_key=los_by_key, nbands=nbands, kfracs=kfracs,
                     kw=kw, sg=sg, rho_symm=rho_symm, atom_orbits=atom_orbits, d_ops=d_ops,
                     at_by_sym=at_by_sym, vat_by_sym=vat_by_sym, R_by_key=R_by_key,
                     rr_by_key=rr_by_key, mask_by_key=mask_by_key, acart=acart,
                     theta_i=theta_i, kerker_K=kerker_K, core_map=core_map, lmax=lmax,
                     ecut=ecut, smearing=smearing, fullpot=fullpot,
                     fullpot_lmax=fullpot_lmax, el_override=el_override, kworkers=kworkers,
                     subspace_reuse=subspace_reuse, subspace_tol=subspace_tol,
                     shift_invert=shift_invert, lo_cond_tol=lo_cond_tol,
                     verbose=verbose, caches={"geom": {}, "ylm": {}})


def _shutdown_ctx_pool(ctx: _MultiCtx) -> None:
    """Shut down a ctx's persistent k-worker pool (no-op when none is live)."""
    pool = ctx.caches.get("pool")
    if pool is not None:
        pool.shutdown(wait=False)
        ctx.caches["pool"] = None
        ctx.caches["pool_nw"] = 0


def _multi_init_state(ctx: _MultiCtx, v_start) -> _MultiState:
    """Build the initial ``_MultiState`` from ``v_start`` (None, a per-key radial-potential
    dict, or ``{"__full_state__": info["state"]}`` — semantics exactly as documented on
    ``crystal_scf_multi``)."""
    st = _MultiState()
    fs = v_start.get("__full_state__") if isinstance(v_start, dict) else None
    _vsrc = fs["v_by_key"] if fs is not None else v_start
    for k, s in zip(ctx.keys, ctx.syms, strict=True):
        if _vsrc is not None and k in _vsrc:
            st.v_by_key[k] = torch.as_tensor(np.asarray(_vsrc[k]), dtype=torch.float64).clone()
        else:
            st.v_by_key[k] = ctx.vat_by_sym[s].clone()

    # staged start: a cold fullpot SCF couples the spherical and aspherical loops from iteration 0
    # and can diverge violently (observed: span bouncing 20 eV, r_v ~100). Converge the muffin-tin
    # problem first, then switch the aspherical potential on from that state (what every
    # well-behaved fullpot run implicitly did via warm-starting).
    st.mt_phase = bool(ctx.fullpot)
    if fs is not None:
        # full-state resume skips the MT staging: a full state IS a coupled fullpot state
        st.v_nsph, st.v_grid_prev, st.warp_state, st.v_i0_prev = _restore_full_state(
            fs, ctx.theta_i, ctx.nfft)
        if ctx.fullpot:
            st.mt_phase = False
    from gradwave.flapw.recorder import FLAPWRecorder
    st.recorder = FLAPWRecorder()
    st.c_prev_by_k = [None] * len(ctx.kfracs)
    st.ev_prev_by_k = [None] * len(ctx.kfracs)
    return st


def _sigma_shift(evp):
    # shift-invert σ: just below the previous iteration's occupied window bottom (the
    # measured-winning placement; the internal inertia certificate adapts if the window moved).
    if evp is None or len(evp) == 0:
        return None
    return float(evp[0]) - max(1.0, 0.02 * float(evp[-1] - evp[0]))


def _build_species(ctx: _MultiCtx, st: _MultiState):
    """Per-key muffin-tin potential (``vmt``, referenced to the interstitial-zero ``v0``),
    linearization energies ``El``, and the ``species`` dicts the k-solve consumes. Reads the
    current spherical potential ``st.v_by_key`` and the previous interstitial reference
    ``st.v_i0_prev``; pure (no mutation). Returns
    ``(species, El_by_key, vmt_by_key, v0_by_key)``."""
    r, r_np, lmax = ctx.r, ctx.r_np, ctx.lmax
    species, El_by_key, vmt_by_key, v0_by_key = {}, {}, {}, {}
    for k, s in zip(ctx.keys, ctx.syms, strict=True):
        R = ctx.R_by_key[k]
        v0 = float(st.v_by_key[k].numpy()[np.argmin(np.abs(r_np - R))])
        vmt = torch.where(r <= R, st.v_by_key[k] - v0, torch.zeros_like(r))
        nl = _VALENCE_NL[s]
        El = {lang: ctx.at_by_sym[s].get(nl.get(lang, ""), -5.0) - v0
              for lang in range(max(lmax, 2) + 1)}
        for lang, spec in (ctx.el_override or {}).get(s, {}).items():
            El[lang] = (ctx.at_by_sym[s][spec] if isinstance(spec, str) else float(spec)) - v0
        species[k] = {"R": R, "v": vmt, "El": El,
                      "v0off": (v0 - st.v_i0_prev) if st.v_i0_prev is not None else 0.0}
        El_by_key[k], vmt_by_key[k], v0_by_key[k] = El, vmt, v0
    return species, El_by_key, vmt_by_key, v0_by_key


def _build_iteration_radials(ctx: _MultiCtx, species, El_by_key, vmt_by_key, v0_by_key, v_nsph):
    """The k-independent radial builds done once per iteration and reused across the k-mesh:
    the solve-band padding ``nb_solve``, the batched radial channels ``chan`` (one Numerov per
    species), the local-orbital data ``lodat``, the aspherical radial integrals ``nsph_int``
    (fullpot only), and the sphere radial functions ``us_by_key``."""
    r, dx, lmax = ctx.r, ctx.dx, ctx.lmax
    npad = max(2, ctx.nbands // 2) if ctx.smearing > 0 else 4
    nb_solve = ctx.nbands + npad
    # radial channels are k-independent — build once per iteration, reuse across the k-mesh
    # (one batched Numerov per species: every (l, E±h) row in a single recurrence loop).
    chan = {key: radial_channels_all(lmax, sp["El"], r, dx, sp["v"], sp["R"])
            for key, sp in species.items()}
    lodat = (_build_lodat(ctx.los_by_key, chan, El_by_key, vmt_by_key, ctx.R_by_key,
                          ctx.at_by_sym, ctx.key_sym, v0_by_key, r, dx, cond_tol=ctx.lo_cond_tol)
             if ctx.los_by_key else None)
    # aspherical radial integrals are k-independent — build once per iteration, like chan.
    nsph_int = (_nsph_radial_integrals(v_nsph, species, lmax, r, dx, lodat=lodat, chan=chan)
                if v_nsph else None)
    # the sphere radial functions are k-independent too — hoist them out of the per-k density
    # accumulation (they were re-Numerov'd for every (k, atom, l))
    us_by_key = {k: _us_ext(El_by_key[k], lmax, r, dx, vmt_by_key[k], ctx.R_by_key[k],
                            (lodat or {}).get(k), chan_sp=chan[k]) for k in ctx.keys}
    return nb_solve, chan, lodat, nsph_int, us_by_key


def _solve_all_k(ctx: _MultiCtx, st: _MultiState, species, chan, lodat, nsph_int, nb_solve,
                 v_nsph, warp_state, full_iter):
    """Pass 1 of the iteration: solve the generalized eigenproblem at every k (serial, or via
    the persistent spawn ``ProcessPool`` on ``ctx.caches``), plus a Γ reporting solve when Γ is
    not already in the mesh. Threads the shift-invert σ and the per-k warm-start seeds through
    ``st`` (``c_prev_by_k``/``ev_prev_by_k``/``c_prev_gamma``/``ev_prev_gamma``) and updates
    them in place. Returns ``(kdata, ev_all, ev_gamma)`` for the density pass / convergence gate."""
    A, acart, lmax, ecut, r, dx = ctx.A, ctx.acart, ctx.lmax, ctx.ecut, ctx.r, ctx.dx
    kfracs, kw, subspace_tol = ctx.kfracs, ctx.kw, ctx.subspace_tol
    si_on = ctx.shift_invert
    cprevs = [None if full_iter else st.c_prev_by_k[ik] for ik in range(len(kfracs))]
    sigmas = [(_sigma_shift(st.ev_prev_by_k[ik]) if si_on else None) for ik in range(len(kfracs))]
    # shift-invert reuses last iter's eigenvectors only as a deterministic start seed (not the
    # subspace-reuse acceptance path); pass them regardless of the full_iter gate.
    solve_cprev = [(st.c_prev_by_k[ik] if si_on else cprevs[ik]) for ik in range(len(kfracs))]
    geom_cache = ctx.caches["geom"]

    def _geom_for(gk, kf_g):
        # per-k geometry is potential-independent: build once per RUN, not per iteration.
        # Serial path only — shipping ~npw² matrices through the pool would cost more than
        # the workers' rebuild.
        if gk not in geom_cache:
            geom_cache[gk] = _k_geometry(kf_g, A, acart,
                                         [species[key]["R"] for _t, key in acart], lmax, ecut)
        return geom_cache[gk]

    if ctx.kworkers > 1 and len(kfracs) > 1:
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        argl = [SolveKArgs(kf=kf, cell=A, acart=acart, species=species, lmax=lmax,
                           ecut=ecut, r=r, dx=dx, nb_solve=nb_solve, v_nsph=v_nsph,
                           chan=chan, lodat=lodat, nsph_int=nsph_int,
                           c_prev=solve_cprev[ik], subspace_tol=subspace_tol,
                           warp=warp_state, shift_invert=si_on, sigma=sigmas[ik])
                for ik, kf in enumerate(kfracs)]
        # spawn, not fork: forked children inherit live OpenMP/BLAS state and can compute
        # silently wrong numbers (observed: a 6 eV span error). The pool lives on ctx.caches —
        # created ONCE per run (and shared across Newton F evaluations) instead of per
        # iteration: the spawn+re-import tax (~seconds) was negligible on multi-minute fullpot
        # iterations but is not on ~10 s ones. _multi_finalize / newton_polish shut it down.
        nw = min(ctx.kworkers, len(argl))
        if ctx.caches.get("pool") is None or ctx.caches.get("pool_nw") != nw:
            _shutdown_ctx_pool(ctx)
            ctx.caches["pool"] = ProcessPoolExecutor(max_workers=nw,
                                                     mp_context=_mp.get_context("spawn"))
            ctx.caches["pool_nw"] = nw
        res_list = list(ctx.caches["pool"].map(_solve_k_args, argl))
    else:
        res_list = [_lapw_multi_k(kf, A, acart, species, lmax, ecut, r, dx, nb_solve,
                                  v_nsph=v_nsph, chan=chan, lodat=lodat, nsph_int=nsph_int,
                                  c_prev=solve_cprev[ik], subspace_tol=subspace_tol,
                                  warp=warp_state, geom=_geom_for(ik, kf),
                                  shift_invert=si_on, sigma=sigmas[ik])
                    for ik, kf in enumerate(kfracs)]
    kdata, ev_all, ev_gamma = [], [], None
    for ik, (kf, w, res) in enumerate(zip(kfracs, kw, res_list, strict=True)):
        kdata.append((kf, w, res))
        ev_all.append(res[0])
        st.c_prev_by_k[ik] = res[1]
        st.ev_prev_by_k[ik] = res[0]
        if np.all(np.abs(kf) < 1e-9):
            ev_gamma = res[0]
            st.ev_prev_gamma = res[0]
    if ev_gamma is None:
        sigma_g = _sigma_shift(st.ev_prev_gamma) if si_on else None
        res_g = _lapw_multi_k((0, 0, 0), A, acart, species, lmax, ecut, r, dx, nb_solve,
                              v_nsph=v_nsph, chan=chan, lodat=lodat, nsph_int=nsph_int,
                              c_prev=(st.c_prev_gamma if si_on
                                      else (None if full_iter else st.c_prev_gamma)),
                              subspace_tol=subspace_tol, warp=warp_state,
                              geom=_geom_for("gamma", (0, 0, 0)),
                              shift_invert=si_on, sigma=sigma_g)
        ev_gamma = res_g[0]
        st.c_prev_gamma = res_g[1]
        st.ev_prev_gamma = res_g[0]
    ev_all = np.array(ev_all)
    return kdata, ev_all, ev_gamma


def _occupations(ctx: _MultiCtx, ev_all):
    """Fermi level (``smearing>0``) and the per-k occupations that weight the density pass.
    Returns ``(e_fermi, occ_by_k)``; ``e_fermi`` is ``None`` in the sharp (degenerate) case."""
    if ctx.smearing > 0:
        e_fermi = _fermi_level(ev_all, ctx.kw, 2 * ctx.nbands, ctx.smearing)
        occ_by_k = [2.0 / (1.0 + np.exp(np.clip((ev - e_fermi) / ctx.smearing, -60, 60)))
                    for ev in ev_all]
    else:
        e_fermi = None
        occ_by_k = [_occ_degenerate_aware(ev, ctx.nbands) for ev in ev_all]
    return e_fermi, occ_by_k


def _accumulate_density(ctx: _MultiCtx, st: _MultiState, kdata, occ_by_k, El_by_key,
                        vmt_by_key, us_by_key, lodat, nb_solve):
    """Pass 2 of the iteration: BZ-integrate the interstitial density ``rho_I`` and the per-key
    sphere valence density (and, in the full-potential loop, the aspherical multipoles
    ``rho_2m``), restore full-BZ symmetry (interstitial in G-space, muffin-tin averaged over
    equivalent atoms, ρ_LM star-unfolded), add the core density, and assemble the ``spheres``
    for Weinert. Returns ``(rho_I, rho_2m, spheres, rho_sph_by_key, sym_dev, lset_pot)``;
    ``lset_pot`` (the aspherical (L,M) set) is ``None`` outside the fullpot pass."""
    nfft, keys, syms, acart = ctx.nfft, ctx.keys, ctx.syms, ctx.acart
    lmax, r, dx = ctx.lmax, ctx.r, ctx.dx
    rr_by_key, mask_by_key, R_by_key = ctx.rr_by_key, ctx.mask_by_key, ctx.R_by_key
    fullpot, mt_phase, fullpot_lmax = ctx.fullpot, st.mt_phase, ctx.fullpot_lmax
    rho_symm, atom_orbits, d_ops, sg = ctx.rho_symm, ctx.atom_orbits, ctx.d_ops, ctx.sg
    core_map, v_by_key = ctx.core_map, st.v_by_key

    rho_I = np.zeros((nfft, nfft, nfft))
    rho_val = {k: np.zeros_like(rr_by_key[k]) for k in keys}
    sym_dev = 0.0
    lset_pot = None
    # The aspherical (l=2) density feeds the SCF only in the full-potential loop; for a plain
    # muffin-tin EFG run it is a pure diagnostic (it does not change the spherical potential,
    # and the l=2 Weinert pseudocharge has zero monopole so it leaves v_sph untouched). So
    # accumulate it once after convergence (_efg_density_pass, below) instead of paying the
    # per-k sphere_density_multipoles at every SCF iteration.
    from gradwave.flapw.efg import sphere_density_multipoles_bands

    rho_2m: dict[Any, Any] | None = None
    lset2: list[tuple[int, int]] = []
    if fullpot and not mt_phase:
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
    ylm_cache = ctx.caches["ylm"]
    for ik, ((_kf, w, res), occ) in enumerate(zip(kdata, occ_by_k, strict=True)):
        _, c, mill, ks, abl_all, vol = res
        occl = list(occ)
        npw_k = len(mill)
        rho_I += w * _interstitial_density(c[:npw_k, :nb_solve], occl, mill, vol, nfft)
        ylm_by_l = ylm_cache.get(ik)           # pure geometry — cached across iterations
        if ylm_by_l is None or len(ylm_by_l) < lmax + 1:
            ylm_by_l = {lang: _ylm_star(lang, ks) for lang in range(lmax + 1)}
            ylm_cache[ik] = ylm_by_l
        for ai, k in enumerate(keys):
            phase = np.exp(1j * (ks @ np.asarray(acart[ai][0])))
            cp = c[:npw_k, :nb_solve] * phase[:, None]
            los_k = (lodat or {}).get(k)
            nlo_k = sum(2 * lo["l"] + 1 for lo in los_k or [])
            lo_block = (c[npw_k + lo_off[k]:npw_k + lo_off[k] + nlo_k, :nb_solve]
                        if nlo_k else None)
            _, rk = _sphere_valence_density(cp, occl, ks, abl_all[ai], El_by_key[k], lmax,
                                            vol, r, dx, vmt_by_key[k], R_by_key[k],
                                            lo_block=lo_block, los=los_k, ylm_by_l=ylm_by_l,
                                            us=us_by_key[k])
            rho_val[k] += w * rk
            if fullpot and not mt_phase:
                assert rho_2m is not None      # bound together with lset2 under this guard
                amps_all = _bands_amps(cp, ks, abl_all[ai], lmax, vol, lo_block=lo_block,
                                       los=los_k, ylm_by_l=ylm_by_l)
                rlm = sphere_density_multipoles_bands(amps_all, occl, us_by_key[k],
                                                      rr_by_key[k], lmax, lset2)
                for lm in lset2:
                    rho_2m[k][lm] += w * rlm[lm]

    if rho_symm is not None:                        # restore full-BZ symmetry lost to IBZ:
        rho_I = np.fft.ifftn(                        #   interstitial ρ symmetrized in G-space,
            rho_symm.apply(torch.tensor(np.fft.fftn(rho_I))).numpy()).real
    if atom_orbits is not None:                     #   muffin-tin ρ averaged over equiv. atoms
        for orbit in atom_orbits:
            ok = [keys[i] for i in orbit]
            avg = cast("np.ndarray", sum(rho_val[k] for k in ok) / len(ok))
            scale = max(float(np.abs(avg).max()), 1e-30)
            for k in ok:
                sym_dev = max(sym_dev, float(np.abs(rho_val[k] - avg).max()) / scale)
                rho_val[k] = avg.copy()
    if fullpot and not mt_phase and d_ops is not None:   # ρ_LM star-unfolded (Wigner-D)
        raw = rho_2m
        assert raw is not None                       # a dict whenever this guard holds
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
                        "rho_2m": (cast("dict[Any, Any]", rho_2m)[k]
                                   if fullpot and not mt_phase else None)})
    return rho_I, rho_2m, spheres, rho_sph_by_key, sym_dev, lset_pot


def _multi_iterate(ctx: _MultiCtx, st: _MultiState, it: int, iters: int, tol: float) -> bool:
    """One damped SCF iteration of ``crystal_scf_multi`` — the fixed-point map applied to the
    evolving state ``st`` under the static context ``ctx``. Mutates ``st`` in place (potentials,
    mixing history, and the last-iteration outputs ``_multi_finalize`` consumes) and returns
    True when the convergence gate fires (the caller breaks). ``iters``/``tol`` are the
    surrounding run's loop controls (the MT→fullpot staging cap and the convergence gate).

    A thin orchestrator over cohesive module-level helpers: ``_build_species`` /
    ``_build_iteration_radials`` (the potential-dependent and k-independent per-iteration
    builds), ``_solve_all_k`` (pass 1: every-k eigensolve), ``_occupations``,
    ``_accumulate_density`` (pass 2: BZ density + symmetry + spheres), then the
    Weinert/non-spherical-potential/joint-Anderson update and the convergence + state
    write-back inline below."""
    A, vol, nfft, r, dx = ctx.A, ctx.vol, ctx.nfft, ctx.r, ctx.dx
    keys, acart, nbands = ctx.keys, ctx.acart, ctx.nbands
    fullpot = ctx.fullpot
    R_by_key, rr_by_key, mask_by_key = ctx.R_by_key, ctx.rr_by_key, ctx.mask_by_key
    atom_orbits = ctx.atom_orbits
    theta_i, kerker_K, verbose = ctx.theta_i, ctx.kerker_K, ctx.verbose
    v_by_key, recorder = st.v_by_key, st.recorder
    mt_phase, conv, hist = st.mt_phase, st.conv, st.hist
    v_nsph, vns_new, r_nsph = st.v_nsph, st.vns_new, st.r_nsph
    warp_state, v_grid_prev, v_i0_prev = st.warp_state, st.v_grid_prev, st.v_i0_prev

    # The plain previous-span projection was blind to states entering the window while the
    # aspherical potential ramps (observed blow-ups at [subsp] iterations), so fullpot
    # iterations used to force exact solves. The AUGMENTED span (previous eigenvectors +
    # their current residuals, solve_geneig_subspace_aug) carries the first-order rotation,
    # so fullpot iterations may reuse too; every 5th iteration stays a forced exact solve
    # and convergence is only ever accepted on an exact iteration (`done` below).
    full_iter = (not ctx.subspace_reuse) or (it % 5 == 0)
    t_it = time.time()
    r_sph = 0.0
    vnew_by_key = {}

    # pass 1 — solve every k, keep the results; pass 2 — occupy and accumulate the density.
    species, El_by_key, vmt_by_key, v0_by_key = _build_species(ctx, st)
    nb_solve, chan, lodat, nsph_int, us_by_key = _build_iteration_radials(
        ctx, species, El_by_key, vmt_by_key, v0_by_key, v_nsph)
    kdata, ev_all, ev_gamma = _solve_all_k(
        ctx, st, species, chan, lodat, nsph_int, nb_solve, v_nsph, warp_state, full_iter)
    e_fermi, occ_by_k = _occupations(ctx, ev_all)
    rho_I, rho_2m, spheres, rho_sph_by_key, sym_dev, lset_pot = _accumulate_density(
        ctx, st, kdata, occ_by_k, El_by_key, vmt_by_key, us_by_key, lodat, nb_solve)

    v_sph_list, v_i0, v_grid, v_hart, qmt_by_sphere = _weinert_multi(rho_I, spheres, A, nfft)
    v_i0_prev = v_i0

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
            # harmonic whose value at R matches the EXTERNAL part of the interstitial boundary,
            # C_LM = (v_bc − 4πE2 q^MT_LM/((2L+1)R^{L+1}))/R^L — Elk zpotcoul's homogeneous
            # solution z1 = (zlm − zvclmt(R))/R^l. The analytic subtraction removes the ENTIRE
            # own-region field (true-moment pseudocharge + the fictitious ρ_I continuation
            # inside the own sphere, which together carry exactly q^MT), so C_LM is the others'
            # field alone. Projected from the HARTREE-ONLY grid: external XC does not continue
            # into the sphere (XC is local; the sphere's own aspherical XC is already in vns).
            v_bc = interstitial_boundary_multi(v_hart, acart[ai][0], R, A, lset_pot)
            qmt = qmt_by_sphere[ai]
            for (bl, m) in lset_pot:
                own = (4 * math.pi * E2 / (2 * bl + 1)) * qmt.get((bl, m), 0.0) / R ** (bl + 1)
                c_lm = (v_bc[(bl, m)] - own) / R**bl
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
    vi_new = v_grid.reshape(-1)[ti_flat]
    if kerker_K is not None and v_grid_prev is not None:
        vi_new = _kerker_screened_interstitial(vi_new, v_grid_prev.reshape(-1)[ti_flat],
                                               ti_flat, kerker_K, nfft)
    news.append(torch.from_numpy(vi_new))
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
    done = conv is not None and full_iter and nsph_ok and d_span < tol and r_sph < 0.1

    # write back the evolved state (rebound locals) + the last-iteration outputs
    st.mt_phase, st.conv, st.hist = mt_phase, new, hist
    st.v_nsph, st.vns_new, st.r_nsph = v_nsph, vns_new, r_nsph
    st.warp_state, st.v_grid_prev, st.v_i0_prev = warp_state, v_grid_prev, v_i0_prev
    st.kdata, st.occ_by_k, st.lodat = kdata, occ_by_k, lodat
    st.El_by_key, st.vmt_by_key, st.nb_solve = El_by_key, vmt_by_key, nb_solve
    st.rho_I, st.spheres, st.rho_2m = rho_I, spheres, rho_2m
    st.v_grid, st.v_hart, st.qmt = v_grid, v_hart, qmt_by_sphere
    st.sym_dev = sym_dev if atom_orbits is not None else None
    return done


def _full_state(ctx: _MultiCtx, st: _MultiState) -> dict[str, Any]:
    """The complete fixed-point state dict (``info["state"]``; pass back as
    ``v_start={"__full_state__": ...}``)."""
    return {
        "v_by_key": {k: st.v_by_key[k].numpy().copy() for k in ctx.keys},
        "v_nsph": None if st.v_nsph is None else
            {k: {lm: np.asarray(v).copy() for lm, v in d.items()}
             for k, d in st.v_nsph.items()},
        "v_grid": None if st.v_grid_prev is None else st.v_grid_prev.copy(),
        "v_i0": st.v_i0_prev,
    }


def _multi_finalize(ctx: _MultiCtx, st: _MultiState, efg: bool = False):
    """Build ``crystal_scf_multi``'s ``(bands, info)`` return from the final state — the
    result summary, the full fixed-point state, and (opt-in) the EFG diagnostic pass."""
    _shutdown_ctx_pool(ctx)                    # release the persistent k-worker pool
    conv = st.conv
    info = {"nbands": ctx.nbands, "symbols": ctx.syms,
            "e_fermi": cast("dict[str, Any]", conv).get("e_fermi"),
            "v_by_key": {k: st.v_by_key[k].numpy().copy() for k in ctx.keys}}
    # complete fixed-point state (pass back as v_start={"__full_state__": ...})
    info["state"] = _full_state(ctx, st)
    info["recorder"] = st.recorder
    if ctx.atom_orbits is not None:
        # relative asymmetry of the RAW final-iteration density across symmetry-equivalent atoms
        # (and, for fullpot, the star-unfolding projector residual). Near convergence this should
        # be small; an order-unity value means the wedge sum sits in a symmetry-broken state —
        # multistable SCF basin or a symmetry bug. Surfaced so it can never fail silently.
        info["symmetry_dev"] = st.sym_dev
    if efg:
        rho_2m, v_hart, qmt = st.rho_2m, st.v_hart, st.qmt
        if not ctx.fullpot or st.mt_phase:
            # Build the aspherical density once from the converged wavefunctions. On an IBZ run the
            # reduced k-sum misses the star of each k, so the l=2 multipoles are star-unfolded with
            # the Wigner-D group average (_symmetrize_rho_lm) — no full-mesh re-solve needed. Then
            # recompute the interstitial potential with the l=2 Weinert pseudocharge (lattice term).
            rho_2m = _efg_density_pass(st.kdata, st.occ_by_k, ctx.keys, ctx.acart,
                                       st.El_by_key, st.vmt_by_key, ctx.R_by_key,
                                       ctx.rr_by_key, ctx.lmax, ctx.r, ctx.dx, st.nb_solve,
                                       lodat=st.lodat)
            if ctx.d_ops is not None:
                rho_2m = _symmetrize_rho_lm(rho_2m, ctx.keys, ctx.sg, ctx.d_ops)
            spheres = cast("list[dict[str, Any]]", st.spheres)
            for ai, k in enumerate(ctx.keys):
                spheres[ai]["rho_2m"] = rho_2m[k]
            _, _, _, v_hart, qmt = _weinert_multi(st.rho_I, st.spheres, ctx.A, ctx.nfft)
        info["efg"] = _efg_from_multipoles(rho_2m, v_hart, ctx.acart, ctx.keys, ctx.R_by_key,
                                           ctx.rr_by_key, ctx.dx, ctx.A, qmt_by_sphere=qmt)
    # Initial-state core levels: re-solve each site's core states in its converged
    # spherical MT potential and keep the eigenvalues (the SCF already does this
    # solve to build ρ_core, then discards them). Cheap; always on. Only the
    # within-cell SHIFT is physical — see flapw.core_levels.
    from gradwave.flapw.core_levels import core_level_shifts, core_levels_from_state

    cl = core_levels_from_state(st.v_by_key, ctx.syms, ctx.keys, ctx.core_map,
                                ctx.r, ctx.dx)
    info["core_levels"] = cl
    info["core_level_shifts"] = core_level_shifts(cl)
    return conv, info


def crystal_scf_multi(a_bohr=None, atoms=None, radii=None, ecut: float = 200.0, lmax: int = 2,
                      iters: int = 40, tol: float = 3e-3, kmesh=(1, 1, 1), smearing: float = 0.0,
                      efg: bool = False, fullpot: bool = False, use_symmetry: bool = True,
                      fullpot_lmax: int = 2, los=None, val_e=None, core=None, el_override=None,
                      v_start=None, kworkers: int = 1, subspace_reuse: bool = False,
                      subspace_tol: float = 1e-4, cell=None, kerker: float | None = None,
                      shift_invert: bool | str = "auto", lo_cond_tol: float = _LO_COND_TOL,
                      verbose: bool = False):
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
    ``spec`` may instead be a ``{"e": <label-or-eV>, "confine": False}`` mapping, which builds an
    unconfined **HELO** (high-energy local orbital): ``φ = a·u(E_l)+c·u(E₂)`` with ``φ(R)=0`` only
    (slope free) at a high ``E₂``. Where the confined LO is near-redundant with the valence span
    (e.g. the l=1 p-channel of a small anion sphere), the HELO supplies the genuinely distinct
    in-sphere aspherical radial the on-site EFG density needs — it corrects the anion-undershoot /
    cation-overshoot on-site l=2 bias and gates at production damping (see
    ``experiments/autoapw/efg_helo_l1_fix.md``).
    When an LO carries semicore electrons, raise that species' valence count with
    ``val_e = {symbol: n}`` and drop the state from the frozen core with ``core = {symbol: [...]}``
    (both override the module defaults) — the LO does not change electron bookkeeping by itself.
    ``el_override = {symbol: {l: spec}}`` moves an energy parameter (e.g. Ti l=1 into the valence
    region once a 3p LO carries the semicore). ``lo_cond_tol`` (default ``_LO_COND_TOL``) is the
    overlap-conditioning threshold: an LO whose norm fraction orthogonal to its valence ``{u, u̇}``
    subspace falls below it is Löwdin-orthogonalized against that subspace before entering the
    secular problem, curing the near-linear-dependence that otherwise makes the extended overlap
    ``S`` singular (the shallow O 2s semicore LO); deep, well-separated LOs (Ti 3s/3p) are
    untouched. ``lo_cond_tol=0`` disables the conditioning entirely.

    ``v_start`` = ``{key: radial potential array}`` warm-starts the SCF from a previous run's
    converged spherical potentials (returned as ``info["v_by_key"]``) instead of the atomic ones —
    a convergence-sweep point then starts nearly converged (~2-3x fewer iterations).
    ``kerker`` (Å⁻¹, default None=off) screens the interstitial residual by G²/(G²+kerker²)
    inside the joint Anderson — the measured cure for the long-wavelength interstitial
    sloshing instability of hard fullpot configs (see the DMD note at kerker_K below).

    Alternatively ``v_start = {"__full_state__": info["state"]}`` restores the COMPLETE
    fixed-point state of a fullpot run — spherical + aspherical potentials, interstitial grid
    and reference — so the resumed run continues the coupled fullpot iteration exactly (the
    plain per-key form silently restarts the aspherical channel from zero). Every run returns
    its final full state as ``info["state"]``.

    ``kworkers > 1`` solves the k-points of each iteration in a process pool (each k's secular
    build+solve is independent). Results are accumulated in mesh order, so the density is identical
    to the serial run. Divide the BLAS threads accordingly (e.g. ``OMP_NUM_THREADS=3`` with
    ``kworkers=6`` on ~20 cores) — the pool multiplies whatever thread count each worker uses.
    The workers are spawned (fork inherits live BLAS state and computes wrong numbers), so the
    CALLING script must be import-safe: guard its executable body with ``if __name__ ==
    "__main__":`` or the workers re-run it on import.

    ``subspace_reuse`` (default False — exact solves keep the fixed-point map F deterministic
    for the Newton machinery, see ``flapw.newton``) solves most iterations by AUGMENTED
    Rayleigh-Ritz in span[previous eigenvectors, their current residuals] (residual-gated,
    exact-solve fallback), including fullpot iterations; every 5th iteration is a forced exact
    solve and convergence is only accepted on exact-solve iterations, so the reported state
    never rests on a projected solve. ``subspace_tol`` (eV) gates acceptance on the max
    eigenvalue-error bound ``||Hc−εSc||/||Sc||`` over the solved bands; the 1e-4 default sits
    10x under the degenerate-occupation tolerance (1e-3 eV), so a gated projected solve cannot
    flip an occupation decision.

    ``shift_invert`` (``True`` / ``False`` / ``"auto"``, default ``"auto"``) solves each warm
    iteration's secular problem by shift-invert Lanczos on the ``(H,S)`` pencil
    (``lapw.solve_geneig_shift_invert``): one ``LDL^H`` factorization of ``H−σS`` with σ just below
    the occupied window, reused as the OPinv of a deterministic ARPACK (implicitly-restarted)
    Lanczos on ``(H−σS)^{-1}S``, and a two-shift Sylvester-inertia certificate that forces a LOUD
    dense fallback on any incompleteness (a missed Γ-point multiplet, a mis-placed σ). σ is warmed
    from the previous iteration's eigenvalues; the first iteration and any cold/degenerate failure
    fall back to the exact dense solve, so the reported spectrum is never a silently-incomplete
    window. ``"auto"`` engages it per-solve only when the pencil dim exceeds the measured crossover
    ``_SHIFT_INVERT_CROSSOVER_DIM`` (below it dense eigh is faster); ``True`` always attempts it
    (certificate-guarded), ``False`` never (exact dense only). Determinism-required contexts keep
    exact solves: Newton's F forces ``shift_invert=False`` (``flapw.newton``), and the pool
    bit-equality path constructs its solve with the ``False`` default. Measured per-solve win grows
    with the basis: ~2.3×/2.9× at pencil dim 737/1559 (``experiments/autoapw/si_ab.py``).

    Internally this is ``_multi_setup`` (state-independent context) + ``_multi_init_state`` +
    a loop over ``_multi_iterate`` (the fixed-point map) + ``_multi_finalize`` — split so the
    Newton polish (``flapw.newton``) can pay the setup once and call the map directly."""
    ctx = _multi_setup(a_bohr=a_bohr, atoms=atoms, radii=radii, ecut=ecut, lmax=lmax,
                       kmesh=kmesh, smearing=smearing, fullpot=fullpot,
                       use_symmetry=use_symmetry, fullpot_lmax=fullpot_lmax, los=los,
                       val_e=val_e, core=core, el_override=el_override, kworkers=kworkers,
                       subspace_reuse=subspace_reuse, subspace_tol=subspace_tol, cell=cell,
                       kerker=kerker, shift_invert=shift_invert, lo_cond_tol=lo_cond_tol,
                       verbose=verbose)
    st = _multi_init_state(ctx, v_start)
    for it in range(iters):
        if _multi_iterate(ctx, st, it, iters=iters, tol=tol):
            break
    return _multi_finalize(ctx, st, efg=efg)


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


def _kerker_screen(nfft: int, cell: np.ndarray, k0: float) -> np.ndarray:
    """Interstitial Kerker screen K(G) = G²/(G²+k0²) on the (nfft,)*3 grid, floored at 0.1.

    Opt-in via ``crystal_scf_multi(kerker=k0)``. DMD mode analysis of the hard TiO2
    config (experiments/autoapw/mode_analysis.py, asus 2026-08-19) measured the bare
    map's dominant instability as a complex pair at |rho|=1.02 living 94% in the
    interstitial channel — classic long-wavelength sloshing. K(G) damps exactly that
    mode inside the joint Anderson (preconditioned residual, same fixed point). The
    0.1 floor keeps the interstitial MEAN mobile (the G=0 component is a real dof
    here, unlike a charge-neutral density mix).
    """
    fr = np.fft.fftfreq(nfft) * nfft
    mm = np.meshgrid(fr, fr, fr, indexing="ij")
    gmat = 2.0 * np.pi * np.linalg.inv(cell).T
    g2 = sum((gmat[i, 0] * mm[0] + gmat[i, 1] * mm[1] + gmat[i, 2] * mm[2]) ** 2
             for i in range(3))
    return np.maximum(g2 / (g2 + float(k0) ** 2), 0.1)


def _kerker_screened_interstitial(vi_new: np.ndarray, vi_prev: np.ndarray,
                                  ti_flat: np.ndarray, kerker_K: np.ndarray,
                                  nfft: int) -> np.ndarray:
    """Screen the interstitial residual vi_new - vi_prev in G-space (see _kerker_screen).

    Embedding the masked residual with zeros inside the spheres is the standard
    approximation — the screen only needs to damp the long-wavelength component,
    which the mask barely perturbs.
    """
    r_full = np.zeros(nfft * nfft * nfft)
    r_full[np.flatnonzero(ti_flat)] = vi_new - vi_prev
    r_scr = np.fft.ifftn(kerker_K * np.fft.fftn(r_full.reshape(nfft, nfft, nfft))).real
    return vi_prev + r_scr.reshape(-1)[ti_flat]


def _fft_resample_cube(v: np.ndarray, n_new: int) -> np.ndarray:
    """Band-limited resample of a periodic cube field ``(n_old,)*3 → (n_new,)*3`` by copying the
    common signed-frequency block in G-space (exact for band-limited fields when upsampling;
    Nyquist truncation when downsampling — fine for a warm-start seed, which mixing corrects)."""
    n_old = v.shape[0]
    if n_old == n_new:
        return v
    f_old = np.fft.fftn(v) / n_old**3
    n = min(n_old, n_new)
    ix = np.fft.fftfreq(n, 1.0 / n).astype(int)           # signed frequencies of the smaller cube
    sel = np.ix_(ix, ix, ix)                              # negative indices wrap in both cubes
    out = np.zeros((n_new,) * 3, dtype=complex)
    out[sel] = f_old[sel]
    return np.fft.ifftn(out).real * n_new**3


def _restore_full_state(fs: dict[str, Any], theta_i: np.ndarray, nfft: int):
    """Unpack a ``v_start={"__full_state__": ...}`` resume into loop state.

    Returns ``(v_nsph, v_grid_prev, warp_state, v_i0_prev)``, each None where the
    saved state has no such component. The warp is rebuilt from the restored grid
    (zero-mean over Θ_I, the same convention as the in-loop update).

    The saved ``v_grid`` carries the *saving* run's nfft, which need not match this run's
    (the pseudocharge-resolvability rule raises nfft with ``fullpot_lmax``, e.g. 28³→32³ at
    fp_lmax=6) — resample it band-limited to the current grid instead of letting the Θ_I mask
    fail on the shape mismatch."""
    v_nsph = None
    if fs.get("v_nsph"):
        v_nsph = {k: {lm: np.asarray(v).astype(complex).copy() for lm, v in d.items()}
                  for k, d in fs["v_nsph"].items()}
    v_grid_prev, warp_state = None, None
    if fs.get("v_grid") is not None:
        v_grid_prev = np.asarray(fs["v_grid"], dtype=float).copy()
        if v_grid_prev.shape != (nfft,) * 3:
            v_grid_prev = _fft_resample_cube(v_grid_prev, nfft)
        vi_mean = float(v_grid_prev.reshape(-1)[theta_i.reshape(-1)].mean())
        u0 = np.where(theta_i, v_grid_prev - vi_mean, 0.0)
        warp_state = (np.fft.fftn(u0) / nfft**3, nfft)
    return v_nsph, v_grid_prev, warp_state, fs.get("v_i0")


class SolveKArgs(NamedTuple):
    """Typed payload for the k-point process pool. A NamedTuple, not a bare
    tuple: this crossed the spawn boundary as a positional 15-tuple once and
    silently broke its test when the 16th element (warp) was added — keyword
    construction makes any future drift a loud TypeError at every call site."""

    kf: np.ndarray | tuple[float, float, float]
    cell: np.ndarray | float                  # lattice rows, or the cubic edge (Å)
    acart: list[tuple[np.ndarray, str]]
    species: dict[str, dict[str, Any]]
    lmax: int
    ecut: float
    r: torch.Tensor
    dx: float
    nb_solve: int
    v_nsph: dict[Any, Any] | None
    chan: dict[Any, Any] | None
    lodat: dict[Any, Any] | None
    nsph_int: dict[Any, Any] | None
    c_prev: np.ndarray | None                 # previous eigvecs (subspace reuse / SI seed)
    subspace_tol: float
    warp: tuple[Any, int] | None              # (FFT of (v_grid-v_i0)·Θ_I, nfft)
    shift_invert: bool | str = False          # True / False / "auto" (default False = exact dense)
    sigma: float | None = None                # shift-invert shift (below the window bottom)


def _solve_k_args(args: SolveKArgs):
    """Picklable per-k secular solve for the k-point process pool (``kworkers``)."""
    a = args
    return _lapw_multi_k(a.kf, a.cell, a.acart, a.species, a.lmax, a.ecut, a.r, a.dx,
                         a.nb_solve, v_nsph=a.v_nsph, chan=a.chan, lodat=a.lodat,
                         nsph_int=a.nsph_int, c_prev=a.c_prev,
                         subspace_tol=a.subspace_tol, warp=a.warp,
                         shift_invert=a.shift_invert, sigma=a.sigma)


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
            rlm = sphere_density_multipoles_bands(amps_all, occl, us_by_key[k],
                                                  rr_by_key[k], lmax, lset2)
            for lm in lset2:
                rho_2m[k][lm] += w * rlm[lm]
    return rho_2m


def _efg_from_multipoles(rho_by_key, v_hart, acart, keys, R_by_key, rr_by_key, dx, A,
                         qmt_by_sphere=None):
    """Per-atom EFG from the converged BZ-averaged aspherical density (``rho_by_key`` = the l=2 (and
    l=0) sphere-density multipoles accumulated over the k-mesh with the SCF occupations, i.e. the
    same density that produced the potential — not a fresh Γ solve at occ=2). The lattice term is
    the l=2 surface projection of the COULOMB-only interstitial grid minus the analytic own-sphere
    multipole field ``4πE2 q^MT_2M/(5R³)`` (the exact Elk-style own-field subtraction — the surface
    projection contains the own pseudocharge + the fictitious in-sphere ρ_I continuation, which
    together carry exactly ``q^MT``). Returns the valence V_zz (on-site l=2 sphere Poisson), the
    full V_zz adding the external boundary term, the asymmetry η, the tensor, and the in-sphere
    valence charge."""
    from gradwave.flapw.efg import (
        efg_tensor,
        efg_tensor_full,
        interstitial_l2_boundary,
        valence_efg_moments,
    )
    out = {}
    for ai, k in enumerate(keys):
        rho, rr, drw = rho_by_key[k], rr_by_key[k], rr_by_key[k] * dx
        R = R_by_key[k]
        q, q2 = valence_efg_moments(rho, rr, drw)
        _, v_zz_val, eta_val = efg_tensor(rho, rr, drw)
        v_bc = interstitial_l2_boundary(v_hart, acart[ai][0], R, A)
        qmt = qmt_by_sphere[ai] if qmt_by_sphere else {}
        v_bc = {m: v_bc[m] - (4 * math.pi * E2 / 5.0) * qmt.get((2, m), 0.0) / R**3
                for m in range(-2, 3)}
        tensor, v_zz, eta = efg_tensor_full(rho, rr, drw, v_bc, R)
        charge = float(math.sqrt(4 * math.pi) * np.sum(rho[(0, 0)].real * rr**2 * drw))
        out[k] = {"Q2": q2, "V_zz": v_zz, "eta": eta, "V_zz_valence": v_zz_val,
                  "eta_valence": eta_val, "tensor": tensor, "sphere_charge": charge,
                  "q_moments": {m: complex(v) for m, v in q.items()}}
    return out
