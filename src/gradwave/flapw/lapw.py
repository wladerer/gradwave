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


def enumerate_kg(kfrac, B, ecut):
    """The plane-wave basis at wavevector k: Miller indices + Cartesian ``k+G`` with
    ``(hbar^2/2m)|k+G|^2 <= ecut``, for reciprocal cell ``B`` (rows). The single k+G enumerator —
    this logic was quadruplicated across the matrix builders with an inconsistent search margin
    (+1 vs +2 shells); the wider margin is kept (never under-covers the ecut sphere)."""
    kf = np.asarray(kfrac, dtype=float)
    b_arr = np.asarray(B, dtype=float)
    bmin = float(np.linalg.norm(b_arr, axis=1).min())
    nmax = int(math.ceil(math.sqrt(ecut / HBAR2_2M) / bmin)) + 2
    mill, ks = [], []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for m in range(-nmax, nmax + 1):
                kg = (np.array([i, j, m]) + kf) @ b_arr
                if HBAR2_2M * (kg @ kg) <= ecut:
                    mill.append([i, j, m])
                    ks.append(kg)
    return np.array(mill), np.array(ks)


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
    return _finish_channel(l, El, r_np, inside, rr, drw, dx, v, R, u, udot)


def radial_channels_all(lmax, El_by_l, r, dx, v, R):
    """Every l-channel of one sphere in ONE batched Numerov call — ``{l: radial_channel dict}``.

    The (lmax+1)·3 rows (each channel's E, E±h) share the mesh and the potential, so they run in a
    single recurrence loop; the per-row arithmetic is elementwise, so the result is bit-identical to
    looping ``radial_channel`` over l. This is the per-iteration builder for the SCF's ``chan``
    (the per-l Numerov loops were ~30% of a production fullpot iteration)."""
    r_np = r.detach().numpy()
    inside = r_np <= R
    rr = r_np[inside]
    drw = rr * dx
    n_cut = int(np.searchsorted(r_np, 2.0 * R)) + 5
    ls, es, hes = [], [], {}
    for lang in range(lmax + 1):
        el = El_by_l[lang]
        he = max(abs(el) * 1e-4, 1e-3)
        hes[lang] = he
        ls += [lang, lang, lang]
        es += [el, el + he, el - he]
    uraw = numerov_log_np(np.array(ls), np.array(es), r, dx, v, n_cut=n_cut)
    un = uraw / np.sqrt((uraw[:, inside] ** 2 * drw).sum(axis=1))[:, None]
    out = {}
    for lang in range(lmax + 1):
        u = un[3 * lang]
        udot = (un[3 * lang + 1] - un[3 * lang + 2]) / (2 * hes[lang])
        out[lang] = _finish_channel(lang, El_by_l[lang], r_np, inside, rr, drw, dx, v, R, u, udot)
    return out


def _finish_channel(l, El, r_np, inside, rr, drw, dx, v, R, u, udot):
    """The post-Numerov tail of ``radial_channel``: boundary value/slope, overlaps, weak-form
    kinetic and potential integrals from the normalized ``u``/``u̇`` mesh arrays. Also carries the
    in-sphere radial arrays (``u_in``/``ud_in``) so downstream consumers (sphere density,
    aspherical integrals, local orbitals) reuse them instead of re-running Numerov."""
    def val_slope(f):
        idx = np.sort(np.argsort(np.abs(r_np - R))[:7])
        c = np.polyfit(r_np[idx] - R, f[idx], 3)
        return float(c[-1]), float(c[-2])

    uR, upR = val_slope(u)
    udR, udpR = val_slope(udot)
    ui, udi = u[inside], udot[inside]
    v_in = (v.detach().numpy() if hasattr(v, "detach") else np.asarray(v))[inside]
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
            "Vudud": float(pot("ud", "ud")), "El": El, "l": l,
            "u_in": ui, "ud_in": udi}


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


def match_ab_vec(ch, q, R):
    """Value+slope match over a whole array of ``q=|k+G|`` at once — the vectorized ``match_ab``.
    One batched ``spherical_jn`` per l replaces one scalar scipy call per plane wave (the LAPW
    build's hot loop). Returns ``(a, b)`` arrays. Bit-identical to looping ``match_ab``."""
    from scipy.special import spherical_jn
    l = ch["l"]
    q = np.asarray(q, dtype=float)
    x = q * R
    jl = spherical_jn(l, x)
    djl = -spherical_jn(1, x) if l == 0 else (spherical_jn(l - 1, x) - (l + 1) / x * jl)
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
    _, ks = enumerate_kg(kfrac, b * np.eye(3), ecut)
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
        a, bb = match_ab_vec(ch, ksafe, R)
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
    _, ks = enumerate_kg(kfrac, b * np.eye(3), ecut)
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
            a, bb = match_ab_vec(ch, ksafe, R)
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
    from scipy.linalg import eigh as scipy_eigh
    w, u = np.linalg.eigh(S)
    return _canonical_solve(H, S, w, u, nbands, with_vecs, tol, scipy_eigh)


def _canonical_solve(H, S, w, u, nbands, with_vecs, tol, scipy_eigh):
    keep = w > tol * w[-1]                                # w ascending; w[-1] = largest
    x = u[:, keep] * (w[keep] ** -0.5)                   # npw × nkeep, S-orthonormal columns
    m = x.conj().T @ H @ x
    m = 0.5 * (m + m.conj().T)
    rank = m.shape[0]
    nb = min(nbands, rank)
    # only the lowest nbands eigenpairs are consumed — the MRRR subset solver skips the rest
    # (the S diagonalization above stays full: the null-space detection needs the whole spectrum)
    ea, va = scipy_eigh(m, subset_by_index=(0, nb - 1))
    evals = ea.real
    if len(evals) < nbands:                              # rank-deficient: pad empty (high) states
        evals = np.concatenate([evals, np.full(nbands - len(evals), 1e10)])
    if with_vecs:
        vecs = x @ va
        if vecs.shape[1] < nbands:
            vecs = np.concatenate([vecs, np.zeros((x.shape[0], nbands - vecs.shape[1]),
                                                  dtype=vecs.dtype)], axis=1)
        return evals, vecs
    return evals


def solve_geneig_subspace_aug(H, S, c_prev, nbands, tol=1e-8):
    """Augmented Rayleigh-Ritz solve of ``H c = ε S c`` in ``span[c_prev, R]`` where ``R`` is the
    residual block of last iteration's eigenvectors under the CURRENT pencil.

    ``solve_geneig_subspace`` projects into the previous span alone, which is blind to the
    first-order rotation of the eigenvectors when the potential moves (the observed fullpot-ramp
    blow-ups); one Davidson-style augmentation captures exactly that direction: Ritz-solve in
    ``span[c_prev]``, form ``R = H c − S c ε`` for the Ritz pairs, and re-solve in the doubled
    span. Cost stays a handful of ``dim × dim × nkeep`` GEMMs + an ``O((2·nkeep)³)`` dense solve —
    far below the ``O(dim³)`` full diagonalization for ``nkeep ≪ dim``.

    Returns ``(evals, vecs, resid)`` with ``resid`` the max eV-scale residual
    (``_pencil_resid``, a first-order bound on the eigenvalue error) over the ``nbands`` kept
    states — same acceptance contract as ``solve_geneig_subspace`` (the caller gates on it and
    falls back to the exact dense solve)."""
    hp = H @ c_prev
    sp = S @ c_prev
    hs = c_prev.conj().T @ hp
    ss = c_prev.conj().T @ sp
    hs = 0.5 * (hs + hs.conj().T)
    ss = 0.5 * (ss + ss.conj().T)
    nk = c_prev.shape[1]
    ev0, y0 = solve_geneig(hs, ss, nk, with_vecs=True, tol=tol)
    m = min(nk, len(ev0))
    resid_blk = (hp @ y0[:, :m]) - (sp @ y0[:, :m]) * ev0[None, :m]
    nrm = np.linalg.norm(resid_blk, axis=0)
    keep = nrm > 1e-14 * max(float(nrm.max()), 1e-300)     # drop numerically-null residuals
    q = np.concatenate([c_prev, resid_blk[:, keep] / nrm[keep]], axis=1)
    hq = H @ q
    sq = S @ q
    hqq = q.conj().T @ hq
    sqq = q.conj().T @ sq
    hqq = 0.5 * (hqq + hqq.conj().T)
    sqq = 0.5 * (sqq + sqq.conj().T)
    ev, y = solve_geneig(hqq, sqq, nbands, with_vecs=True, tol=tol)
    vecs = q @ y
    hv = hq @ y
    sv = sq @ y
    return ev, vecs, _pencil_resid(hv, sv, ev)


def _pencil_resid(hv, sv, ev):
    """Max eV-scale residual ``||H c − ε S c|| / ||S c||`` over the kept states. Normalizing by
    ``||S c||`` (not ``||H c||``) keeps the metric meaningful for eigenvalues near zero — FLAPW
    eigenvalues are referenced to the interstitial zero, so bands DO cross ε≈0 and a
    ``||r||/||H c||`` metric reports O(1) "relative residual" on them regardless of accuracy
    (measured on a production TiO2 pencil: 0.97 at a 3.5e-3 eV true eigenvalue error — the old
    gate could never engage). For S-normalized eigenvectors this bounds the eigenvalue error
    (eV) to first order."""
    r = hv - sv * ev[None, :]
    scale = np.maximum(np.linalg.norm(sv, axis=0), 1e-12)
    return float((np.linalg.norm(r, axis=0) / scale).max())


def solve_geneig_subspace(H, S, c_prev, nbands, tol=1e-8):
    """Rayleigh-Ritz solve of ``H c = ε S c`` inside the span of a previous iteration's
    eigenvectors ``c_prev`` (dim × nkeep, nkeep ≥ nbands). Near SCF self-consistency the
    eigenvectors barely rotate between iterations, so projecting into last iteration's subspace
    (two thin GEMMs + an nkeep×nkeep dense solve) replaces the O(dim³) full diagonalization.

    Returns ``(evals, vecs, resid)`` where ``resid`` is the max eV-scale residual
    (``_pencil_resid``) over the ``nbands`` kept states — the caller accepts the step only when
    ``resid`` is below its threshold and falls back to the exact solve otherwise, so a drifting
    subspace (band crossings, early iterations, large mixing steps) can never silently corrupt
    the result."""
    hp = H @ c_prev
    sp = S @ c_prev
    hs = c_prev.conj().T @ hp
    ss = c_prev.conj().T @ sp
    hs = 0.5 * (hs + hs.conj().T)
    ss = 0.5 * (ss + ss.conj().T)
    ev, y = solve_geneig(hs, ss, nbands, with_vecs=True, tol=tol)
    vecs = c_prev @ y
    hv = hp @ y
    sv = sp @ y
    return ev, vecs, _pencil_resid(hv, sv, ev)


# ---------------------------------------------------------------------------
# Shift-invert Lanczos secular path (OPT-IN; exact dense stays the default).
# ---------------------------------------------------------------------------


def _inertia_below(mmat):
    """Sylvester inertia of the Hermitian shift matrix ``M = H − σS``: the number of NEGATIVE
    eigenvalues of ``M``, which by Sylvester's law of inertia (S is SPD) equals the number of
    generalized eigenvalues of the pencil ``(H, S)`` below σ. Computed from the block-diagonal ``D``
    of the Bunch-Kaufman ``LDL^H`` factorization (``scipy.linalg.ldl`` → LAPACK ``hetrf``): a
    single ``O(dim³/3)`` factorization, and only the 1×1/2×2 pivot blocks' signs are inspected."""
    from scipy.linalg import ldl
    _lu, d, _perm = ldl(mmat, hermitian=True)
    n, neg, i = d.shape[0], 0, 0
    while i < n:
        if i + 1 < n and abs(d[i + 1, i]) > 0.0:              # 2×2 pivot block
            neg += int((np.linalg.eigvalsh(d[i:i + 2, i:i + 2]) < 0).sum())
            i += 2
        else:                                                 # 1×1 pivot
            neg += int(d[i, i].real < 0.0)
            i += 1
    return neg


def solve_geneig_shift_invert(hmat, smat, nbands, sigma, c_prev=None, buffer=None, maxiter=None):
    """Shift-invert Lanczos generalized eigensolve for the lowest ``nbands`` of ``H c = ε S c``
    (OPT-IN; the exact dense ``solve_geneig`` stays the SCF default). One ``LDL^H`` factorization of
    ``M = H − σS`` (``_shift_factor``) is reused BOTH as the ARPACK shift-invert operator
    ``(H−σS)^{-1}`` (implicitly-restarted Lanczos on ``M^{-1}S`` in the S-inner product,
    ``scipy.sparse.linalg.eigsh``, deterministic under a fixed start vector) AND for the Sylvester
    inertia of the completeness certificate.

    ``sigma`` is placed just BELOW the occupied+buffer window (the caller warms it from the previous
    iteration's lowest eigenvalue, or an atomic estimate cold) — the measured-winning shift
    (``pencil_bench``: 2.5× at dim 737, 5.4× at 1559; ARPACK converges the lowest bands, the
    largest ``|1/(ε−σ)|``, first). The **Sylvester-inertia completeness certificate** guards a LOUD
    dense fallback: one factorization at a midgap ``σ_hi`` in the window's top gap gives the inertia
    ``n_hi`` = number of pencil eigenvalues below σ_hi; requiring ``n_hi == nbands`` (with all
    returned eigenvalues below σ_hi and a real gap) certifies the returned set is *exactly* the
    lowest ``nbands`` — a missed Γ multiplet copy (ARPACK returns fewer than ``nbands`` below σ_hi ⇒
    ``n_hi > nbands``) or a set that isn't the bottom (``n_hi ≠ nbands``) breaks it, and it returns
    ``None``. ARPACK is deterministic under the fixed start vector ``v0``.

    ``c_prev`` (dim×nb) seeds a deterministic start vector rich in the occupied subspace. Returns
    ``(evals, vecs)`` matching ``solve_geneig(..., with_vecs=True)``, or ``None`` on any
    certificate/robustness failure."""
    from scipy.sparse.linalg import eigsh
    dim = hmat.shape[0]
    nb = min(nbands, dim)
    if buffer is None:
        buffer = max(8, nb // 4)
    k = min(nb + buffer, dim - 2)
    if dim < 48 or k >= dim - 1:                          # no headroom → let the caller use dense
        return None
    if c_prev is not None and c_prev.shape[0] == dim and c_prev.shape[1] > 0:
        v0 = c_prev.sum(axis=1).astype(complex)          # deterministic warm start
    else:
        v0 = np.ones(dim, dtype=complex)
    try:                                                 # ARPACK shift-invert: its own fast dense LU
        w, vecs = eigsh(hmat, k=k, M=smat, sigma=sigma,  # of H−σS internally; which='LM' ⇒ near σ
                        which="LM", v0=v0, tol=0.0, maxiter=maxiter or dim * 20)
    except Exception:                                    # ARPACK non-convergence → dense fallback
        return None
    order = np.argsort(w.real)
    eps_ext, vecs = w.real[order], vecs[:, order]
    if len(eps_ext) <= nb or not np.isfinite(eps_ext).all():
        return None
    # --- Sylvester-inertia completeness certificate (one factorization) ---
    gap = eps_ext[nb] - eps_ext[nb - 1]
    if gap <= 1e-9 * max(1.0, abs(eps_ext[nb - 1])):     # window top not isolated → cannot certify
        return None
    sigma_hi = 0.5 * (eps_ext[nb - 1] + eps_ext[nb])
    try:
        n_hi = _inertia_below(hmat - sigma_hi * smat)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if n_hi != nb:                                       # exactly nb eigenvalues below σ_hi ⇒ the
        return None                                      # returned nb (all < σ_hi) are the lowest
    return eps_ext[:nb], vecs[:, :nb]                    # vecs already S-orthonormal (ARPACK)
