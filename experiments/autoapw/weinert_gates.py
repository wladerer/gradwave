"""Numerical gates for the exact G-space Weinert chain (D2 own-field + D4 high-L near-field).

A) pseudocharge moment exactness (Bessel-measured vs prescribed) vs npow, nfft, L
B) isolated-sphere boundary null (the D2 gate): C_LM = (v_bc - analytic own)/R^L must
   vanish for ALL L for a lone sphere — the old numerical own-pseudocharge subtraction
   left ~20% and retained the in-sphere rho_I continuation entirely.
C) near-touching rutile pair (the D4 gate): receiver-surface field vs the analytic
   image-summed multipole potential at the 0.024 A O|Ti gap geometry.

    uv run python experiments/autoapw/weinert_gates.py [A|B|C ...]
"""
import math
import sys

import numpy as np

from gradwave.constants import E2
from gradwave.flapw.coulomb import (
    cell_matrix,
    gvec_ylm_tables,
    sphere_interstitial_moments,
    sphere_pseudocharge_ft,
)
from gradwave.flapw.efg import _angular_grid, interstitial_boundary_multi
from gradwave.flapw.radial import log_mesh
from gradwave.flapw.scf import _weinert_multi

CELL = 10.0  # A cubic


def sphere(R, tau, moments, r_np, dx, z=0.0):
    rr = r_np[r_np <= R]
    rho_sph = np.exp(-((rr / 0.25) ** 2))
    drw = rr * dx
    q = float(np.sum(4 * math.pi * rho_sph * rr**2 * drw))
    rho_sph *= z / q if z else 0.0  # neutral: electrons = Z (or empty)
    rho_2m = {(0, 0): np.zeros_like(rr, dtype=complex)}
    for (L, M), qlm in moments.items():
        shape = rr**L * np.exp(-((rr / (0.35 * R)) ** 2))
        shape /= np.sum(shape * rr ** (L + 2) * drw)
        rho_2m[(L, M)] = (qlm * shape).astype(complex)
    return {"tau": np.asarray(tau, float), "rr": rr, "dx": dx, "rho_sph": rho_sph,
            "Z": z, "R": R, "rho_2m": rho_2m}


def own_field(qlm, R):
    return {lm: (4 * math.pi * E2 / (2 * lm[0] + 1)) * q / R ** (lm[0] + 1)
            for lm, q in qlm.items()}


def bench_moments(nfft, R=0.824, lmax=6):
    """Pseudocharge truncated-series moments vs prescribed."""
    a = cell_matrix(CELL)
    vol = float(abs(np.linalg.det(a)))
    gvec, gnorm, ylm = gvec_ylm_tables(a, nfft, lmax)
    tau = np.array([CELL / 2] * 3)
    qlm = {(L, min(L, 2)): 1.0 + 0.0j for L in range(lmax + 1)}
    gmax = math.pi * nfft / CELL
    for npow in (3, 5, 8):
        ps = sphere_pseudocharge_ft(qlm, R, tau, vol, npow, gvec, gnorm, ylm)
        meas = sphere_interstitial_moments(ps, R, tau, gvec, gnorm, ylm, list(range(lmax + 1)))
        errs = {L: abs(meas[(L, min(L, 2))] - qlm[(L, min(L, 2))]) for L in range(lmax + 1)}
        print(f"  nfft={nfft} npow={npow} (gmax*R={gmax*R:.1f}): "
              + " ".join(f"L{L}={errs[L]:.1e}" for L in range(lmax + 1)), flush=True)


def hermitian_moments(lmax):
    """A real (physical) aspherical density's moments: q_{L,-M} = (-1)^M conj(q_{L,M})."""
    out = {}
    for L in range(1, lmax + 1):
        M = min(L, 2)
        q = 1.0 + (0.3j if M else 0.0)
        out[(L, M)] = q
        if M:
            out[(L, -M)] = (-1) ** M * np.conj(q)
    return out


def bench_null(nfft, R=0.824, lmax=6):
    """B1: lone sphere with real aspherical density, rho_I = 0 — the external boundary field
    (v_bc - analytic own multipole field) must vanish for ALL (L,M)."""
    r, dx = log_mesh(1e-5, 28.0, 2500)
    r_np = r.numpy()
    moments = hermitian_moments(lmax)
    sp = sphere(R, [CELL / 2] * 3, moments, r_np, dx, z=6.0)
    rho_I = np.zeros((nfft, nfft, nfft))
    _, _, _, v_hart, qmt = _weinert_multi(rho_I, [sp], CELL, nfft)
    lset = [(L, M) for L in range(lmax + 1) for M in range(-L, L + 1)]
    v_bc = interstitial_boundary_multi(v_hart, sp["tau"], R, cell_matrix(CELL), lset)
    ownf = own_field(qmt[0], R)
    gmax = math.pi * nfft / CELL
    print(f"  nfft={nfft} (gmax*R={gmax*R:.1f}, npow={int(np.clip(round(R*gmax/4), 2, 12))})",
          flush=True)
    for L in range(lmax + 1):
        M = min(L, 2) if L else 0
        own = ownf.get((L, M), 0.0)
        res = v_bc[(L, M)] - own
        ref = abs(own) if abs(own) > 1e-12 else 1.0
        print(f"    L={L} M={M}: own={abs(own):9.3e}  resid={abs(res):9.3e}  "
              f"rel={abs(res)/ref:9.2e}", flush=True)


def bench_rho_i(nfft, R=0.824, lmax=4):
    """B2 (the retained-rho_I D2 case): EMPTY sphere immersed in a plane-wave interstitial
    density. q^MT = 0, so the external boundary field must equal
    [analytic plane-wave surface projection] - [multipole field of the pw's in-sphere moments
    q^I] — the second term is exactly what the old chain retained (its D2 defect)."""
    from gradwave.flapw.coulomb import reciprocal
    r, dx = log_mesh(1e-5, 28.0, 2500)
    r_np = r.numpy()
    sp = sphere(R, [CELL / 2] * 3, {(L, 0): 0.0 for L in range(1, lmax + 1)}, r_np, dx, z=0.0)
    a = cell_matrix(CELL)
    b = reciprocal(a)
    g1 = b[0] + 2 * b[1] + b[2]
    amp = 0.05
    ax = np.arange(nfft) * (CELL / nfft)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts_all = np.stack([X, Y, Z], -1)
    rho_I = amp * np.cos(pts_all @ g1)
    _, _, _, v_hart, qmt = _weinert_multi(rho_I, [sp], CELL, nfft)
    lset = [(L, M) for L in range(lmax + 1) for M in range(-L, L + 1)]
    v_bc = interstitial_boundary_multi(v_hart, sp["tau"], R, cell_matrix(CELL), lset)
    # analytic total pw potential on the surface, projected
    from scipy.special import sph_harm_y
    th, ph, wgt = _angular_grid(24, 36)
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    pts = sp["tau"] + R * dirs.reshape(-1, 3)
    vpw = 4 * math.pi * E2 * amp * np.cos(pts @ g1) / (g1 @ g1)
    v_tot_lm = {lm: complex(np.sum(vpw.reshape(th.shape) * wgt
                                   * np.conj(sph_harm_y(lm[0], lm[1], th, ph))))
                for lm in lset}
    # q^I of the pw inside R (analytic Bessel = exact for the band-limited density)
    gvec, gnorm, ylm = gvec_ylm_tables(a, nfft, lmax)
    rho_g = (np.fft.fftn(rho_I) / nfft**3).reshape(-1)
    q_i = sphere_interstitial_moments(rho_g, R, sp["tau"], gvec, gnorm, ylm,
                                      list(range(lmax + 1)))
    own_i = own_field(q_i, R)
    print(f"  nfft={nfft}: external field vs [pw analytic - own(q^I)] "
          f"(old chain retained own(q^I)):", flush=True)
    for L in range(lmax + 1):
        for M in range(-L, L + 1):
            ref = v_tot_lm[(L, M)] - own_i.get((L, M), 0.0)
            num = v_bc[(L, M)]
            d2_term = abs(own_i.get((L, M), 0.0))
            if abs(ref) < 1e-8 and d2_term < 1e-8:
                continue
            scale = max(abs(ref), 1e-12)
            print(f"    L={L}M{M:+d}: ext_num={num:.5e} ext_ref={ref:.5e} "
                  f"rel={abs(num-ref)/scale:.2e}  [retained-rho_I term was {d2_term:.3e}]",
                  flush=True)


def bench_pair(nfft, lmax=6):
    """Rutile pair: Ti-like source R1=1.098 with unit L-multipoles, O-like receiver
    R2=0.824, centers 1.946 A apart (gap 0.024 A). Receiver v_bc vs analytic image sum."""
    from scipy.special import sph_harm_y
    r, dx = log_mesh(1e-5, 28.0, 2500)
    r_np = r.numpy()
    d = 1.946
    tau1 = np.array([CELL / 2 - d / 2, CELL / 2, CELL / 2])
    tau2 = np.array([CELL / 2 + d / 2, CELL / 2, CELL / 2])
    moments = {(L, min(L, 2)): 1.0 + 0.0j for L in range(2, lmax + 1)}
    sp1 = sphere(1.098, tau1, moments, r_np, dx, z=0.0)
    sp2 = sphere(0.824, tau2, {}, r_np, dx, z=0.0)
    rho_I = np.zeros((nfft, nfft, nfft))
    _, _, _, v_hart, qmt = _weinert_multi(rho_I, [sp1, sp2], CELL, nfft)
    lset = [(L, M) for L in range(lmax + 1) for M in range(-L, L + 1)]
    v_bc2 = interstitial_boundary_multi(v_hart, tau2, sp2["R"], cell_matrix(CELL), lset)
    # analytic: V(x) = sum_img sum_LM 4piE2/(2L+1) q_LM Y_LM(u)/|u|^{L+1}, u = x - tau1 - img
    th, ph, wgt = _angular_grid(24, 36)
    dirs = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    pts = tau2 + sp2["R"] * dirs.reshape(-1, 3)
    v_ref = np.zeros(pts.shape[0])
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            for iz in range(-2, 3):
                u = pts - tau1 - CELL * np.array([ix, iy, iz])
                un = np.linalg.norm(u, axis=1)
                thu = np.arccos(np.clip(u[:, 2] / un, -1, 1))
                phu = np.arctan2(u[:, 1], u[:, 0])
                for (L, M), q in moments.items():
                    v_ref += (4 * math.pi * E2 / (2 * L + 1)) * np.real(
                        q * sph_harm_y(L, M, thu, phu)) / un ** (L + 1)
    v_ref_lm = {lm: complex(np.sum(v_ref.reshape(th.shape) * wgt
                                   * np.conj(sph_harm_y(lm[0], lm[1], th, ph))))
                for lm in lset}
    print(f"  nfft={nfft}: receiver-surface projections (num vs analytic):", flush=True)
    worst = 0.0
    scale = max(abs(v) for v in v_ref_lm.values())
    for lm in lset:
        num, ref = v_bc2[lm], v_ref_lm[lm]
        if abs(ref) > 1e-3 * scale:
            rel = abs(num - ref) / abs(ref)
            worst = max(worst, rel)
            if lm[0] <= 3 or rel > 0.02:
                print(f"    L{lm[0]}M{lm[1]:+d}: num={num:.4e} ref={ref:.4e} rel={rel:.2e}",
                      flush=True)
    print(f"    worst rel (significant comps): {worst:.2e}", flush=True)


def bench_npow(nfft=64, R=0.824, lmax=6):
    """N) npow policy scan: own-surface field residual per L for flat vs Elk-style
    (fixed-Bessel-order, npow_L = lnpsd - L - 1) pseudocharge orders, at fixed grid.
    Images are negligible for the high L that matter here."""
    a = cell_matrix(CELL)
    vol = float(abs(np.linalg.det(a)))
    gvec, gnorm, ylm = gvec_ylm_tables(a, nfft, lmax)
    tau = np.array([CELL / 2] * 3)
    gmax = math.pi * nfft / CELL
    print(f"  nfft={nfft} gmax*R={gmax*R:.1f}", flush=True)
    qref = {}
    for L in range(1, lmax + 1):
        M = min(L, 2)
        qref[(L, M)] = 1.0 + (0.3j if M else 0)
        if M:
            qref[(L, -M)] = (-1) ** M * np.conj(qref[(L, M)])
    policies = {"flat2": {L: 2 for L in range(lmax + 1)},
                "flat3": {L: 3 for L in range(lmax + 1)},
                "flat4": {L: 4 for L in range(lmax + 1)},
                "flat6": {L: 6 for L in range(lmax + 1)},
                "elk9": {L: max(1, 9 - L - 1) for L in range(lmax + 1)},
                "elk11": {L: max(1, 11 - L - 1) for L in range(lmax + 1)},
                "bord": {L: max(1, int(round(R * gmax / 2)) - L - 1)
                         for L in range(lmax + 1)}}
    lset = [(L, min(L, 2)) for L in range(1, lmax + 1)]
    for name, pol in policies.items():
        ps = np.zeros(nfft**3, dtype=complex)
        for L in range(1, lmax + 1):
            sub = {lm: q for lm, q in qref.items() if lm[0] == L}
            ps += sphere_pseudocharge_ft(sub, R, tau, vol, pol[L], gvec, gnorm, ylm)
        g2 = gnorm**2
        v_g = np.where(g2 > 1e-12, 4 * math.pi * E2 * ps / np.where(g2 > 1e-12, g2, 1.0), 0.0)
        v = np.fft.ifftn((v_g * nfft**3).reshape(nfft, nfft, nfft)).real
        v_bc = interstitial_boundary_multi(v, tau, R, a, lset)
        rels = []
        for (L, M) in lset:
            own = (4 * math.pi * E2 / (2 * L + 1)) * qref[(L, M)] / R ** (L + 1)
            rels.append(abs(v_bc[(L, M)] - own) / abs(own))
        print(f"  {name:6s} npow(L)={[pol[L] for L in range(1, lmax+1)]}: "
              + " ".join(f"L{L}={r:.1e}" for L, r in zip(range(1, lmax + 1), rels,
                                                         strict=True)), flush=True)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["A", "B", "C"]
    if "A" in stages:
        print("A) pseudocharge moment exactness", flush=True)
        for n in (32, 48, 64):
            bench_moments(n)
    if "B" in stages:
        print("\nB) isolated-sphere boundary null (D2 gate)", flush=True)
        for n in (32, 48, 64):
            bench_null(n)
    if "B2" in stages:
        print("\nB2) retained-rho_I external-field identity (D2 gate)", flush=True)
        for n in (32, 48, 64):
            bench_rho_i(n)
    if "N" in stages:
        print("\nN) npow policy scan (own-surface residual per L)", flush=True)
        for n in (48, 64, 80):
            bench_npow(n)
    if "C" in stages:
        print("\nC) near-touching rutile pair (D4 gate)", flush=True)
        for n in (48, 64, 80):
            bench_pair(n)
