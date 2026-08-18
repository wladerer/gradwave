"""PROD-D — the LAPW secular equation in production units (eV/Å) on a log mesh, fed a REAL
self-consistent atomic potential from PROD-C.

Ports the verified single-atom LAPW assembly (mixed_basis.py, Hartree a.u. + uniform mesh + toy
well) to gradwave eV/Å units with:
  - radial functions u_l, u̇_l from the log-mesh solver (radial_log.numerov_log),
  - kinetic prefactor HBAR2_2M (interstitial HBAR2_2M k·k', weak-form muffin-tin kinetic),
  - the self-consistent all-electron atomic potential V(r) from PROD-C (atomic_scf).

Two checks:
  1. EMPTY LATTICE (V=0): bands must reproduce the free-electron bands HBAR2_2M|k+G|² (confirms
     the unit port is correct).
  2. ISOLATED ATOM: with the muffin-tin filled by a real self-consistent atomic potential, the
     Γ-point LAPW valence eigenvalue must reproduce the atom's own KS valence eigenvalue (computed
     independently by PROD-C's radial solver) — real bands on a real potential.

    uv run python experiments/autoapw/prod_lapw.py
"""

from __future__ import annotations

import math

import numpy as np
import torch
from radial_log import numerov_log

from gradwave.constants import HBAR2_2M


def sph_jn(l, x):
    from scipy.special import spherical_jn
    return spherical_jn(l, np.asarray(x))


def ball_ff_np(g, R):
    x = g * R
    small = x < 1e-2
    gs = np.where(small, 1.0, g)
    full = 4 * math.pi * (np.sin(x) - x * np.cos(x)) / gs**3
    series = 4 * math.pi * R**3 * (1 / 3 - x**2 / 30 + x**4 / 840)
    return np.where(small, series, full)


def radial_channel(l, El, r, dx, v, R):
    """u_l, u̇_l on the log mesh (eV/Å) + value/slope at R, overlaps, weak-form kinetic (HBAR2_2M),
    and potential integrals. d/dr = (1/r) d/dx on the log mesh; dr = r·dx."""
    r_np = r.detach().numpy()
    inside = r_np <= R
    rr = r_np[inside]
    drw = rr * dx                       # ∫ dr weight on the log mesh

    def norm_u(E):
        u = numerov_log(l, torch.tensor(E, dtype=torch.float64), r, dx, v).detach().numpy()
        return u / np.sqrt((u[inside] ** 2 * drw).sum())

    u = norm_u(El)
    hE = max(abs(El) * 1e-4, 1e-3)
    udot = (norm_u(El + hE) - norm_u(El - hE)) / (2 * hE)

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
    Rf = {"u": ui / rr, "ud": udi / rr}
    # d/dr on log mesh: du/dr = (1/r) du/dx ; dR/dr = (du/dx - u)/r²
    dRf = {"u": (np.gradient(ui, dx) - ui) / rr**2, "ud": (np.gradient(udi, dx) - udi) / rr**2}

    def T(i, j):   # weak-form muffin-tin kinetic: HBAR2_2M ∫[R_i'R_j' r² + l(l+1)R_iR_j] dr
        return HBAR2_2M * ((dRf[i] * dRf[j] * rr**2 + ll * Rf[i] * Rf[j]) * drw).sum()

    def V(i, j):
        f = {"u": ui, "ud": udi}
        return (f[i] * v_in * f[j] * drw).sum()

    return {"uR": uR, "upR": upR, "udR": udR, "udpR": udpR,
            "uu": float(ov["uu"]), "uud": float(ov["uud"]), "udud": float(ov["udud"]),
            "Tuu": float(T("u", "u")), "Tuud": float(T("u", "ud")), "Tudud": float(T("ud", "ud")),
            "Vuu": float(V("u", "u")), "Vuud": float(V("u", "ud")), "Vudud": float(V("ud", "ud")),
            "El": El, "l": l}


def match_ab(ch, q, R):
    """Match (u_l, u̇_l) to r·j_l(qr) at R (the u=r·R factor); analytic j_l slope."""
    x = q * R
    jl = float(sph_jn(ch["l"], x))
    djl = -float(sph_jn(1, x)) if ch["l"] == 0 else (
        float(sph_jn(ch["l"] - 1, x)) - (ch["l"] + 1) / x * float(sph_jn(ch["l"], x)))
    tval, tslope = R * jl, jl + R * q * djl
    w = ch["uR"] * ch["udpR"] - ch["upR"] * ch["udR"]
    a = (tval * ch["udpR"] - tslope * ch["udR"]) / w
    b = (ch["uR"] * tslope - ch["upR"] * tval) / w
    return a, b


def build_matrices(kfrac, L, R, lmax, El_by_l, ecut, r, dx, v):
    """Assemble LAPW S, H (eV/Å) over HBAR2_2M|k+G|² < ecut for one origin atom."""
    from scipy.special import eval_legendre
    vol = L**3
    b = 2 * math.pi / L
    kgmax = math.sqrt(ecut / HBAR2_2M)
    nmax = int(math.ceil(kgmax / b)) + 1
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
    H = HBAR2_2M * kdot * inter           # interstitial kinetic HBAR2_2M k·k' × interstitial ov
    for lang in range(lmax + 1):
        ch = radial_channel(lang, El_by_l[lang], r, dx, v, R)
        ab = np.array([match_ab(ch, ksafe[g], R) for g in range(npw)])
        a, bb = ab[:, 0], ab[:, 1]
        aa, bbo = np.outer(a, a), np.outer(bb, bb)
        ab_s = np.outer(a, bb) + np.outer(bb, a)
        Ms = aa * ch["uu"] + ab_s * ch["uud"] + bbo * ch["udud"]
        Tk = aa * ch["Tuu"] + ab_s * ch["Tuud"] + bbo * ch["Tudud"]
        Vk = aa * ch["Vuu"] + ab_s * ch["Vuud"] + bbo * ch["Vudud"]
        pref = (4 * math.pi / vol) * (2 * lang + 1) * eval_legendre(lang, cost)
        S += pref * Ms
        H += pref * (Tk + Vk)
    return H, S, knorm


def _sym_geneig(H, S, nbands):
    w, U = np.linalg.eigh(S)
    Sinv2 = U @ np.diag(np.clip(w, 1e-12, None) ** -0.5) @ U.T
    return np.sort(np.linalg.eigvalsh(Sinv2 @ H @ Sinv2))[:nbands]


def free_electron_ref(kfrac, L, ecut, nbands):
    b = 2 * math.pi / L
    nmax = int(math.ceil(math.sqrt(ecut / HBAR2_2M) / b)) + 1
    e = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for m in range(-nmax, nmax + 1):
                kg = b * (np.array([i, j, m]) + np.asarray(kfrac))
                if HBAR2_2M * (kg @ kg) <= ecut:
                    e.append(HBAR2_2M * (kg @ kg))
    return np.sort(e)[:nbands]


def empty_lattice_check():
    """V=0 must give free-electron bands HBAR2_2M|k+G|² (confirms the eV/Å + log-mesh port)."""
    from radial_log import log_mesh
    L, R, lmax, ecut = 6.0, 1.0, 6, 120.0     # ecut in eV
    r, dx = log_mesh(1e-4, R + 1.0, 1500)
    v = torch.zeros_like(r)
    print("  EMPTY LATTICE (eV/Å): LAPW vs free-electron HBAR2_2M|k+G|²")
    ok = True
    for kfrac in ([0.0, 0.0, 0.0], [0.1, 0.0, 0.0]):
        # linearization energy in the band range being checked (0–~4 eV) to minimise LAPW error
        H, S, _ = build_matrices(kfrac, L, R, lmax, {lg: 2.5 for lg in range(lmax + 1)}, ecut,
                                 r, dx, v)
        H = 0.5 * (H + H.T)
        ev = _sym_geneig(H, S, 6)
        ref = free_electron_ref(kfrac, L, ecut, 6)
        err = float(np.abs(ev - ref).max())
        ok = ok and err < 5e-3
        print(f"    k={kfrac}  max|Δ| = {err:.2e} eV   ({np.array2string(ev[:4], precision=3)})")
    return ok


def isolated_atom_check(symbol="Ne", R=2.0, L=7.0, ecut=140.0):
    """Fill the muffin tin with a REAL self-consistent atomic potential (PROD-C) and check the
    Γ-point LAPW valence eigenvalues reproduce the atom's own KS valence eigenvalues."""
    from atomic_scf import atomic_scf
    from radial_log import log_mesh
    r, dx = log_mesh(1e-5, 28.0, 2500)
    at_eigs, v_at = atomic_scf(symbol, r, dx)              # self-consistent atomic V(r) + eigs

    # muffin-tin: shift so V(R_MT)=0 (interstitial zero); LAPW eig is then relative to V(R_MT).
    iR = int(np.argmin(np.abs(r.numpy() - R)))
    v0 = float(v_at[iR])
    v_mt = torch.where(r <= R, v_at - v0, torch.zeros_like(r))

    # linearization energies at the atomic valence levels (relative to the muffin-tin zero)
    val = {0: "2s", 1: "2p"}
    lmax = 2
    El = {lg: (at_eigs.get(val.get(lg, ""), -5.0) - v0) for lg in range(lmax + 1)}
    H, S, _ = build_matrices([0.0, 0.0, 0.0], L, R, lmax, El, ecut, r, dx, v_mt)
    H = 0.5 * (H + H.T)
    ev = _sym_geneig(H, S, 12) + v0                       # back to the vacuum-referenced scale

    print(f"\n  ISOLATED ATOM ({symbol}, real self-consistent potential, R_MT={R} Å, L={L} Å):")
    print(f"  {'valence':>8} | {'atomic KS (eV)':>14} | {'nearest LAPW (eV)':>18} | {'|Δ|':>8}")
    ok = True
    for lvl in ("2s", "2p"):
        if lvl not in at_eigs:
            continue
        e_at = at_eigs[lvl]
        e_lapw = float(ev[np.argmin(np.abs(ev - e_at))])
        d = abs(e_lapw - e_at)
        ok = ok and d < 1.0
        print(f"  {lvl:>8} | {e_at:>14.3f} | {e_lapw:>22.3f} | {d:>8.3f}")
    return ok


def main():
    print("\nPROD-D — LAPW in production units (eV/Å, log mesh) on a real atomic potential\n")
    el = empty_lattice_check()
    at = isolated_atom_check("Ne")
    print("\n  VERDICT:")
    print(f"    empty-lattice port correct (free-electron bands)     : {'PASS' if el else 'FAIL'}")
    print(f"    real atomic potential -> valence bands match atomic  : {'PASS' if at else 'FAIL'}")


if __name__ == "__main__":
    main()
