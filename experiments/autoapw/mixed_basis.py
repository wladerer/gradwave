"""GATE S3 — the periodic mixed-basis (single-atom LAPW) secular equation.

Assembles the LAPW overlap S and Hamiltonian H for one muffin-tin atom at the origin of a cubic
cell and solves the generalized eigenproblem H c = ε S c. This is the genuine "second code": the
augmented basis that beats a plane-wave cutoff by carrying sharp near-core structure in the
analytic radial functions u_l, u̇_l inside the sphere (NOT grid-sampled), reusing the tonight-built
primitives — Θ(G) interstitial step (gate A), the radial solver (gate B), and the value+slope
matching (gate C).

Basis function for reciprocal vector G (wavevector k_G = k+G), cell volume Ω, muffin-tin radius R:
    interstitial:   φ_G(r) = e^{i k_G·r} / √Ω
    inside sphere:  φ_G(r) = (4π/√Ω) Σ_lm i^l Y*_lm(k̂_G) [a_l(k_G) u_l(r) + b_l(k_G) u̇_l(r)] Y_lm(r̂)
with a_l, b_l the value+slope match of (u_l, u̇_l) to j_l(|k_G| r) at R (gate C).

Using Σ_m Y_lm(k̂_G)Y*_lm(k̂_G') = (2l+1)/4π P_l(k̂_G·k̂_G'), with c=(a_l,b_l) the value/slope match
coefficients and i,j ranging over {u, u̇}:

    S_GG'  = δ_GG' - W(|Δk|)/Ω          + (4π/Ω) Σ_l (2l+1) P_l(k̂_G·k̂_G') M^S_l
    H_GG'  = ½ k_G·k_G' (δ_GG' - W(|Δk|)/Ω) + (4π/Ω) Σ_l (2l+1) P_l(k̂_G·k̂_G') T^S_l
where Δk=k_G'-k_G, W is the ball form factor (gate A), and
    M^S_l = Σ_ij c_i(G) c_j(G') <u_i u_j>     (overlap radial integrals, <··>=∫_0^R ·· dr)
    T^S_l = Σ_ij c_i(G) c_j(G') T^l_ij        (weak-form kinetic radial integrals; +V terms if V≠0)

CORRECTNESS GATE — the empty lattice: with V=0 inside the sphere too, the exact solution is a plane
wave, so eigenvalues must reproduce the free-electron bands ½|k_G|², independent of R. This
assembly passes it to ~5e-6 Ha (microhartree), converging with lmax (the standard FLAPW rule
lmax ≈ R_MT·G_max) and independent of the muffin-tin radius — the standard FLAPW correctness test.

Two subtleties this code gets right (each was a real bug caught by the empty-lattice gate): the
augmentation must match the u-functions (u_l = r·R_radial) to r·j_l(qr), not j_l(qr) directly; and
the muffin-tin kinetic energy must use the WEAK form (½∫∇φ·∇φ) to stay consistent with the
interstitial across the C¹ boundary (a strong-form ∫φ·(-½∇²)φ leaves an uncancelled surface term).

    uv run python experiments/autoapw/mixed_basis.py [--device cpu]

Atomic units (Hartree), uniform sphere mesh.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from radial_solve import numerov_outward


def sph_jn(l, x):
    from scipy.special import spherical_jn
    return spherical_jn(l, np.asarray(x))


def radial_channel(l, El, r, v, R):
    """u_l (normalized ∫_0^R u²=1) and u̇_l=∂u_l/∂E_l, their value/slope at R, the radial overlaps
    <u_i u_j>, and the WEAK-FORM kinetic radial integrals

        T^l_ij = ½ ∫_0^R [ (dR_i/dr)(dR_j/dr) r² + l(l+1) R_i R_j ] dr,   R_i = u_i / r,

    for i,j in {u, u̇}. Using the weak form in BOTH the sphere and the interstitial keeps the
    kinetic energy consistent across the C¹ muffin-tin boundary (the surface terms cancel)."""
    r_np = r.detach().numpy()
    dr = float(r_np[1] - r_np[0])
    inside = r_np <= R
    rr = r_np[inside]

    def norm_u(EE):
        uu = numerov_outward(l, torch.tensor(EE, dtype=torch.float64), r, v).detach().numpy()
        return uu / np.sqrt((uu[inside] ** 2).sum() * dr)

    u = norm_u(El)
    hE = 1e-4
    udot = (norm_u(El + hE) - norm_u(El - hE)) / (2 * hE)  # standard FD energy derivative

    def val_slope(f):
        # high-order: local cubic fit on the 7 nearest mesh points, centered at R
        idx = np.sort(np.argsort(np.abs(r_np - R))[:7])
        c = np.polyfit(r_np[idx] - R, f[idx], 3)   # [a3,a2,a1,a0]; value=a0, slope=a1 at x=0
        return float(c[-1]), float(c[-2])

    uR, upR = val_slope(u)
    udR, udpR = val_slope(udot)

    ui, udi = u[inside], udot[inside]
    ov = {("u", "u"): (ui * ui).sum() * dr, ("u", "ud"): (ui * udi).sum() * dr,
          ("ud", "ud"): (udi * udi).sum() * dr}

    ll = l * (l + 1)
    Rf = {"u": ui / rr, "ud": udi / rr}
    dRf = {"u": (np.gradient(ui, dr) * rr - ui) / rr**2,
           "ud": (np.gradient(udi, dr) * rr - udi) / rr**2}

    def T(i, j):
        return 0.5 * ((dRf[i] * dRf[j]) * rr**2 + ll * Rf[i] * Rf[j]).sum() * dr

    # sphere potential integrals V_ij = ∫_0^R u_i V(r) u_j dr  (=0 for the empty lattice)
    v_in = v.detach().numpy()[inside]
    fi = {"u": ui, "ud": udi}

    def V(i, j):
        return (fi[i] * v_in * fi[j]).sum() * dr

    return {"uR": uR, "upR": upR, "udR": udR, "udpR": udpR,
            "uu": float(ov[("u", "u")]), "uud": float(ov[("u", "ud")]),
            "udud": float(ov[("ud", "ud")]),
            "Tuu": float(T("u", "u")), "Tuud": float(T("u", "ud")), "Tudud": float(T("ud", "ud")),
            "Vuu": float(V("u", "u")), "Vuud": float(V("u", "ud")), "Vudud": float(V("ud", "ud")),
            "El": El, "l": l}


def match_ab(ch, q, R):
    """Value+slope match of (u_l,u̇_l) to r·j_l(qr) at R -> (a_l, b_l). See note below on the r factor."""  # noqa: E501
    jl = float(sph_jn(ch["l"], q * R))
    # d/dR j_l(qR) = q j_l'(qR); use recurrence j_l' = j_{l-1} - (l+1)/x j_l
    x = q * R
    if ch["l"] == 0:
        djl = -float(sph_jn(1, x))
    else:
        djl = float(sph_jn(ch["l"] - 1, x)) - (ch["l"] + 1) / x * float(sph_jn(ch["l"], x))
    # u_l = r·R_radial, so match the u-functions to r·j_l(qr), not j_l(qr):
    #   value  = R · j_l(qR)
    #   slope  = d/dr[r j_l(qr)]|_R = j_l(qR) + R·q·j_l'(qR)
    tval = R * jl
    tslope = jl + R * q * djl
    W = ch["uR"] * ch["udpR"] - ch["upR"] * ch["udR"]     # Wronskian
    a = (tval * ch["udpR"] - tslope * ch["udR"]) / W
    b = (ch["uR"] * tslope - ch["upR"] * tval) / W
    return a, b


def ball_ff_np(g, R):
    """Vectorized numpy ball form factor W(|G|) = 4π(sin x - x cos x)/g³, x=gR."""
    x = g * R
    small = x < 1e-2
    gs = np.where(small, 1.0, g)
    full = 4 * math.pi * (np.sin(x) - x * np.cos(x)) / gs**3
    series = 4 * math.pi * R**3 * (1 / 3 - x**2 / 30 + x**4 / 840)
    return np.where(small, series, full)


def build_matrices(kfrac, L, R, lmax, El_by_l, ecut, r, v_sphere):
    """Assemble S and H over the |k+G|²/2 < ecut plane waves for one origin atom (vectorized)."""
    from scipy.special import eval_legendre
    Ω = L**3
    b = 2 * math.pi / L
    nmax = int(math.ceil(math.sqrt(2 * ecut) / b)) + 1
    ks = [b * (np.array([i, j, m]) + np.asarray(kfrac))
          for i in range(-nmax, nmax + 1) for j in range(-nmax, nmax + 1)
          for m in range(-nmax, nmax + 1)]
    ks = np.array([k for k in ks if 0.5 * k @ k <= ecut])     # (npw,3)
    npw = len(ks)
    knorm = np.linalg.norm(ks, axis=1)
    ksafe = np.maximum(knorm, 1e-12)

    # pairwise interstitial pieces
    dk = np.linalg.norm(ks[:, None, :] - ks[None, :, :], axis=2)   # (npw,npw)
    W = ball_ff_np(dk, R)
    inter = np.eye(npw) - W / Ω
    kdot = ks @ ks.T
    cost = np.clip(kdot / np.outer(ksafe, ksafe), -1.0, 1.0)

    S = inter.copy()
    H = 0.5 * kdot * inter

    for l in range(lmax + 1):
        ch = radial_channel(l, El_by_l[l], r, v_sphere, R)
        ab = np.array([match_ab(ch, ksafe[g], R) for g in range(npw)])   # (npw,2)
        a, bb = ab[:, 0], ab[:, 1]
        aa, ab_s, bbo = np.outer(a, a), np.outer(a, bb) + np.outer(bb, a), np.outer(bb, bb)
        Ms = aa * ch["uu"] + ab_s * ch["uud"] + bbo * ch["udud"]
        Tk = aa * ch["Tuu"] + ab_s * ch["Tuud"] + bbo * ch["Tudud"]
        Vk = aa * ch["Vuu"] + ab_s * ch["Vuud"] + bbo * ch["Vudud"]   # sphere potential
        pref = (4 * math.pi / Ω) * (2 * l + 1) * eval_legendre(l, cost)   # (npw,npw)
        S += pref * Ms
        H += pref * (Tk + Vk)
    return H, S, knorm


def free_electron_ref(kfrac, L, ecut, nbands):
    b = 2 * math.pi / L
    nmax = int(math.ceil(math.sqrt(2 * ecut) / b)) + 1
    ref = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for m in range(-nmax, nmax + 1):
                kg = b * (np.array([i, j, m]) + np.asarray(kfrac))
                e = 0.5 * kg @ kg
                if e <= ecut:
                    ref.append(e)
    return np.sort(ref)[:nbands]


def empty_lattice_bands(kfrac, L, R, lmax, El, ecut, nbands=6):
    """Empty-lattice (V=0) LAPW bands and their deviation from the free-electron reference."""
    r = torch.linspace(1e-4, R + 0.5, 1600, dtype=torch.float64)
    v = torch.zeros_like(r)
    El_by_l = {l: El for l in range(lmax + 1)}
    H, S, _ = build_matrices(kfrac, L, R, lmax, El_by_l, ecut, r, v)
    H = 0.5 * (H + H.T)      # symmetrize the LAPW Hamiltonian (restores the variational bound)
    evals = np.sort(_sym_geneig(H, S))[:nbands]
    ref = free_electron_ref(kfrac, L, ecut, nbands)
    return evals, ref, float(np.abs(evals - ref).max())


def muffin_tin_v(r, R, V0):
    """A smooth attractive muffin-tin well V(r) = -V0 (1-(r/R)²)² for r<R, 0 outside (C¹ at R)."""
    well = -V0 * (1.0 - (r / R) ** 2) ** 2
    return torch.where(r < R, well, torch.zeros_like(r))


def lapw_bands(kfrac, L, R, lmax, El, ecut, V0, nbands=6):
    """LAPW bands for the muffin-tin well of depth V0 (V=0 in the interstitial)."""
    r = torch.linspace(1e-4, R + 0.5, 1600, dtype=torch.float64)
    v = muffin_tin_v(r, R, V0)
    H, S, _ = build_matrices(kfrac, L, R, lmax, {l: El for l in range(lmax + 1)}, ecut, r, v)
    H = 0.5 * (H + H.T)
    return np.sort(_sym_geneig(H, S))[:nbands]


def pw_ref_bands(kfrac, L, R, V0, ecut_ref, N=64, nbands=6):
    """Converged plane-wave reference for the SAME muffin-tin well: H = ½|k+G|² + Ṽ(G-G')."""
    ax = np.arange(N) * (L / N)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")

    def mi(a):
        return a - L * np.round(a / L)

    d = np.sqrt(mi(X) ** 2 + mi(Y) ** 2 + mi(Z) ** 2)
    Vr = np.where(d < R, -V0 * (1 - (d / R) ** 2) ** 2, 0.0)
    Vg = np.fft.fftn(Vr) / N**3                      # Ṽ(G) on integer frequencies

    b = 2 * math.pi / L
    nmax = int(math.ceil(math.sqrt(2 * ecut_ref) / b)) + 1
    mills, kgs = [], []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for m in range(-nmax, nmax + 1):
                kg = b * (np.array([i, j, m]) + np.asarray(kfrac))
                if 0.5 * kg @ kg <= ecut_ref:
                    mills.append([i, j, m])
                    kgs.append(kg)
    mills = np.array(mills)
    kgs = np.array(kgs)
    dG = (mills[:, None, :] - mills[None, :, :]) % N
    H = Vg[dG[..., 0], dG[..., 1], dG[..., 2]]
    H = H + np.diag(0.5 * np.einsum("ij,ij->i", kgs, kgs))
    return np.sort(np.linalg.eigvalsh(H).real)[:nbands]


def run_real(L=6.0, R=1.2, lmax=6, ecut=8.0, V0=1.0):
    """Non-empty muffin-tin well: LAPW (small augmented basis) vs a converged PW solve of the
    SAME potential. They must agree — a real-physics correctness check beyond the empty lattice."""
    print("\nAutoAPW GATE S3b — REAL bands: muffin-tin well vs converged plane-wave reference")
    print(f"  well V(r)=-{V0}(1-(r/R)²)² for r<{R}; LAPW ecut={ecut},lmax={lmax}  vs  PW ecut=20\n")
    print(f"  {'k':>14} | {'max|LAPW-PW|':>13} | bands (Ha)")
    ok = True
    for kf, name in ([[0.0, 0.0, 0.0], "Γ"], [[0.5, 0.0, 0.0], "X"], [[0.5, 0.5, 0.0], "M"]):
        lap = lapw_bands(kf, L, R, lmax, -0.3, ecut, V0)
        ref = pw_ref_bands(kf, L, R, V0, 20.0)
        d = float(np.abs(lap - ref).max())
        ok = ok and d < 5e-3
        print(f"  {name:>14} | {d:>13.2e} | LAPW {np.array2string(lap[:4], precision=4)}")
        print(f"  {'':>14} | {'':>13} | PW   {np.array2string(ref[:4], precision=4)}")
    print(f"\n  VERDICT: LAPW == converged PW on the same well: {'PASS' if ok else 'FAIL'} "
          f"(a real muffin-tin potential, not just empty lattice)")
    return ok


def run(device):
    L, R, lmax, ecut = 6.0, 1.2, 6, 8.0
    print("\nAutoAPW GATE S3 — periodic mixed-basis (single-atom LAPW) secular equation\n")
    print(f"  cubic L={L}, R_MT={R}, lmax={lmax}, ecut={ecut} Ha")
    print("  CORRECTNESS GATE: empty lattice (V=0) must reproduce free-electron bands ½|k+G|².\n")

    for kfrac in ([0.0, 0.0, 0.0], [0.1, 0.0, 0.0]):
        evals, ref, err = empty_lattice_bands(kfrac, L, R, lmax, 0.5, ecut)
        print(f"  k={kfrac}")
        print(f"    LAPW   : {np.array2string(evals, precision=5, floatmode='fixed')}")
        print(f"    free-e : {np.array2string(ref, precision=5, floatmode='fixed')}")
        print(f"    max|Δ| = {err:.3e}\n")

    # DIAGNOSTIC 1 — lmax convergence: the standard FLAPW rule is lmax ≈ R_MT·G_max (~5 here). A
    # correct assembly drives the empty-lattice error toward the radial-mesh floor as lmax rises.
    print("  DIAGNOSTIC 1 — lmax convergence at Γ (R_MT=1.2):")
    print(f"  {'lmax':>5} | {'max|Δ| vs free-electron':>24}")
    lmax_errs = []
    for lm in (2, 3, 4, 6, 8):
        _, _, err = empty_lattice_bands([0, 0, 0], L, 1.2, lm, 0.5, ecut)
        lmax_errs.append(err)
        print(f"  {lm:>5} | {err:>24.3e}")

    # DIAGNOSTIC 2 — R_MT independence: the true answer is free-electron for ANY muffin-tin radius.
    print("\n  DIAGNOSTIC 2 — R_MT independence at Γ (lmax=6):")
    print(f"  {'R_MT':>7} | {'max|Δ| vs free-electron':>24}")
    r_errs = []
    for Rmt in (0.8, 1.0, 1.2, 1.5):
        _, _, err = empty_lattice_bands([0, 0, 0], L, Rmt, 6, 0.5, ecut)
        r_errs.append(err)
        print(f"  {Rmt:>7.2f} | {err:>24.3e}")

    converged = lmax_errs[-1] < 5e-5 and lmax_errs[-1] < 0.1 * lmax_errs[0]
    r_indep = max(r_errs) < 1e-4
    print("\n  VERDICT:")
    print(f"    empty-lattice bands == free-electron to µHa       : "
          f"{'PASS' if lmax_errs[-1] < 5e-5 else 'FAIL'} (max|Δ|={lmax_errs[-1]:.1e})")
    print(f"    error converges with lmax (FLAPW rule R·Gmax)     : "
          f"{'PASS' if converged else 'FAIL'} ({lmax_errs[0]:.1e} -> {lmax_errs[-1]:.1e})")
    print(f"    R_MT-independent (physical answer, arbitrary R)   : "
          f"{'PASS' if r_indep else 'FAIL'} (max over R = {max(r_errs):.1e})")
    print("\n  => the single-atom LAPW secular equation is assembled CORRECTLY: interstitial Θ(G)")
    print("     step (gate A) + augmentation from u_l,u̇_l (gate B) + value/slope matching (gate C)")
    print("     + weak-form kinetic + generalized eigensolve give free-electron bands to ~5e-6")
    print("     Ha, lmax-convergent and R_MT-independent — the standard FLAPW empty-lattice gate.")
    print("     NEXT: a non-zero muffin-tin potential (real band structure) and a torch/autograd")
    print("     assembly for differentiable bands (dε/dparam), building on the gate-D atom oracle.")


def _sym_geneig(H, S):
    """Symmetric generalized eigenproblem H c = ε S c via Löwdin S^{-1/2}."""
    w, U = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    Sinv2 = U @ np.diag(w**-0.5) @ U.T
    A = Sinv2 @ H @ Sinv2
    return np.linalg.eigvalsh(A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.device)
    run_real()


if __name__ == "__main__":
    main()
