"""DG-ALB spike (kill point 1): does the SIPG interior-penalty assembly of an
adaptive local basis reproduce the exact plane-wave spectrum in 1D?

The whole moonshot rests on the symmetric interior penalty (SIPG) form giving a
correct, variational global operator out of a DISCONTINUOUS per-element basis.
This standalone 1D test settles that before any 3D infrastructure is built:

  * A periodic 1D Kohn-Sham-like Hamiltonian H = -HB d^2/dx^2 + V(x), V a sum of
    periodic Gaussian wells ("atoms"). Reference eigenvalues from a dense
    plane-wave diagonalisation (exact to grid resolution).
  * DG-ALB: partition [0,L) into N_elem elements. On each EXTENDED element
    (core + buffer halo) solve the local problem by plane waves, keep M lowest
    states, core-restrict + Loewdin-orthonormalise -> adaptive local basis (ALB).
  * Assemble the global operator with the SIPG form
        a(u,v) = sum_K  HB (u')* v'                              (volume)
               - sum_F  HB ( {u'}*[v] + [u]* {v'} )              (consistency+symmetry)
               + sum_F  HB (sigma/h) [u]* [v]                    (penalty)
    with jump [w]=w_L-w_R and average {w'}=½(w'_L+w'_R) at each element face
    (a POINT in 1D). ALBs are orthonormal by the non-overlapping core partition
    -> standard eigenproblem, no overlap matrix.
  * Compare the lowest DG eigenvalues to the plane-wave reference; sweep the
    penalty sigma (coercivity vs conditioning) and M (basis convergence).

PASS = DG eigenvalues converge to the reference to ~sub-meV with a modest M and a
stable sigma window, and the assembled operator is Hermitian to ~1e-12.

Atomic-like units: HB = hbar^2/2m = 1/2, energies "Ha-like". Self-contained
(numpy + torch only) — this is the pre-integration spike, not gradwave code.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
CD = torch.complex128
HB = 0.5  # hbar^2 / 2m


# ---------------------------------------------------------------- potential
def make_Vfunc(L, n_at, V0, width):
    centers = (np.arange(n_at) + 0.5) * L / n_at

    def V(x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for c in centers:
            d = np.abs(x - c)
            d = np.minimum(d, L - d)          # periodic distance
            out += -V0 * np.exp(-(d / width) ** 2)
        return out

    return V


# ---------------------------------------------------------------- reference
def reference_eigs(L, Ng, Vfunc):
    """Exact eigenvalues via a dense plane-wave Hamiltonian on the full box."""
    xs = np.arange(Ng) * L / Ng
    Vx = Vfunc(xs)
    m = np.round(np.fft.fftfreq(Ng, d=1.0 / Ng)).astype(int)   # integer modes
    k = 2 * np.pi * m / L
    Vhat = np.fft.fft(Vx) / Ng                                 # coeff of e^{i k_j x}
    H = np.zeros((Ng, Ng), dtype=complex)
    diff = (m[:, None] - m[None, :]) % Ng
    H = Vhat[diff]
    H[np.diag_indices(Ng)] += HB * k ** 2
    w = np.linalg.eigvalsh(H)
    return np.sort(w.real)


# ---------------------------------------------------------------- local solve
def local_states(x0, w, Vfunc, Nloc, M):
    """M lowest eigenstates of the extended element [x0, x0+w] (periodic box),
    returned as plane-wave coeffs + wavevectors so they evaluate anywhere."""
    xs = x0 + np.arange(Nloc) * w / Nloc
    Vx = Vfunc(xs)
    m = np.round(np.fft.fftfreq(Nloc, d=1.0 / Nloc)).astype(int)
    k = 2 * np.pi * m / w
    Vhat = np.fft.fft(Vx) / Nloc
    diff = (m[:, None] - m[None, :]) % Nloc
    H = Vhat[diff]
    H[np.diag_indices(Nloc)] += HB * k ** 2
    wv, U = np.linalg.eigh(H)
    idx = np.argsort(wv.real)[:M]
    return k, U[:, idx]          # k (Nloc,), coeffs (Nloc, M)


def evalf(coeffs, k, x0, xs, deriv=False):
    """phi_j(xs) = sum_G coeffs[G,j] e^{i k_G (xs-x0)}  (or its x-derivative)."""
    ph = np.exp(1j * np.outer(np.asarray(xs) - x0, k))     # (P, Nk)
    if deriv:
        ph = ph * (1j * k)[None, :]
    return ph @ coeffs                                     # (P, M)


# ---------------------------------------------------------------- ALB per element
def build_alb(xe, h, buf, Vfunc, Nloc, Ncore, M, orth_tol=1e-8):
    """Core-restricted, canonically-orthonormalised ALBs for the core [xe, xe+h].
    Core-restriction makes the M extended-element states near-linearly-dependent
    on a small core, so the overlap S is rank-deficient: keep only directions
    with eigenvalue > orth_tol*max (canonical orthogonalisation). The surviving
    count n_keep is the element's true ALB count. Returns core-grid values &
    derivatives, core potential samples, and endpoint (value, derivative)."""
    x0 = xe - buf
    w = h + 2 * buf
    k, C = local_states(x0, w, Vfunc, Nloc, M)

    xc = xe + (np.arange(Ncore) + 0.5) * h / Ncore         # midpoint core grid
    dx = h / Ncore
    Phi = evalf(C, k, x0, xc)                              # (Ncore, M)
    dPhi = evalf(C, k, x0, xc, deriv=True)

    S = (Phi.conj().T @ Phi) * dx                          # (M, M) core overlap
    ev, Q = np.linalg.eigh(S)
    keep = ev > orth_tol * ev.max()
    X = Q[:, keep] * (ev[keep] ** -0.5)[None, :]           # (M, n_keep) canonical
    Phi = Phi @ X
    dPhi = dPhi @ X

    ends_x = np.array([xe, xe + h])
    valE = evalf(C, k, x0, ends_x) @ X                     # (2, n_keep): [left, right]
    derE = evalf(C, k, x0, ends_x, deriv=True) @ X
    Vc = Vfunc(xc)
    return {
        "Phi": torch.tensor(Phi, dtype=CD), "dPhi": torch.tensor(dPhi, dtype=CD),
        "Vc": torch.tensor(Vc), "dx": dx, "n": int(keep.sum()),
        "valL": torch.tensor(valE[0], dtype=CD), "derL": torch.tensor(derE[0], dtype=CD),
        "valR": torch.tensor(valE[1], dtype=CD), "derR": torch.tensor(derE[1], dtype=CD),
    }


# ---------------------------------------------------------------- global assembly
def assemble(elems, h, sigma):
    N = len(elems)
    sizes = [el["n"] for el in elems]
    off = np.cumsum([0] + sizes)
    D = int(off[-1])
    H = torch.zeros(D, D, dtype=CD)

    def rng(e):
        return torch.arange(off[e], off[e + 1])

    # diagonal (volume) blocks: kinetic + potential
    for e, el in enumerate(elems):
        T = HB * (el["dPhi"].conj().T @ el["dPhi"]) * el["dx"]
        Vm = (el["Phi"].conj().T @ (el["Vc"].to(CD)[:, None] * el["Phi"])) * el["dx"]
        idx = rng(e)
        H[idx.unsqueeze(1), idx.unsqueeze(0)] += T + Vm

    # face terms: face f sits between element e (left) and e+1 (right), periodic
    for e in range(N):
        eR = (e + 1) % N
        nL, nR = sizes[e], sizes[eR]
        vL, dL = elems[e]["valR"], elems[e]["derR"]        # left element, right end
        vR, dR = elems[eR]["valL"], elems[eR]["derL"]      # right element, left end
        val = torch.cat([vL, vR])                          # (nL+nR,)
        der = torch.cat([dL, dR])
        orient = torch.cat([torch.ones(nL, dtype=CD), -torch.ones(nR, dtype=CD)])
        jump = orient * val                                # [b_p]
        flux = 0.5 * der                                   # {b_p'}
        F = HB * (
            -torch.outer(flux.conj(), jump)
            - torch.outer(jump.conj(), flux)
            + (sigma / h) * torch.outer(jump.conj(), jump)
        )
        idx = torch.cat([rng(e), rng(eR)])
        H[idx.unsqueeze(1), idx.unsqueeze(0)] += F

    herm = (H - H.conj().T).abs().max().item()
    H = 0.5 * (H + H.conj().T)
    return H, herm


# ---------------------------------------------------------------- driver
def run(L, n_at, V0, width, N_elem, buf_frac, Nloc, Ncore, M, sigma, ref, n_check):
    Vfunc = make_Vfunc(L, n_at, V0, width)
    h = L / N_elem
    buf = buf_frac * h
    elems = [build_alb(e * h, h, buf, Vfunc, Nloc, Ncore, M) for e in range(N_elem)]
    H, herm = assemble(elems, h, sigma)
    w = torch.linalg.eigvalsh(H).real.numpy()
    w = np.sort(w)[:n_check]
    err = np.abs(w - ref[:n_check])
    return w, err, herm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-at", type=int, default=6, help="atoms (Gaussian wells) in the box")
    ap.add_argument("--spacing", type=float, default=2.0, help="atom spacing -> L = n_at*spacing")
    ap.add_argument("--v0", type=float, default=1.0)
    ap.add_argument("--width", type=float, default=0.4)
    ap.add_argument("--n-elem", type=int, default=6)
    ap.add_argument("--buf-frac", type=float, default=1.0, help="buffer halo in units of h")
    ap.add_argument("--nloc", type=int, default=96, help="extended-element grid")
    ap.add_argument("--ncore", type=int, default=64, help="core quadrature grid")
    ap.add_argument("--ng-ref", type=int, default=256)
    ap.add_argument("--m-list", default="4,8,12,16,24")
    ap.add_argument("--sigma-list", default="1,5,20,50,200,500,2000")
    ap.add_argument("--sigma-scale", type=float, default=2.0,
                    help="coercive penalty scales with basis richness: sigma = scale*M^2")
    ap.add_argument("--n-check", type=int, default=8, help="lowest states compared")
    args = ap.parse_args()

    L = args.n_at * args.spacing
    Vfunc = make_Vfunc(L, args.n_at, args.v0, args.width)
    ref = reference_eigs(L, args.ng_ref, Vfunc)
    print(f"# 1D DG-ALB spike: L={L} n_at={args.n_at} N_elem={args.n_elem} "
          f"buf={args.buf_frac}h  (HB={HB}, Ha-like units)", flush=True)
    print(f"# plane-wave reference lowest {args.n_check}: "
          f"{np.array2string(ref[:args.n_check], precision=4)}", flush=True)

    m_list = [int(x) for x in args.m_list.split(",")]
    sig_list = [float(x) for x in args.sigma_list.split(",")]

    print(f"\n# sigma sweep at M={m_list[-1]} (coercivity vs accuracy):", flush=True)
    print(f"{'sigma':>8} {'max_err':>11} {'e0_err':>11} {'min_eig':>11} {'ref_e0':>11} "
          f"{'herm':>9}", flush=True)
    for s in sig_list:
        w, err, herm = run(L, args.n_at, args.v0, args.width, args.n_elem, args.buf_frac,
                           args.nloc, args.ncore, m_list[-1], s, ref, args.n_check)
        print(f"{s:>8.1f} {err.max():>11.2e} {err[0]:>11.2e} {w[0]:>11.5f} "
              f"{ref[0]:>11.5f} {herm:>9.1e}", flush=True)

    print(f"\n# M convergence at coercive sigma = {args.sigma_scale}*M^2 "
          f"(threshold grows with basis richness):", flush=True)
    print(f"{'M':>4} {'M/atom':>7} {'sigma':>8} {'max_err':>11} {'e0_err':>11} "
          f"{'herm':>9}", flush=True)
    for M in m_list:
        sig = args.sigma_scale * M ** 2
        w, err, herm = run(L, args.n_at, args.v0, args.width, args.n_elem, args.buf_frac,
                           args.nloc, args.ncore, M, sig, ref, args.n_check)
        print(f"{M:>4} {M*args.n_elem/args.n_at:>7.1f} {sig:>8.0f} {err.max():>11.2e} "
              f"{err[0]:>11.2e} {herm:>9.1e}", flush=True)


if __name__ == "__main__":
    main()
