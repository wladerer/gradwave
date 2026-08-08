"""DG-ALB spike, 3D (kill point 2, core risk): does the SIPG interior-penalty
assembly reproduce the plane-wave spectrum in 3D, with 2D surface quadrature on
element faces and a real 3D potential?

The 1D spike (dgalb_spike_1d.py) validated the SIPG form: correct signs
(Hermitian), coercivity threshold, sub-meV vs a plane-wave reference. 3D adds the
genuinely new machinery the report flagged — 6 faces per cubic element, each a 2D
face needing SURFACE quadrature of the jump / average-normal-flux / penalty
terms. This standalone test settles whether that machinery converges, before it
is ported into gradwave (nonlocal pseudopotentials are then an additive term that
does not touch the face assembly).

Setup: periodic cubic box, one Gaussian well per element (n_side^3 "atoms").
Reference: dense plane-wave diagonalisation. DG-ALB: per extended-element
(core + buffer halo) plane-wave solve -> M lowest -> core-restrict (3D indicator)
-> canonical-orthonormalise -> assemble with the 3D SIPG form:

    a(u,v) = sum_K  HB grad(u)*.grad(v)                               (volume)
           - sum_F  HB int_F ( {d_n u}* [v] + [u]* {d_n v} ) dA       (consistency+sym)
           + sum_F  HB (sigma/h) int_F [u]* [v] dA                    (penalty)

face F = a 2D square shared by two cubic elements; d_n = normal derivative along
the face axis; int_F is a 2D quadrature. ALBs orthonormal by the non-overlapping
core partition -> standard eigenproblem.

PASS = DG eigenvalues converge to the reference to ~sub-meV at a modest M above a
coercivity sigma, operator Hermitian to ~1e-12.

Atomic-like units (HB=1/2, Ha-like). numpy + torch only (pre-integration spike).
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
CD = torch.complex128
HB = 0.5


# ---------------------------------------------------------------- potential
def make_Vfunc(L, centers, V0, width):
    def V(pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        out = np.zeros(pts.shape[0])
        for c in centers:
            d = pts - c
            d -= L * np.round(d / L)                      # min-image (periodic)
            out += -V0 * np.exp(-(d ** 2).sum(1) / width ** 2)
        return out
    return V


# ---------------------------------------------------------------- PW modes / eval
def modes(n, w):
    """Integer modes and Cartesian G for an n-per-axis PW box of side w."""
    m1 = np.round(np.fft.fftfreq(n, d=1.0 / n)).astype(int)
    mm = np.array(list(itertools.product(m1, m1, m1)))    # (n^3, 3) integer modes
    G = 2 * np.pi * mm / w
    return mm, G


def beta_vals(center, pts, rc):
    """Localized Gaussian KB projector beta_a(r) = exp(-|r-tau|^2/rc^2).
    No periodic wrap: projectors are used well inside their box (rc << box)."""
    return np.exp(-((np.asarray(pts) - center) ** 2).sum(-1) / rc ** 2)


def pw_hamiltonian(mm, G, w, n, Vfunc, corner, vnl=None):
    """Dense PW Hamiltonian H[a,b] = Vhat[m_a-m_b] + HB|G_a|^2 delta on the box
    [corner, corner+w]^3 sampled with n points/axis. Optional separable nonlocal
    term vnl=(centers, d0, rc): H += d0 * Vbox * sum_a bhat_a outer conj(bhat_a),
    bhat_a = fft(beta_a)/n^3 (coeffs in the e^{iG.(r-corner)} basis the states use)."""
    lin = np.arange(n) * w / n
    grid = np.array(list(itertools.product(lin, lin, lin))) + corner
    Vx = Vfunc(grid).reshape(n, n, n)
    Vhat = np.fft.fftn(Vx) / n ** 3                       # coeff of e^{i G.r}
    a = (mm[:, None, :] - mm[None, :, :]) % n             # (Nk,Nk,3)
    H = Vhat[a[..., 0], a[..., 1], a[..., 2]].astype(complex)
    H[np.diag_indices(len(mm))] += HB * (G ** 2).sum(1)
    if vnl is not None:
        centers, d0, rc = vnl
        Vbox = w ** 3
        for c in centers:
            bhat = (np.fft.fftn(beta_vals(c, grid, rc).reshape(n, n, n)) / n ** 3).reshape(-1)
            H += d0 * Vbox * np.outer(bhat, bhat.conj())
    return H


def eval3d(coeffs, G, corner, pts, dax=None):
    """f(pts) = sum_G coeffs[G,:] e^{iG.(pts-corner)}, or its dax-derivative."""
    ph = np.exp(1j * ((np.asarray(pts) - corner) @ G.T))  # (P, Nk)
    if dax is not None:
        ph = ph * (1j * G[:, dax])[None, :]
    return ph @ coeffs                                    # (P, M)


# ---------------------------------------------------------------- ALB per element
def build_alb(xe, h, buf, Vfunc, nmode, ncore, nface, M, orth_tol=1e-8, vnl=None):
    corner = xe - buf
    w = h + 2 * buf
    mm, G = modes(nmode, w)
    # the element's own atom sits at the core centre; its projector is included
    # in the local solve so the ALBs adapt to the nonlocal potential.
    atom = xe + h / 2
    vnl_loc = (([atom],) + tuple(vnl[1:])) if vnl is not None else None
    Hloc = pw_hamiltonian(mm, G, w, nmode, Vfunc, corner, vnl=vnl_loc)
    ev, U = np.linalg.eigh(Hloc)
    C = U[:, np.argsort(ev.real)[:M]]                     # (Nk, M) PW coeffs

    off1 = (np.arange(ncore) + 0.5) * h / ncore
    xc = np.array(list(itertools.product(off1, off1, off1))) + xe  # (ncore^3, 3)
    dV = (h / ncore) ** 3
    Phi = eval3d(C, G, corner, xc)                        # (Nc, M)
    dPhi = [eval3d(C, G, corner, xc, dax=a) for a in range(3)]

    S = (Phi.conj().T @ Phi) * dV
    s, Q = np.linalg.eigh(S)
    keep = s > orth_tol * s.max()
    X = Q[:, keep] * (s[keep] ** -0.5)[None, :]           # (M, n_keep)
    Phi = Phi @ X
    dPhi = [d @ X for d in dPhi]

    # nonlocal becp in the ALB basis: B[i] = <b_i|beta_atom> = sum_core b_i* beta dV.
    # block contribution (added in assemble) is d0 * B outer conj(B).
    bnl = None
    if vnl is not None:
        _, d0, rc = vnl
        beta_c = torch.tensor(beta_vals(atom, xc, rc), dtype=CD)
        B = (torch.tensor(Phi, dtype=CD).conj().T @ beta_c) * dV  # (n_keep,)
        bnl = (B, float(d0))

    # face quadrature grids: 6 faces (axis, side). side 0 = low (xe), 1 = high (xe+h).
    fl = (np.arange(nface) + 0.5) * h / nface
    dA = (h / nface) ** 2
    faces = {}
    for ax in range(3):
        tan = [t for t in range(3) if t != ax]
        for side in (0, 1):
            pts = np.zeros((nface * nface, 3))
            tg = np.array(list(itertools.product(fl, fl)))
            pts[:, tan[0]] = xe[tan[0]] + tg[:, 0]
            pts[:, tan[1]] = xe[tan[1]] + tg[:, 1]
            pts[:, ax] = xe[ax] + (h if side else 0.0)
            val = eval3d(C, G, corner, pts) @ X           # (nface^2, n_keep)
            nder = eval3d(C, G, corner, pts, dax=ax) @ X  # normal derivative
            faces[(ax, side)] = (torch.tensor(val, dtype=CD),
                                 torch.tensor(nder, dtype=CD))

    Vc = Vfunc(xc)
    return {
        "Phi": torch.tensor(Phi, dtype=CD),
        "dPhi": [torch.tensor(d, dtype=CD) for d in dPhi],
        "Vc": torch.tensor(Vc), "dV": dV, "n": int(keep.sum()),
        "faces": faces, "dA": dA, "bnl": bnl,
    }


# ---------------------------------------------------------------- global assembly
def assemble(elems, idx_of, n_side, h, sigma):
    sizes = [el["n"] for el in elems]
    off = np.cumsum([0] + sizes)
    D = int(off[-1])
    H = torch.zeros(D, D, dtype=CD)

    def rng(e):
        return torch.arange(off[e], off[e + 1])

    for e, el in enumerate(elems):
        T = HB * sum((d.conj().T @ d) for d in el["dPhi"]) * el["dV"]
        Vm = (el["Phi"].conj().T @ (el["Vc"].to(CD)[:, None] * el["Phi"])) * el["dV"]
        i = rng(e)
        H[i.unsqueeze(1), i.unsqueeze(0)] += T + Vm
        if el["bnl"] is not None:                          # nonlocal KB: d0 * B outer B^H
            B, d0 = el["bnl"]
            H[i.unsqueeze(1), i.unsqueeze(0)] += d0 * torch.outer(B, B.conj())

    # each interior face counted once: for every element, its +axis face with the
    # neighbour in +axis. e uses its high face (ax,1); neighbour uses low (ax,0).
    grid = list(itertools.product(range(n_side), repeat=3))
    for cell in grid:
        e = idx_of[cell]
        for ax in range(3):
            nb = list(cell); nb[ax] = (nb[ax] + 1) % n_side
            eR = idx_of[tuple(nb)]
            vL, dL = elems[e]["faces"][(ax, 1)]           # e high face, normal +ax
            vR, dR = elems[eR]["faces"][(ax, 0)]          # neighbour low face, same plane
            nL, nR = vL.shape[1], vR.shape[1]
            dA = elems[e]["dA"]
            jump = torch.cat([vL, -vR], dim=1)            # (Npts, nL+nR): [b], orient +/-
            flux = 0.5 * torch.cat([dL, dR], dim=1)       # {d_n b}, fixed +ax orientation
            F = HB * dA * (
                -flux.conj().T @ jump
                - jump.conj().T @ flux
                + (sigma / h) * jump.conj().T @ jump
            )
            ii = torch.cat([rng(e), rng(eR)])
            H[ii.unsqueeze(1), ii.unsqueeze(0)] += F

    herm = (H - H.conj().T).abs().max().item()
    return 0.5 * (H + H.conj().T), herm


# ---------------------------------------------------------------- driver
def reference(L, nmode, Vfunc, vnl=None):
    mm, G = modes(nmode, L)
    H = pw_hamiltonian(mm, G, L, nmode, Vfunc, np.zeros(3), vnl=vnl)
    return np.sort(np.linalg.eigvalsh(H).real)


def build_all(L, n_side, buf_frac, Vfunc, nmode, ncore, nface, M, vnl=None):
    h = L / n_side
    idx_of = {}
    cells = list(itertools.product(range(n_side), repeat=3))
    for i, c in enumerate(cells):
        idx_of[c] = i
    elems = [build_alb(np.array(c) * h, h, buf_frac * h, Vfunc, nmode, ncore, nface, M,
                       vnl=vnl)
             for c in cells]
    return elems, idx_of, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-side", type=int, default=2, help="elements/atoms per axis")
    ap.add_argument("--spacing", type=float, default=2.2)
    ap.add_argument("--v0", type=float, default=1.0)
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--buf-frac", type=float, default=1.0)
    ap.add_argument("--nmode", type=int, default=14, help="PW modes/axis (local & ref)")
    ap.add_argument("--ncore", type=int, default=10, help="core quad points/axis")
    ap.add_argument("--nface", type=int, default=10, help="face quad points/axis")
    ap.add_argument("--m-list", default="8,16,24,32")
    ap.add_argument("--sigma-list", default="20,100,500,2000")
    ap.add_argument("--sigma-scale", type=float, default=2.0, help="coercive sigma = scale*M^2")
    ap.add_argument("--n-check", type=int, default=6)
    ap.add_argument("--vnl-d0", type=float, default=0.0,
                    help="nonlocal KB strength (0 = off; e.g. -0.5 attractive)")
    ap.add_argument("--vnl-rc", type=float, default=0.35, help="KB projector radius")
    args = ap.parse_args()

    L = args.n_side * args.spacing
    centers = [(np.array(c) + 0.5) * args.spacing
               for c in itertools.product(range(args.n_side), repeat=3)]
    Vfunc = make_Vfunc(L, centers, args.v0, args.width)
    vnl = (centers, args.vnl_d0, args.vnl_rc) if args.vnl_d0 != 0.0 else None
    ref = reference(L, args.nmode, Vfunc, vnl=vnl)
    print(f"# 3D DG-ALB spike: L={L:.2f} n_side={args.n_side} ({args.n_side**3} atoms) "
          f"buf={args.buf_frac}h nmode={args.nmode} "
          f"VNL(d0={args.vnl_d0},rc={args.vnl_rc}) (HB={HB}, Ha-like)", flush=True)
    print(f"# plane-wave reference lowest {args.n_check}: "
          f"{np.array2string(ref[:args.n_check], precision=4)}", flush=True)

    m_list = [int(x) for x in args.m_list.split(",")]
    sig_list = [float(x) for x in args.sigma_list.split(",")]

    print(f"\n# sigma sweep at M={m_list[-1]}:", flush=True)
    print(f"{'sigma':>8} {'max_err':>11} {'e0_err':>11} {'min_eig':>11} {'ref_e0':>11} "
          f"{'herm':>9}", flush=True)
    elems, idx_of, h = build_all(L, args.n_side, args.buf_frac, Vfunc, args.nmode,
                                 args.ncore, args.nface, m_list[-1], vnl=vnl)
    for s in sig_list:
        H, herm = assemble(elems, idx_of, args.n_side, h, s)
        w = np.sort(torch.linalg.eigvalsh(H).real.numpy())[:args.n_check]
        err = np.abs(w - ref[:args.n_check])
        print(f"{s:>8.1f} {err.max():>11.2e} {err[0]:>11.2e} {w[0]:>11.5f} "
              f"{ref[0]:>11.5f} {herm:>9.1e}", flush=True)

    print(f"\n# M convergence at coercive sigma = {args.sigma_scale}*M^2:", flush=True)
    print(f"{'M':>4} {'sigma':>8} {'D':>5} {'max_err':>11} {'e0_err':>11} {'herm':>9}",
          flush=True)
    for M in m_list:
        elems, idx_of, h = build_all(L, args.n_side, args.buf_frac, Vfunc, args.nmode,
                                     args.ncore, args.nface, M, vnl=vnl)
        s = args.sigma_scale * M ** 2
        H, herm = assemble(elems, idx_of, args.n_side, h, s)
        w = np.sort(torch.linalg.eigvalsh(H).real.numpy())[:args.n_check]
        err = np.abs(w - ref[:args.n_check])
        print(f"{M:>4} {s:>8.0f} {H.shape[0]:>5} {err.max():>11.2e} {err[0]:>11.2e} "
              f"{herm:>9.1e}   err={np.array2string(err, precision=1)}", flush=True)


if __name__ == "__main__":
    main()
