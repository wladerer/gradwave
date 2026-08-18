"""GATE D — a working DIFFERENTIABLE all-electron atom (the AutoAPW payoff, in miniature).

Gates A-C validated the differentiable primitives of an augmented basis. This gate shows the
capability those primitives exist to serve — and the user's north star directly: a differentiable
all-electron reference whose eigenvalues have EXACT autograd gradients w.r.t. the potential
parameters you would fit a pseudopotential (or a functional) against.

We solve the true all-electron radial problem (bare Coulomb V=-Z/r, no pseudopotential) by
root-finding the bound-state condition u(R_box; E)=0 on a batched Numerov shooting solution
(the recurrence vectorized over trial energies, so an eigenvalue scan is one pass). Then:

  1. eigenvalues E_nl reproduce the analytic hydrogen spectrum E_n = -Z²/2n²,
  2. the eigenvalue is DIFFERENTIABLE in Z: dE*/dZ via the implicit-function theorem with autograd
     partials of the shooting residual — matches the analytic -Z/n²,
  3. for a screened (Yukawa) potential V=-Z e^{-λr}/r — a stand-in for a tunable pseudopotential
     parameter — dE*/dλ falls out of the same machinery (checked vs finite difference). THIS is
     "fit a pseudo/functional parameter against the all-electron oracle by gradient descent".

    uv run python experiments/autoapw/allelectron_atom.py [--device cpu]

Atomic units (Hartree), uniform mesh.
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor


def endpoint(l: int, energies: Tensor, r: Tensor, v: Tensor) -> Tensor:
    """u(R_box; E) for a batch of trial energies, via a rolling batched Numerov.

    energies: (B,), r: (N,), v: (N,)  ->  (B,) endpoint values. Autograd-clean in energies and
    in any parameter v depends on (the seed u~r^{l+1} is an E/param-independent constant)."""
    h = r[1] - r[0]
    cent = l * (l + 1) / (r * r)                              # (N,)
    g = cent[None, :] + 2.0 * (v[None, :] - energies[:, None])  # (B, N)
    f = 1.0 - (h * h / 12.0) * g
    a = torch.full_like(energies, float(r[0] ** (l + 1)))     # u_0
    b = torch.full_like(energies, float(r[1] ** (l + 1)))     # u_1
    for i in range(2, r.shape[0]):
        c = ((12.0 - 10.0 * f[:, i - 1]) * b - f[:, i - 2] * a) / f[:, i]
        a, b = b, c
    return b


def find_eigenvalues(l, r, v, e_lo, e_hi, n_scan=160, want=2, n_bisect=50):
    """Bracket eigenvalues by sign changes of u(R_box;E); refine all brackets by batched
    bisection."""
    grid = torch.linspace(e_lo, e_hi, n_scan, dtype=r.dtype, device=r.device)
    with torch.no_grad():
        vals = endpoint(l, grid, r, v)
    los, his = [], []
    for i in range(n_scan - 1):
        if float(vals[i]) == 0.0 or float(vals[i]) * float(vals[i + 1]) < 0:
            los.append(grid[i])
            his.append(grid[i + 1])
            if len(los) >= want:
                break
    if not los:
        return []
    lo = torch.stack(los)
    hi = torch.stack(his)
    with torch.no_grad():
        flo = endpoint(l, lo, r, v)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            fmid = endpoint(l, mid, r, v)
            left = (flo * fmid) <= 0
            hi = torch.where(left, mid, hi)
            lo = torch.where(left, lo, mid)
            flo = torch.where(left, flo, fmid)
    return list(0.5 * (lo + hi))


def dEstar_dparam(l, e_star, r, v_of_param, param):
    """dE*/dparam by the implicit function theorem: E* solves F(E,param)=u(R_box)=0, so
    dE*/dparam = -(∂F/∂param)/(∂F/∂E), both from autograd at the converged root."""
    e = e_star.detach().reshape(1).clone().requires_grad_(True)
    p = param.detach().clone().requires_grad_(True)
    f = endpoint(l, e, r, v_of_param(p, r))[0]
    (f_e,) = torch.autograd.grad(f, e, retain_graph=True)
    (f_p,) = torch.autograd.grad(f, p)
    return float(-f_p / f_e[0])


def run(device):
    device = torch.device(device)
    r = torch.linspace(1e-4, 25.0, 3000, dtype=torch.float64, device=device)

    print(f"\nAutoAPW GATE D — differentiable all-electron atom   [{device}]\n")

    # ---- 1 & 2: hydrogen spectrum + exact dE/dZ ------------------------------------------
    Z = torch.tensor(1.0, dtype=torch.float64, device=device)
    print("  Hydrogen (Z=1), bare Coulomb -Z/r:")
    print(f"  {'state':>6} | {'E numerov':>12} | {'E exact':>12} | {'dE/dZ autograd':>15} | "
          f"{'dE/dZ exact':>12}")
    ok_spec = ok_grad = True
    for l, e_lo, e_hi, ns in [(0, -0.6, -0.02, 2), (1, -0.2, -0.02, 1)]:
        eigs = find_eigenvalues(l, r, -Z / r, e_lo, e_hi, want=ns)
        for k, e_star in enumerate(eigs):
            n = (k + 1) if l == 0 else (k + 2)
            e_exact = -0.5 * float(Z) ** 2 / n**2
            dez = dEstar_dparam(l, e_star, r, lambda zz, rr: -zz / rr, Z)
            dez_exact = -float(Z) / n**2
            print(f"  {f'{n}{chr(115 + l)}' if l < 1 else f'{n}p':>6} | {float(e_star):>12.6f} | "
                  f"{e_exact:>12.6f} | {dez:>15.6f} | {dez_exact:>12.6f}")
            ok_spec = ok_spec and abs(float(e_star) - e_exact) < 5e-4
            ok_grad = ok_grad and abs(dez - dez_exact) < 2e-3

    # ---- 3: screened (Yukawa) potential — dE/dλ = the pseudo-fitting gradient -------------
    Zy = torch.tensor(2.0, dtype=torch.float64, device=device)
    lam = torch.tensor(0.5, dtype=torch.float64, device=device)

    def yuk(ll, rr):
        return -Zy * torch.exp(-ll * rr) / rr

    e1s = find_eigenvalues(0, r, yuk(lam, r), -2.2, -0.05, want=1)[0]
    dlam = dEstar_dparam(0, e1s, r, yuk, lam)
    h = 1e-5
    ep = find_eigenvalues(0, r, yuk(lam + h, r), -2.2, -0.05, want=1)[0]
    em = find_eigenvalues(0, r, yuk(lam - h, r), -2.2, -0.05, want=1)[0]
    dlam_fd = float((ep - em) / (2 * h))
    print("\n  Screened Coulomb -Z e^{-λr}/r  (Z=2, λ=0.5) — the 'tunable pseudo param' case:")
    print(f"    1s eigenvalue           = {float(e1s):.6f}")
    print(f"    dE(1s)/dλ  autograd     = {dlam:.6f}")
    print(f"    dE(1s)/dλ  finite diff  = {dlam_fd:.6f}")
    ok_yuk = abs(dlam - dlam_fd) < 1e-4

    print("\n  VERDICT:")
    print(f"    hydrogen spectrum E_nl == -Z^2/2n^2          : {'PASS' if ok_spec else 'FAIL'}")
    print(f"    exact autograd dE_nl/dZ == -Z/n^2            : {'PASS' if ok_grad else 'FAIL'}")
    print(f"    differentiable oracle dE/dλ (autograd == FD) : {'PASS' if ok_yuk else 'FAIL'}")
    print("\n  => a differentiable all-electron reference: eigenvalue gradients w.r.t. potential")
    print("     parameters come out exact, ready to fit a pseudopotential/functional against.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.device)


if __name__ == "__main__":
    main()
