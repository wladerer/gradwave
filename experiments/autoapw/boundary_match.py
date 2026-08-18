"""GATE C — the (L)APW value+slope boundary match, differentiable by construction.

This ties GATE A (the interstitial region) to GATE B (the muffin-tin interior). For one
plane wave of wavenumber q = |k+G|, the LAPW radial amplitude in channel l inside the sphere is

    R_l(r) = a_l u_l(r) + b_l u̇_l(r),     u̇_l = ∂u_l/∂E_l,

with (a_l, b_l) fixed by matching value AND slope to the plane wave's Rayleigh component j_l(qr)
at the muffin-tin radius R:

    a_l u_l(R)  + b_l u̇_l(R)  = j_l(qR)
    a_l u_l'(R) + b_l u̇_l'(R) = q j_l'(qR)

Two things make this the AutoAPW capstone:
  1. u̇_l — the energy derivative every LAPW code derives and codes by hand — is obtained here
     for free as an autograd derivative of the GATE-B Numerov solution w.r.t. E_l.
  2. j_l(qR) and its slope q j_l'(qR) are obtained as autograd derivatives of a closed-form j_l
     w.r.t. R, so R_MT flows differentiably into the match.
Hence (a_l, b_l) — and any energy built from them — are differentiable in E_l and R_MT with no
hand-derived Pulay/surface algebra. We verify C¹ continuity (machine precision, by construction)
and that ∂(a_l,b_l)/∂E_l and ∂(a_l,b_l)/∂R_MT are exact (autograd vs finite difference).

    uv run python experiments/autoapw/boundary_match.py [--device cpu]

Atomic units, uniform mesh (same prototype conventions as radial_solve.py).
"""

from __future__ import annotations

import argparse

import torch
from radial_solve import _lin_interp, numerov_outward
from torch import Tensor


def sph_jn_torch(l: int, x: Tensor) -> Tensor:
    """Closed-form spherical Bessel j_l for l in {0,1,2}, differentiable in x (x>0)."""
    if l == 0:
        return torch.sin(x) / x
    if l == 1:
        return torch.sin(x) / x**2 - torch.cos(x) / x
    if l == 2:
        return (3.0 / x**3 - 1.0 / x) * torch.sin(x) - (3.0 / x**2) * torch.cos(x)
    raise ValueError("prototype supports l<=2")


def _val_slope(u: Tensor, r: Tensor, r_mt: Tensor):
    """u(R_MT) and u'(R_MT) via a differentiable interpolant + central difference."""
    eps = r[1] - r[0]
    val = _lin_interp(r, u, r_mt)
    up = _lin_interp(r, u, r_mt + eps)
    um = _lin_interp(r, u, r_mt - eps)
    slope = (up - um) / (2 * eps)
    return val, slope


def lapw_match(l: int, q: float, energy: Tensor, r: Tensor, v: Tensor, r_mt: Tensor):
    """Return (a_l, b_l) and the value/slope continuity residuals, all differentiable in
    ``energy`` and ``r_mt``. u̇_l value/slope come from autograd of the GATE-B solution."""
    u = numerov_outward(l, energy, r, v)
    val, slope = _val_slope(u, r, r_mt)

    # u̇_l value and slope = ∂/∂E of u_l value and slope (create_graph so a_l,b_l stay diffable).
    (udot_val,) = torch.autograd.grad(val, energy, create_graph=True, retain_graph=True)
    (udot_slope,) = torch.autograd.grad(slope, energy, create_graph=True, retain_graph=True)

    # plane-wave Rayleigh component value and slope at R: j_l(qR) and q j_l'(qR) = d/dR j_l(qR).
    tval = sph_jn_torch(l, q * r_mt)
    (tslope,) = torch.autograd.grad(tval, r_mt, create_graph=True, retain_graph=True)

    w = val * udot_slope - slope * udot_val          # Wronskian
    a = (tval * udot_slope - tslope * udot_val) / w
    b = (val * tslope - slope * tval) / w

    res_val = a * val + b * udot_val - tval
    res_slope = a * slope + b * udot_slope - tslope
    return a, b, res_val, res_slope


def _fd_rel(autograd_val, plus, minus, h):
    fd = (plus - minus) / (2 * h)
    denom = max(abs(float(autograd_val)), 1e-8)
    return abs(float(autograd_val) - float(fd)) / denom


def run(device, N=1200, Rmax=6.0, r_min=1e-3, q=2.0):
    device = torch.device(device)
    r = torch.linspace(r_min, Rmax, N, dtype=torch.float64, device=device)
    v = torch.zeros_like(r)  # free-particle interior keeps the check analytic-friendly
    out = {}
    for l in (0, 1, 2):
        r_mt0 = 3.30
        E0 = 0.5

        # continuity residuals (should be ~machine eps: (a,b) solve the 2x2 exactly)
        E = torch.tensor(E0, dtype=torch.float64, device=device, requires_grad=True)
        r_mt = torch.tensor(r_mt0, dtype=torch.float64, device=device, requires_grad=True)
        a, b, rv, rs = lapw_match(l, q, E, r, v, r_mt)

        # differentiability of (a,b) w.r.t E and R_MT: autograd vs finite difference (relative)
        (da_dE,) = torch.autograd.grad(a, E, retain_graph=True)
        (db_dR,) = torch.autograd.grad(b, r_mt, retain_graph=True)

        def leaf(x):
            return torch.tensor(x, dtype=torch.float64, device=device, requires_grad=True)

        def ab(el, rm, _l=l):
            aa, bb, _, _ = lapw_match(_l, q, el, r, v, rm)
            return aa.detach(), bb.detach()

        hE, hR = 1e-5, 1e-5
        a_Ep, _ = ab(leaf(E0 + hE), leaf(r_mt0))
        a_Em, _ = ab(leaf(E0 - hE), leaf(r_mt0))
        err_a_E = _fd_rel(da_dE, a_Ep, a_Em, hE)

        _, b_Rp = ab(leaf(E0), leaf(r_mt0 + hR))
        _, b_Rm = ab(leaf(E0), leaf(r_mt0 - hR))
        err_b_R = _fd_rel(db_dR, b_Rp, b_Rm, hR)

        out[l] = {
            "res_val": float(rv.detach().abs()),
            "res_slope": float(rs.detach().abs()),
            "err_da_dE_rel": err_a_E,
            "err_db_dR_rel": err_b_R,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = run(args.device)
    print(f"\nAutoAPW GATE C — LAPW value+slope boundary match   [{args.device}]\n")
    print(f"  {'l':>2} | {'C1 value resid':>15} | {'C1 slope resid':>15} | "
          f"{'da/dE rel(ad-fd)':>17} | {'db/dR rel(ad-fd)':>17}")
    cont_ok = True
    diff_ok = True
    for l, rec in out.items():
        print(f"  {l:>2} | {rec['res_val']:>15.2e} | {rec['res_slope']:>15.2e} | "
              f"{rec['err_da_dE_rel']:>17.2e} | {rec['err_db_dR_rel']:>17.2e}")
        cont_ok = cont_ok and rec["res_val"] < 1e-10 and rec["res_slope"] < 1e-10
        diff_ok = diff_ok and rec["err_da_dE_rel"] < 1e-4 and rec["err_db_dR_rel"] < 1e-4

    print("\n  VERDICT:")
    print(f"    C1 continuity of the matched mixed function (by construction): "
          f"{'PASS' if cont_ok else 'FAIL'}")
    print(f"    match coefficients differentiable in E_l and R_MT (autograd) : "
          f"{'PASS' if diff_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
