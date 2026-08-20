"""GATE B — a differentiable radial Schrödinger solver (the muffin-tin interior).

Inside a muffin tin an APW/LAPW basis carries the radial functions u_l(r; E_l) (and, for LAPW,
the energy derivative u̇_l = ∂u_l/∂E_l) obtained by integrating the scalar radial Schrödinger
equation at a linearization energy E_l:

    u''(r) = [ l(l+1)/r² + 2(V(r) - E) ] u(r),      u(r) = r · R_l(r),

in Hartree atomic units (ℏ = m = 1, so the kinetic prefactor makes V and E enter as 2(V-E),
and the free-particle wavenumber is k = √(2E)). We integrate outward with Numerov on a uniform
mesh, seeding the origin with the regular behaviour u ~ r^{l+1}.

The whole point for AutoAPW: E and R_MT enter the recurrence / the boundary evaluation as
ordinary torch ops, so ∂u(R_MT)/∂E, the log-derivative D_l(R_MT) = R u'(R)/u(R), and their
gradients w.r.t. E_l and R_MT all fall out of autograd — the "differentiable linearization
energies and muffin-tin radii" the pitch promises. Validated here against the closed-form
free-particle solution u_l(r) = r j_l(kr) and a Coulomb (hydrogen 1s) case.

    uv run python experiments/autoapw/radial_solve.py [--device cpu]

Prototype (atomic units, uniform mesh). Promotion to src/gradwave with gradwave's eV/Å units
(HBAR2_2M) and UPF log meshes is GATE B's follow-on; see STATUS.md.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import Tensor


def numerov_outward(l: int, energy: Tensor, r: Tensor, v: Tensor) -> Tensor:
    """Integrate u'' = [l(l+1)/r² + 2(V-E)] u outward on a uniform mesh via Numerov.

    Args:
        l: angular momentum (int).
        energy: scalar tensor E (Hartree). May require grad.
        r: (N,) uniform radial mesh, r[0] > 0 (Hartree bohr).
        v: (N,) potential V(r) (Hartree).

    Returns:
        u: (N,) the regular solution, seeded u ~ r^{l+1} near the origin (arbitrary overall
        scale, fixed by the seed).
    """
    n = r.shape[0]
    h = r[1] - r[0]
    g = l * (l + 1) / (r * r) + 2.0 * (v - energy)          # (N,)
    f = 1.0 - (h * h / 12.0) * g                             # Numerov F_n

    u = torch.zeros(n, dtype=r.dtype, device=r.device)
    # Regular seed: u ~ r^{l+1} on the first two points (leading order).
    u = u.clone()
    seed0 = r[0] ** (l + 1)
    seed1 = r[1] ** (l + 1)
    us = [seed0, seed1]
    for i in range(2, n):
        nxt = ((12.0 - 10.0 * f[i - 1]) * us[i - 1] - f[i - 2] * us[i - 2]) / f[i]
        us.append(nxt)
    return torch.stack(us)


def _lin_interp(r: Tensor, y: Tensor, x: Tensor) -> Tensor:
    """Differentiable linear interpolation of y(r) at query point x (scalar tensor)."""
    # index of the mesh cell containing x
    idx = torch.searchsorted(r, x).clamp(1, r.shape[0] - 1)
    r0 = r[idx - 1]
    r1 = r[idx]
    y0 = y[idx - 1]
    y1 = y[idx]
    t = (x - r0) / (r1 - r0)
    return y0 + t * (y1 - y0)


def value_and_logderiv(l: int, energy: Tensor, r: Tensor, v: Tensor, r_mt: Tensor):
    """u(R_MT) and the log-derivative D_l = R_MT · u'(R_MT) / u(R_MT), both differentiable in
    energy and r_mt. u'(R_MT) via a differentiable central difference of the interpolant."""
    u = numerov_outward(l, energy, r, v)
    val = _lin_interp(r, u, r_mt)
    eps = (r[1] - r[0])
    up = _lin_interp(r, u, r_mt + eps)
    um = _lin_interp(r, u, r_mt - eps)
    deriv = (up - um) / (2 * eps)
    dlog = r_mt * deriv / val
    return val, dlog, u


# --------------------------------------------------------------------------- validation
def _spherical_jn(l: int, x):
    import numpy as np
    from scipy.special import spherical_jn

    return spherical_jn(l, np.asarray(x))


def check_free_particle(device, N=4000, Rmax=6.0, r_min=1e-3):
    """V=0: the regular solution is u_l(r) = C · r j_l(kr), k=√(2E).

    (a) shape: Numerov u_l matches r j_l(kr) to ~mesh-order accuracy;
    (b) differentiability: torch.autograd.gradcheck confirms ∂u(R_MT)/∂E and ∂u(R_MT)/∂R_MT are
        exact (gradcheck compares autograd to a carefully-scaled finite difference).
    """
    r = torch.linspace(r_min, Rmax, N, dtype=torch.float64, device=device)
    v = torch.zeros_like(r)
    results = {}
    for l in (0, 1, 2):
        E = torch.tensor(0.5, dtype=torch.float64, device=device)  # k=1
        k = math.sqrt(2 * E.item())
        u = numerov_outward(l, E, r, v)
        rn = r.detach().cpu().numpy()
        ana = rn * _spherical_jn(l, k * rn)
        iref = int(0.55 * N)
        scale = u[iref] / torch.tensor(ana[iref], device=device, dtype=torch.float64)
        u_scaled = u / scale
        ana_t = torch.tensor(ana, device=device, dtype=torch.float64)
        mask = r > (Rmax * 0.1)
        rel = ((u_scaled[mask] - ana_t[mask]).abs().max() / ana_t[mask].abs().max()).item()

        # gradcheck ∂u(R_MT)/∂(E, R_MT) on a coarser mesh (speed). r_mt kept mid-cell so the FD
        # perturbation never crosses a mesh knot of the piecewise-linear interpolant.
        Ng = 500
        rg = torch.linspace(r_min, Rmax, Ng, dtype=torch.float64, device=device)
        vg = torch.zeros_like(rg)
        r_mt0 = float(rg[int(0.55 * Ng)] + 0.5 * (rg[1] - rg[0]))

        def val_fn(e, rmt, _l=l, _r=rg, _v=vg):
            return _lin_interp(_r, numerov_outward(_l, e, _r, _v), rmt)

        e_in = torch.tensor(0.5, dtype=torch.float64, device=device, requires_grad=True)
        rmt_in = torch.tensor(r_mt0, dtype=torch.float64, device=device, requires_grad=True)
        gc = torch.autograd.gradcheck(val_fn, (e_in, rmt_in), eps=1e-6, atol=1e-5, rtol=1e-3)

        # log-derivative D_l is finite and differentiable (used by the GATE C boundary match)
        rmt_c = torch.tensor(r_mt0, dtype=torch.float64, device=device, requires_grad=True)
        _, dlog, _ = value_and_logderiv(l, E, r, v, rmt_c)
        (dD_dR,) = torch.autograd.grad(dlog, rmt_c)
        dlog_finite = bool(torch.isfinite(dlog) and torch.isfinite(dD_dR))

        results[l] = {"shape_rel_err": rel, "gradcheck": bool(gc),
                      "logderiv": float(dlog), "dlog_finite": dlog_finite}
    return results


def check_hydrogen_1s(device, N=6000, Rmax=25.0, r_min=1e-4, Z=1.0):
    """Coulomb V=-Z/r at E = -Z²/2: the regular l=0 outward solution must match the 1s shape
    u_1s(r) ∝ r e^{-Zr} (before the classically-forbidden blow-up at large r)."""
    r = torch.linspace(r_min, Rmax, N, dtype=torch.float64, device=device)
    v = -Z / r
    E = torch.tensor(-0.5 * Z * Z, dtype=torch.float64, device=device)
    u = numerov_outward(0, E, r, v)
    ana = r * torch.exp(-Z * r)
    # match scale in the physically meaningful region (before the outward integration's
    # exponential-growth contamination near Rmax); compare on r in [0.5, 8].
    mask = (r > 0.5) & (r < 8.0)
    iref = int(mask.nonzero()[len(mask.nonzero()) // 2])
    scale = u[iref] / ana[iref]
    rel = ((u[mask] / scale - ana[mask]).abs().max() / ana[mask].abs().max()).item()
    return {"shape_rel_err": rel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f"\nAutoAPW GATE B — differentiable radial Schrödinger solver   [{device}]\n")

    fp = check_free_particle(device)
    print("  Free particle V=0  (regular solution u_l = C·r j_l(kr), k=1):")
    print(f"  {'l':>2} | {'shape rel.err':>14} | {'gradcheck ∂u/∂(E,R)':>22} | {'D_l finite':>12}")
    shape_ok = True
    grad_ok = True
    for l, rec in fp.items():
        print(f"  {l:>2} | {rec['shape_rel_err']:>14.2e} | "
              f"{('PASS' if rec['gradcheck'] else 'FAIL'):>22} | {str(rec['dlog_finite']):>12}")
        shape_ok = shape_ok and rec["shape_rel_err"] < 1e-4
        grad_ok = grad_ok and rec["gradcheck"] and rec["dlog_finite"]

    h = check_hydrogen_1s(device)
    print(f"\n  Coulomb V=-1/r, hydrogen 1s at E=-0.5:  shape rel.err = {h['shape_rel_err']:.2e}")
    hyd_ok = h["shape_rel_err"] < 5e-3

    v_shape = "PASS" if shape_ok else "FAIL"
    v_grad = "PASS" if grad_ok else "FAIL"
    v_hyd = "PASS" if hyd_ok else "FAIL"
    print("\n  VERDICT:")
    print(f"    Numerov matches analytic free-particle j_l         : {v_shape}")
    print(f"    autograd d u(R_MT)/dE, d u/d R_MT exact (gradcheck): {v_grad}")
    print(f"    Coulomb hydrogen-1s outward solution shape         : {v_hyd}")


if __name__ == "__main__":
    main()
