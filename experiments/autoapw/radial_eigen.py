"""PROD-B — robust radial eigensolver via inward-outward matching (production-grade).

Outward-only shooting (radial_log.bound_states) confines diffuse states and contaminates deep-core
states, because the growing solution dominates the classically-forbidden tail. The standard cure,
used by every atomic code, integrates OUTWARD from the origin and INWARD from r_max and matches
value + log-derivative at the classical turning point r_c (where V_eff(r_c)=E):

    Δ(E) = u_out'(r_c)/u_out(r_c) - u_in'(r_c)/u_in(r_c)   ->   0 at an eigenvalue.

Root-finding Δ(E) with a node count (nodes = n-l-1) gives every bound state, deep or diffuse.
Works on the eV/Å log mesh with the q = u/√r transform from radial_log.

Verified against hydrogen (Z=1) AND a deep hydrogenic Z=8 1s at -870 eV, which outward-only
shooting gets wrong.

    uv run python experiments/autoapw/radial_eigen.py
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import E2, HBAR2_2M, RY_EV


def _match_mismatch(l, E, r, dx, v):
    """Integrate q outward and inward, match at the turning point; return (Δ, q_matched)."""
    n = r.shape[0]
    veff = v + HBAR2_2M * l * (l + 1) / (r * r)
    allowed = (veff < E).nonzero().flatten()          # classically allowed region
    i_c = int(allowed[-1]) if len(allowed) else n // 2
    i_c = max(3, min(i_c, n - 4))

    g = (l + 0.5) ** 2 + r * r * (v - E) / HBAR2_2M
    f = 1.0 - (dx * dx / 12.0) * g

    qo = torch.zeros(n, dtype=torch.float64)
    qo[0] = r[0] ** (l + 0.5)
    qo[1] = r[1] ** (l + 0.5)
    for i in range(2, i_c + 2):
        qo[i] = ((12.0 - 10.0 * f[i - 1]) * qo[i - 1] - f[i - 2] * qo[i - 2]) / f[i]

    qi = torch.zeros(n, dtype=torch.float64)
    kappa = math.sqrt(max(float(v[-1] - E), 1e-6) / HBAR2_2M)
    qi[-1] = math.exp(-kappa * float(r[-1])) / math.sqrt(float(r[-1]))
    qi[-2] = math.exp(-kappa * float(r[-2])) / math.sqrt(float(r[-2]))
    for i in range(n - 3, i_c - 2, -1):
        qi[i] = ((12.0 - 10.0 * f[i + 1]) * qi[i + 1] - f[i + 2] * qi[i + 2]) / f[i]

    scale = qo[i_c] / qi[i_c]
    qi = qi * scale
    # log-derivative mismatch (index-derivative; the common 1/2dx cancels in the root condition)
    dqo = qo[i_c + 1] - qo[i_c - 1]
    dqi = qi[i_c + 1] - qi[i_c - 1]
    mism = float((dqo - dqi) / qo[i_c])

    q = torch.cat([qo[:i_c + 1], qi[i_c + 1:]])
    return mism, q, i_c


def radial_eigs_tridiag(l, r, dx, v, k):
    """Direct radial eigensolve: the KS operator on the log mesh is symmetric tridiagonal after
    q=u/√r then w=u√r, so ALL k lowest bound states come from one compiled LAPACK call — no
    shooting, no scan/bisection, robust for deep-core AND diffuse states.

    -q'' + [(l+½)² + r²V/ℏ²2m] q = (E/ℏ²2m) r² q  ->  standard tridiagonal M w = (E/ℏ²2m) w with
        M_ii   = [2/dx² + (l+½)²]/r_i²  +  V_i/ℏ²2m
        M_i,i+1 = -1/(dx² r_i r_{i+1}),      w = u·√r.

    Returns (E [k] eV, u [N,k] each normalized ∫u²dr=1, dr=r·dx). Not autograd (LAPACK); for
    gradients use first-order perturbation dε = <ψ|dH|ψ> as in gate D.
    """
    import numpy as np
    from scipy.linalg import eigh_tridiagonal

    rn = r.detach().numpy() if torch.is_tensor(r) else np.asarray(r)
    vn = v.detach().numpy() if torch.is_tensor(v) else np.asarray(v)
    invr2 = 1.0 / (rn * rn)
    diag = (2.0 / dx**2 + (l + 0.5) ** 2) * invr2 + vn / HBAR2_2M
    off = -1.0 / (dx**2 * rn[:-1] * rn[1:])
    w, W = eigh_tridiagonal(diag, off, select="i", select_range=(0, k - 1))
    E = w * HBAR2_2M
    u = W / np.sqrt(rn)[:, None]
    u = u / np.sqrt((u**2 * (rn * dx)[:, None]).sum(axis=0))[None, :]
    return E, u


def turning_point(l, E, r, v):
    """Outermost classically-allowed index (V_eff < E)."""
    veff = v + HBAR2_2M * l * (l + 1) / (r * r)
    allowed = (veff < E).nonzero().flatten()
    n = r.shape[0]
    i_c = int(allowed[-1]) if len(allowed) else n // 2
    return max(3, min(i_c, n - 4))


def mismatch_batch(l, energies, r, dx, v, i_c):
    """Log-derivative mismatch Δ(E) for a BATCH of energies at a fixed match index i_c (vectorized
    inward+outward Numerov). One pass replaces a whole energy scan; valid over a narrow window
    where i_c stays in the classically-allowed region."""
    n = r.shape[0]
    rr2 = (r * r)[None, :]
    g = (l + 0.5) ** 2 + rr2 * (v[None, :] - energies[:, None]) / HBAR2_2M
    f = 1.0 - (dx * dx / 12.0) * g                    # (B, N)
    cap = {}
    a = torch.full_like(energies, float(r[0] ** (l + 0.5)))
    b = torch.full_like(energies, float(r[1] ** (l + 0.5)))
    if 0 in (i_c - 1, i_c, i_c + 1):
        cap[0] = a.clone()
    if 1 in (i_c - 1, i_c, i_c + 1):
        cap[1] = b.clone()
    for i in range(2, i_c + 2):
        a, b = b, ((12.0 - 10.0 * f[:, i - 1]) * b - f[:, i - 2] * a) / f[:, i]
        if i in (i_c - 1, i_c, i_c + 1):
            cap[i] = b.clone()
    qo_lo, qo_c, qo_hi = cap[i_c - 1], cap[i_c], cap[i_c + 1]

    kappa = torch.sqrt(torch.clamp(v[-1] - energies, min=1e-6) / HBAR2_2M)
    ai = torch.exp(-kappa * float(r[-1])) / math.sqrt(float(r[-1]))
    bi = torch.exp(-kappa * float(r[-2])) / math.sqrt(float(r[-2]))
    capi = {n - 1: ai.clone(), n - 2: bi.clone()}
    for i in range(n - 3, i_c - 2, -1):
        ai, bi = bi, ((12.0 - 10.0 * f[:, i + 1]) * bi - f[:, i + 2] * ai) / f[:, i]
        if i in (i_c - 1, i_c, i_c + 1):
            capi[i] = bi.clone()
    qi_lo, qi_c, qi_hi = capi[i_c - 1], capi[i_c], capi[i_c + 1]

    scale = qo_c / qi_c
    return ((qo_hi - qo_lo) - scale * (qi_hi - qi_lo)) / qo_c


def _nodes(q, r, rmax):
    m = r < rmax
    s = torch.sign(q[m])
    s = s[s != 0]
    return int((s[1:] * s[:-1] < 0).sum())


def eigenstates(l, r, dx, v, e_lo, e_hi, want, n_scan=120):
    """All bound states in [e_lo,e_hi] via inward-outward matching + Δ(E) root-finding."""
    grid = torch.linspace(e_lo, e_hi, n_scan, dtype=torch.float64)
    ms = [_match_mismatch(l, float(e), r, dx, v)[0] for e in grid]
    out = []
    for i in range(n_scan - 1):
        if ms[i] * ms[i + 1] < 0 and abs(ms[i] - ms[i + 1]) < 1.0:   # skip pole sign flips
            lo, hi = float(grid[i]), float(grid[i + 1])
            flo = ms[i]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fmid = _match_mismatch(l, mid, r, dx, v)[0]
                if flo * fmid <= 0:
                    hi = mid
                else:
                    lo, flo = mid, fmid
            e = 0.5 * (lo + hi)
            _, q, _ = _match_mismatch(l, e, r, dx, v)
            out.append((e, _nodes(q, r, float(r[-1]))))
            if len(out) >= want:
                break
    return out


def main():
    from radial_log import log_mesh

    print("\nPROD-B — robust radial eigensolver (inward-outward matching)\n")

    ok = True
    for Z, rmax, cases in [
        (1.0, 40.0, [(0, -15.0, -0.2, 2), (1, -4.0, -0.2, 1)]),
        (8.0, 8.0, [(0, -1000.0, -50.0, 2), (1, -400.0, -50.0, 1)]),   # DEEP Z=8: 1s at -870 eV
    ]:
        r, dx = log_mesh(1e-4, rmax, 1500)
        v = -Z * E2 / r
        print(f"  hydrogenic Z={Z:.0f}:")
        print(f"  {'state':>6} | {'E (eV)':>13} | {'exact -Z²Ry/n²':>15} | {'nodes':>5} | "
              f"{'|Δ| (meV)':>10}")
        for l, e_lo, e_hi, want in cases:
            for e, nc in eigenstates(l, r, dx, v, e_lo, e_hi, want):
                n = nc + l + 1
                e_exact = -Z**2 * RY_EV / n**2
                dmev = abs(e - e_exact) * 1e3
                ok = ok and dmev < 20.0
                print(f"  {f'{n}{chr(115 + l)}' if l < 1 else f'{n}p':>6} | {e:>13.4f} | "
                      f"{e_exact:>15.4f} | {nc:>5} | {dmev:>10.2f}")
    print(f"\n  VERDICT: robust eigensolver correct incl. deep Z=8 1s (-870 eV): "
          f"{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
