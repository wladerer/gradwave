"""Production radial Schrödinger solver: gradwave eV/Å units + UPF log meshes.

The gate-B/D prototype used Hartree atomic units on a uniform mesh. Real pseudopotential and
all-electron data (parse_upf / parse_upf_paw) live on a LOGARITHMIC mesh r_i = r_0 e^{i·dx} in
eV/Å units. On a log mesh the radial equation u''=[l(l+1)/r² + (V-E)/ℏ²2m]u picks up a first
derivative under x=ln r; the standard cure is q = u/√r, which integrates cleanly with Numerov:

    q''(x) = [ (l+½)² + r² (V(r) - E) / HBAR2_2M ] q,      u = q·√r,

on a UNIFORM x-grid (step dx). Units: r,R in Å; V,E in eV; HBAR2_2M = ℏ²/2m in eV·Å²; the
Coulomb factor is E2 = e² in eV·Å (so V_H(r) = -Z·E2/r). Seed q ~ r^{l+½} (u ~ r^{l+1}).

Verified here against the analytic hydrogen spectrum E_n = -Z²·Ry/n² (in eV).

    uv run python experiments/autoapw/radial_log.py
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gradwave.constants import E2, HBAR2_2M, RY_EV


def log_mesh(rmin: float, rmax: float, n: int, device=None):
    """UPF-style logarithmic mesh r_i = rmin·e^{i·dx}; returns (r, dx)."""
    x = torch.linspace(math.log(rmin), math.log(rmax), n, dtype=torch.float64, device=device)
    return torch.exp(x), float(x[1] - x[0])


def numerov_log(l: int, energy: Tensor, r: Tensor, dx: float, v: Tensor) -> Tensor:
    """Outward Numerov for q'' = [(l+½)² + r²(V-E)/ℏ²2m] q on a log mesh; returns u = q·√r.

    Autograd-clean in `energy` and in anything `v` depends on."""
    g = (l + 0.5) ** 2 + r * r * (v - energy) / HBAR2_2M
    f = 1.0 - (dx * dx / 12.0) * g
    a = r[0] ** (l + 0.5)
    b = r[1] ** (l + 0.5)
    qs = [a, b]
    for i in range(2, r.shape[0]):
        qs.append(((12.0 - 10.0 * f[i - 1]) * qs[i - 1] - f[i - 2] * qs[i - 2]) / f[i])
    return torch.stack(qs) * torch.sqrt(r)


def numerov_log_batch(l: int, energies: Tensor, r: Tensor, dx: float, v: Tensor) -> Tensor:
    """Batched over trial energies: returns u(r_max) for each E (for eigenvalue scans)."""
    g = (l + 0.5) ** 2 + (r * r)[None, :] * (v[None, :] - energies[:, None]) / HBAR2_2M
    f = 1.0 - (dx * dx / 12.0) * g
    a = torch.full_like(energies, float(r[0] ** (l + 0.5)))
    b = torch.full_like(energies, float(r[1] ** (l + 0.5)))
    for i in range(2, r.shape[0]):
        a, b = b, ((12.0 - 10.0 * f[:, i - 1]) * b - f[:, i - 2] * a) / f[:, i]
    return b * math.sqrt(float(r[-1]))


def bound_states(l, r, dx, v, e_lo, e_hi, want, n_scan=300, n_bisect=60):
    """Bound-state eigenvalues (eV) by sign changes of u(r_max;E), refined by batched bisection."""
    grid = torch.linspace(e_lo, e_hi, n_scan, dtype=torch.float64)
    with torch.no_grad():
        vals = numerov_log_batch(l, grid, r, dx, v)
    los, his = [], []
    for i in range(n_scan - 1):
        if float(vals[i]) * float(vals[i + 1]) < 0:
            los.append(grid[i])
            his.append(grid[i + 1])
            if len(los) >= want:
                break
    if not los:
        return []
    lo, hi = torch.stack(los), torch.stack(his)
    with torch.no_grad():
        flo = numerov_log_batch(l, lo, r, dx, v)
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            fmid = numerov_log_batch(l, mid, r, dx, v)
            left = (flo * fmid) <= 0
            hi = torch.where(left, mid, hi)
            lo = torch.where(left, lo, mid)
            flo = torch.where(left, flo, fmid)
    return list(0.5 * (lo + hi))


def _node_count(u, r, rmax=6.0):
    m = r < rmax
    s = torch.sign(u[m])
    return int((s[1:] * s[:-1] < 0).sum())


def run_oxygen():
    """Ingest REAL pseudopotential data (oxygen PAW, native log mesh) and verify the production
    solver on that mesh. Finding: this dataset's stored `ae_vloc` is essentially the IONIC
    -z_valence·e²/r local potential (it matches -6·e²/r to <1% for r>0.5 Å), NOT the screened
    neutral-atom all-electron KS potential — so it yields a Z=6 Coulomb Rydberg series, not O's
    1s/2s/2p. A real neutral-atom all-electron spectrum needs an atomic KS self-consistent solve
    (the next production piece); this check verifies the solver+units+real-log-mesh are correct by
    reproducing the analytic Coulomb spectrum for the potential's actual -z_val/r form."""
    import numpy as np

    from gradwave.constants import E2 as _E2
    from gradwave.pseudo.upf_paw import parse_upf_paw
    p = parse_upf_paw("tests/fixtures/qe/pseudos/O.pbe-n-kjpaw_psl.1.0.0.UPF")
    r = torch.tensor(np.asarray(p.r), dtype=torch.float64)
    dx = float(math.log(p.r[2] / p.r[1]))
    v = torch.tensor(np.asarray(p.ae_vloc), dtype=torch.float64)
    zval = float(p.z_valence)
    msh = int(p.msh)

    print("\nProduction radial solver — REAL pseudopotential data (oxygen PAW, native log mesh)\n")
    print(f"  native log mesh: {msh} pts, r∈[{float(r[0]):.1e},{float(r[msh - 1]):.1f}] Å, "
          f"dx={dx:.4f}")

    # (1) characterise the stored potential: is it the ionic -z_val·e²/r?
    m = (r > 0.5) & (r < 5.0)
    coul = -zval * _E2 / r
    rel = float(((v[m] - coul[m]).abs() / coul[m].abs()).max())
    print(f"  stored ae_vloc vs -z_val·e²/r (z_val={zval:.0f}): max rel dev (0.5–5 Å) = {rel:.2e}")
    print("    -> ae_vloc is the IONIC local potential (not screened AE); real AE needs an SCF")

    # (2) verify the solver on THIS mesh reproduces the analytic Coulomb Rydberg series (Z=z_val),
    # for states well-contained in the mesh (n=3,4). Deep states and diffuse states both stress
    # outward-only shooting; robust handling needs inward-outward matching (a production TODO).
    rt = r[:msh]
    vc = -zval * _E2 / rt
    print(f"\n  solver check — clean -{zval:.0f}·e²/r on the real O mesh vs analytic -Z²·Ry/n²:")
    print(f"  {'state':>6} | {'E numerov (eV)':>15} | {'E exact (eV)':>13} | {'|Δ| (meV)':>10}")
    ok = True
    eigs = bound_states(0, rt, dx, vc, -80.0, -25.0, 2)   # n=3,4: contained in the 5.3 Å mesh
    got = sorted(float(e) for e in eigs)
    for e in got:
        n = int(round(math.sqrt(-zval**2 * RY_EV / e)))
        e_exact = -zval**2 * RY_EV / n**2
        dmev = abs(e - e_exact) * 1e3
        ok = ok and dmev < 5.0
        print(f"  {f'n={n}':>6} | {e:>15.4f} | {e_exact:>13.4f} | {dmev:>10.2f}")
    print(f"\n  VERDICT: production solver correct on the real O log mesh (Coulomb, contained "
          f"states): {'PASS' if ok and got else 'FAIL'}")
    print("  PRODUCTION TODO surfaced here: (a) robust eigenvalues for deep-core AND diffuse")
    print("  states need inward-outward matching (outward-only shooting confines/contaminates);")
    print("  (b) a real screened all-electron potential needs an atomic KS self-consistent solve —")
    print("  this dataset's ae_vloc is the ionic -z_val/r, not the neutral-atom AE potential.")
    return ok and bool(got)


def main():
    r, dx = log_mesh(1e-4, 40.0, 2400)
    print("\nProduction radial solver (eV/Å, log mesh) — hydrogen spectrum check\n")
    print(f"  {'state':>6} | {'E numerov (eV)':>15} | {'E exact (eV)':>13} | {'|Δ| (meV)':>10}")
    Z = 1.0
    v = -Z * E2 / r
    ok = True
    for l, e_lo, e_hi, want in [(0, -15.0, -0.2, 2), (1, -4.0, -0.2, 1)]:
        eigs = bound_states(l, r, dx, v, e_lo, e_hi, want)
        for k, e in enumerate(eigs):
            n = (k + 1) if l == 0 else (k + 2)
            e_exact = -Z**2 * RY_EV / n**2
            label = f"{n}{'spdf'[l]}"
            dmev = abs(float(e) - e_exact) * 1e3
            ok = ok and dmev < 5.0
            print(f"  {label:>6} | {float(e):>15.5f} | {e_exact:>13.5f} | {dmev:>10.2f}")
    print(f"\n  VERDICT: log-mesh eV/Å radial solver matches hydrogen: {'PASS' if ok else 'FAIL'}")
    run_oxygen()


if __name__ == "__main__":
    main()
