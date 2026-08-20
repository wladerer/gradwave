"""GATE A — does autograd recover a moving muffin-tin boundary's surface term?

The single make-or-break claim under the whole AutoAPW (differentiable LAPW) idea:

    A two-region energy  E(τ) = ∫_ball(τ) g(r) d³r  has a τ-derivative that is a PURE SURFACE
    INTEGRAL over the sphere |r-τ| = R.  On a fixed FFT grid this term is INVISIBLE to naive
    grid-based autograd (the region membership is a Heaviside the grid never resolves), but is
    recovered EXACTLY when the ball indicator is expressed through its analytic Fourier
    transform Θ(G) = W(G)·e^{-iG·τ}, so τ enters only through a smooth phase.

We make g(r) a sum of a few plane-wave modes g(r) = Σ_j a_j cos(k_j·r + φ_j) on reciprocal
lattice vectors, so the whole thing has a closed form to check against:

    E(τ)      = Σ_j a_j W(k_j) cos(k_j·τ + φ_j)
    dE/dτ     = -Σ_j a_j W(k_j) sin(k_j·τ + φ_j) k_j       (this IS the surface integral)

Three numbers must agree, and one control must fail:
    (1) autograd ∂E_Θ/∂τ            — the proposed recipe
    (2) central finite-difference of E_Θ(τ)
    (3) analytic dE/dτ              — grid-independent ground truth
    (4) autograd ∂E_grid/∂τ  == 0   — CONTROL: naive real-space hard mask fails
    (5) central FD of E_grid(τ)     — CONTROL: a biased, N-dependent staircase

Self-contained; also cross-checks that gradwave's core.fftbox.g_to_r_box reproduces the
band-limited indicator (integration with the real machinery).

    uv run python experiments/autoapw/surface_term_toy.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from gradwave.core.sphere_ff import ball_ff

torch.set_default_dtype(torch.float64)

# ------------------------------------------------------------------ problem setup
L = 8.0          # cubic box side (Å)
R_MT = 1.8       # muffin-tin radius (Å)
TAU0 = torch.tensor([4.13, 3.87, 4.31])   # ball centre, deliberately off-grid (Å)

# Field g(r) = Σ_j a_j cos(k_j·r + φ_j) with k_j = (2π/L)·m_j on reciprocal lattice vectors.
_MODES = [
    # (m_j  integer triple,     a_j,   φ_j)
    ((1, 0, 0), 1.0, 0.30),
    ((1, 2, -1), 0.7, 1.10),
    ((2, -1, 3), 0.5, -0.40),
    ((3, 3, 0), 0.35, 2.20),
]


def _mode_arrays(device):
    twopi_L = 2.0 * math.pi / L
    m_idx = torch.tensor([m for (m, _, _) in _MODES], dtype=torch.float64, device=device)
    k = m_idx * twopi_L                                                    # (M,3) reciprocal vecs
    a = torch.tensor([a for (_, a, _) in _MODES], dtype=torch.float64, device=device)   # (M,)
    phi = torch.tensor([p for (_, _, p) in _MODES], dtype=torch.float64, device=device)  # (M,)
    return k, a, phi


def analytic_energy_and_grad(tau, k, a, phi):
    """Closed form E(τ) and dE/dτ for the multi-mode field. τ: (3,)."""
    kdotr = k @ tau                       # (M,)
    kR = k.norm(dim=1)                    # (M,)
    W = ball_ff(kR, R_MT)                 # (M,)  analytic ball FT at each shell
    e = (a * W * torch.cos(kdotr + phi)).sum()
    # dE/dτ = -Σ a W sin(k·τ+φ) k
    g = -(a * W * torch.sin(kdotr + phi)).unsqueeze(1) * k
    return e, g.sum(dim=0)


# ------------------------------------------------------------------ grids
def real_grid(n, device):
    """Real-space sample points r_n = (L/n)·n_idx, shape (n,n,n,3)."""
    ax = torch.arange(n, device=device) * (L / n)
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
    return torch.stack([X, Y, Z], dim=-1)      # (n,n,n,3)


def g_grid(n, device):
    """Reciprocal vectors G = (2π/L)·fftfreq(n)·n, matched to torch.fft ordering. (n,n,n,3)."""
    freq = torch.fft.fftfreq(n, d=1.0 / n, device=device) * (2.0 * math.pi / L)  # (n,)
    GX, GY, GZ = torch.meshgrid(freq, freq, freq, indexing="ij")
    return torch.stack([GX, GY, GZ], dim=-1)   # (n,n,n,3)


def field_on_grid(r, k, a, phi):
    """g(r) = Σ_j a_j cos(k_j·r+φ_j) sampled on the grid. r:(n,n,n,3) -> (n,n,n)."""
    kr = torch.einsum("xyzc,mc->xyzm", r, k)   # (n,n,n,M)
    return (a * torch.cos(kr + phi)).sum(dim=-1)


# ------------------------------------------------------------------ the two routes
def energy_theta(tau, n, device, k, a, phi):
    """E(τ) via the Θ(G) recipe: band-limited ball indicator built from W(G)·e^{-iG·τ}.

    χ(r;τ) = (1/Ω) Σ_G W(G) e^{iG·(r-τ)}  ->  χ_box(G) = W(G) e^{-iG·τ} / Ω, then N·ifftn.
    τ enters only through the phase, so ∂E/∂τ is autograd-clean.
    """
    G = g_grid(n, device)                          # (n,n,n,3)
    Gn = G.norm(dim=-1)                             # (n,n,n)
    W = ball_ff(Gn, R_MT)                           # (n,n,n)   real, analytic
    phase = torch.exp(-1j * (G @ tau))             # (n,n,n)   complex, carries τ
    vol = L**3
    chi_box = (W.to(torch.complex128) * phase) / vol
    ntot = n**3
    chi_r = torch.fft.ifftn(chi_box).real * ntot   # gradwave convention: f(r) = N·ifftn(box)
    r = real_grid(n, device)
    g = field_on_grid(r, k, a, phi)
    h3 = vol / ntot
    return h3 * (chi_r * g).sum()


def energy_gridmask(tau, n, device, k, a, phi):
    """CONTROL: E(τ) via a naive real-space hard mask 1[|r_n-τ| < R]. τ enters a Heaviside."""
    r = real_grid(n, device)                       # (n,n,n,3)
    d = r - tau                                     # broadcasts (n,n,n,3)
    d = d - L * torch.round(d / L)                 # minimum image
    dist = d.norm(dim=-1)                           # (n,n,n)
    chi = (dist < R_MT).to(torch.float64)          # <-- non-differentiable threshold
    g = field_on_grid(r, k, a, phi)
    h3 = (L**3) / (n**3)
    return h3 * (chi * g).sum()


def central_fd_grad(fn, tau, h=1e-4):
    """Central finite-difference gradient of a scalar fn(tau) w.r.t. the 3-vector tau."""
    grad = torch.zeros(3, device=tau.device, dtype=tau.dtype)
    for j in range(3):
        ep = tau.clone()
        ep[j] += h
        em = tau.clone()
        em[j] -= h
        grad[j] = (fn(ep) - fn(em)) / (2 * h)
    return grad


# ------------------------------------------------------------------ fftbox cross-check
def fftbox_crosscheck(n, device, k, a, phi):
    """Confirm gradwave's core.fftbox.g_to_r_box reproduces the same band-limited indicator."""
    try:
        from gradwave.core.fftbox import g_to_r_box
    except Exception as exc:  # pragma: no cover - environment guard
        return {"available": False, "reason": repr(exc)}
    tau = TAU0.to(device)
    G = g_grid(n, device)
    Gn = G.norm(dim=-1)
    W = ball_ff(Gn, R_MT)
    phase = torch.exp(-1j * (G @ tau))
    chi_box = (W.to(torch.complex128) * phase) / (L**3)
    mine = torch.fft.ifftn(chi_box).real * (n**3)
    theirs = g_to_r_box(chi_box, real=True)
    return {"available": True, "max_abs_diff": float((mine - theirs).abs().max())}


# ------------------------------------------------------------------ driver
def run(device="cpu"):
    device = torch.device(device)
    k, a, phi = _mode_arrays(device)
    tau0 = TAU0.to(device)

    e_an, g_an = analytic_energy_and_grad(tau0, k, a, phi)

    out = {"device": str(device), "L": L, "R_MT": R_MT, "tau0": TAU0.tolist(),
           "analytic": {"E": float(e_an), "grad": g_an.tolist()}, "by_n": {}}

    for n in (40, 64, 96):
        # Θ(G) route: autograd force
        tau = tau0.clone().requires_grad_(True)
        e_theta = energy_theta(tau, n, device, k, a, phi)
        (g_theta_ad,) = torch.autograd.grad(e_theta, tau)

        # Θ(G) route: finite difference of the same functional
        g_theta_fd = central_fd_grad(
            lambda t, n=n: energy_theta(t, n, device, k, a, phi).detach(), tau0)

        # control: grid mask autograd (expected exactly 0) + its finite difference
        taum = tau0.clone().requires_grad_(True)
        e_mask = energy_gridmask(taum, n, device, k, a, phi)
        try:
            (g_mask_ad,) = torch.autograd.grad(e_mask, taum, allow_unused=True)
            g_mask_ad = torch.zeros(3, device=device) if g_mask_ad is None else g_mask_ad
        except RuntimeError:
            g_mask_ad = torch.zeros(3, device=device)
        g_mask_fd = central_fd_grad(
            lambda t, n=n: energy_gridmask(t, n, device, k, a, phi).detach(), tau0)

        rec = {
            "E_theta": float(e_theta.detach()),
            "grad_theta_autograd": g_theta_ad.detach().tolist(),
            "grad_theta_fd": g_theta_fd.tolist(),
            "grad_mask_autograd": g_mask_ad.detach().tolist(),
            "grad_mask_fd": g_mask_fd.tolist(),
            "err_theta_autograd_vs_analytic": float((g_theta_ad.detach() - g_an).norm()),
            "err_theta_fd_vs_analytic": float((g_theta_fd - g_an).norm()),
            "err_theta_autograd_vs_fd": float((g_theta_ad.detach() - g_theta_fd).norm()),
            "norm_grad_mask_autograd": float(g_mask_ad.norm()),
            "err_mask_fd_vs_analytic": float((g_mask_fd - g_an).norm()),
        }
        out["by_n"][n] = rec

    out["fftbox_crosscheck"] = fftbox_crosscheck(96, device, k, a, phi)
    return out


def _fmt(v):
    return "[" + ", ".join(f"{x:+.3e}" for x in v) + "]"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = run(args.device)
    g_an = out["analytic"]["grad"]

    print(f"\nAutoAPW GATE A — moving-boundary surface term via Θ(G)   [{out['device']}]")
    print(f"  box L={L} Å,  R_MT={R_MT} Å,  τ0={TAU0.tolist()}")
    print(f"  analytic dE/dτ = {_fmt(g_an)}   (grid-independent ground truth)\n")
    print(f"  {'N':>4} | {'‖Θ.autograd-analytic‖':>22} | {'‖Θ.fd-analytic‖':>17} | "
          f"{'‖Θ.autograd-Θ.fd‖':>18} | {'‖mask.autograd‖':>15} | {'‖mask.fd-analytic‖':>18}")
    for n, rec in out["by_n"].items():
        print(f"  {n:>4} | {rec['err_theta_autograd_vs_analytic']:>22.3e} | "
              f"{rec['err_theta_fd_vs_analytic']:>17.3e} | "
              f"{rec['err_theta_autograd_vs_fd']:>18.3e} | "
              f"{rec['norm_grad_mask_autograd']:>15.3e} | {rec['err_mask_fd_vs_analytic']:>18.3e}")

    cc = out["fftbox_crosscheck"]
    if cc.get("available"):
        print(f"\n  fftbox.g_to_r_box cross-check (N=96): max|Δ| = {cc['max_abs_diff']:.3e}")
    else:
        print(f"\n  fftbox cross-check unavailable: {cc.get('reason')}")

    # verdicts
    worst_theta = max(r["err_theta_autograd_vs_analytic"] for r in out["by_n"].values())
    worst_ad_fd = max(r["err_theta_autograd_vs_fd"] for r in out["by_n"].values())
    worst_mask = max(r["norm_grad_mask_autograd"] for r in out["by_n"].values())
    gan_norm = torch.tensor(g_an).norm().item()
    theta_ok = worst_theta < 1e-8 * max(gan_norm, 1.0) or worst_theta < 1e-9
    ad_fd_ok = worst_ad_fd < 1e-5
    mask_fails = worst_mask < 1e-10
    mask_verdict = "PASS (fails as predicted)" if mask_fails else "UNEXPECTED"
    print("\n  VERDICT:")
    print(f"    Θ(G) autograd == analytic surface term : {'PASS' if theta_ok else 'FAIL'} "
          f"(worst ‖err‖ = {worst_theta:.2e})")
    print(f"    Θ(G) autograd == finite difference      : {'PASS' if ad_fd_ok else 'FAIL'} "
          f"(worst ‖err‖ = {worst_ad_fd:.2e})")
    print(f"    control: naive grid mask force == 0     : {mask_verdict} "
          f"(worst ‖autograd‖ = {worst_mask:.2e})")

    out["verdict"] = {"theta_matches_analytic": bool(theta_ok),
                      "theta_autograd_matches_fd": bool(ad_fd_ok),
                      "gridmask_force_is_zero": bool(mask_fails)}
    res_path = Path(__file__).with_name("results_gate_a.json")
    res_path.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {res_path}")


if __name__ == "__main__":
    main()
