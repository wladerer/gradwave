"""Tests for the ball-indicator form factor W(G) = Θ(G) and the moving-boundary surface-term
identity it enables (the make-or-break primitive under differentiable augmented / LAPW bases).
"""

from __future__ import annotations

import math

import torch

from gradwave.core.sphere_ff import ball_ff

torch.manual_seed(0)


def test_ball_ff_zero_is_volume():
    R = 1.8
    w0 = ball_ff(torch.zeros(1, dtype=torch.float64), R)
    assert torch.allclose(w0, torch.tensor([4.0 / 3.0 * math.pi * R**3], dtype=torch.float64))


def test_ball_ff_small_g_branch_continuous():
    """Series and closed-form branches agree across the switchover, with no NaNs/Infs."""
    R = 2.1
    g = torch.linspace(1e-6, 0.05, 400, dtype=torch.float64)  # straddles _SMALL/R
    w = ball_ff(g, R)
    assert torch.isfinite(w).all()
    # Monotone-smooth near 0: finite differences should not jump at the branch seam.
    d2 = (w[2:] - 2 * w[1:-1] + w[:-2]).abs().max()
    assert float(d2) < 1e-3


def test_ball_ff_matches_numerical_transform():
    """W(G) vs a direct real-space quadrature of the ball indicator, ∫_{|r|<R} cos(G·r) dV."""
    R = 1.8
    # Fine Cartesian grid over a cube enclosing the ball.
    n = 160
    half = R * 1.25
    ax = torch.linspace(-half, half, n, dtype=torch.float64)
    dv = (ax[1] - ax[0]) ** 3
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
    inside = (X * X + Y * Y + Z * Z) <= R * R
    for gmag in (0.5, 1.7, 3.4, 6.0):
        # G along z; W is real and isotropic in |G|.
        integ = (torch.cos(gmag * Z) * inside).sum() * dv
        w = ball_ff(torch.tensor(gmag, dtype=torch.float64), R)
        assert torch.allclose(w, integ, rtol=2e-3, atol=2e-3), (gmag, float(w), float(integ))


def test_ball_ff_grad_finite_including_near_zero():
    g = torch.tensor([0.0, 1e-4, 0.02, 1.0, 5.0], dtype=torch.float64, requires_grad=True)
    ball_ff(g, 1.5).sum().backward()
    assert torch.isfinite(g.grad).all()


# --------------------------------------------------------------------------- surface term
_L = 8.0
_R = 1.8
_KVEC = (1, 2, -1)   # reciprocal lattice index; k = (2π/L)·_KVEC
_PHI = 0.7
_A = 1.3


def _k(device):
    return torch.tensor(_KVEC, dtype=torch.float64, device=device) * (2.0 * math.pi / _L)


def _theta_energy(tau, n, device):
    """E(τ) = ∫_ball(τ) A cos(k·r+φ) dV built from the analytic Θ(G) form factor."""
    freq = torch.fft.fftfreq(n, d=1.0 / n, device=device, dtype=torch.float64)
    freq = freq * (2.0 * math.pi / _L)
    GX, GY, GZ = torch.meshgrid(freq, freq, freq, indexing="ij")
    G = torch.stack([GX, GY, GZ], dim=-1)
    W = ball_ff(G.norm(dim=-1), _R)
    phase = torch.exp(-1j * (G @ tau))
    chi_box = (W.to(torch.complex128) * phase) / (_L**3)
    chi_r = torch.fft.ifftn(chi_box).real * (n**3)
    ax = torch.arange(n, device=device, dtype=torch.float64) * (_L / n)
    RX, RY, RZ = torch.meshgrid(ax, ax, ax, indexing="ij")
    r = torch.stack([RX, RY, RZ], dim=-1)
    k = _k(device)
    g = _A * torch.cos(r @ k + _PHI)
    return ((_L**3) / (n**3)) * (chi_r * g).sum()


def _gridmask_energy(tau, n, device):
    """CONTROL: same energy via a real-space hard mask 1[|r-τ|<R] — τ enters a Heaviside."""
    ax = torch.arange(n, device=device, dtype=torch.float64) * (_L / n)
    RX, RY, RZ = torch.meshgrid(ax, ax, ax, indexing="ij")
    r = torch.stack([RX, RY, RZ], dim=-1)
    d = r - tau
    d = d - _L * torch.round(d / _L)
    chi = (d.norm(dim=-1) < _R).to(torch.float64)
    k = _k(device)
    g = _A * torch.cos(r @ k + _PHI)
    return ((_L**3) / (n**3)) * (chi * g).sum()


def test_theta_route_recovers_analytic_surface_term():
    """autograd ∂E_Θ/∂τ == the analytic surface integral -A W(k) sin(k·τ+φ) k, to ~machine eps."""
    device = torch.device("cpu")
    n = 48
    tau0 = torch.tensor([4.13, 3.87, 4.31], dtype=torch.float64, device=device)
    k = _k(device)
    W = ball_ff(k.norm(), _R)
    analytic = -_A * W * torch.sin(k @ tau0 + _PHI) * k

    tau = tau0.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(_theta_energy(tau, n, device), tau)
    assert torch.allclose(grad, analytic, atol=1e-10), (grad.tolist(), analytic.tolist())


def test_gridmask_route_force_is_zero():
    """CONTROL: the naive grid-mask energy carries no autograd gradient in τ, proving the
    surface term is invisible without the Θ(G) reformulation. The hard threshold detaches τ
    entirely, so the gradient is either None/zero or autograd refuses (energy has no grad_fn) —
    both mean zero surface force."""
    device = torch.device("cpu")
    n = 48
    tau = torch.tensor(
        [4.13, 3.87, 4.31], dtype=torch.float64, device=device, requires_grad=True
    )
    e = _gridmask_energy(tau, n, device)
    try:
        (grad,) = torch.autograd.grad(e, tau, allow_unused=True)
    except RuntimeError as exc:  # "does not require grad and does not have a grad_fn"
        assert "does not require grad" in str(exc)
        return
    assert grad is None or float(grad.norm()) == 0.0
