"""Differentiable NEGF transport (postscf/transport.py): surface Green's function,
Caroli transmission, and autograd forces on the transmission."""

from __future__ import annotations

import numpy as np
import torch

from gradwave.postscf.transport import (
    conductance,
    lead_self_energy,
    surface_green_function,
    transmission,
)


def _tb(e0: float = 0.0, t: float = 1.0):
    """Scalar 1-band tight-binding lead: onsite e0, hopping -t; band [e0-2t, e0+2t]."""
    return torch.tensor([[e0]], dtype=torch.float64), torch.tensor([[-t]], dtype=torch.float64)


def _analytic_sigma(energies, t, e0, eta=1e-7):
    """Retarded surface self-energy of the scalar lead: Σ = t² g, g the decaying
    root of t²g² − (E−e0)g + 1 = 0 (branch Im g ≤ 0 with E→E+iη)."""
    d = (np.asarray(energies) + 1j * eta) - e0
    root = np.sqrt(d * d - 4 * t * t + 0j)
    g1 = (d - root) / (2 * t * t)
    g2 = (d + root) / (2 * t * t)
    g = np.where(g1.imag <= 0.0, g1, g2)
    return t * t * g


def test_surface_gf_matches_analytic_scalar_lead():
    """Sancho-Rubio surface self-energy reproduces the analytic scalar lead."""
    t, e0 = 1.3, -0.4
    h00 = torch.tensor([[e0 + 2 * t]], dtype=torch.float64)  # band-centred onsite
    h01 = torch.tensor([[-t]], dtype=torch.float64)
    energies = torch.tensor([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=torch.float64)
    g = surface_green_function(h00, h01, energies, eta=1e-7)
    sigma = (h01.to(torch.complex128).conj().T @ g @ h01.to(torch.complex128))[:, 0, 0]
    ref = _analytic_sigma(energies.numpy(), t, e0 + 2 * t)
    assert np.allclose(sigma.numpy(), ref, atol=1e-8)


def test_perfect_conductor_integer_transmission():
    """A homogeneous device (device == lead) transmits perfectly: T(E) = number of
    open channels (=1 in-band for a scalar chain, 0 in the gap)."""
    h00, h01 = _tb(0.0, 1.0)                       # band [-2, 2]
    onsites = [h00] * 4
    e_in = torch.tensor([-1.5, -0.8, 0.0, 0.8, 1.5], dtype=torch.float64)
    t_in = transmission(onsites, h01, e_in)
    assert torch.allclose(t_in, torch.ones_like(t_in), atol=1e-4)
    t_gap = transmission(onsites, h01, torch.tensor([2.5, 3.0], dtype=torch.float64))
    assert torch.all(t_gap < 1e-6)


def test_transmission_bounds_and_nonnegative():
    h00, h01 = _tb(0.0, 1.0)
    e = torch.linspace(-3.0, 3.0, 41, dtype=torch.float64)
    t = transmission([h00] * 3, h01, e)
    assert torch.all(t >= -1e-9)
    assert torch.all(t <= 1.0 + 1e-4)


def test_barrier_suppresses_transmission():
    """Raising a middle device onsite (a scattering barrier) drops T below 1."""
    h00, h01 = _tb(0.0, 1.0)
    e = torch.tensor([0.0], dtype=torch.float64)
    t0 = transmission([h00, h00, h00, h00, h00], h01, e)
    tb = transmission([h00, h00, h00 + 1.5, h00, h00], h01, e)
    assert float(t0[0]) > 0.99
    assert float(tb[0]) < 0.9 * float(t0[0])


def test_transmission_differentiable_vs_finite_difference():
    """dT/d(barrier height) via autograd matches central finite difference — the
    inverse-design superpower (differentiable transport)."""
    h00, h01 = _tb(0.0, 1.0)
    e = torch.tensor([0.3], dtype=torch.float64)
    eye = torch.eye(1, dtype=torch.float64)

    def t_of(b):
        return transmission([h00, h00, h00 + b * eye, h00, h00], h01, e)[0]

    b = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    t = t_of(b)
    t.backward()
    grad_ad = float(b.grad)
    db = 1e-4
    grad_fd = (float(t_of(torch.tensor(0.6 + db))) - float(t_of(torch.tensor(0.6 - db)))) / (2 * db)
    assert abs(grad_ad - grad_fd) <= 1e-5 + 1e-3 * abs(grad_fd)


def test_two_channel_perfect_conductor():
    """A 2-orbital lead: in-band T = number of open channels (1 or 2)."""
    h00 = torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.float64))   # two bands, offset
    h01 = -0.6 * torch.eye(2, dtype=torch.float64)                     # bands [-2.2,0.2],[−0.2,2.2]
    # E = -1.0 lies in the lower band only (1 channel); E = 0.0 lies in both (2)
    t = transmission([h00] * 3, h01, torch.tensor([-1.0, 0.0], dtype=torch.float64))
    assert abs(float(t[0]) - 1.0) < 1e-3
    assert abs(float(t[1]) - 2.0) < 1e-3


def test_conductance_quantum():
    """Perfect single channel at E_F → G = G₀ (one conductance quantum)."""
    h00, h01 = _tb(0.0, 1.0)
    g = conductance([h00] * 3, h01, fermi_energy=0.0)
    assert abs(g - 7.748091729e-5) < 1e-8


def test_left_right_self_energy_hermitian_gamma():
    """Γ = i(Σ − Σ†) is Hermitian and positive-semidefinite in the band (open
    channel), ~0 in the gap (evanescent)."""
    h00, h01 = _tb(0.0, 1.0)
    e = torch.tensor([0.5], dtype=torch.float64)
    sig = lead_self_energy(h00, h01, e, "left")
    gam = 1j * (sig - sig.conj().transpose(-1, -2))
    assert torch.allclose(gam, gam.conj().transpose(-1, -2), atol=1e-10)
    assert float(gam[0, 0, 0].real) > 0.0
    sig_gap = lead_self_energy(h00, h01, torch.tensor([3.0], dtype=torch.float64), "left")
    gam_gap = 1j * (sig_gap - sig_gap.conj().transpose(-1, -2))
    assert float(gam_gap.abs().max()) < 1e-4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all transport tests passed")
