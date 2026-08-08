"""ESM open-boundary Hartree potential (core/energies/esm.py).

Phase 1 of the open-boundary surface engine (docs/design/dtn-3d-engine.md):
periodic in-plane, open (isolated, no z-images) in the vacuum direction. Each
check is independent of how the solver computes v_H — the Poisson equation, the
analytic dipole step, box-independence, and reduction to the periodic solve.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from gradwave.constants import E2
from gradwave.core.energies.esm import hartree_potential_esm

FOURPI_E2 = 4.0 * math.pi * E2


def _neutral_dipole_density(cell: np.ndarray, shape: tuple[int, int, int]) -> torch.Tensor:
    """A +/- gaussian pair, periodic in-plane and localized near the z-center;
    net-neutral so the open G∥=0 channel is well defined."""
    na, nb, nz = shape
    ax, ay, az = (float(np.linalg.norm(cell[i])) for i in range(3))
    xs = (np.arange(na) + 0.5) / na * ax
    ys = (np.arange(nb) + 0.5) / nb * ay
    zs = (np.arange(nz) + 0.5) / nz * az
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    def blob(cx, cy, cz, amp, sig=0.9):
        dx = X - cx
        dx -= ax * np.round(dx / ax)
        dy = Y - cy
        dy -= ay * np.round(dy / ay)
        return amp * np.exp(-(dx**2 + dy**2 + (Z - cz) ** 2) / (2 * sig**2))

    zc = az / 2
    rho = blob(ax * 0.4, ay * 0.5, zc - 2.0, +1.0) + blob(ax * 0.6, ay * 0.5, zc + 2.0, -1.0)
    rho -= rho.mean()  # exact net neutrality
    return torch.as_tensor(rho, dtype=torch.float64)


def test_satisfies_poisson_equation():
    """∇²v_H = −4πe²ρ in the interior — a finite-difference Laplacian that is
    blind to how v_H was solved."""
    cell = np.diag([6.0, 6.0, 22.0])
    na, nb, nz = 30, 30, 110
    rho = _neutral_dipole_density(cell, (na, nb, nz))
    v = hartree_potential_esm(rho, cell)

    hx, hy, hz = 6.0 / na, 6.0 / nb, 22.0 / nz
    lap = (torch.roll(v, 1, 0) + torch.roll(v, -1, 0) - 2 * v) / hx**2
    lap += (torch.roll(v, 1, 1) + torch.roll(v, -1, 1) - 2 * v) / hy**2
    lap[:, :, 1:-1] += (v[:, :, 2:] + v[:, :, :-2] - 2 * v[:, :, 1:-1]) / hz**2

    target = -FOURPI_E2 * rho
    m = rho.abs() > 0.02 * rho.abs().max()
    m[:, :, :6] = False
    m[:, :, -6:] = False
    rel = float((lap - target)[m].abs().max() / target[m].abs().max())
    assert rel < 2e-2, f"Poisson residual {rel:.2e}"


def test_box_independent_in_slab():
    """The open potential in the slab is independent of vacuum thickness — the
    defining ESM property a periodic box cannot deliver."""
    ax = ay = 6.0
    dz = 0.2

    def run(az):
        nz = int(round(az / dz))
        cell = np.diag([ax, ay, az])
        rho = _neutral_dipole_density(cell, (30, 30, nz))
        return hartree_potential_esm(rho, cell), az

    v1, az1 = run(20.0)
    v2, az2 = run(30.0)

    def window(v, az):  # a 10 A window centered on the slab, same physical z
        nz = v.shape[2]
        z = (np.arange(nz) + 0.5) / nz * az
        i0 = int(np.argmin(np.abs(z - (az / 2 - 5.0))))
        w = v[:, :, i0:i0 + 50]
        return w - w.mean()  # drop the arbitrary additive gauge

    d = float((window(v1, az1) - window(v2, az2)).abs().max())
    assert d < 1e-6, f"box dependence {d:.2e} eV"


def test_dipole_sheet_step_matches_analytic():
    """Two uniform +/- sheets give the textbook potential step Δv = 4πe²σd, with
    zero field outside (open BC) — impossible in a periodic box."""
    ax = ay = 5.0
    az = 40.0
    nz = 400
    cell = np.diag([ax, ay, az])
    dz = az / nz
    sigma = 0.05  # e/A^2
    z1, z2 = 15.0, 25.0
    i1, i2 = int(z1 / dz), int(z2 / dz)
    rho = torch.zeros((2, 2, nz), dtype=torch.float64)
    rho[:, :, i1] = sigma / dz
    rho[:, :, i2] = -sigma / dz
    v = hartree_potential_esm(rho, cell)

    line = v[0, 0]
    step = float(line[i2 + 5:].mean() - line[:i1 - 5].mean())
    analytic = -FOURPI_E2 * sigma * (z2 - z1)
    assert abs(step - analytic) / abs(analytic) < 2e-2, (step, analytic)
    # field is ~flat (zero) outside the capacitor — open, not periodic
    outside = line[i2 + 10:].diff().abs().max()
    assert float(outside) < 1e-6


def test_reduces_to_periodic_for_uniform_neutral_slab():
    """For a laterally-uniform density (only the G∥=0 channel), the open solve is
    the 1D isolated Poisson: v is flat in-plane and the field vanishes in the
    vacuum on both sides."""
    ax = ay = 4.0
    az = 30.0
    nz = 300
    cell = np.diag([ax, ay, az])
    dz = az / nz
    z = (np.arange(nz) + 0.5) * dz
    # a localized neutral double layer (+ slab adjacent to − slab); net zero and
    # confined to [12,18] so there is genuine vacuum at both z-edges.
    rho1d = np.where((z >= 12.0) & (z < 15.0), 1.0, 0.0)
    rho1d += np.where((z >= 15.0) & (z < 18.0), -1.0, 0.0)
    rho = torch.as_tensor(np.broadcast_to(rho1d, (8, 8, nz)).copy(), dtype=torch.float64)
    v = hartree_potential_esm(rho, cell)
    # flat in-plane
    assert float(v.std(dim=(0, 1)).max()) < 1e-9
    # zero field deep in vacuum on both sides (isolated slab)
    line = v[0, 0]
    assert float(line[:20].diff().abs().max()) < 1e-6
    assert float(line[-20:].diff().abs().max()) < 1e-6


def test_differentiable_in_density():
    """Autograd flows through the per-G∥ recursion — the differentiable-ESM point."""
    cell = np.diag([6.0, 6.0, 20.0])
    rho = _neutral_dipole_density(cell, (16, 16, 100)).requires_grad_(True)
    v = hartree_potential_esm(rho, cell)
    (g,) = torch.autograd.grad(v.pow(2).sum(), rho)
    assert torch.isfinite(g).all()
    assert float(g.abs().max()) > 0


def test_rejects_non_orthogonal_open_axis():
    """ESM requires the open axis ⊥ the periodic plane; a sheared cell is refused."""
    cell = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [1.0, 0.0, 20.0]])
    rho = torch.zeros((8, 8, 40), dtype=torch.float64)
    with pytest.raises(ValueError, match="orthogonal"):
        hartree_potential_esm(rho, cell)
