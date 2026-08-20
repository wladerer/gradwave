"""Radial Schrödinger solvers on a logarithmic mesh, in gradwave eV/Å units.

Two solvers, both used by the FLAPW stack:

- ``numerov_log`` / ``numerov_log_batch`` — outward Numerov integration of the radial equation at a
  *fixed* energy, returning ``u_l(r; E) = r·R_l``. This is the muffin-tin linearization solver: the
  augmentation carries ``u_l(r; E_l)`` and its energy derivative. Differentiable in ``E`` and in
  anything the potential depends on. On a log mesh ``x = ln r`` the first-derivative term is removed
  by ``q = u/√r``, giving ``q'' = [(l+½)² + r²(V-E)/ℏ²2m] q`` on a uniform ``x`` grid.

- ``radial_eigs_tridiag`` — the bound-state *spectrum*: the KS radial operator is symmetric
  tridiagonal after ``q = u/√r`` then ``w = u√r``, so ``scipy.linalg.eigh_tridiagonal`` returns all
  ``k`` lowest states in one LAPACK call — no shooting, robust for deep-core and diffuse states.

Units: ``r`` in Å, ``V``/``E`` in eV, ``HBAR2_2M = ℏ²/2m`` in eV·Å².
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from gradwave.constants import HBAR2_2M


def log_mesh(rmin: float, rmax: float, n: int, device=None) -> tuple[Tensor, float]:
    """UPF-style logarithmic mesh ``r_i = rmin·e^{i·dx}``; returns ``(r, dx)``."""
    x = torch.linspace(math.log(rmin), math.log(rmax), n, dtype=torch.float64, device=device)
    return torch.exp(x), float(x[1] - x[0])


def numerov_log(l: int, energy: Tensor, r: Tensor, dx: float, v: Tensor) -> Tensor:
    """Outward Numerov for ``q'' = [(l+½)² + r²(V-E)/ℏ²2m] q`` on a log mesh; returns ``u = q·√r``.

    Autograd-clean in ``energy`` and in anything ``v`` depends on."""
    g = (l + 0.5) ** 2 + r * r * (v - energy) / HBAR2_2M
    f = 1.0 - (dx * dx / 12.0) * g
    a = r[0] ** (l + 0.5)
    b = r[1] ** (l + 0.5)
    qs = [a, b]
    for i in range(2, r.shape[0]):
        qs.append(((12.0 - 10.0 * f[i - 1]) * qs[i - 1] - f[i - 2] * qs[i - 2]) / f[i])
    return torch.stack(qs) * torch.sqrt(r)


def numerov_log_np(l, energies, r, dx: float, v, n_cut: int | None = None) -> np.ndarray:
    """Pure-numpy outward Numerov, batched over trial energies — the fast *forward* path for the SCF
    (no autograd). Same recurrence as ``numerov_log``; ``energies`` is a scalar or ``(nE,)`` array,
    returning ``u`` of shape ``(N,)`` or ``(nE, N)``. The N-step recurrence runs once, vectorized
    over energies, in numpy — no per-element torch dispatch. Use ``numerov_log`` when a gradient in
    ``E`` or the potential is needed.

    ``l`` may be a scalar or an ``(nE,)`` array (one angular momentum per row), so a caller can
    integrate every (l, E) channel of a sphere in ONE recurrence loop — the per-row arithmetic is
    elementwise, so a batched call is bit-identical to the per-l calls it replaces.

    ``n_cut``: integrate only the first ``n_cut`` mesh points (zeros beyond). The augmentation uses
    ``u`` only inside R_MT, so integrating out to r_max would just risk float overflow for deep
    linearization energies (the classically-forbidden solution grows exponentially) — pass a cutoff
    a little past the sphere."""
    rn = (r.detach().numpy() if torch.is_tensor(r) else np.asarray(r)).astype(float)
    vn = (v.detach().numpy() if torch.is_tensor(v) else np.asarray(v)).astype(float)
    scalar = np.ndim(energies) == 0
    e = np.atleast_1d(np.asarray(energies, dtype=float))
    lhalf = np.asarray(l, dtype=float) + 0.5                  # scalar or per-row (nE,)
    if lhalf.ndim:
        lhalf = lhalf[:, None]
    n = rn.shape[0]
    nc = n if n_cut is None else min(int(n_cut), n)
    f = 1.0 - (dx * dx / 12.0) * (lhalf**2
                                  + (rn * rn)[None, :] * (vn[None, :] - e[:, None]) / HBAR2_2M)
    q = np.zeros((e.shape[0], n))
    q[:, 0] = rn[0] ** np.squeeze(lhalf)
    q[:, 1] = rn[1] ** np.squeeze(lhalf)
    for i in range(2, nc):
        q[:, i] = ((12.0 - 10.0 * f[:, i - 1]) * q[:, i - 1] - f[:, i - 2] * q[:, i - 2]) / f[:, i]
    u = q * np.sqrt(rn)[None, :]
    return u[0] if scalar else u


def numerov_log_batch(l: int, energies: Tensor, r: Tensor, dx: float, v: Tensor) -> Tensor:
    """Batched over trial energies: ``u(r_max)`` for each ``E`` (for eigenvalue scans)."""
    g = (l + 0.5) ** 2 + (r * r)[None, :] * (v[None, :] - energies[:, None]) / HBAR2_2M
    f = 1.0 - (dx * dx / 12.0) * g
    a = torch.full_like(energies, float(r[0] ** (l + 0.5)))
    b = torch.full_like(energies, float(r[1] ** (l + 0.5)))
    for i in range(2, r.shape[0]):
        a, b = b, ((12.0 - 10.0 * f[:, i - 1]) * b - f[:, i - 2] * a) / f[:, i]
    return b * math.sqrt(float(r[-1]))


def radial_eigs_tridiag(l: int, r: Tensor, dx: float, v: Tensor, k: int):
    """The ``k`` lowest bound states of the l-channel via a symmetric-tridiagonal diagonalization.

    ``-q'' + [(l+½)² + r²V/ℏ²2m] q = (E/ℏ²2m) r² q`` becomes ``M w = (E/ℏ²2m) w`` with
        ``M_ii = [2/dx²+(l+½)²]/r_i² + V_i/ℏ²2m``, ``M_i,i+1 = -1/(dx² r_i r_{i+1})``, ``w = u·√r``.

    Returns ``(E [k] eV, u [N,k])`` with each column normalized to ``∫u² dr = 1`` (``dr = r·dx``).
    Not autograd (LAPACK); for gradients use first-order perturbation ``dε = <ψ|dH|ψ>``.
    """
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
