"""FLAPW radial solvers: the log-mesh Numerov and the tridiagonal eigensolver vs analytic H-like."""

from __future__ import annotations

import torch

from gradwave.constants import E2, RY_EV
from gradwave.flapw.radial import log_mesh, radial_eigs_tridiag


def test_tridiag_hydrogen_spectrum():
    """Z=1 Coulomb: the l=0 spectrum reproduces E_n = -Ry/n²."""
    r, dx = log_mesh(1e-5, 40.0, 2000)
    v = -1.0 * E2 / r
    e, _ = radial_eigs_tridiag(0, r, dx, v, 3)
    # 3-point O(dx²) stencil: ~60 meV on the 1s cusp; 3s is diffuse (mesh-rmax limited).
    for n in (1, 2):
        assert abs(e[n - 1] - (-RY_EV / n**2)) < 0.1


def test_tridiag_deep_core_z8():
    """Deep hydrogenic Z=8 1s (~-870 eV) — the regime that breaks outward shooting."""
    r, dx = log_mesh(1e-5, 10.0, 3000)
    v = -8.0 * E2 / r
    e, u = radial_eigs_tridiag(0, r, dx, v, 2)
    assert abs(e[0] - (-64 * RY_EV)) < 1.0                 # 1s = -Z²·Ry, to <1 eV
    # eigenvectors normalized on the log mesh (∫u²dr=1, dr=r·dx)
    norm = float((torch.tensor(u[:, 0]) ** 2 * r * dx).sum())
    assert abs(norm - 1.0) < 1e-6
