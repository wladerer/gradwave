"""FLAPW radial solvers: the log-mesh Numerov and the tridiagonal eigensolver vs analytic H-like."""

from __future__ import annotations

import torch

from gradwave.constants import E2, RY_EV
from gradwave.flapw.lapw import radial_channel
from gradwave.flapw.radial import log_mesh, radial_eigs_tridiag
from gradwave.flapw.scf import _build_lo, _radial_u


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


def _p_sphere(Z=4.0, R=2.0):
    """A screened-Coulomb l=1 sphere: mesh, potential, channel, and matching u/u̇ radials."""
    r, dx = log_mesh(1e-5, 28.0, 2500)
    v = -Z * E2 / r
    El = -(Z**2) * RY_EV / 4.0                              # 2p of a Z-Coulomb well (n=2)
    ch = radial_channel(1, El, r, dx, v, R)
    u, ud = _radial_u(1, El, r, dx, v, R)
    return r, dx, v, R, ch, u, ud


def test_helo_is_unconfined_and_confined_to_sphere():
    """The HELO (``confine=False``) drops the u̇ term (b=0), stays confined to the sphere (φ(R)≈0),
    and is a normalized, distinct radial — the second-energy l=1 fix for the on-site EFG density."""
    r, dx, v, R, ch, u, ud = _p_sphere()
    e2 = 60.0                                              # a high scattering energy
    helo = _build_lo(1, e2, ch, u, ud, r, dx, v, R, cond_tol=0.0, confine=False)
    conf = _build_lo(1, e2, ch, u, ud, r, dx, v, R, cond_tol=0.0, confine=True)
    inside = r.numpy() <= R
    drw = r.numpy()[inside] * dx
    # (1) unconfined: no u̇ component at all
    assert helo["b"] == 0.0
    # (2) still a local orbital — value at the boundary is ~0 relative to its interior amplitude
    assert abs(helo["phi"][-1]) < 1e-2 * float(abs(helo["phi"]).max())
    # (3) normalized on the log mesh, self-overlap 1
    assert abs(float((helo["phi"] ** 2 * drw).sum()) - 1.0) < 1e-6
    assert helo["S_pp"] == 1.0
    # (4) genuinely different radial from the confined construction (which pulls in a large u̇)
    assert conf["b"] != 0.0
    assert float((torch.as_tensor(helo["phi"]) - torch.as_tensor(conf["phi"])).abs().max()) > 1e-3
