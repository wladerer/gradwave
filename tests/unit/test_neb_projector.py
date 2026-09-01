"""CI-NEB force-projector validation on analytic 2D surfaces (no SCF).

The band math in ``gradwave.opt.neb`` is exercised against surfaces whose
saddle point is known in closed form:

* a symmetric quartic double well, whose saddle sits exactly at the origin, and
* the Müller–Brown potential, the standard curved-MEP benchmark, whose upper
  saddle SP2 ≈ (0.212, 0.293) the climbing image must find even though the
  linear initial band does not pass through it.

The optimizer here is a self-contained FIRE loop driving the projected forces —
deliberately independent of ASE so the test proves the projector, not the
driver.
"""

from __future__ import annotations

import numpy as np
import pytest

from gradwave.opt.neb import neb_forces, neb_tangent


# --------------------------------------------------------------------------
# analytic surfaces: V(x, y) and its true force −∇V
# --------------------------------------------------------------------------
def _double_well(xy: np.ndarray) -> tuple[float, np.ndarray]:
    """V = (x²−1)² + y²  (minima at (±1, 0), saddle exactly at (0, 0), E=1)."""
    x, y = float(xy[0]), float(xy[1])
    v = (x * x - 1.0) ** 2 + y * y
    grad = np.array([2.0 * (x * x - 1.0) * 2.0 * x, 2.0 * y])
    return v, -grad


# Müller–Brown parameters (Müller & Brown, Theor. Chim. Acta 53, 75 (1979)).
_MB_A = np.array([-200.0, -100.0, -170.0, 15.0])
_MB_a = np.array([-1.0, -1.0, -6.5, 0.7])
_MB_b = np.array([0.0, 0.0, 11.0, 0.6])
_MB_c = np.array([-10.0, -10.0, -6.5, 0.7])
_MB_x0 = np.array([1.0, 0.0, -0.5, -1.0])
_MB_y0 = np.array([0.0, 0.5, 1.5, 1.0])


def _muller_brown(xy: np.ndarray) -> tuple[float, np.ndarray]:
    x, y = float(xy[0]), float(xy[1])
    dx = x - _MB_x0
    dy = y - _MB_y0
    ex = _MB_A * np.exp(_MB_a * dx * dx + _MB_b * dx * dy + _MB_c * dy * dy)
    v = float(ex.sum())
    dvdx = float((ex * (2.0 * _MB_a * dx + _MB_b * dy)).sum())
    dvdy = float((ex * (_MB_b * dx + 2.0 * _MB_c * dy)).sum())
    return v, -np.array([dvdx, dvdy])


def _relax_point(surf, xy0: np.ndarray, steps: int = 400) -> np.ndarray:
    """Plain gradient descent to pin an endpoint on its true minimum."""
    xy = np.array(xy0, dtype=float)
    for _ in range(steps):
        _, f = surf(xy)
        xy = xy + 1.0e-3 * f
        if np.linalg.norm(f) < 1e-8:
            break
    return xy


def _run_neb(surf, ini, fin, n_images=11, spring_k=1.0, climb=True,
             n_steps=2000, dt=0.02):
    """FIRE-optimize a band of 2D images under the projected NEB force."""
    band = np.linspace(ini, fin, n_images)          # linear initial path
    vel = np.zeros_like(band)
    # FIRE parameters (Bitzek 2006)
    alpha, alpha0 = 0.1, 0.1
    finc, fdec, falpha = 1.1, 0.5, 0.99
    n_min, npos = 5, 0
    dt_max = dt * 10.0
    for _ in range(n_steps):
        energies = np.array([surf(band[i])[0] for i in range(n_images)])
        forces = np.array([surf(band[i])[1] for i in range(n_images)])
        proj = neb_forces(band, energies, forces, spring_k, climb=climb)
        # FIRE on the interior images
        p = float((proj * vel).sum())
        if p > 0.0:
            npos += 1
            vnorm = np.linalg.norm(vel)
            fnorm = np.linalg.norm(proj)
            if fnorm > 0:
                vel = (1.0 - alpha) * vel + alpha * (vnorm / fnorm) * proj
            if npos > n_min:
                dt = min(dt * finc, dt_max)
                alpha *= falpha
        else:
            npos = 0
            dt *= fdec
            alpha = alpha0
            vel[:] = 0.0
        vel = vel + dt * proj
        band = band + dt * vel
        if np.abs(proj).max() < 1e-4:
            break
    energies = np.array([surf(band[i])[0] for i in range(n_images)])
    return band, energies


def test_tangent_endpoints_zero_and_unit_interior():
    """Endpoint tangents are zero; interior tangents are unit vectors."""
    rng = np.random.default_rng(0)
    band = np.cumsum(rng.normal(size=(6, 4, 3)), axis=0)
    energies = np.array([0.0, 1.0, 2.0, 1.5, 0.5, 0.0])
    tau = neb_tangent(band.reshape(6, -1), energies)
    assert np.allclose(tau[0], 0.0)
    assert np.allclose(tau[-1], 0.0)
    for i in range(1, 5):
        assert np.linalg.norm(tau[i]) == pytest.approx(1.0)


def test_endpoints_never_move():
    """The projector returns exactly zero force on both endpoints."""
    rng = np.random.default_rng(1)
    band = np.cumsum(rng.normal(size=(7, 3)), axis=0)  # (n_images, ndof)
    energies = rng.normal(size=7)
    forces = rng.normal(size=(7, 3))
    proj = neb_forces(band, energies, forces, 0.5)
    assert np.allclose(proj[0], 0.0)
    assert np.allclose(proj[-1], 0.0)


def test_climbing_image_has_no_net_parallel_when_perp_relaxed():
    """A climbing image whose perpendicular force is zero keeps only the
    inverted parallel force (it climbs), and no spring force leaks in."""
    band = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    energies = np.array([0.0, 1.0, 0.0])  # image 1 is the top → climbs
    tangent = np.array([1.0, 0.0])
    # a purely parallel true force of magnitude 3 along +x
    forces = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0]])
    proj = neb_forces(band, energies, forces, 5.0, climb=True)
    # F_c - 2 (F_c·τ̂) τ̂  =  3x̂ - 2·3 x̂ = -3 x̂  (inverted, no spring)
    assert np.allclose(proj[1], -forces[1])
    assert np.allclose(proj[1], -3.0 * tangent)


def test_double_well_saddle_exact():
    """Symmetric double well: the climbing image lands on the origin saddle."""
    ini = _relax_point(_double_well, np.array([-1.0, 0.05]))
    fin = _relax_point(_double_well, np.array([1.0, -0.05]))
    band, energies = _run_neb(_double_well, ini, fin, n_images=11,
                              spring_k=1.0, climb=True)
    top = band[int(np.argmax(energies))]
    assert np.linalg.norm(top - np.array([0.0, 0.0])) < 5e-3
    # the barrier height matches the analytic saddle energy V(0,0) = 1
    assert energies.max() == pytest.approx(1.0, abs=5e-3)


def test_muller_brown_curved_saddle():
    """Müller–Brown SP2: the climbing image finds the curved-path saddle that
    the straight-line initial band misses."""
    # endpoints: the intermediate shallow minimum and the right-hand minimum,
    # whose connecting MEP crosses exactly one saddle (SP2).
    ini = _relax_point(_muller_brown, np.array([-0.05, 0.47]))
    fin = _relax_point(_muller_brown, np.array([0.62, 0.03]))
    band, energies = _run_neb(_muller_brown, ini, fin, n_images=13,
                              spring_k=2.0, climb=True, n_steps=4000, dt=2e-4)
    top = band[int(np.argmax(energies))]
    sp2 = np.array([0.2124, 0.2930])
    # the climbing image sits on the known saddle
    assert np.linalg.norm(top - sp2) < 3e-2
    # and the true force there has vanished (it is a stationary point)
    _, f_top = _muller_brown(top)
    assert np.linalg.norm(f_top) < 2.0  # |∇V| small vs O(100) surface scale
    # the climbing-image energy matches the analytic saddle energy V(SP2)
    v_sp2, _ = _muller_brown(sp2)
    assert energies.max() == pytest.approx(v_sp2, abs=1.5)


def test_climb_vs_noclimb_barrier_ordering():
    """Without the climbing image the band underestimates the barrier; the CI
    raises the top image onto the saddle so the estimate is at least as high."""
    ini = _relax_point(_double_well, np.array([-1.0, 0.05]))
    fin = _relax_point(_double_well, np.array([1.0, -0.05]))
    _, e_climb = _run_neb(_double_well, ini, fin, climb=True)
    _, e_plain = _run_neb(_double_well, ini, fin, climb=False)
    assert e_climb.max() >= e_plain.max() - 1e-6
    assert e_climb.max() == pytest.approx(1.0, abs=5e-3)
