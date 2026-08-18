"""The aspherical-density angular projection (EFG step 1), with synthetic amplitudes:
a pure s state has no l=2 asphericity, a p_z state does, and a filled p shell is spherical again."""

from __future__ import annotations

import numpy as np

from gradwave.flapw.efg import efg_tensor, sphere_density_multipoles

NR = 8
_US = {l: (np.ones(NR), np.zeros(NR)) for l in range(3)}     # flat radial parts (angular test only)
_LSET = [(0, 0)] + [(2, m) for m in range(-2, 3)]


def _const_multipoles(vals):
    """l=2 multipoles with constant values over a small radial mesh; returns (rho, rr, drw)."""
    rr = np.linspace(0.05, 1.3, 40)
    drw = np.full_like(rr, float(rr[1] - rr[0]))
    rho = {(2, m): np.full(rr.shape, vals.get(m, 0.0), dtype=complex) for m in range(-2, 3)}
    return rho, rr, drw


def _amp(l, m):
    """One band with unit amplitude in the (l, m) channel (u_l part), zero elsewhere."""
    a = {0: np.zeros(1, complex), 1: np.zeros(3, complex), 2: np.zeros(5, complex)}
    a[l][m + l] = 1.0
    b = {ll: np.zeros(2 * ll + 1, complex) for ll in range(3)}
    return (1.0, a, b)


def _l2_magnitude(rho):
    return float(np.sqrt(sum(np.abs(rho[(2, m)][0]) ** 2 for m in range(-2, 3))))


def test_pure_s_has_no_l2_asphericity():
    """A pure s state is spherical: nonzero monopole, ~zero l=2 density."""
    rho = sphere_density_multipoles([_amp(0, 0)], _US, 2, _LSET)
    assert abs(rho[(0, 0)][0]) > 0.1
    assert _l2_magnitude(rho) < 1e-10


def test_pz_state_is_aspherical():
    """A p_z (l=1, m=0) state carries a real l=2, M=0 density and no M=±1, ±2."""
    rho = sphere_density_multipoles([_amp(1, 0)], _US, 2, _LSET)
    assert abs(rho[(2, 0)][0]) > 0.05
    assert abs(rho[(2, 1)][0]) < 1e-10
    assert abs(rho[(2, 2)][0]) < 1e-10


def test_filled_p_shell_is_spherical():
    """One electron in each of p_{-1}, p_0, p_{+1} (filled shell) is spherical again: ~zero l=2."""
    amps = [_amp(1, m) for m in (-1, 0, 1)]
    rho = sphere_density_multipoles(amps, _US, 2, _LSET)
    assert abs(rho[(0, 0)][0]) > 0.1
    assert _l2_magnitude(rho) < 1e-10


def test_efg_tensor_is_traceless_and_symmetric():
    """The EFG tensor from any l=2 density is symmetric and traceless (Laplace: ∇²V=0)."""
    rho, rr, drw = _const_multipoles({0: 0.5, 1: 0.3 + 0.1j, 2: 0.2 - 0.4j})
    t, _, _ = efg_tensor(rho, rr, drw)
    assert np.allclose(t, t.T)
    assert abs(np.trace(t)) < 1e-10 * np.abs(t).max()


def test_efg_axial_from_rho20():
    """A pure ρ_20 density is axial: V_xx=V_yy=-V_zz/2, no off-diagonal, asymmetry η≈0."""
    rho, rr, drw = _const_multipoles({0: 1.0})
    t, v_zz, eta = efg_tensor(rho, rr, drw)
    assert abs(v_zz) > 1e-6
    assert eta < 1e-9
    assert abs(t[0, 1]) < 1e-12 and abs(t[0, 2]) < 1e-12 and abs(t[1, 2]) < 1e-12
    assert abs(t[0, 0] - t[1, 1]) < 1e-12          # V_xx = V_yy
    assert abs(t[2, 2] + 2 * t[0, 0]) < 1e-12      # V_zz = -2 V_xx


def test_efg_asymmetric_from_rho22():
    """A pure real ρ_22 density gives a fully asymmetric EFG (η≈1) with zero cell-frame V_zz."""
    rho, rr, drw = _const_multipoles({2: 1.0})
    t, v_zz, eta = efg_tensor(rho, rr, drw)
    assert abs(t[2, 2]) < 1e-12                    # no ρ_20 -> zero cell-frame V_zz
    assert eta > 0.99


def test_efg_from_pz_amplitudes_is_axial():
    """End to end: the l=2 density of a p_z state, through the sphere Poisson, is an axial EFG."""
    rr = np.linspace(0.05, 1.2, 30)
    drw = np.full_like(rr, float(rr[1] - rr[0]))
    us = {l: (np.ones(rr.shape), np.zeros(rr.shape)) for l in range(3)}
    a = {0: np.zeros(1, complex), 1: np.zeros(3, complex), 2: np.zeros(5, complex)}
    a[1][1] = 1.0
    b = {ll: np.zeros(2 * ll + 1, complex) for ll in range(3)}
    rho = sphere_density_multipoles([(1.0, a, b)], us, 2, [(2, m) for m in range(-2, 3)])
    _, v_zz, eta = efg_tensor(rho, rr, drw)
    assert abs(v_zz) > 1e-6
    assert eta < 1e-6


def _cubic_grid(n, L):
    """Cartesian displacement components (x,y,z) from the cell centre on an n^3 cubic grid."""
    ax = np.arange(n) * (L / n)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
    c = L / 2

    def mi(a):
        return (a - c) - L * np.round((a - c) / L)

    return mi(xx), mi(yy), mi(zz), np.array([c, c, c])


def test_interstitial_l2_boundary_recovers_l2():
    """A pure r²Y_20 interstitial potential projects onto v_bc_20 only (the boundary sampling)."""
    from gradwave.flapw.efg import interstitial_l2_boundary
    n, L = 32, 6.0
    x, y, z, center = _cubic_grid(n, L)
    vgrid = 2 * z**2 - x**2 - y**2                       # proportional to r² Y_20
    vbc = interstitial_l2_boundary(vgrid, center, 1.5, np.eye(3) * L)
    mags = {m: abs(vbc[m]) for m in range(-2, 3)}
    assert mags[0] > 0.1
    assert all(mags[m] < 1e-3 * mags[0] for m in (-2, -1, 1, 2))


def test_interstitial_l2_boundary_isotropic_is_zero():
    """An isotropic interstitial potential has no l=2 component at the boundary."""
    from gradwave.flapw.efg import interstitial_l2_boundary
    n, L = 32, 6.0
    x, y, z, center = _cubic_grid(n, L)
    vgrid = x**2 + y**2 + z**2                            # isotropic
    vbc = interstitial_l2_boundary(vgrid, center, 1.5, np.eye(3) * L)
    assert all(abs(vbc[m]) < 1e-3 * L**2 for m in range(-2, 3))
