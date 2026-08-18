"""The aspherical-density angular projection (EFG step 1), with synthetic amplitudes:
a pure s state has no l=2 asphericity, a p_z state does, and a filled p shell is spherical again."""

from __future__ import annotations

import numpy as np

from gradwave.flapw.efg import sphere_density_multipoles

NR = 8
_US = {l: (np.ones(NR), np.zeros(NR)) for l in range(3)}     # flat radial parts (angular test only)
_LSET = [(0, 0)] + [(2, m) for m in range(-2, 3)]


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
