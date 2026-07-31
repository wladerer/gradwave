"""4-channel energy-metric estimator for the spinor path
(postscf._response.kernel_energy_error_noncollinear).

The estimator returns the spinor residual's exact second-order energy error
1/2 <r|K_Hxc|r>, so it is validated against a finite-difference second difference
of the actual Hartree+XC energy (the repo's oracle convention) and, at a moment
along a single axis, against the validated collinear kernel_energy_error to
machine precision. No SCF here, these run fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gradwave.core.energies.hartree import hartree_energy
from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.core.xc.noncollinear import NoncollinearXC, energy_with_grid
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
from gradwave.grids import build_fft_grid
from gradwave.postscf._response import (
    kernel_energy_error,
    kernel_energy_error_noncollinear,
)
from tests.helpers import RY


def _grid(a=6.0, ecut_ry=10.0):
    return build_fft_grid(np.eye(3) * a, ecut_ry * RY, equal_dims=None)


def _smooth_rho(grid, n_elec=8.0, seed=0):
    torch.manual_seed(seed)
    x = torch.rand(grid.shape, dtype=torch.float64) * 0.1 + 0.3
    cell = grid.volume / grid.n_points
    return x / (x.sum() * cell) * n_elec


def _charge_conserving(grid, seed):
    """A random real grid field with zero G=0 component (∫δρ = 0)."""
    torch.manual_seed(seed)
    d = torch.rand(grid.shape, dtype=torch.float64) - 0.5
    dg = r_to_g(d.to(torch.complex128))
    dg.reshape(-1)[0] = 0.0
    return g_to_r_box(dg, real=True)


def _ehxc_nc(grid, xc, rho, m):
    eh = hartree_energy(r_to_g(rho.to(torch.complex128)), grid.g2, grid.volume)
    return float(eh) + float(energy_with_grid(xc, rho, m, grid))


# --------------------------------------------------------------------------- #
#  FD oracle: 1/2 <r|K_Hxc|r> == second difference of E_Hxc                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("xc_cls, tol", [(LSDA_PW92, 1e-3), (SpinPBE, 3e-2)])
def test_estimator_matches_second_difference(xc_cls, tol):
    """The 4-channel estimate equals 1/2[E(x+r) − 2E(x) + E(x−r)] to the FD
    truncation floor. The moment is along z and proportional to ρ, so |m⃗| stays
    away from zero wherever the residual lives (the transverse f_xc kernel has a
    geometric 1/|m⃗| factor that is only near-singular where the moment vanishes,
    the magnon-soft regime the campaign identifies)."""
    grid = _grid()
    xc = NoncollinearXC(xc_cls())
    rho = _smooth_rho(grid)
    mz = 0.3 * rho
    zero = torch.zeros_like(rho)
    m = torch.stack([zero, zero, mz])
    r_rho = _charge_conserving(grid, seed=1) * 1e-3
    r_m = torch.stack([
        _charge_conserving(grid, 2) * 1e-3,
        _charge_conserving(grid, 3) * 1e-3,
        _charge_conserving(grid, 4) * 1e-3,
    ])
    e_total, *_ = kernel_energy_error_noncollinear(grid, xc, r_rho, r_m, rho, m)
    oracle = 0.5 * (_ehxc_nc(grid, xc, rho + r_rho, m + r_m)
                    - 2 * _ehxc_nc(grid, xc, rho, m)
                    + _ehxc_nc(grid, xc, rho - r_rho, m - r_m))
    assert abs(e_total / oracle - 1.0) < tol


@pytest.mark.parametrize("xc_cls", [LSDA_PW92, SpinPBE])
def test_reduces_to_collinear_kernel_along_z(xc_cls):
    """A moment purely along +z is the collinear limit. The noncollinear total
    then equals the collinear kernel_energy_error under the ρ± = (ρ ± m_z)/2
    channel map, to machine precision (both differentiate the same locally
    collinear energy)."""
    grid = _grid()
    nc = NoncollinearXC(xc_cls())
    rho = _smooth_rho(grid)
    mz = 0.3 * rho
    zero = torch.zeros_like(rho)
    m = torch.stack([zero, zero, mz])
    r_rho = _charge_conserving(grid, 1) * 1e-3
    r_mz = _charge_conserving(grid, 4) * 1e-3
    r_m = torch.stack([zero, zero, r_mz])
    e_nc, *_ = kernel_energy_error_noncollinear(grid, nc, r_rho, r_m, rho, m)
    # collinear map: ρ↑,↓ = (ρ ± m_z)/2, residual r↑,↓ = (r_ρ ± r_mz)/2
    ru, rd = (rho + mz) / 2, (rho - mz) / 2
    r_up, r_dn = (r_rho + r_mz) / 2, (r_rho - r_mz) / 2
    e_coll, _, _ = kernel_energy_error(grid, xc_cls(), [r_up, r_dn], [ru, rd], None, 2)
    assert abs(e_nc - e_coll) <= 1e-10 * abs(e_coll) + 1e-14


def test_channels_are_pure():
    """A pure-charge residual (r_m⃗ = 0) has exactly zero magnetization energy; a
    pure-magnetization residual (r_ρ = 0) has exactly zero charge energy (its
    Hartree and f_xc charge blocks contract against a zero charge residual)."""
    grid = _grid()
    xc = NoncollinearXC(SpinPBE())
    rho = _smooth_rho(grid)
    mz = 0.3 * rho
    zero = torch.zeros_like(rho)
    m = torch.stack([zero, zero, mz])
    d = _charge_conserving(grid, 5) * 1e-3
    zerom = torch.zeros_like(m)
    # pure charge
    _, e_charge_c, e_mag_c, e_long_c, e_trans_c = kernel_energy_error_noncollinear(
        grid, xc, d, zerom, rho, m)
    assert e_mag_c == 0.0 and e_long_c == 0.0 and e_trans_c == 0.0
    assert abs(e_charge_c) > 0.0
    # pure magnetization
    r_m = torch.stack([zero, zero, d])
    _, e_charge_m, e_mag_m, *_ = kernel_energy_error_noncollinear(
        grid, xc, zero.clone(), r_m, rho, m)
    assert e_charge_m == 0.0
    assert abs(e_mag_m) > 0.0


def test_longitudinal_transverse_split():
    """With the moment along z, a z-only magnetization residual is purely
    longitudinal and an x-only residual is purely transverse."""
    grid = _grid()
    xc = NoncollinearXC(SpinPBE())
    rho = _smooth_rho(grid)
    mz = 0.3 * rho
    zero = torch.zeros_like(rho)
    m = torch.stack([zero, zero, mz])
    d = _charge_conserving(grid, 6) * 1e-3
    # z-only residual: longitudinal
    r_long = torch.stack([zero, zero, d])
    _, _, _, e_long, e_trans = kernel_energy_error_noncollinear(
        grid, xc, zero.clone(), r_long, rho, m)
    assert abs(e_trans) < 1e-10 * abs(e_long)
    # x-only residual: transverse
    r_trans = torch.stack([d, zero, zero])
    _, _, _, e_long2, e_trans2 = kernel_energy_error_noncollinear(
        grid, xc, zero.clone(), r_trans, rho, m)
    assert abs(e_long2) < 1e-10 * abs(e_trans2)


def test_no_moment_is_all_transverse():
    """A nonmagnetic state (m⃗ ≡ 0) has no longitudinal axis, so the whole
    magnetization residual is reported transverse and the charge channel is
    still Hartree-amplified."""
    grid = _grid()
    xc = NoncollinearXC(SpinPBE())
    rho = _smooth_rho(grid)
    m = torch.zeros(3, *rho.shape, dtype=torch.float64)
    d = _charge_conserving(grid, 7) * 1e-3
    r_m = torch.stack([d, torch.zeros_like(rho), torch.zeros_like(rho)])
    e_total, e_charge, e_mag, e_long, e_trans = kernel_energy_error_noncollinear(
        grid, xc, _charge_conserving(grid, 8) * 1e-3, r_m, rho, m)
    assert e_long == 0.0
    assert abs(e_trans - e_mag) < 1e-12 * abs(e_mag)
    assert abs(e_charge) > 0.0


def test_metagga_is_rejected():
    """The kernel omits the τ response, so a meta-GGA raises rather than return a
    silently-wrong estimate."""
    grid = _grid()
    rho = _smooth_rho(grid)
    m = torch.zeros(3, *rho.shape, dtype=torch.float64)
    r_rho = _charge_conserving(grid, 9) * 1e-3
    r_m = torch.zeros_like(m)

    class _FakeMetaGGA:
        needs_tau = True
        needs_gradient = True

    with pytest.raises(NotImplementedError, match="meta-GGA"):
        kernel_energy_error_noncollinear(grid, _FakeMetaGGA(), r_rho, r_m, rho, m)
