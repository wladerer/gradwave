"""r2SCAN meta-GGA validated pointwise against libxc (via pyscf).

libxc is the reference implementation r2SCAN is transcribed from, and QE's
`input_dft='r2scan'` *is* libxc, so a pointwise match to libxc's e_xc, vρ, vσ,
vτ on random (ρ, σ, τ) grids is the strongest correctness gate — it isolates the
functional from the SCF, and pins every regime (α from the single-orbital limit
through the slowly-varying and large-α branches) to machine precision.

The oracle is optional: pyscf ships a manylinux wheel bundling libxc, but a bare
checkout may lack it, so these skip when pyscf is absent. A small oracle-free
sanity test guards the τ_flat/limit behaviour unconditionally.
"""

import math

import numpy as np
import pytest
import torch

from gradwave.constants import BOHR_ANG, HARTREE_EV
from gradwave.core.xc.r2scan import R2SCAN, SpinR2SCAN

libxc = pytest.importorskip("pyscf.dft.libxc",
                            reason="pyscf/libxc oracle not installed")


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-8)


def _grids():
    """A spread of (ρ, |∇ρ|, τ) with α spanning all three r2SCAN branches."""
    den = np.array([0.35, 0.12, 0.03, 0.6, 0.02, 0.15, 0.5, 0.9])
    gmag = np.array([0.05, 0.03, 0.005, 0.2, 0.001, 0.08, 0.3, 0.02])
    sigma = gmag**2
    tau_w = sigma / (8 * den)
    tau_unif = 0.3 * (3 * math.pi**2) ** (2 / 3) * den ** (5 / 3)
    alpha = np.array([0.0, 0.3, 0.8, 1.0, 1.5, 3.0, 0.5, 1.2])  # spans the branches
    tau = tau_w + alpha * tau_unif
    return den, gmag, sigma, tau


def test_unpolarized_matches_libxc():
    """e_xc and all partials (vρ, vσ, vτ) equal libxc r2scan to machine precision."""
    den, gmag, sigma, tau = _grids()
    n = len(den)
    rho6 = np.vstack([den, gmag, np.zeros(n), np.zeros(n), np.zeros(n), tau])
    exc, vxc, _, _ = libxc.eval_xc("r2scan", rho6, spin=0, deriv=1)
    vrho, vsig, _, vtau = vxc

    dt = torch.tensor(den, dtype=torch.float64, requires_grad=True)
    st = torch.tensor(sigma, dtype=torch.float64, requires_grad=True)
    tt = torch.tensor(tau, dtype=torch.float64, requires_grad=True)
    # a.u. inputs → Å units for the functional; energy density → Ha/bohr³
    e = R2SCAN().energy_density(dt / BOHR_ANG**3, st / BOHR_ANG**8, tt / BOHR_ANG**5)
    e_haub = e / HARTREE_EV * BOHR_ANG**3
    g_rho, g_sig, g_tau = torch.autograd.grad(e_haub.sum(), (dt, st, tt))

    assert _rel(e_haub.detach().numpy(), exc * den).max() < 1e-12
    assert _rel(g_rho.numpy(), vrho).max() < 1e-11
    assert _rel(g_sig.numpy(), vsig).max() < 1e-10
    assert _rel(g_tau.numpy(), vtau).max() < 1e-11


def test_exchange_and_correlation_separately():
    """Each channel matches libxc's standalone mgga_x/mgga_c_r2scan."""
    from gradwave.core.xc import r2scan as R

    den, gmag, sigma, tau = _grids()
    n = len(den)
    rho6 = np.vstack([den, gmag, np.zeros(n), np.zeros(n), np.zeros(n), tau])
    nt = torch.tensor(den, dtype=torch.float64)
    st = torch.tensor(sigma, dtype=torch.float64)
    tt = torch.tensor(tau, dtype=torch.float64)
    for name, mine in [
        ("mgga_x_r2scan", R._ex_unpol(nt, st, tt)),
        ("mgga_c_r2scan", R._ec(nt, st, tt, torch.zeros_like(nt))),
    ]:
        ref, _, _, _ = libxc.eval_xc(name, rho6, spin=0, deriv=0)
        assert _rel(mine.numpy(), ref * den).max() < 1e-12


def test_spin_polarized_matches_libxc():
    """Collinear r2SCAN equals libxc spin=1 (energy and vτ per channel)."""
    nu = np.array([0.30, 0.05, 0.4, 0.02, 0.25, 0.5])
    nd = np.array([0.10, 0.05, 0.25, 0.015, 0.25, 0.1])
    gu = np.array([0.04, 0.01, 0.15, 0.001, 0.06, 0.2])
    gd = np.array([0.02, 0.01, 0.08, 0.001, 0.06, 0.05])
    suu, sdd, sud = gu**2, gd**2, gu * gd  # gradients parallel (all along x)
    stot = suu + 2 * sud + sdd
    n = nu + nd
    tuU = 0.3 * (6 * math.pi**2) ** (2 / 3) * nu ** (5 / 3)
    tdU = 0.3 * (6 * math.pi**2) ** (2 / 3) * nd ** (5 / 3)
    tu = suu / (8 * nu) + 0.8 * tuU
    td = sdd / (8 * nd) + 1.2 * tdU
    N = len(nu)
    Z = np.zeros(N)
    rho_a = np.array([nu, gu, Z, Z, Z, tu])
    rho_b = np.array([nd, gd, Z, Z, Z, td])
    exc, vxc, _, _ = libxc.eval_xc("r2scan", (rho_a, rho_b), spin=1, deriv=1)
    vtau = vxc[3]  # (N, 2)

    def leaf(a):
        return torch.tensor(a, dtype=torch.float64, requires_grad=True)

    nut, ndt = leaf(nu / BOHR_ANG**3), leaf(nd / BOHR_ANG**3)
    suut, sddt = leaf(suu / BOHR_ANG**8), leaf(sdd / BOHR_ANG**8)
    stott = leaf(stot / BOHR_ANG**8)
    tut, tdt = leaf(tu / BOHR_ANG**5), leaf(td / BOHR_ANG**5)
    e = SpinR2SCAN().energy_density(nut, ndt, suut, sddt, stott, tut, tdt)
    e_haub = e / HARTREE_EV * BOHR_ANG**3
    g_tu, g_td = torch.autograd.grad(e_haub.sum(), (tut, tdt))

    assert _rel(e_haub.detach().numpy(), exc * n).max() < 1e-12
    assert _rel((g_tu / BOHR_ANG**5).numpy(), vtau[:, 0]).max() < 1e-10
    assert _rel((g_td / BOHR_ANG**5).numpy(), vtau[:, 1]).max() < 1e-10
