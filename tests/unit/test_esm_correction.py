"""ESM electrostatic correction (core/energies/esm.py): the SCF-facing energy
ΔE and potential ΔV = δΔE/δρ, plus the Gaussian ion charge.

The correction is the open-minus-periodic electrostatic energy of the neutral
total charge (electrons − Gaussian ions), computed with a discretization-matched
difference (linear vs circular z-convolution of the SAME kernel) so it is
independent of the ion Gaussian width and its potential is the exact functional
derivative of its energy — the properties an SCF needs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gradwave.core.energies.esm import (
    esm_energy,
    esm_energy_strained,
    esm_potential,
    gaussian_ion_density,
    hartree_potential_capacitor,
)
from gradwave.core.energies.hartree import hartree_potential_r
from gradwave.grids import build_fft_grid


def _grid(az, n_open):
    return build_fft_grid(np.diag([6.0, 6.0, az]), ecut=1.0, shape_override=(30, 30, n_open))


def _elec(grid, centers, charges, width=0.9):
    """A localized electron density = Σ Z·Gaussian at fixed positions (∫ = ΣZ)."""
    return gaussian_ion_density(torch.as_tensor(centers, dtype=torch.float64),
                                torch.as_tensor(charges, dtype=torch.float64), grid, width)


def test_gaussian_ion_integrates_to_Z():
    grid = _grid(20.0, 100)
    pos = torch.tensor([[3.0, 3.0, 8.0], [3.0, 3.0, 12.0]], dtype=torch.float64)
    z = torch.tensor([4.0, 6.0], dtype=torch.float64)
    rho_ion = gaussian_ion_density(pos, z, grid, 0.8)
    dvol = grid.volume / rho_ion.numel()
    assert abs(float(rho_ion.sum() * dvol) - float(z.sum())) < 1e-6


def test_potential_is_functional_derivative_of_energy():
    """esm_potential must equal δ(esm_energy)/δρ (finite difference) — this is
    what keeps the SCF variational and Hellmann-Feynman forces consistent."""
    grid = _grid(20.0, 100)
    pos = torch.tensor([[3.0, 3.0, 8.0], [3.0, 3.0, 12.0]], dtype=torch.float64)
    z = torch.tensor([4.0, 6.0], dtype=torch.float64)
    rho = _elec(grid, [[3, 3, 8.5], [3, 3, 11.5]], [4.0, 6.0])
    dv = esm_potential(rho, pos, z, grid)
    dvol = grid.volume / rho.numel()
    h = 1e-6
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(6):
        ix = tuple(int(rng.integers(n)) for n in rho.shape)
        rp = rho.clone()
        rp[ix] += h
        rm = rho.clone()
        rm[ix] -= h
        fd = (float(esm_energy(rp, pos, z, grid)) - float(esm_energy(rm, pos, z, grid))) / (2 * h)
        worst = max(worst, abs(fd - float(dv[ix]) * dvol))
    assert worst < 1e-7, worst


def test_force_matches_finite_difference():
    """Autograd δΔE/δR (the ESM force term) matches a finite difference of ΔE."""
    grid = _grid(20.0, 100)
    pos = torch.tensor([[3.0, 3.0, 8.0], [3.0, 3.0, 12.0]], dtype=torch.float64)
    z = torch.tensor([4.0, 6.0], dtype=torch.float64)
    rho = _elec(grid, [[3, 3, 8.5], [3, 3, 11.5]], [4.0, 6.0])
    p = pos.clone().requires_grad_(True)
    (fgrad,) = torch.autograd.grad(esm_energy(rho, p, z, grid), p)
    h = 1e-5
    pp = pos.clone()
    pp[1, 2] += h
    pm = pos.clone()
    pm[1, 2] -= h
    fd = (float(esm_energy(rho, pp, z, grid)) - float(esm_energy(rho, pm, z, grid))) / (2 * h)
    assert abs(float(fgrad[1, 2]) - fd) / (abs(fd) + 1e-12) < 1e-5


def test_energy_independent_of_ion_width():
    """The discretization-matched difference makes ΔE independent of the Gaussian
    ion width β (the short-range self-field cancels open-vs-periodic)."""
    grid = _grid(24.0, 120)
    pos = torch.tensor([[3.0, 3.0, 10.0], [3.0, 3.0, 14.0]], dtype=torch.float64)
    z = torch.tensor([4.0, 6.0], dtype=torch.float64)
    rho = _elec(grid, [[3, 3, 11], [3, 3, 13]], [4.0, 6.0])
    des = [float(esm_energy(rho, pos, z, grid, beta=b)) for b in (0.2, 0.4, 0.6)]
    assert max(des) - min(des) < 1e-3, des


def test_total_energy_box_independent():
    """E_open = E_periodic + ΔE is independent of vacuum thickness while the
    periodic part alone drifts — the surface-energetics payoff, at fixed dz."""
    ions = torch.tensor([[3.0, 3.0, 8.0], [3.0, 3.0, 12.0]], dtype=torch.float64)
    elec = [[3, 3, 9.0], [3, 3, 11.0]]  # charge-transfer dipole
    z = torch.tensor([4.0, 6.0], dtype=torch.float64)

    def eo_ep(az, n_open):
        grid = _grid(az, n_open)
        rho = _elec(grid, elec, [4.0, 6.0])
        de = float(esm_energy(rho, ions, z, grid))
        rho_tot = rho - gaussian_ion_density(ions, z, grid, 2 * (az / n_open))
        dvol = grid.volume / rho.numel()
        e_per = 0.5 * float((rho_tot * hartree_potential_r(rho_tot, grid.g2)).sum() * dvol)
        return e_per + de, e_per

    eo1, ep1 = eo_ep(24.0, 120)
    eo2, ep2 = eo_ep(36.0, 180)
    assert abs(eo2 - eo1) < 5e-3, (eo1, eo2)
    assert abs(ep2 - ep1) > 5 * abs(eo2 - eo1)  # periodic drifts, open does not


@pytest.mark.parametrize(("mode", "bias"), [("vacuum", 0.0), ("capacitor", 0.0),
                                            ("capacitor", 2.0)])
def test_strained_energy_matches_and_stress_matches_fd(mode, bias):
    """esm_energy_strained equals esm_energy at zero strain, and its autograd
    w.r.t. an in-plane strain (the ESM stress contribution) matches finite
    difference — the term postscf.stress adds, for both vacuum and capacitor."""
    grid = _grid(20.0, 100)
    pos = torch.tensor([[3.0, 3.0, 8.0], [3.0, 3.0, 12.0]], dtype=torch.float64)
    z = torch.tensor([4.0, 6.0], dtype=torch.float64)
    rho = _elec(grid, [[3, 3, 8.6], [3, 3, 11.4]], [4.0, 6.0])
    cell0 = torch.tensor(np.asarray(grid.cell), dtype=torch.float64)
    beta = 0.8

    assert float(esm_energy_strained(rho, pos, z, cell0, grid.shape, beta,
                                     mode=mode, bias=bias)) == \
        pytest.approx(float(esm_energy(rho, pos, z, grid, beta=beta,
                                       mode=mode, bias=bias)), abs=1e-9)

    eye = torch.eye(3, dtype=torch.float64)
    mask = torch.ones(3, 3, dtype=torch.float64)
    mask[2, :] = 0.0
    mask[:, 2] = 0.0  # in-plane strain only (z is open)

    def energy(eps):
        f = eye + eps * mask
        return esm_energy_strained(rho, pos @ f.T, z, cell0 @ f.T, grid.shape,
                                   beta, mode=mode, bias=bias)

    eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(energy(eps), eps)
    h = 1e-5
    for a, b in [(0, 0), (1, 1), (0, 1)]:
        ep = torch.zeros(3, 3, dtype=torch.float64)
        ep[a, b] += h
        fd = (float(energy(ep)) - float(energy(-ep))) / (2 * h)
        assert float(g[a, b]) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("bias", [0.0, 2.0])
def test_capacitor_potential_and_force_match_fd(bias):
    """In capacitor mode (metal planes, optional bias), esm_potential = δΔE/δρ and
    the ESM force = δΔE/δR by finite difference — the grounded (quadratic) and
    applied-bias (linear) parts are both handled consistently."""
    grid = _grid(20.0, 100)
    pos = torch.tensor([[3.0, 3.0, 8.0], [3.0, 3.0, 12.0]], dtype=torch.float64)
    z = torch.tensor([4.0, 6.0], dtype=torch.float64)
    rho = _elec(grid, [[3, 3, 8.6], [3, 3, 11.4]], [4.0, 6.0])
    dvol = grid.volume / rho.numel()

    dv = esm_potential(rho, pos, z, grid, mode="capacitor", bias=bias)
    rng = np.random.default_rng(5)
    h = 1e-6
    worst = 0.0
    for _ in range(6):
        ix = tuple(int(rng.integers(n)) for n in rho.shape)
        rp = rho.clone()
        rp[ix] += h
        rm = rho.clone()
        rm[ix] -= h
        ep = float(esm_energy(rp, pos, z, grid, mode="capacitor", bias=bias))
        em = float(esm_energy(rm, pos, z, grid, mode="capacitor", bias=bias))
        worst = max(worst, abs((ep - em) / (2 * h) - float(dv[ix]) * dvol))
    assert worst < 1e-7, worst

    p = pos.clone().requires_grad_(True)
    (fg,) = torch.autograd.grad(esm_energy(rho, p, z, grid, mode="capacitor", bias=bias), p)
    pp = pos.clone()
    pp[1, 2] += 1e-5
    pm = pos.clone()
    pm[1, 2] -= 1e-5
    fd = (float(esm_energy(rho, pp, z, grid, mode="capacitor", bias=bias))
          - float(esm_energy(rho, pm, z, grid, mode="capacitor", bias=bias))) / 2e-5
    assert float(fg[1, 2]) == pytest.approx(fd, abs=1e-5)


def test_capacitor_charged_cell_potential_is_functional_derivative():
    """In the capacitor mode a NET-CHARGED cell (N_e ≠ ΣZ, the constant-potential
    regime — the plates carry the counter-charge) is well defined, and
    esm_potential = δΔE/δρ still holds by finite difference, so a fixed-µ SCF on
    it would be variational."""
    grid = _grid(20.0, 100)
    pos = torch.tensor([[3.0, 3.0, 8.0], [3.0, 3.0, 12.0]], dtype=torch.float64)
    z = torch.tensor([4.0, 6.0], dtype=torch.float64)
    rho = _elec(grid, [[3, 3, 8.6], [3, 3, 11.4]], [4.0, 6.0])
    dvol = grid.volume / rho.numel()
    rho = rho * (10.4 / (float(rho.sum()) * dvol))  # N_e=10.4 vs ΣZ=10 → +0.4 e

    dv = esm_potential(rho, pos, z, grid, mode="capacitor")
    rng = np.random.default_rng(7)
    h = 1e-6
    worst = 0.0
    for _ in range(6):
        ix = tuple(int(rng.integers(n)) for n in rho.shape)
        rp = rho.clone()
        rp[ix] += h
        rm = rho.clone()
        rm[ix] -= h
        ep = float(esm_energy(rp, pos, z, grid, mode="capacitor"))
        em = float(esm_energy(rm, pos, z, grid, mode="capacitor"))
        worst = max(worst, abs((ep - em) / (2 * h) - float(dv[ix]) * dvol))
    assert worst < 1e-7, worst


def test_capacitor_charged_reference_calibrated_to_plates():
    """The net-charge reference is calibrated to the plates. Isolate it via the
    charged−neutral difference (which cancels the ion self-energy, correctly
    excluded from v_eff but present in v_cap): the ESM effective electrostatic
    potential then tracks the plate-referenced v_cap to ~1e-3. Without the
    calibration this difference is O(1 eV) in z-structure."""
    from gradwave.core.energies.esm import _default_beta
    from gradwave.core.energies.hartree import hartree_potential_r

    grid = _grid(20.0, 100)
    pos = torch.tensor([[3.0, 3.0, 8.0], [3.0, 3.0, 12.0]], dtype=torch.float64)
    z = torch.tensor([4.0, 6.0], dtype=torch.float64)
    e0 = _elec(grid, [[3, 3, 8.6], [3, 3, 11.4]], [4.0, 6.0])
    dvol = grid.volume / e0.numel()
    rho_ion = gaussian_ion_density(pos, z, grid, _default_beta(grid, 2))

    def veff_vcap(n_e):
        rho = e0 * (n_e / (float(e0.sum()) * dvol))
        rho_tot = rho - rho_ion
        v_eff = (hartree_potential_r(rho_tot, grid.g2)
                 + esm_potential(rho, pos, z, grid, mode="capacitor"))
        return v_eff, hartree_potential_capacitor(rho_tot, grid.cell)

    ve_n, vc_n = veff_vcap(10.0)  # neutral (N=ΣZ)
    ve_c, vc_c = veff_vcap(10.4)  # +0.4 e charged
    d = (ve_c - ve_n) - (vc_c - vc_n)
    assert float(d.std()) < 5e-3


def test_input_rejects_bad_boundary():
    """inputs validation guards the boundary knob."""
    from gradwave.inputs import InputError, SCFParams

    SCFParams(boundary="open_z")  # ok
    SCFParams(boundary="periodic")  # ok
    with pytest.raises(InputError, match="boundary"):
        SCFParams(boundary="open_x")
