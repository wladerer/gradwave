"""d-band center / width from projected-DOS moments (postscf.band_center).

No SCF: the moment math is validated on synthetic d-PDOS with a KNOWN 1st/2nd
moment (shifted Gaussians and discrete states). The real-material cross-check
(bulk-Pt ε_d ≈ −2.25 eV vs QE projwfc) needs a slab/bulk SCF and is DEFERRED to
an integration test.
"""

import numpy as np
import pytest
import torch

from gradwave.postscf.band_center import BandMoments, band_center, band_moments
from gradwave.postscf.pdos import ProjectedDOS


def _gauss(grid, mu, sigma):
    return np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _pdos(groups, fermi=0.0, nspin=1, grid=None):
    if grid is None:
        grid = np.linspace(-15.0, 15.0, 4001)
    return ProjectedDOS(
        energy_eV=grid, total=sum(groups.values()), groups=groups,
        spilling=0.0, fermi_eV=fermi, nspin=nspin, group_by="l")


# ----------------------------------------------------------------------------
# band_moments — the differentiable primitive
# ----------------------------------------------------------------------------

def test_moments_recover_gaussian_mean_and_width():
    # a Gaussian DOS sampled on a symmetric grid: 1st moment = mu, width = sigma
    mu, sigma = -2.25, 1.3
    grid = np.linspace(mu - 12 * sigma, mu + 12 * sigma, 6001)
    e = torch.as_tensor(grid)
    w = torch.as_tensor(_gauss(grid, mu, sigma))
    m = band_moments(e, w)
    assert isinstance(m, BandMoments)
    assert m.abs_center.item() == pytest.approx(mu, abs=1e-9)
    assert m.width.item() == pytest.approx(sigma, abs=1e-4)
    assert m.variance.item() == pytest.approx(sigma**2, abs=1e-4)


def test_moments_reference_shift():
    # referencing subtracts the Fermi level from the 1st moment only
    mu, sigma, ef = 1.0, 0.8, 3.5
    grid = np.linspace(mu - 12 * sigma, mu + 12 * sigma, 6001)
    e, w = torch.as_tensor(grid), torch.as_tensor(_gauss(grid, mu, sigma))
    m = band_moments(e, w, ref=ef)
    assert m.center.item() == pytest.approx(mu - ef, abs=1e-9)
    assert m.abs_center.item() == pytest.approx(mu, abs=1e-9)
    assert m.width.item() == pytest.approx(sigma, abs=1e-4)  # width is ref-free


def test_moments_discrete_states_exact():
    # per-state populations: center is the population-weighted mean of eigenvalues
    e = torch.tensor([-3.0, -1.0, 0.5, 2.0])
    w = torch.tensor([0.4, 1.0, 0.7, 0.2])
    m = band_moments(e, w)
    expect = float((w * e).sum() / w.sum())
    assert m.abs_center.item() == pytest.approx(expect)


def test_moments_empty_selection_raises():
    e = torch.linspace(-5, 5, 100)
    with pytest.raises(ValueError, match="non-positive"):
        band_moments(e, torch.zeros_like(e))


# ----------------------------------------------------------------------------
# differentiability — the strain/potential-tunable-descriptor angle
# ----------------------------------------------------------------------------

def test_center_is_differentiable_in_a_band_shift():
    # a d-band whose center rides on a parameter theta (a proxy for strain / U):
    # dε_d/dtheta must be exactly 1, and dW_d/dtheta exactly 0.
    grid = np.linspace(-20.0, 20.0, 8001)
    e = torch.as_tensor(grid)
    theta = torch.tensor(-2.25, dtype=torch.float64, requires_grad=True)
    sigma = 1.1
    w = torch.exp(-0.5 * ((e - theta) / sigma) ** 2)
    m = band_moments(e, w)
    (dcenter,) = torch.autograd.grad(m.center, theta, retain_graph=True)
    (dwidth,) = torch.autograd.grad(m.width, theta)
    assert dcenter.item() == pytest.approx(1.0, abs=1e-6)
    assert dwidth.item() == pytest.approx(0.0, abs=1e-6)


def test_center_gradient_wrt_state_energy():
    # dε_d/dε_i = w_i / Σw for a discrete moment — autograd matches the analytic form
    e = torch.tensor([-3.0, -1.0, 0.5, 2.0], requires_grad=True)
    w = torch.tensor([0.4, 1.0, 0.7, 0.2])
    m = band_moments(e, w)
    (grad,) = torch.autograd.grad(m.center, e)
    assert torch.allclose(grad, w / w.sum(), atol=1e-12)


# ----------------------------------------------------------------------------
# band_center — the ProjectedDOS convenience
# ----------------------------------------------------------------------------

def test_band_center_selects_d_and_references_fermi():
    grid = np.linspace(-15.0, 15.0, 6001)
    # an s band and a d band at different energies; band_center must pick d only
    groups = {
        "atom1:4S": _gauss(grid, mu=-6.0, sigma=1.0),
        "atom1:3D": _gauss(grid, mu=-2.25, sigma=1.4),
    }
    p = _pdos(groups, fermi=0.0, grid=grid)
    ed = band_center(p, l="d", ref="fermi", moment=1)
    assert ed == pytest.approx(-2.25, abs=1e-3)
    wd = band_center(p, l="d", moment=2)
    assert wd == pytest.approx(1.4, abs=1e-3)
    # the s band sits elsewhere, proving the l-selection is real
    es = band_center(p, l="s", moment=1)
    assert es == pytest.approx(-6.0, abs=1e-3)


def test_band_center_fermi_offset():
    grid = np.linspace(-15.0, 15.0, 6001)
    p = _pdos({"atom1:3D": _gauss(grid, mu=-1.0, sigma=1.0)}, fermi=2.0, grid=grid)
    assert band_center(p, l="d", ref="fermi") == pytest.approx(-3.0, abs=1e-3)
    assert band_center(p, l="d", ref="none") == pytest.approx(-1.0, abs=1e-3)
    assert band_center(p, l="d", ref=0.5) == pytest.approx(-1.5, abs=1e-3)


def test_band_center_atom_selection():
    grid = np.linspace(-15.0, 15.0, 6001)
    groups = {
        "atom1:3D": _gauss(grid, mu=-3.0, sigma=1.0),
        "atom2:3D": _gauss(grid, mu=-1.0, sigma=1.0),
    }
    p = _pdos(groups, fermi=0.0, grid=grid)
    # atoms are 0-based; atom index 0 -> the 'atom1' label
    assert band_center(p, l="d", atoms=[0]) == pytest.approx(-3.0, abs=1e-3)
    assert band_center(p, l="d", atoms=[1]) == pytest.approx(-1.0, abs=1e-3)
    # both atoms: the two equal-area Gaussians average to the midpoint
    assert band_center(p, l="d", atoms=[0, 1]) == pytest.approx(-2.0, abs=1e-3)


def test_band_center_spin_channels():
    grid = np.linspace(-15.0, 15.0, 6001)
    up = _gauss(grid, mu=-3.0, sigma=1.0)
    dn = _gauss(grid, mu=-1.0, sigma=1.0)
    groups = {"atom1:3D": np.stack([up, dn])}  # (nspin=2, npoints)
    p = _pdos(groups, fermi=0.0, nspin=2, grid=grid)
    assert band_center(p, l="d", spin=0) == pytest.approx(-3.0, abs=1e-3)
    assert band_center(p, l="d", spin=1) == pytest.approx(-1.0, abs=1e-3)
    # summed spins: equal areas → midpoint
    assert band_center(p, l="d", spin=None) == pytest.approx(-2.0, abs=1e-3)


def test_band_center_errors():
    grid = np.linspace(-15.0, 15.0, 2001)
    p = _pdos({"atom1:3D": _gauss(grid, -2.0, 1.0)}, fermi=None, grid=grid)
    with pytest.raises(ValueError, match="no fermi_eV"):
        band_center(p, l="d", ref="fermi")
    with pytest.raises(ValueError, match="no 'S' groups"):
        band_center(p, l="s", ref="none")
    with pytest.raises(ValueError, match="moment must be"):
        band_center(p, l="d", ref="none", moment=3)
