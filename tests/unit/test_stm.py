"""Tersoff-Hamann STM (postscf/stm.py): LDOS assembly on a real grid + tip planes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.dtypes import CDTYPE
from gradwave.postscf.stm import (
    ldos_grid,
    spin_ldos_grid,
    stm_constant_height,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system

RY = 13.605693122994


def _tiny_system():
    from tests.helpers import pseudo
    cell = np.eye(3) * 5.0
    pos = np.array([[0.0, 0.0, 0.0]])
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    return setup_system(cell, pos, [0], [upf], ecut=6 * RY, kmesh=(1, 1, 1),
                        use_symmetry=False)


def _mock_result(system, coeffs, eigs, fermi=0.0):
    return SimpleNamespace(system=system, fermi=fermi, nspin=1,
                           coeffs=[coeffs], eigenvalues=eigs)


def test_g0_state_gives_uniform_ldos():
    """A pure G=0 plane wave is spatially constant, so its |ψ|² LDOS is uniform."""
    system = _tiny_system()
    npw = system.spheres[0].npw
    c = torch.zeros(1, npw, dtype=CDTYPE)
    c[0, 0] = 1.0                                   # G=0 is the lowest |k+G|² (index 0 at Γ)
    res = _mock_result(system, c, torch.zeros(1, 1))
    ldos = ldos_grid(res, energy=0.0, sigma=1.0)
    assert float(ldos.mean()) > 0
    assert float(ldos.std() / ldos.mean()) < 1e-8   # uniform


def test_energy_window_excludes_far_states():
    """A state far outside the energy window contributes ~nothing."""
    system = _tiny_system()
    npw = system.spheres[0].npw
    c = torch.zeros(1, npw, dtype=CDTYPE)
    c[0, 0] = 1.0
    near = ldos_grid(_mock_result(system, c, torch.zeros(1, 1)), energy=0.0, sigma=0.2)
    far = ldos_grid(_mock_result(system, c, torch.full((1, 1), 50.0)), energy=0.0, sigma=0.2)
    assert float(far.abs().max()) < 1e-6 * float(near.abs().max())


def test_symmetry_reduced_ldos_is_space_group_symmetric():
    """A symmetry-reduced SCF sums over the IBZ; ldos_grid symmetrizes the map over
    the space group, so the result is invariant under the group (re-symmetrizing is
    a no-op) — this is what makes an IBZ STM match a full-BZ one."""
    from tests.helpers import pseudo
    cell = 5.43 / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # fcc Si primitive
    pos = np.array([[0.0, 0, 0], [0.25, 0.25, 0.25]]) @ cell
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    system = setup_system(cell, pos, [0, 0], [upf], ecut=8 * RY, kmesh=(2, 2, 2),
                          use_symmetry=True)
    assert system.sym is not None and system.sym.n_ops > 1
    torch.manual_seed(0)
    coeffs = [torch.randn(2, s.npw, dtype=CDTYPE) for s in system.spheres]
    eigs = torch.zeros(len(system.spheres), 2)
    res = SimpleNamespace(system=system, fermi=0.0, nspin=1, coeffs=coeffs, eigenvalues=eigs)
    ldos = ldos_grid(res, energy=0.0, sigma=1.0)
    # re-applying the same G-space symmetrizer is a no-op on an already-symmetric map
    resym = g_to_r_box(system.rho_symmetrizer.apply(r_to_g(ldos.to(CDTYPE))), real=True)
    assert torch.allclose(ldos, resym, atol=1e-9)


def test_spin_ldos_nonmagnetic_has_zero_spin():
    """nspin=1: spin_ldos_grid returns m = 0 and rho = the charge ldos."""
    system = _tiny_system()
    npw = system.spheres[0].npw
    c = torch.zeros(1, npw, dtype=CDTYPE)
    c[0, 0] = 1.0
    res = _mock_result(system, c, torch.zeros(1, 1))
    rho, m = spin_ldos_grid(res, energy=0.0, sigma=1.0)
    assert torch.allclose(m, torch.zeros_like(m))
    assert torch.allclose(rho, ldos_grid(res, energy=0.0, sigma=1.0))


def _mock_nspin2(system, up, dn, eigs, fermi=0.0):
    return SimpleNamespace(system=system, fermi=fermi, nspin=2,
                           coeffs=[[up], [dn]], eigenvalues=eigs)


def test_collinear_charge_and_spin_channels():
    """Collinear: rho = ρ↑+ρ↓, m = (0,0,ρ↑−ρ↓); ldos_grid(spin=0/1) = ρ↑/ρ↓."""
    system = _tiny_system()                              # use_symmetry=False -> no symm
    npw = system.spheres[0].npw
    torch.manual_seed(1)
    up = torch.randn(2, npw, dtype=CDTYPE)
    dn = torch.randn(2, npw, dtype=CDTYPE)
    eigs = torch.zeros(2, 1, 2)                          # (nspin, nk, nb)
    res = _mock_nspin2(system, up, dn, eigs)
    rho, m = spin_ldos_grid(res, energy=0.0, sigma=1.0)
    rho_up = ldos_grid(res, energy=0.0, sigma=1.0, spin=0)
    rho_dn = ldos_grid(res, energy=0.0, sigma=1.0, spin=1)
    assert torch.allclose(m[0], torch.zeros_like(m[0]))   # collinear -> no in-plane spin
    assert torch.allclose(m[1], torch.zeros_like(m[1]))
    assert torch.allclose(rho, rho_up + rho_dn, atol=1e-10)
    assert torch.allclose(m[2], rho_up - rho_dn, atol=1e-10)
    assert torch.allclose(ldos_grid(res, energy=0.0, sigma=1.0), rho)  # spin=None -> charge


def test_tip_polarization_projects_spin():
    """A magnetized tip images ρ + P·m; P=(0,0,1) gives ρ + m_z."""
    system = _tiny_system()
    npw = system.spheres[0].npw
    torch.manual_seed(2)
    up = torch.randn(2, npw, dtype=CDTYPE)
    dn = torch.randn(2, npw, dtype=CDTYPE)
    res = _mock_nspin2(system, up, dn, torch.zeros(2, 1, 2))
    rho, m = spin_ldos_grid(res, energy=0.0, sigma=1.0)
    img, z_tip = stm_constant_height(res, height=1.0, sigma=1.0, tip_polarization=(0.0, 0.0, 1.0))
    length = float(system.grid.cell[2, 2])
    nz = system.grid.shape[2]
    iz = int(round((z_tip % length) / length * nz)) % nz
    assert torch.allclose(img, (rho + m[2])[:, :, iz], atol=1e-10)


def test_constant_height_returns_2d_plane():
    system = _tiny_system()
    npw = system.spheres[0].npw
    c = torch.zeros(1, npw, dtype=CDTYPE)
    c[0, 0] = 1.0
    res = _mock_result(system, c, torch.zeros(1, 1))
    img, z_tip = stm_constant_height(res, height=1.5, sigma=1.0)
    n1, n2, n3 = system.grid.shape
    assert img.shape == (n1, n2)
    assert z_tip > 0
