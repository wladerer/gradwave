"""Tersoff-Hamann STM (postscf/stm.py): LDOS assembly on a real grid + tip planes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from gradwave.dtypes import CDTYPE
from gradwave.postscf.stm import ldos_grid, stm_constant_height
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system

RY = 13.605693122994


def _tiny_system():
    from tests.helpers import pseudo  # noqa: PLC0415
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
