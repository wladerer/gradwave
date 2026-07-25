"""Phase 1 of the differentiable-composition channel.

A per-atom weight lambda blends two endpoint pseudopotentials for the local
potential and the ionic charge, and dE/dlambda is the alchemical derivative of
those terms. The tests check endpoint exactness (lambda=0 is species A, lambda=1
is species B) and the derivative against finite difference on both the Ewald
charge channel and the Hellmann-Feynman local term. This is not the virtual
crystal approximation. The endpoints are real species and the derivative is
taken there. Si and Ge are the pair, so both the local potential (different form
factors) and the charge (4 vs 14 valence) vary with lambda.
"""

from pathlib import Path

import numpy as np
import torch

from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.dtypes import RDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.alchemical import (
    alchemical_charges,
    blend_local_table,
    endpoint_local_tables,
    per_atom_local_tables,
)
from gradwave.scf.loop import setup_system
from gradwave.scf.setup_common import _unique_shells

RY = 13.605693122994
DG = Path(__file__).resolve().parents[2] / "benchmarks" / "delta_gauge" / "pseudos"


def _si_cell():
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    return cell, pos


def _setup():
    cell, pos = _si_cell()
    si = parse_upf(DG / "Si.upf")
    ge = parse_upf(DG / "Ge.upf")
    system = setup_system(cell, pos, [0, 0], [si], ecut=20 * RY, kmesh=(1, 1, 1))
    return system, si, ge, cell, pos


def _shells(grid):
    g_flat = np.sqrt(grid.g2.reshape(-1).numpy())
    return _unique_shells(g_flat)


def test_endpoint_tables_match_plain_setup():
    # the A endpoint table must equal the per-species table a plain setup builds
    system, si, ge, _, _ = _setup()
    uniq, inv = _shells(system.grid)
    si_tab, _ = endpoint_local_tables(si, ge, uniq, inv, system.grid.shape)
    assert torch.allclose(si_tab, system.vloc_tables[0], atol=1e-10)


def test_blend_endpoints_are_exact():
    system, si, ge, _, _ = _setup()
    uniq, inv = _shells(system.grid)
    si_tab, ge_tab = endpoint_local_tables(si, ge, uniq, inv, system.grid.shape)
    b0 = blend_local_table(si_tab, ge_tab, torch.tensor(0.0, dtype=RDTYPE))
    b1 = blend_local_table(si_tab, ge_tab, torch.tensor(1.0, dtype=RDTYPE))
    assert torch.allclose(b0, si_tab, atol=1e-12)
    assert torch.allclose(b1, ge_tab, atol=1e-12)


def test_identity_blend_has_zero_gradient():
    # blending a species with itself is lambda-independent
    system, si, _, _, _ = _setup()
    uniq, inv = _shells(system.grid)
    si_tab, _ = endpoint_local_tables(si, si, uniq, inv, system.grid.shape)
    lam = torch.tensor(0.4, dtype=RDTYPE, requires_grad=True)
    blend_local_table(si_tab, si_tab, lam).sum().backward()
    assert lam.grad.abs().item() < 1e-12


def test_charge_channel_ewald_gradient_vs_fd():
    # atom 0 transmutes Si -> Ge (Z 4 -> 14); atom 1 stays Si
    system, si, ge, cell, pos = _setup()
    pos_t = torch.tensor(pos, dtype=RDTYPE)
    z_si, z_ge = float(si.z_valence), float(ge.z_valence)

    def e_ewald(x):
        q = torch.stack([alchemical_charges(z_si, z_ge, x),
                         torch.tensor(z_si, dtype=RDTYPE)])
        return ewald_energy(pos_t, q, cell)

    lam0 = 0.3
    lam = torch.tensor(lam0, dtype=RDTYPE, requires_grad=True)
    e_ewald(lam).backward()
    g_ad = lam.grad.item()
    h = 1e-5
    g_fd = (e_ewald(torch.tensor(lam0 + h, dtype=RDTYPE)).item()
            - e_ewald(torch.tensor(lam0 - h, dtype=RDTYPE)).item()) / (2 * h)
    assert abs(g_ad - g_fd) < 1e-4 * max(1.0, abs(g_fd)), (g_ad, g_fd)


def test_local_potential_gradient_vs_fd():
    # d(E_loc)/dlambda at a fixed density, autograd vs finite difference
    system, si, ge, _, pos = _setup()
    grid = system.grid
    uniq, inv = _shells(grid)
    si_tab, ge_tab = endpoint_local_tables(si, ge, uniq, inv, grid.shape)
    torch.manual_seed(0)
    rho_g = torch.randn(grid.shape, dtype=torch.complex128)

    def e_loc(x):
        vatom = per_atom_local_tables(system.vloc_tables, system.species_index,
                                      {0: (si_tab, ge_tab, x)})
        vg = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume,
                               vloc_atom=vatom)
        return local_energy(rho_g, vg, grid.volume)

    lam0 = 0.35
    lam = torch.tensor(lam0, dtype=RDTYPE, requires_grad=True)
    e_loc(lam).backward()
    g_ad = lam.grad.item()
    h = 1e-5
    g_fd = (e_loc(torch.tensor(lam0 + h, dtype=RDTYPE)).item()
            - e_loc(torch.tensor(lam0 - h, dtype=RDTYPE)).item()) / (2 * h)
    assert abs(g_ad - g_fd) < 1e-5 * max(1.0, abs(g_fd)), (g_ad, g_fd)
