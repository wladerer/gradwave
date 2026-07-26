"""Joint geometry+electronic minimization (opt/joint.py) — fast checks.

The consistency anchor: the joint energy functional at (ε=0, reference
fractional positions, converged orbitals) must reproduce the SCF total —
same contract postscf/stress.py's strained energy honours. Everything else
(descent, orthonormality, H-apply accounting) is checked on a Γ-only Si cell
kept small enough for the fast gate.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.dtypes import RDTYPE
from gradwave.opt.joint import (
    count_h_applies,
    joint_energy,
    joint_relax,
    lowdin,
    teter_precond,
)
from gradwave.pseudo.radial_torch import RadialTables
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_upf

A0 = 5.43


def _si_system(ecut=12 * RY, kmesh=(1, 1, 1), pos=None, cell=None):
    cell = (A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
            if cell is None else cell)
    pos = np.array([[0.0, 0, 0], [A0 / 4] * 3]) if pos is None else pos
    return cell, pos, setup_system(
        cell=cell, positions=pos, species_of_atom=[0, 0], upfs=[si_upf()],
        ecut=ecut, kmesh=kmesh, use_symmetry=False)


def test_lowdin_orthonormalizes():
    g = torch.Generator().manual_seed(7)
    z = torch.randn(4, 50, 2, generator=g, dtype=RDTYPE)
    c = lowdin(torch.view_as_complex(z))
    s = c @ c.conj().T
    assert torch.allclose(s, torch.eye(4, dtype=s.dtype), atol=1e-12)


def test_joint_energy_matches_scf_total():
    cell, pos, system = _si_system()
    xc = LDA_PW92()
    with count_h_applies() as ctr:
        res = scf(system, xc, verbose=False)
    assert res.converged
    assert ctr.count > 0  # the counter saw Davidson's H-applies

    nocc = 4
    coeffs = [c[:nocc] for c in res.coeffs]
    occ = torch.full((len(system.spheres), nocc), 2.0, dtype=RDTYPE)
    tabs = [RadialTables(u) for u in system.upfs]
    eps = torch.zeros(3, 3, dtype=RDTYPE)
    frac = torch.tensor(pos @ np.linalg.inv(cell), dtype=RDTYPE)
    e = joint_energy(system, xc, tabs, eps, frac, coeffs, occ)
    assert float(e) == pytest.approx(float(res.energies.total), abs=1e-6)


def test_joint_energy_strain_gradient_matches_stress():
    """dE/dε of the joint functional at the SCF point = Ω·σ (fixed basis)."""
    from gradwave.postscf.stress import stress

    cell, pos, system = _si_system()
    xc = LDA_PW92()
    res = scf(system, xc, verbose=False)
    nocc = 4
    coeffs = [c[:nocc].detach() for c in res.coeffs]
    occ = torch.full((len(system.spheres), nocc), 2.0, dtype=RDTYPE)
    tabs = [RadialTables(u) for u in system.upfs]
    eps = torch.zeros(3, 3, dtype=RDTYPE, requires_grad=True)
    frac = torch.tensor(pos @ np.linalg.inv(cell), dtype=RDTYPE)
    e = joint_energy(system, xc, tabs, eps, frac, coeffs, occ)
    (g,) = torch.autograd.grad(e, eps)
    sigma_joint = 0.5 * (g + g.T) / system.grid.volume
    sigma_ref = stress(res, xc, symmetrize=False)
    assert torch.allclose(sigma_joint, sigma_ref, atol=1e-8)


def test_teter_precond_range():
    kpg2 = torch.tensor([0.0, 1.0, 100.0], dtype=RDTYPE)
    p = teter_precond(kpg2, 10.0)
    assert float(p[0]) == 1.0
    assert (p > 0).all() and (p <= 1.0).all()
    assert p[2] < p[1]


@pytest.mark.standard
def test_joint_relax_fixed_cell_recovers_si_bond():
    """Positions-only joint descent pulls a displaced Si atom back to the
    ideal bond, matching the SCF+forces answer without any nested SCF."""
    cell = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [A0 / 4] * 3])
    pos[1] += [0.10, -0.06, 0.04]
    res = joint_relax(cell, pos, [0, 0], [si_upf()], LDA_PW92(),
                      ecut=12 * RY, kmesh=(2, 2, 2), fix_cell=True,
                      fmax=0.005, max_closures=400)
    assert res.converged
    bond = np.linalg.norm(res.positions[1] - res.positions[0])
    assert bond == pytest.approx(A0 * np.sqrt(3) / 4, abs=2e-3)
    assert res.fmax < 0.005
