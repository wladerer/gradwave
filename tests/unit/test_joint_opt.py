"""Joint geometry+electronic minimization (opt/joint.py) — fast checks.

The consistency anchor: the joint energy functional at (ε=0, reference
fractional positions, converged orbitals) must reproduce the SCF total —
same contract postscf/stress.py's strained energy honours. Everything else
(descent, orthonormality, H-apply accounting) is checked on a Γ-only Si cell
kept small enough for the fast gate.

The metal path (opt/_metals.py, joint_free_energy) adds the same anchor at the
free-energy level: F = E − σS at converged orbitals must reproduce the SCF
free energy, and dF/dε at fixed occupations must reproduce stress() — checked
on a small smeared Al cell. GGA/PBE reuses the insulator anchors with PBE xc.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import RDTYPE
from gradwave.opt.joint import (
    count_h_applies,
    joint_energy,
    joint_free_energy,
    joint_relax,
    lowdin,
    teter_precond,
)
from gradwave.pseudo.radial_torch import RadialTables
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo, si_upf

A0 = 5.43
AL_A = 4.05
AL_ONCV = "Al_ONCV_PBE-1.2.upf"


def _si_system(ecut=12 * RY, kmesh=(1, 1, 1), pos=None, cell=None):
    cell = (A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
            if cell is None else cell)
    pos = np.array([[0.0, 0, 0], [A0 / 4] * 3]) if pos is None else pos
    return cell, pos, setup_system(
        cell=cell, positions=pos, species_of_atom=[0, 0], upfs=[si_upf()],
        ecut=ecut, kmesh=kmesh, use_symmetry=False)


@pytest.fixture(scope="module")
def si_scf():
    """One Γ-only Si SCF shared by the consistency tests (with H-count)."""
    cell, pos, system = _si_system()
    xc = LDA_PW92()
    with count_h_applies() as ctr:
        res = scf(system, xc, verbose=False)
    return cell, pos, system, xc, res, ctr.count


def test_lowdin_orthonormalizes():
    g = torch.Generator().manual_seed(7)
    z = torch.randn(4, 50, 2, generator=g, dtype=RDTYPE)
    c = lowdin(torch.view_as_complex(z))
    s = c @ c.conj().T
    assert torch.allclose(s, torch.eye(4, dtype=s.dtype), atol=1e-12)


def test_joint_energy_matches_scf_total(si_scf):
    cell, pos, system, xc, res, h_count = si_scf
    assert res.converged
    assert h_count > 0  # the counter saw Davidson's H-applies

    nocc = 4
    coeffs = [c[:nocc] for c in res.coeffs]
    occ = torch.full((len(system.spheres), nocc), 2.0, dtype=RDTYPE)
    tabs = [RadialTables(u) for u in system.upfs]
    eps = torch.zeros(3, 3, dtype=RDTYPE)
    frac = torch.tensor(pos @ np.linalg.inv(cell), dtype=RDTYPE)
    e = joint_energy(system, xc, tabs, eps, frac, coeffs, occ)
    assert float(e) == pytest.approx(float(res.energies.total), abs=1e-6)


def test_joint_energy_strain_gradient_matches_stress(si_scf):
    """dE/dε of the joint functional at the SCF point = Ω·σ (fixed basis)."""
    from gradwave.postscf.stress import stress

    cell, pos, system, xc, res, _ = si_scf
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


def test_joint_energy_pbe_matches_scf_total():
    """GGA/PBE is wired through the same joint functional: at ε=0 with the
    converged PBE orbitals it reproduces the PBE SCF total energy."""
    cell, pos, system = _si_system()
    xc = PBE()
    res = scf(system, xc, verbose=False)
    assert res.converged
    coeffs = [c[:4] for c in res.coeffs]
    occ = torch.full((len(system.spheres), 4), 2.0, dtype=RDTYPE)
    tabs = [RadialTables(u) for u in system.upfs]
    eps = torch.zeros(3, 3, dtype=RDTYPE)
    frac = torch.tensor(pos @ np.linalg.inv(cell), dtype=RDTYPE)
    e = joint_energy(system, xc, tabs, eps, frac, coeffs, occ)
    assert float(e) == pytest.approx(float(res.energies.total), abs=1e-5)


def test_joint_energy_pbe_strain_gradient_matches_stress():
    """dE/dε of the PBE joint functional at the SCF point = Ω·σ (GGA sigma term
    on the strained graph reproduces stress())."""
    from gradwave.postscf.stress import stress

    cell, pos, system = _si_system()
    xc = PBE()
    res = scf(system, xc, verbose=False)
    coeffs = [c[:4].detach() for c in res.coeffs]
    occ = torch.full((len(system.spheres), 4), 2.0, dtype=RDTYPE)
    tabs = [RadialTables(u) for u in system.upfs]
    eps = torch.zeros(3, 3, dtype=RDTYPE, requires_grad=True)
    frac = torch.tensor(pos @ np.linalg.inv(cell), dtype=RDTYPE)
    e = joint_energy(system, xc, tabs, eps, frac, coeffs, occ)
    (g,) = torch.autograd.grad(e, eps)
    sigma_joint = 0.5 * (g + g.T) / system.grid.volume
    sigma_ref = stress(res, xc, symmetrize=False)
    assert torch.allclose(sigma_joint, sigma_ref, atol=1e-7)


def _al_system(ecut=12 * RY, kmesh=(2, 2, 2)):
    cell = AL_A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0]])
    upf = parse_upf(pseudo(AL_ONCV))
    system = setup_system(cell=cell, positions=pos, species_of_atom=[0],
                          upfs=[upf], ecut=ecut, kmesh=kmesh, use_symmetry=False)
    return cell, pos, system


@pytest.fixture(scope="module")
def al_scf():
    """A small smeared Al SCF (a metal) shared by the free-energy anchors."""
    cell, pos, system = _al_system()
    xc = LDA_PW92()
    res = scf(system, xc, smearing="gaussian", width=0.3, verbose=False)
    return cell, pos, system, xc, res


@pytest.mark.standard
def test_metal_stress_at_fractional_occupations(al_scf):
    """The joint energy fed the SCF's FRACTIONAL metal occupations reproduces
    stress() — the entropy has no explicit ε-dependence, so dF/dε = dE/dε."""
    from gradwave.postscf.stress import stress

    cell, pos, system, xc, res = al_scf
    tabs = [RadialTables(u) for u in system.upfs]
    eps = torch.zeros(3, 3, dtype=RDTYPE, requires_grad=True)
    frac = torch.tensor(pos @ np.linalg.inv(cell), dtype=RDTYPE)
    coeffs = [c.detach() for c in res.coeffs]
    e = joint_energy(system, xc, tabs, eps, frac, coeffs, res.occupations)
    e_val = float(e.detach())
    (g,) = torch.autograd.grad(e, eps)
    sigma_joint = 0.5 * (g + g.T) / system.grid.volume
    sigma_ref = stress(res, xc, symmetrize=False)
    assert e_val == pytest.approx(float(res.energies.total), abs=1e-5)
    assert torch.allclose(sigma_joint, sigma_ref, atol=1e-7)


@pytest.mark.standard
def test_joint_free_energy_matches_scf_free_energy(al_scf):
    """F = E − σS from the subspace-diagonalisation occupations reproduces the
    SCF free energy at the converged orbitals — the metal anchor mirroring the
    insulator's energy anchor."""
    cell, pos, system, xc, res = al_scf
    tabs = [RadialTables(u) for u in system.upfs]
    lmax = max((b.l for u in system.upfs for b in u.betas), default=0)
    frac = torch.tensor(pos @ np.linalg.inv(cell), dtype=RDTYPE)
    coeffs = [lowdin(c.detach()) for c in res.coeffs]  # all n_bands, orthonormal
    eps = torch.zeros(3, 3, dtype=RDTYPE)
    f, occ, mu, eigs = joint_free_energy(system, xc, tabs, eps, frac, coeffs,
                                         "gaussian", 0.3, lmax=lmax, n_inner=8)
    assert float(f) == pytest.approx(float(res.energies.free_energy), abs=1e-4)
    assert mu == pytest.approx(res.fermi, abs=1e-2)
    ne = float((occ * system.kweights[:, None]).sum())
    assert ne == pytest.approx(system.n_electrons, abs=1e-6)


@pytest.mark.standard
def test_joint_free_energy_strain_gradient_matches_stress(al_scf):
    """dF/dε from the metal free-energy functional (occupations detached) equals
    Ω·stress at the converged smeared point."""
    from gradwave.postscf.stress import stress

    cell, pos, system, xc, res = al_scf
    tabs = [RadialTables(u) for u in system.upfs]
    lmax = max((b.l for u in system.upfs for b in u.betas), default=0)
    frac = torch.tensor(pos @ np.linalg.inv(cell), dtype=RDTYPE)
    coeffs = [lowdin(c.detach()) for c in res.coeffs]
    eps = torch.zeros(3, 3, dtype=RDTYPE, requires_grad=True)
    f, *_ = joint_free_energy(system, xc, tabs, eps, frac, coeffs,
                              "gaussian", 0.3, lmax=lmax, n_inner=8)
    (g,) = torch.autograd.grad(f, eps)
    sigma_joint = 0.5 * (g + g.T) / system.grid.volume
    sigma_ref = stress(res, xc, symmetrize=False)
    assert torch.allclose(sigma_joint, sigma_ref, atol=2e-4)


@pytest.mark.standard
def test_joint_free_energy_position_gradient_matches_forces():
    """−dF/dτ from the LIVE metal free-energy functional (robust-eigh subspace
    rotation, live occupations, #129) equals the Hellmann–Feynman forces at a
    converged smeared point with NONZERO forces (displaced 2-atom Al). At
    self-consistency the occupation/rotation response channels carry zero
    prefactor, so the live gradient must reduce to postscf.forces exactly —
    the anchor the frozen-occupation trajectory bias motivated."""
    from gradwave.postscf.forces import forces

    cell = AL_A * np.eye(3)
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) * AL_A
    pos[1] += [0.10, -0.06, 0.04]
    upf = parse_upf(pseudo(AL_ONCV))
    system = setup_system(cell=cell, positions=pos, species_of_atom=[0, 0],
                          upfs=[upf], ecut=12 * RY, kmesh=(2, 2, 2),
                          use_symmetry=False)
    xc = LDA_PW92()
    res = scf(system, xc, smearing="gaussian", width=0.3, verbose=False)
    assert res.converged
    tabs = [RadialTables(u) for u in system.upfs]
    lmax = max((b.l for u in system.upfs for b in u.betas), default=0)
    eps = torch.zeros(3, 3, dtype=RDTYPE)
    frac = torch.tensor(pos @ np.linalg.inv(cell), dtype=RDTYPE,
                        requires_grad=True)
    coeffs = [lowdin(c.detach()) for c in res.coeffs]
    # n_inner=200: this 2×2×2 Fermi shell charge-sloshes, and every response
    # channel of the live gradient carries a (λ−ε) prefactor that only vanishes
    # at the inner fixed point — 60 damped sweeps leave a 4e-4 eV/Å residual,
    # 200 reach ~1e-7 (the force error is a direct readout of that residual).
    f, *_ = joint_free_energy(system, xc, tabs, eps, frac, coeffs,
                              "gaussian", 0.3, lmax=lmax, n_inner=200)
    (g,) = torch.autograd.grad(f, frac)
    de_dpos = g.numpy() @ np.linalg.inv(cell).T
    de_dpos = de_dpos - de_dpos.mean(axis=0, keepdims=True)
    f_ref = forces(res, xc=xc).numpy()
    assert np.abs(-de_dpos - f_ref).max() < 1e-5


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
