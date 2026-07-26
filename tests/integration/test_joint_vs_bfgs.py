"""Joint (cell, positions, orbitals) descent vs nested BFGS+SCF on silicon.

The research question (#122): gradwave is differentiable end-to-end, so a
variable-cell relaxation can descend on strain, positions, and orbital
coefficients SIMULTANEOUSLY instead of running a full SCF inside every
geometry step. Success bar: same minimum as the nested reference (energy
within 1e-5 Ha, geometry within 1e-3 Å) with fewer Hamiltonian applications.

Both paths use the same physics settings (LDA, 15 Ry, 2x2x2 mesh, no
symmetry). Final energies are compared via a fresh, fully converged SCF at
each method's relaxed geometry so neither method's own energy accounting
(fixed-basis strain vs rebuilt basis) biases the comparison.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS

from gradwave.calculator import GradWave
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.opt.joint import count_h_applies, joint_relax
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, SI_ONCV, pseudo

HA = 27.211386245988
A0 = 5.43
ECUT = 15 * RY
KPTS = (2, 2, 2)


def _perturbed_si():
    """2-atom Si, one atom displaced 0.1 Å and the cell strained ~1.5%."""
    cell = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    strain = np.array([[1.015, 0.004, 0.0],
                       [0.004, 0.985, 0.0],
                       [0.0, 0.0, 1.010]])
    cell = cell @ strain.T
    pos = np.array([[0.0, 0, 0], [A0 / 4] * 3]) @ strain.T
    pos[1] += [0.08, -0.05, 0.03]
    return cell, pos


def _scf_at(cell, pos):
    """Fresh, fully converged SCF free energy [eV] at a given geometry."""
    upf = parse_upf(pseudo(SI_ONCV))
    system = setup_system(cell=cell, positions=pos, species_of_atom=[0, 0],
                          upfs=[upf], ecut=ECUT, kmesh=KPTS,
                          use_symmetry=False)
    res = scf(system, LDA_PW92(), verbose=False)
    assert res.converged
    return float(res.energies.free_energy)


@pytest.mark.slow
def test_joint_relax_matches_nested_bfgs_scf():
    torch.set_num_threads(8)
    cell0, pos0 = _perturbed_si()

    # ---- reference: nested SCF-inside-BFGS, variable cell
    atoms = Atoms("Si2", positions=pos0.copy(), cell=cell0.copy(), pbc=True)
    atoms.calc = GradWave(ecut=ECUT, pseudopotentials={"Si": pseudo(SI_ONCV)},
                          xc="lda", kpts=KPTS, use_symmetry=False)
    with count_h_applies() as ref_counter:
        opt = BFGS(FrechetCellFilter(atoms), logfile=None)
        assert opt.run(fmax=0.005, steps=80)
    h_ref = ref_counter.count
    cell_ref = atoms.cell.array.copy()
    pos_ref = atoms.get_positions().copy()

    # ---- joint descent
    res = joint_relax(cell0, pos0, [0, 0], [parse_upf(pseudo(SI_ONCV))],
                      LDA_PW92(), ecut=ECUT, kmesh=KPTS,
                      fmax=0.005, max_closures=800)
    assert res.converged

    # ---- same minimum?
    bond_ref = np.linalg.norm(pos_ref[1] - pos_ref[0])
    bond_joint = np.linalg.norm(res.positions[1] - res.positions[0])
    a_ref = np.linalg.norm(cell_ref[0])
    a_joint = np.linalg.norm(res.cell[0])
    e_ref = _scf_at(cell_ref, pos_ref)
    e_joint = _scf_at(res.cell, res.positions)

    print(f"\nbond: joint {bond_joint:.5f} vs ref {bond_ref:.5f} A")
    print(f"|a1|: joint {a_joint:.5f} vs ref {a_ref:.5f} A")
    print(f"E(SCF@final): joint {e_joint:.6f} vs ref {e_ref:.6f} eV "
          f"(diff {abs(e_joint - e_ref) / HA:.2e} Ha)")
    print(f"H-applies: ref {h_ref}, joint {res.h_equiv} "
          f"(seed {res.h_seed} + {res.n_closures} closures), "
          f"ratio {h_ref / max(res.h_equiv, 1):.2f}x, "
          f"cycles {res.n_cycles}")

    assert abs(bond_joint - bond_ref) < 1e-3
    assert abs(a_joint - a_ref) < 2e-3
    assert abs(e_joint - e_ref) < 1e-5 * HA


# --------------------------------------------------------------- metal (Al)
AL_A = 4.05
AL_ONCV = "Al_ONCV_PBE-1.2.upf"
AL_ECUT = 13 * RY
AL_KPTS = (3, 3, 3)
SMEAR, WIDTH = "gaussian", 0.3


def _perturbed_al2():
    """2-atom bcc-like Al (simple-cubic cell, atoms at corner + body centre),
    one atom displaced — a metal, relaxed at fixed cell (positions only)."""
    cell = AL_A * np.eye(3)
    pos = np.array([[0.0, 0, 0], [0.5, 0.5, 0.5]]) * AL_A
    pos[1] += [0.15, -0.10, 0.06]
    return cell, pos


def _al_scf_free(cell, pos):
    upf = parse_upf(pseudo(AL_ONCV))
    system = setup_system(cell=cell, positions=pos, species_of_atom=[0, 0],
                          upfs=[upf], ecut=AL_ECUT, kmesh=AL_KPTS,
                          use_symmetry=False)
    res = scf(system, LDA_PW92(), smearing=SMEAR, width=WIDTH, verbose=False)
    assert res.converged
    return float(res.energies.free_energy)


@pytest.mark.slow
def test_joint_relax_metal_vs_nested_bfgs_scf():
    """Metal head-to-head (mirrors the Si test): free-energy joint descent with
    LIVE variational occupations (robust-eigh subspace rotation, #129) vs
    nested BFGS+SCF, positions only, against a TIGHT (fmax = 0.005) reference.

    Honest state (see docs/design/joint-geometry-electronic.md): the free-energy
    FUNCTIONAL and its gradient are exact at electronic self-consistency (F to
    ~5e-10 eV, force to ~4e-7 eV/Å vs postscf.forces — anchors in
    tests/unit/test_joint_opt.py), and the descent genuinely moves the atoms
    down the free-energy surface into the reference basin — the frozen-occupation
    scheme this replaces did NOT (it reported "converged" without ever leaving
    the start geometry; caught by exactly the displacement assertion below).
    But away from self-consistency the detached inner occupation solve and the
    Lorentzian-broadened eigh backward leave a trajectory-level force bias, so
    the descent stops short of insulator-grade agreement (~0.02-0.03 Å on this
    soft mode, not 1e-3 Å) and needs several hundred closures — the H-apply
    advantage inverts (~0.4x, reported below, not asserted). The scoped fix is
    the preconditioned Marzari-Vanderbilt occupation block (#129)."""
    torch.set_num_threads(8)
    cell, pos0 = _perturbed_al2()
    d0 = np.linalg.norm(pos0[1] - pos0[0])

    # ---- reference: nested SCF-inside-BFGS at fixed cell, TIGHT
    atoms = Atoms("Al2", positions=pos0.copy(), cell=cell.copy(), pbc=True)
    atoms.calc = GradWave(ecut=AL_ECUT, pseudopotentials={"Al": pseudo(AL_ONCV)},
                          xc="lda", kpts=AL_KPTS, use_symmetry=False,
                          smearing=SMEAR, width=WIDTH)
    with count_h_applies() as ref_counter:
        opt = BFGS(atoms, logfile=None)
        assert opt.run(fmax=0.005, steps=40)
    h_ref = ref_counter.count
    pos_ref = atoms.get_positions().copy()

    # ---- joint free-energy descent (metal path, live occupations)
    res = joint_relax(cell, pos0, [0, 0], [parse_upf(pseudo(AL_ONCV))],
                      LDA_PW92(), ecut=AL_ECUT, kmesh=AL_KPTS,
                      smearing=SMEAR, width=WIDTH, fix_cell=True, fmax=0.01,
                      broaden=WIDTH / 10, max_closures=450, lbfgs_chunk=25)

    d_ref = np.linalg.norm(pos_ref[1] - pos_ref[0])
    d_joint = np.linalg.norm(res.positions[1] - res.positions[0])
    e_ref = _al_scf_free(cell, pos_ref)
    e_joint = _al_scf_free(cell, res.positions)

    print(f"\nmetal pair sep: joint {d_joint:.5f} vs ref {d_ref:.5f} A "
          f"(start {d0:.5f})")
    print(f"metal E(SCF@final): joint {e_joint:.6f} vs ref {e_ref:.6f} eV "
          f"(diff {abs(e_joint - e_ref) / HA:.2e} Ha)")
    print(f"metal H-applies: ref {h_ref}, joint {res.h_equiv} "
          f"(seed {res.h_seed} + {res.n_closures} closures), "
          f"ratio {h_ref / max(res.h_equiv, 1):.2f}x")

    # The descent must actually descend: the atoms move most of the way from
    # the start separation to the tight-reference one (the frozen-occupation
    # regression left d_joint == d0 to 4 decimal places), and the endpoint sits
    # in the reference basin by both geometry and a fresh-SCF free energy.
    assert abs(d_joint - d0) > 0.5 * abs(d_ref - d0)   # really moved
    assert abs(d_joint - d_ref) < 0.05                 # Å — same basin
    assert abs(e_joint - e_ref) < 1e-3 * HA            # ~same free-energy min
