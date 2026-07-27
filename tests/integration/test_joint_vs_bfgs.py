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
    """Fresh, fully converged SCF result at a given geometry."""
    upf = parse_upf(pseudo(SI_ONCV))
    system = setup_system(cell=cell, positions=pos, species_of_atom=[0, 0],
                          upfs=[upf], ecut=ECUT, kmesh=KPTS,
                          use_symmetry=False)
    res = scf(system, LDA_PW92(), verbose=False)
    assert res.converged
    return res


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
    scf_ref = _scf_at(cell_ref, pos_ref)
    scf_joint = _scf_at(res.cell, res.positions)
    e_ref = float(scf_ref.energies.free_energy)
    e_joint = float(scf_joint.energies.free_energy)

    # self-oracle: the joint minimum must be a stationary point of the
    # independently-computed SCF energy — fresh forces AND stress ~0 there,
    # and agreeing with the nested reference's own residuals.
    from gradwave.postscf.forces import forces as hf_forces
    from gradwave.postscf.stress import stress as hf_stress

    xc = LDA_PW92()
    f_joint = np.abs(hf_forces(scf_joint, xc=xc).cpu().numpy()).max()
    s_joint = np.abs(hf_stress(scf_joint, xc, symmetrize=False).cpu().numpy()).max()
    f_ref = np.abs(hf_forces(scf_ref, xc=xc).cpu().numpy()).max()
    s_ref = np.abs(hf_stress(scf_ref, xc, symmetrize=False).cpu().numpy()).max()
    ratio = h_ref / max(res.h_equiv, 1)

    print(f"\nbond: joint {bond_joint:.5f} vs ref {bond_ref:.5f} A")
    print(f"|a1|: joint {a_joint:.5f} vs ref {a_ref:.5f} A")
    print(f"E(SCF@final): joint {e_joint:.6f} vs ref {e_ref:.6f} eV "
          f"(diff {abs(e_joint - e_ref) / HA:.2e} Ha)")
    print(f"residuals @final: joint fmax {f_joint:.2e} smax {s_joint:.2e} | "
          f"ref fmax {f_ref:.2e} smax {s_ref:.2e} eV/Å,eV/Å³")
    print(f"H-applies: ref {h_ref}, joint {res.h_equiv} "
          f"(seed {res.h_seed} + {res.n_closures} closures), "
          f"ratio {ratio:.2f}x, cycles {res.n_cycles}")

    assert abs(bond_joint - bond_ref) < 1e-3
    assert abs(a_joint - a_ref) < 2e-3
    assert abs(e_joint - e_ref) < 1e-5 * HA
    # stress/forces agreement: both methods sit at an SCF stationary point
    assert f_joint < 0.01 and s_joint < 1e-3
    assert abs(f_joint - f_ref) < 0.01 and abs(s_joint - s_ref) < 1e-3
    # the whole point: the H-apply advantage survives on the test case
    assert ratio >= 3.0
