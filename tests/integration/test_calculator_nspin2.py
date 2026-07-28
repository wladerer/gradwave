"""Self-oracle for the ASE calculator's collinear-spin (nspin=2) path.

The GradWave calculator gained an nspin=2 route: atoms carrying nonzero
initial magnetic moments switch the underlying norm-conserving SCF to two
spin channels (the ASE μ_B moments become scf's per-atom start_mag fractions
m = μ/Z_valence, and their sum fixes the spin moment for the default
no-smearing / fixed-spin-moment run). The SCF, forces, and stress spin paths
are each validated elsewhere (test_spin_vs_qe, test_forces_nspin2,
test_stress_nspin2); this test proves the *calculator plumbing* is faithful.

Oracle: build the same ferromagnetic O2 triplet two ways — through the ASE
GradWave calculator (set_initial_magnetic_moments) and through a direct scf()
call that mirrors the calculator's setup and defaults — and assert the total
energy and the converged spin moment agree. Both routes hit the identical
scf() underneath, so the energies are bit-identical (tol 1e-8 eV is slack by
orders of magnitude); the point is that the ASE-side inputs thread through
unchanged. A nonzero moment (~2 μB) confirms the spin channel is actually
live rather than silently collapsing to nspin=1.
"""

import numpy as np
import pytest
import torch
from ase import Atoms

from gradwave.calculator import GradWave
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo

# O2 triplet in a box: two equal moments → ferromagnetic, uniform-per-species
# (so the default use_symmetry path accepts it). Small box + modest ecut keep
# this in the fast tier; physical accuracy is irrelevant — only that the two
# routes take the same code path with the same inputs.
BOX = 5.0
BOND = 1.21
ECUT = 16 * RY
POS = np.array([[0.0, 0.0, -BOND / 2], [0.0, 0.0, BOND / 2]]) + BOX / 2


def test_calculator_nspin2_matches_direct_scf():
    torch.set_num_threads(4)
    pse = pseudo("O_ONCV_PBE-1.2.upf")

    # --- ASE calculator route (nspin auto-detected from the initial moments;
    # default smearing="none" → fixed-spin-moment with M = Σμ = 2 μB) ---
    atoms = Atoms("O2", positions=POS, cell=[BOX, BOX, BOX], pbc=True)
    atoms.set_initial_magnetic_moments([1.0, 1.0])
    atoms.calc = GradWave(ecut=ECUT, pseudopotentials={"O": pse},
                          xc="lda", kpts=(1, 1, 1))
    e_calc = atoms.get_potential_energy()
    m_calc = atoms.get_magnetic_moment()

    # --- direct scf route, mirroring the calculator's system build + defaults ---
    system = setup_system(np.diag([BOX, BOX, BOX]), POS, [0, 0], [parse_upf(pse)],
                          ecut=ECUT, kmesh=(1, 1, 1), use_symmetry=True)
    z_val = system.charges.detach().cpu().numpy().astype(float)  # per-atom valence
    start_mag = [1.0 / z_val[0], 1.0 / z_val[1]]  # μ/Z, as _resolve_spin computes
    res = scf(system, LSDA_PW92(), smearing="none", width=0.1, max_iter=100,
              etol=1e-8, rhotol=1e-7, mixing_alpha=0.7, kerker=None,
              diago_tol=1e-9, eigensolver="davidson", precond="kerker",
              nspin=2, start_mag=start_mag, tot_magnetization=2.0, verbose=False)
    assert res.converged

    # genuinely magnetic: the FSM run pins M = 2 μB, so a nonzero moment here
    # is what distinguishes the live spin channel from a collapsed nspin=1 run
    assert m_calc > 1.5, m_calc
    # same code path underneath → bit-identical (tol is generous slack)
    assert abs(e_calc - float(res.energies.free_energy)) < 1e-8
    assert abs(m_calc - float(res.mag_total)) < 1e-8
    # the calculator advertises and populates the ASE magmom result
    assert "magmom" in atoms.calc.results


def test_calculator_rejects_noncollinear_nspin():
    """The constructor still gates anything past collinear spin (nspin ∈ {1,2});
    noncollinear/SOC has no calculator route."""
    with pytest.raises(ValueError):
        GradWave(ecut=ECUT, pseudopotentials={"O": pseudo("O_ONCV_PBE-1.2.upf")},
                 nspin=4)


@pytest.mark.slow
def test_calculator_uspp_nspin2_matches_direct_scf():
    """Self-oracle for the ASE calculator's USPP/PAW collinear-spin (nspin=2)
    path (issue #147). The gate in _calculate_uspp was stale — scf_uspp,
    paw_forces.forces_uspp, and paw_stress.stress_uspp already thread nspin=2
    (validated in #150) — the calculator just never resolved the spin channel
    and passed it through. Same oracle shape as the NC test above: a
    ferromagnetic O2 triplet through the PAW ASE calculator vs a direct scf_uspp
    call mirroring the calculator's setup, energy + moment bit-for-bit.
    """
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import scf_uspp, setup_uspp

    torch.set_num_threads(8)
    pse = pseudo("O.pbe-n-kjpaw_psl.1.0.0.UPF")

    # --- ASE calculator route (nspin auto from the initial moments; a gaussian
    # smearing lets the shared Fermi level find the moment, no FSM pin) ---
    atoms = Atoms("O2", positions=POS, cell=[BOX, BOX, BOX], pbc=True)
    atoms.set_initial_magnetic_moments([1.0, 1.0])
    atoms.calc = GradWave(
        ecut=ECUT, pseudopotentials={"O": pse}, xc="lda", kpts=(1, 1, 1),
        smearing="gaussian", width=0.1, use_symmetry=False)
    e_calc = atoms.get_potential_energy()
    m_calc = atoms.get_magnetic_moment()

    # --- direct scf_uspp route mirroring the calculator's build + defaults ---
    system = setup_uspp(np.diag([BOX, BOX, BOX]), POS, [0, 0],
                        [parse_upf_paw(pse)], ecut=ECUT, kmesh=(1, 1, 1),
                        use_symmetry=False)
    z_val = system.charges.detach().cpu().numpy().astype(float)
    start_mag = [1.0 / z_val[0], 1.0 / z_val[1]]  # μ/Z, as _resolve_spin computes
    res = scf_uspp(system, LSDA_PW92(), nspin=2, start_mag=start_mag,
                   smearing="gaussian", width=0.1, max_iter=100, etol=1e-8,
                   rhotol=1e-7, diago_tol=1e-9, mixing_scheme="pulay",
                   mixing_alpha=0.7, mixing_history=None, mixing_kerker=None,
                   precond="kerker", verbose=False)
    assert res.converged

    # genuinely magnetic: a live spin channel (not collapsed to nspin=1); the
    # sign of the moment is the arbitrary spontaneous choice of the seed branch
    assert abs(m_calc) > 1.0, m_calc
    # same scf_uspp underneath → bit-identical (tol is generous slack)
    assert abs(e_calc - float(res.energies.free_energy)) < 1e-8
    assert abs(m_calc - float(res.mag_total)) < 1e-8
    assert "magmom" in atoms.calc.results
