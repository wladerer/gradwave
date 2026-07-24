"""Hellmann–Feynman forces for collinear spin (nspin=2 unblock).

The nonlocal force term now sums over both spin channels; local (total
density) and Ewald terms are spin-agnostic. Two checks:

- nonmagnetic limit: nspin=2 forces on a displaced Si cell equal the
  spin-restricted forces (already validated vs QE + finite differences),
  so the two-channel occupation bookkeeping reconstructs the nspin=1 result.
- genuinely magnetic (V↑ ≠ V↓): finite-difference of the nspin=2 free energy
  matches the analytic nspin=2 force for a ferromagnetic 2-atom bcc-Fe cell
  with one atom displaced — the test that actually exercises the per-spin
  nonlocal sum on a genuinely spin-split system.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.spin import LSDA_PW92, SpinPBE
from gradwave.postscf.forces import forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

# Displaced Si cell (same geometry as test_forces_vs_qe: atom 1 off its site).
A = 5.43
CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS = np.array([[0.0, 0.0, 0.0], [0.24, 0.26, 0.255]]) @ CELL

pytestmark = pytest.mark.standard  # full SCF; not a fast-gate test


def test_forces_nspin2_matches_spin_restricted():
    """nspin=2 (start_mag=0) forces reproduce the spin-restricted forces on a
    displaced Si cell to SCF-convergence precision."""
    torch.set_num_threads(4)
    upf = parse_upf(PSEUDOS / "Si_ONCV_PBE-1.2.upf")

    def make():
        return setup_system(CELL, POS, [0, 0], [upf], ecut=15 * RY, kmesh=(2, 2, 2))

    r1 = scf(make(), LDA_PW92(), smearing="gaussian", width=0.05,
             etol=1e-10, rhotol=1e-9, verbose=False)
    r2 = scf(make(), LSDA_PW92(), smearing="gaussian", width=0.05, nspin=2,
             start_mag=[0.0, 0.0], etol=1e-10, rhotol=1e-9, verbose=False)
    assert r1.converged and r2.converged
    f1 = forces(r1).cpu().numpy()
    f2 = forces(r2).cpu().numpy()
    assert np.abs(f2 - f1).max() < 1e-5, f"\nnspin1:\n{f1}\nnspin2:\n{f2}"


@pytest.mark.slow
def test_forces_nspin2_finite_difference_magnetic():
    """Ferromagnetic bcc Fe (2-atom conventional cell, one atom off-site): FD of
    the nspin=2 free energy matches the analytic nspin=2 force. Fe ONCV has no
    NLCC, so the (still-gated) core-correction force term is not exercised.

    Compared with remove_net=False: FD gives the raw −dF/dτ, while the default
    mean-force removal would shift it by the semicore XC-grid egg-box term."""
    torch.set_num_threads(8)
    fe = parse_upf(PSEUDOS / "Fe_ONCV_PBE-1.2.upf")
    a = 2.87  # bcc Fe lattice constant (Å); 60 Ry needed for the magnetic state
    cell = a * np.eye(3)

    def geom(frac_dy):
        # atom 2 displaced off the body centre along y (cartesian dy = a·frac_dy)
        frac = np.array([[0.0, 0, 0], [0.5, 0.52, 0.5]])
        frac[1, 1] += frac_dy
        return frac @ cell

    def run(pos, etol, rhotol):
        system = setup_system(cell, pos, [0, 0], [fe], ecut=60 * RY,
                              kmesh=(2, 2, 2), nbands=20)
        return scf(system, SpinPBE(), smearing="gaussian", width=0.1, nspin=2,
                   start_mag=[0.4, 0.4], etol=etol, rhotol=rhotol, verbose=False)

    res = run(geom(0.0), 1e-9, 1e-8)
    assert res.converged and res.mag_total > 5.0  # ~3.6 μB/atom, genuinely FM
    f = forces(res, remove_net=False).cpu().numpy()

    h = 1e-3  # cartesian displacement (Å)
    e = []
    for sign in (+1, -1):
        r = run(geom(sign * h / a), 1e-10, 1e-9)
        assert r.converged
        e.append(float(r.energies.free_energy))  # variational quantity for smeared forces
    fd = -(e[0] - e[1]) / (2 * h)  # −dF/dy on atom 2
    assert abs(fd - float(f[1, 1])) < 5e-4, (fd, float(f[1, 1]))
