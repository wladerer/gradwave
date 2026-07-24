"""D3(BJ) dispersion through the ASE ``GradWave`` calculator.

Self-oracle (docs/verification.md, Tier 0) for the calculator wiring added
alongside the post-SCF/api dispersion path (PR #56): with dispersion enabled
the calculator's reported energy, forces, and stress must carry the D3(BJ)
contribution, and that contribution must be the derivative of the calculator's
*own* total energy — checked by a central finite difference. With dispersion
off the results reproduce the pre-existing (no-dispersion) behavior.

Why the FD is differenced on/off rather than taken against the raw forces: the
dispersion force here (~6e-4 eV/Å) is *smaller* than the base SCF's real-space
grid egg-box error (the gap between the Hellmann–Feynman forces and a finite
difference of the discretized energy, ~1e-2 eV/Å at this cutoff/box). A raw
FD-vs-forces test would be swamped by that base artifact and blind to the term
this PR adds. Dispersion is a post-SCF additive correction, so at a fixed
geometry the SCF energy is bit-identical between the on and off runs; the
difference E_on − E_off is the dispersion energy alone and the egg-box cancels.
Finite-differencing that difference is a genuine FD of the calculator's total
energy that *sees* the dispersion term — the state where nothing cancels.

A cheap closed-shell CO molecule in a box (Γ-only, norm-conserving) places C
and O — both in the vendored D3 element subset — within dispersion range.
"""

import numpy as np
import pytest
import torch
from ase import Atoms

from gradwave.calculator import GradWave
from gradwave.postscf.dispersion import (
    D3Config,
    dispersion_energy,
    dispersion_forces,
    dispersion_stress,
)
from tests.helpers import RY, pseudo

# rattled, off-center CO in a cubic box: low symmetry, C and O both D3-covered,
# atoms within dispersion range (bond ~1.13 Å) plus periodic images
_POS = np.array([[3.15, 3.20, 3.20], [3.15, 3.20, 4.33]])
_BOX = 6.4
_Z = [6, 8]


def _atoms():
    return Atoms("CO", positions=_POS.copy(), cell=[_BOX] * 3, pbc=True)


def _kw(**extra):
    return dict(
        ecut=24 * RY,
        pseudopotentials={"C": pseudo("C_ONCV_PBE_sr.upf"),
                          "O": pseudo("O_ONCV_PBE-1.2.upf")},
        xc="pbe", kpts=(1, 1, 1),
        etol=1e-10, rhotol=1e-9, diago_tol=1e-11, max_iter=200,
        **extra,
    )


def _calc(**extra):
    atoms = _atoms()
    atoms.calc = GradWave(**_kw(**extra))
    return atoms


def _energy_at(pos, *, dispersion):
    """Total energy from a fresh (cold) calculator at ``pos`` — cold so the SCF
    part is deterministic and cancels in the on/off difference."""
    atoms = _atoms()
    atoms.set_positions(pos)
    atoms.calc = GradWave(**_kw(dispersion=dispersion))
    return atoms.get_potential_energy()


def _d3_reference():
    """The exact D3(BJ) energy/forces/stress the calculator should add, computed
    straight from postscf/dispersion.py on the same geometry (PBE preset)."""
    cfg = D3Config.resolve("pbe", cutoff_ang=21.2, cn_cutoff_ang=10.6)
    cell = np.array(_atoms().cell, dtype=np.float64)
    cell_t = torch.as_tensor(cell, dtype=torch.float64)
    pos_t = torch.tensor(_POS)
    e = float(dispersion_energy(pos_t, cell_t, _Z, cfg))
    f = dispersion_forces(pos_t, cell, _Z, cfg).cpu().numpy()
    sig = dispersion_stress(pos_t, cell, _Z, cfg).cpu().numpy()
    voigt = np.array([sig[0, 0], sig[1, 1], sig[2, 2],
                      sig[1, 2], sig[0, 2], sig[0, 1]])
    return e, f, voigt


@pytest.mark.slow
def test_calculator_dispersion_wiring_and_off():
    """Enabling dispersion shifts energy/forces/stress by exactly the D3(BJ)
    contribution; disabling it reproduces the no-dispersion result."""
    torch.set_num_threads(8)
    e_d3, f_d3, s_d3 = _d3_reference()

    base = _calc()
    e0 = base.get_potential_energy()
    f0 = base.get_forces()
    s0 = base.get_stress()

    # dispersion=False must not perturb the SCF-only result (identical code path;
    # the residual is only the SCF's ~1e-14 float nondeterminism, not this PR)
    off = _calc(dispersion=False)
    assert abs(off.get_potential_energy() - e0) < 1e-9
    np.testing.assert_allclose(off.get_forces(), f0, atol=1e-9)
    np.testing.assert_allclose(off.get_stress(), s0, atol=1e-9)

    on = _calc(dispersion=True)
    e1 = on.get_potential_energy()
    f1 = on.get_forces()
    s1 = on.get_stress()

    assert e_d3 != 0.0  # dispersion is actually contributing on this geometry
    assert abs((e1 - e0) - e_d3) < 1e-8
    np.testing.assert_allclose(f1 - f0, f_d3, atol=1e-8)
    np.testing.assert_allclose(s1 - s0, s_d3, atol=1e-8)


@pytest.mark.slow
def test_calculator_dispersion_forces_match_fd():
    """The dispersion contribution to the calculator's forces equals a central
    finite difference of the (on − off) calculator total energy — an FD of the
    calculator's own energy that isolates the wired-in D3(BJ) term."""
    torch.set_num_threads(8)
    df = _calc(dispersion=True).get_forces() - _calc(dispersion=False).get_forces()
    assert np.abs(df).max() > 1e-4  # dispersion force is non-negligible here

    h = 2e-3
    dfd = np.zeros_like(_POS)
    for i in range(_POS.shape[0]):
        for c in range(3):
            plus = _POS.copy()
            plus[i, c] += h
            minus = _POS.copy()
            minus[i, c] -= h
            de_plus = _energy_at(plus, dispersion=True) - _energy_at(plus, dispersion=False)
            de_minus = (_energy_at(minus, dispersion=True)
                        - _energy_at(minus, dispersion=False))
            dfd[i, c] = -(de_plus - de_minus) / (2 * h)

    assert np.abs(df - dfd).max() < 1e-6
