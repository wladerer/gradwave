"""Metamorphic relations for the PW stack — ported from the FLAPW suite that caught a
spontaneous-symmetry-breaking bug on its first execution (tests/integration/
test_flapw_metamorphic.py). Each relation is an invariance the code must satisfy with NO
reference data; violations localize silent implementation errors that reference comparisons
miss. The PW stack already guards some paths (IBZ rebuild regression); these fill the gaps.
"""

import numpy as np
import pytest
from ase import Atoms

from gradwave.calculator import GradWave
from tests.helpers import RY, SI_ONCV, pseudo

A0 = 5.43


def _si(shift=None):
    cell = A0 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = A0 / 4 * np.array([[0.0, 0, 0], [1, 1, 1]])
    if shift is not None:
        pos = pos + np.asarray(shift, dtype=float)[None, :]
    return Atoms("Si2", positions=pos, cell=cell, pbc=True)


def _calc(**extra):
    kw = dict(ecut=12 * RY, xc="lda", kpts=(2, 2, 2), use_symmetry=True,
              etol=1e-9, rhotol=1e-8, max_iter=120)
    kw.update(extra)
    return GradWave(pseudopotentials={"Si": pseudo(SI_ONCV)}, **kw)


@pytest.mark.standard
def test_translation_invariance():
    """MR: rigidly translating every atom by an arbitrary (incommensurate) vector changes
    nothing physical — energies and force magnitudes must match. Positions enter only through
    structure-factor phases; any absolute-position leak (grid pinning, symmetrizer origin,
    Ewald reference) breaks this."""
    a = _si()
    a.calc = _calc()
    e0 = a.get_potential_energy()
    f0 = a.get_forces()
    b = _si(shift=[0.4321, -0.117, 0.2013])
    b.calc = _calc()
    e1 = b.get_potential_energy()
    f1 = b.get_forces()
    assert abs(e0 - e1) < 5e-5
    assert np.abs(f0 - f1).max() < 5e-4


@pytest.mark.standard
def test_symmetry_on_off_same_fixed_point():
    """MR: IBZ reduction + density symmetrization must reproduce the full-mesh answer (diamond
    Si: a real 48-op reduction — the FLAPW lesson is that a 2-op 'validation' where IBZ equals
    the full mesh validates nothing). Insulating Si has a single SCF fixed point, so cold
    starts are comparable."""
    a = _si()
    a.calc = _calc(use_symmetry=True)
    e_sym = a.get_potential_energy()
    b = _si()
    b.calc = _calc(use_symmetry=False)
    e_full = b.get_potential_energy()
    assert abs(e_sym - e_full) < 1e-4
