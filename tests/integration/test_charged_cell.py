"""Net-charged periodic cell (tot_charge) + Makov-Payne finite-size correction.

A charged supercell is run with ``setup_system(..., tot_charge=q)`` (n_electrons
= ΣZ − q, implicit −q jellium). Its total energy carries the spurious
image+background interaction; the Makov-Payne monopole correction removes the
leading part, so the corrected energy of a charged Na⁺ ion is far less
box-size-dependent than the raw periodic energy.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.charged import makov_payne_correction
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard


def test_tot_charge_sets_electron_count():
    na = parse_upf(PSEUDOS / "Na_ONCV_PBE_sr.upf")
    cell = 9.0 * np.eye(3)
    pos = np.array([[4.5, 4.5, 4.5]])
    z = float(na.z_valence)
    sys0 = setup_system(cell, pos, [0], [na], 30 * RY, (1, 1, 1), use_symmetry=False)
    sysp = setup_system(cell, pos, [0], [na], 30 * RY, (1, 1, 1),
                        use_symmetry=False, tot_charge=1.0)
    assert abs(sys0.n_electrons - z) < 1e-9
    assert abs(sysp.n_electrons - (z - 1.0)) < 1e-9
    with pytest.raises(ValueError, match="n_electrons"):
        setup_system(cell, pos, [0], [na], 30 * RY, (1, 1, 1), tot_charge=z + 1)


def test_makov_payne_reduces_finite_size_spread():
    """The Makov-Payne correction makes the charged-cell energy far more
    box-size-independent — the whole point of the correction."""
    torch.set_num_threads(4)
    na = parse_upf(PSEUDOS / "Na_ONCV_PBE_sr.upf")
    kw = dict(smearing="fermi-dirac", width=0.05, etol=1e-9, rhotol=1e-8,
              max_iter=400, verbose=False)

    def charged(L):
        cell = L * np.eye(3)
        pos = np.array([[0.5, 0.5, 0.5]]) * L
        res = scf(setup_system(cell, pos, [0], [na], 30 * RY, (1, 1, 1),
                               use_symmetry=False, tot_charge=1.0), PBE(), **kw)
        assert res.converged
        e_periodic = float(res.energies.free_energy)
        e_corr = e_periodic + makov_payne_correction(1.0, cell)
        return e_periodic, e_corr

    ep_a, ec_a = charged(9.0)
    ep_b, ec_b = charged(13.0)
    raw_spread = abs(ep_a - ep_b)
    corr_spread = abs(ec_a - ec_b)
    # the raw periodic energies differ by the ~0.4 eV finite-size term; the
    # correction cuts that spread by a large factor.
    assert raw_spread > 0.2
    assert corr_spread < 0.4 * raw_spread
