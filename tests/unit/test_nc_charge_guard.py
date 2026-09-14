"""Charge-conservation guard for the norm-conserving SCF drivers.

The USPP/PAW loops raise on ∫ρ ≠ N_electrons (scf/uspp_loop.py,
scf/uspp_noncollinear.py); the NC collinear loop (scf/loop.py) and the NC
spinor loop (scf/noncollinear.py) now carry the analogous guard. These tests
are an INDEPENDENT gate: they recompute ∫ρ from the converged result density
and assert it equals N_electrons to ~1e-5 (not merely relying on the internal
raise). fcc Al at a small cutoff keeps both in the fast tier.
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from tests.helpers import PSEUDOS, RY

_A = 4.05
_CELL = _A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
_POS = np.array([[0.0, 0.0, 0.0]])


def _make_al(kmesh=(2, 2, 2), **kw):
    al = parse_upf(str(PSEUDOS / "Al_ONCV_PBE-1.2.upf"))
    return setup_system(_CELL, _POS, [0], [al], ecut=12 * RY, kmesh=kmesh, **kw)


def _integrated_charge(rho: torch.Tensor, system) -> float:
    """∫ρ dr from the total real-space density [e/Å³] on the FFT grid."""
    grid = system.grid
    return float(rho.sum()) * grid.volume / grid.n_points


def test_nc_collinear_conserves_charge():
    """A small nspin=2 NC-collinear metal conserves ∫ρ = N_electrons; the
    external gate mirrors the loop's internal raise."""
    torch.set_num_threads(6)
    res = scf(_make_al(), LSDA_PW92(), smearing="gaussian", width=0.1,
              nspin=2, start_mag=[0.2], max_iter=40, etol=1e-9, rhotol=1e-8,
              verbose=False)
    assert res.converged
    n_int = _integrated_charge(res.rho, res.system)
    assert abs(n_int - res.system.n_electrons) < 1e-5, (
        n_int, res.system.n_electrons)


def test_nc_spinor_conserves_charge():
    """The spinor (noncollinear) loop conserves the CHARGE channel Tr ρ =
    ∫ρ = N_electrons. This path previously had NO charge guard at all — the
    guard added here fires on violation, and this test is the independent
    gate on a converged run."""
    torch.set_num_threads(6)
    res = scf_noncollinear(_make_al(time_reversal=False),
                           NoncollinearXC(LSDA_PW92()),
                           mag_vec_init=[[0.0, 0.0, 0.0]], nonmagnetic=True,
                           smearing="gaussian", width=0.1, max_iter=40,
                           etol=1e-9, rhotol=1e-8, verbose=False)
    assert res.converged
    n_int = _integrated_charge(res.rho, res.system)
    assert abs(n_int - res.system.n_electrons) < 1e-5, (
        n_int, res.system.n_electrons)
