"""FM-Ni non-collinear + spin-orbit convergence (issue/PR #79).

Ferromagnetic fcc nickel is a weak, itinerant, near-Stoner-instability magnet:
in a spinor SCF with spin-orbit coupling the *longitudinal* magnetization mode
carries a Stoner-enhanced gain that limit-cycles a naive history mixer. On this
case the stock (``spin_precond=False``) driver pins the density residual in a
~3e-2..1e-1 band with the moment thrashing between roughly -0.4 and +5 uB and
never self-consistifies (it exhausts the iteration budget).

The opt-in Stoner spin-preconditioner on the moment channel
(``spin_precond=True``, plus the collinear "quadratic" diago schedule and a
gentler magnetization step) breaks that limit-cycle: the moment locks to Ni's
physical ~0.6 uB and the residual drops ~30-60x. This is a convergence-behaviour
regression guard, not a QE-accuracy benchmark: a loose k-mesh / modest ecut and
metal-appropriate (loose) tolerances are used deliberately so the assertion is
"did the magnetic branch escape the limit-cycle", which the stock driver cannot.
A separate, much smaller Fermi-surface residual floor (~1e-3 at this coarse
mesh) means the strict 1e-6 gate is not reached here; see PR #79.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from tests.helpers import RY

FIX = __import__("pathlib").Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"
_FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.mark.slow
def test_ni_soc_ferromagnetic_converges():
    torch.set_num_threads(4)
    a = 3.52  # fcc Ni lattice constant (Angstrom)
    cell = a / 2 * _FCC
    pos = np.array([[0.0, 0.0, 0.0]])
    ni = parse_upf(FIX / "Ni_ONCV_PBE_FR-1.0.upf")  # fully-relativistic (has_so)
    system = setup_system(cell, pos, [0], [ni], ecut=40 * RY, kmesh=(4, 4, 4),
                          nbands=16, use_symmetry=False, time_reversal=False)

    res = scf_noncollinear(
        system, NoncollinearXC(SpinPBE()),
        mag_vec_init=[[0, 0, 1.0]],          # ferromagnetic seed along z
        smearing="gaussian", width=0.1,
        max_iter=60, etol=1e-4, rhotol=5e-3,  # metal-appropriate loose gate
        spin_precond=True,                    # <-- the #79 fix
        mag_mixer="pulay", mag_diago_schedule="quadratic",
        mag_mixing_alpha=0.3,
        verbose=False,
    )

    assert res.converged, f"ni_soc did not converge in {res.n_iter} iters"
    assert res.n_iter < 45, f"converged but took {res.n_iter} iters (limit-cycle?)"
    # Ni's bulk FM moment is famously small (~0.6 uB). Assert a physically sane
    # band, sign-agnostic (the spinor moment axis is free): the moment must not
    # have collapsed and must sit mostly along the seed (z) axis.
    mz = abs(float(res.mag_vec[2]))
    assert res.mag_abs > 0.3, f"moment collapsed: |m|={res.mag_abs:.3f}"
    assert mz > 0.3, f"moment not along seed axis: m_vec={res.mag_vec}"
