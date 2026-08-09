"""Norm-conserving inexact-Newton SCF finisher (scf.newton_nc, wired into
scf/loop.py via ``newton_finish=``).

The finisher swaps the linear density mixer for the exact-Jacobian Newton step
``(I − χ₀ K_Hxc) δ = r`` once the residual falls below ``newton_switch``. It is
a root-finding-class change, not a new preconditioner, so the invariant is that
it reaches the SAME fixed point as the mixer (exact at convergence) — that is
what this test pins, alongside the iteration-count reduction that motivates it.

fcc Al (a nearly-free-electron metal with genuine fractional occupations) at a
small mesh, cold smearing — the metal path of ``apply_chi0`` (Fermi-surface
window + charge-conserving δμ term) is exercised.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard  # two full metal SCFs

_FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _al_system(use_symmetry=False):
    upf = parse_upf(str(PSEUDOS / "Al_ONCV_PBE-1.2.upf"))
    cell = 4.05 / 2.0 * _FCC
    return setup_system(cell, np.zeros((1, 3)), [0], [upf], ecut=20 * RY,
                        kmesh=(4, 4, 4), use_symmetry=use_symmetry)


def _run(system, newton):
    return scf(system, PBE(), smearing="cold", width=0.2, max_iter=100,
               etol=1e-11, rhotol=1e-9, verbose=False, newton_finish=newton,
               newton_switch=1e-3, newton_kwargs=dict(inner_tol=1e-6, max_inner=60))


def test_newton_finish_same_fixed_point_and_fewer_iters():
    torch.set_num_threads(4)
    system = _al_system()
    mix = _run(system, newton=False)
    new = _run(system, newton=True)
    e_mix = float(mix.energies.free_energy)
    e_new = float(new.energies.free_energy)
    print(f"\nfcc Al [4x4x4/20Ry cold 0.2]: mixer n_iter={mix.n_iter} "
          f"(conv={mix.converged})  newton n_iter={new.n_iter} "
          f"(conv={new.converged})  dE={e_new - e_mix:+.2e} eV")
    assert mix.converged and new.converged
    # exact at convergence: same fixed point regardless of root-finder
    assert abs(e_new - e_mix) < 1e-8, abs(e_new - e_mix)
    # the whole point: the finisher does not cost extra iterations, and cuts them
    assert new.n_iter <= mix.n_iter


def test_newton_finish_requires_no_symmetry():
    torch.set_num_threads(4)
    system = _al_system(use_symmetry=True)
    with pytest.raises(ValueError, match="use_symmetry=False"):
        scf(system, PBE(), smearing="cold", width=0.2, max_iter=5,
            verbose=False, newton_finish=True)
