"""Convergence of the NC collinear loop with the Stoner spin preconditioner.

A small magnetic metal (bcc Fe, FM) converged with ``spin_precond`` on vs off,
on both the default nspin=2 scheme (johnson) and pulay. The preconditioner is a
residual-space operator, so on/off MUST reach the same fixed point (energy and
moment) — that is the invariant this test guards. The iteration counts are
reported (not asserted as an improvement): the effect is scheme/system
dependent and can be a wash, which is a legitimate outcome for a preconditioner
that is near-identity outside its target regime.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.standard  # full SCF to convergence

_A = 2.87
_CELL = _A * np.eye(3)
_POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ _CELL


def _run(spin_precond, scheme):
    fe = parse_upf(str(PSEUDOS / "Fe_ONCV_PBE-1.2.upf"))
    system = setup_system(_CELL, _POS, [0, 0], [fe], ecut=40 * RY,
                          kmesh=(3, 3, 3), nbands=24)
    return scf(system, SpinPBE(), smearing="gaussian", width=0.1, nspin=2,
               start_mag=[0.5, 0.5], mixing_scheme=scheme, mixing_alpha=0.7,
               max_iter=120, etol=1e-9, rhotol=1e-8, verbose=False,
               spin_precond=spin_precond)


@pytest.mark.parametrize("scheme", ["johnson", "pulay"])
def test_nc_spin_precond_same_fixed_point(scheme):
    torch.set_num_threads(4)
    off = _run(False, scheme)
    on = _run(True, scheme)
    e_off = float(off.energies.free_energy)
    e_on = float(on.energies.free_energy)
    print(f"\nbcc Fe FM NC collinear [{scheme}]: spin_precond "
          f"OFF n_iter={off.n_iter} (conv={off.converged})  "
          f"ON n_iter={on.n_iter} (conv={on.converged})")
    print(f"  E_off={e_off:.8f} eV  E_on={e_on:.8f} eV  dE={e_on - e_off:+.2e}  "
          f"m_off={off.mag_total:+.4f}  m_on={on.mag_total:+.4f}")
    assert off.converged and on.converged
    # the invariant: a residual preconditioner does not move the fixed point
    assert abs(e_on - e_off) < 1e-5
    assert abs(float(on.mag_total) - float(off.mag_total)) < 2e-3
    # sanity guard against gross divergence (NOT an improvement claim)
    assert on.n_iter <= off.n_iter + 8
