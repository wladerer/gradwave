"""Convergence of the NC collinear loop with the Stoner spin preconditioner.

A small magnetic metal (bcc Fe, FM) converged with ``spin_precond`` on vs off.
The preconditioner is a residual-space operator, so both must reach the SAME
fixed point (energy and moment); the iteration count is reported and the on-run
must be no worse than the off-run. This exercises the full port on a genuinely
Stoner-expansive magnetization channel — the regime the preconditioner targets.
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


def _run(spin_precond):
    fe = parse_upf(str(PSEUDOS / "Fe_ONCV_PBE-1.2.upf"))
    system = setup_system(_CELL, _POS, [0, 0], [fe], ecut=40 * RY,
                          kmesh=(3, 3, 3), nbands=24)
    return scf(system, SpinPBE(), smearing="gaussian", width=0.1, nspin=2,
               start_mag=[0.5, 0.5], mixing_scheme="pulay", mixing_alpha=0.7,
               max_iter=100, etol=1e-9, rhotol=1e-8, verbose=False,
               spin_precond=spin_precond)


def test_nc_spin_precond_same_fixed_point_and_reports_iters():
    torch.set_num_threads(4)
    off = _run(False)
    on = _run(True)
    print(f"\nbcc Fe FM NC collinear: spin_precond OFF n_iter={off.n_iter} "
          f"(conv={off.converged})  ON n_iter={on.n_iter} (conv={on.converged})")
    print(f"  E_off={off.free_energy:.8f} eV  E_on={on.free_energy:.8f} eV  "
          f"m_off={off.mag_total:+.4f}  m_on={on.mag_total:+.4f}")
    assert off.converged and on.converged
    # same fixed point — the preconditioner only reshapes the residual
    assert abs(float(on.free_energy) - float(off.free_energy)) < 1e-5
    assert abs(float(on.mag_total) - float(off.mag_total)) < 2e-3
    # must not hurt convergence (allow one iteration of slack for tie-breaking)
    assert on.n_iter <= off.n_iter + 1
