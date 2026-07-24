"""Fixed spin moment (FSM) for collinear nspin=2 without smearing.

scf() gained a tot_magnetization argument: instead of finding the moment from a
shared Fermi level (which smearing needs), the per-channel electron counts are
fixed by M — N↑=(N_e+M)/2, N↓=(N_e−M)/2 — and each channel gets integer
occupations. This is QE's occupations='fixed' / tot_magnetization mode, needed
e.g. for the two-step insulator hp.x procedure.
"""

import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY, si_fcc

pytestmark = pytest.mark.standard

CELL, POS = si_fcc()


def _make():
    upf = parse_upf(PSEUDOS / "Si_ONCV_PBE-1.2.upf")
    return setup_system(CELL, POS, [0, 0], [upf], ecut=15 * RY,
                        kmesh=(2, 2, 2), nbands=12)


def test_fsm_zero_moment_matches_spin_restricted():
    """M=0 fixed-moment reproduces the nspin=1 fixed-occupation result exactly
    (both fill the same states; N↑=N↓=N_e/2)."""
    torch.set_num_threads(4)
    r1 = scf(_make(), LDA_PW92(), smearing="none", etol=1e-10, rhotol=1e-9,
             verbose=False)
    r0 = scf(_make(), LSDA_PW92(), nspin=2, tot_magnetization=0.0, smearing="none",
             etol=1e-10, rhotol=1e-9, verbose=False)
    assert r1.converged and r0.converged
    assert abs(float(r0.energies.total) - float(r1.energies.total)) < 1e-8
    assert abs(float(r0.mag_total)) < 1e-8


def test_fsm_holds_the_moment_fixed():
    """A forced M=2 run on Si converges to an excited state whose moment is held
    exactly at 2 μB (the point of FSM), above the M=0 ground state."""
    torch.set_num_threads(4)
    r0 = scf(_make(), LSDA_PW92(), nspin=2, tot_magnetization=0.0, smearing="none",
             etol=1e-9, rhotol=1e-8, verbose=False)
    r2 = scf(_make(), LSDA_PW92(), nspin=2, tot_magnetization=2.0, smearing="none",
             etol=1e-9, rhotol=1e-8, verbose=False, max_iter=150)
    assert r2.converged
    assert abs(float(r2.mag_total) - 2.0) < 1e-6  # moment fixed, not relaxed
    assert float(r2.energies.total) > float(r0.energies.total)  # excited


def test_fsm_gate_requires_tot_magnetization():
    """nspin=2 without smearing still errors unless tot_magnetization is given,
    and an out-of-range moment is rejected."""
    with pytest.raises(ValueError, match="tot_magnetization"):
        scf(_make(), LSDA_PW92(), nspin=2, smearing="none",
            max_iter=2, verbose=False)
    with pytest.raises(ValueError, match="exceeds n_electrons"):
        scf(_make(), LSDA_PW92(), nspin=2, tot_magnetization=99.0,
            smearing="none", max_iter=2, verbose=False)
