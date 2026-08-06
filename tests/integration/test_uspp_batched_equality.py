"""Batched USPP path == per-k reference path, end to end.

Same displaced-Si PAW SCF through scf_uspp(batched=True) and (batched=False):
free energy, eigenvalues, and forces must agree to solver precision, both
unpolarized and spin-polarized. Guards the three batched-solver subtleties
found during the port: the indefinite-S/cond>1e14 drop-oldest guard, the
positive (kinetic) TPA preconditioner scale, and unit-normalizing expansion
rows so tiny near-converged residuals are not replaced by rank-safety jitter.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.paw_forces import forces_uspp
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS_DISP = np.array([[0.0, 0.0, 0.0], [1.4075, 1.3175, 1.3775]])


# The nspin=1 route-equivalence (batched == per-k) is the every-push guard on
# the batched solver and stays standard-tier. The nspin=2 case adds the
# spin-resolved batched path but costs ~4x (a full magnetic PAW SCF, twice) and
# the PAW pseudo is at its convergence floor here — ecut can't be lowered
# without the SCF failing to converge — so it runs nightly (slow), not on every
# standard CI shard where it was the second-slowest test.
@pytest.mark.parametrize("nspin", [
    pytest.param(1, marks=pytest.mark.standard),
    pytest.param(2, marks=pytest.mark.slow),
])
def test_batched_equals_per_k(nspin):
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    xc = PBE() if nspin == 1 else SpinPBE()
    kw = dict(nspin=nspin)
    if nspin == 2:
        kw.update(start_mag=[0.2], smearing="gaussian", width=0.05)

    res = {}
    for batched in (False, True):
        s = setup_uspp(SI_CELL, SI_POS_DISP, [0, 0], [paw], ecut=15 * RY,
                       kmesh=(2, 2, 1), ecutrho=60 * RY)
        r = scf_uspp(s, xc, etol=1e-10, rhotol=1e-9, verbose=False,
                     max_iter=60, batched=batched, **kw)
        assert r["converged"], f"batched={batched} did not converge"
        res[batched] = r

    dF = abs(float(res[True]["energies"].free_energy)
             - float(res[False]["energies"].free_energy))
    assert dF < 1e-9, f"free energy split {dF:.3e} eV"
    de = float((res[True]["eigenvalues"] - res[False]["eigenvalues"])
               .abs().max())
    assert de < 1e-7, f"eigenvalues split {de:.3e} eV"
    f1 = forces_uspp(res[False], xc).numpy()
    f2 = forces_uspp(res[True], xc).numpy()
    assert np.abs(f2 - f1).max() < 1e-8, np.abs(f2 - f1).max()
