"""Energy-based SCF convergence criterion (criterion="energy").

The free energy is variational, so its error is O(residual²); for smeared
metals the density residual floors at occupation noise while F settles
orders of magnitude earlier. The energy criterion converges on a settled
3-iteration F tail with only a loose residual safety, and must land on the
same fixed point as the strict density criterion."""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994


@pytest.mark.slow
def test_energy_criterion_matches_drho_fixed_point():
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Al.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 4.04
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0]])

    def run(**kw):
        s = setup_uspp(cell, pos, [0], [paw], ecut=20 * RY, kmesh=(2, 2, 2),
                       ecutrho=100 * RY, nbands=8)
        r = scf_uspp(s, PBE(), smearing="gaussian", width=0.5,
                     verbose=False, max_iter=120, **kw)
        assert r["converged"]
        return r

    r_rho = run(etol=1e-10, rhotol=1e-9, criterion="drho")
    r_e = run(etol=1e-10, criterion="energy")
    f_rho = float(r_rho["energies"].free_energy)
    f_e = float(r_e["energies"].free_energy)
    assert abs(f_rho - f_e) < 1e-6, f"criteria disagree: {abs(f_rho - f_e):.2e}"
    # the 3-iteration settled tail may cost an extra iteration on easy
    # cases; its economy is on plateaued metals where rhotol never fires
    assert r_e["n_iter"] <= r_rho["n_iter"] + 3
