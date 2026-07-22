"""Regression for the non-collinear (SOC) band path, band_structure_nc.

band_structure_nc rebuilds the converged (V, B) from the SCF density and runs a
frozen-potential spinor Davidson at arbitrary k. At a k-point that lies in the
SCF mesh it must reproduce that k-point's self-consistent eigenvalues. This
guards the SOC band-structure path (examples/bi2se3_bands_compare.py) against
silent removal, which is how band_structure_nc was lost once already.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import band_structure_nc, scf_noncollinear
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
CELL, POS = si_fcc(5.653)

pytestmark = pytest.mark.standard  # full SOC SCF; not a fast-gate test


def test_bands_nc_reproduces_scf_spectrum_on_mesh():
    torch.set_num_threads(4)
    ga = parse_upf(FIX / "pseudos" / "Ga_ONCV_PBE_FR-1.0.upf")
    as_ = parse_upf(FIX / "pseudos" / "As_ONCV_PBE_FR-1.1.upf")
    ref = json.loads((FIX / "gaas_so_ci" / "reference.json").read_text())
    system = setup_system(CELL, POS, [0, 1], [ga, as_], ecut=40 * RY,
                          kmesh=(2, 2, 2), nbands=13, fft_shape=ref["fft_dims"],
                          time_reversal=False)
    xc = NoncollinearXC(SpinPBE())
    res = scf_noncollinear(system, xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                           smearing="gaussian", width=0.1, etol=1e-7,
                           rhotol=1e-6, verbose=False, nonmagnetic=True)
    assert res.converged

    # Rebuild the spectrum at the SCF mesh k-points with the frozen-potential
    # band solver; it must reproduce the SCF eigenvalues there.
    kfrac = np.array([sp.k_frac for sp in system.spheres], dtype=float)
    eigs = band_structure_nc(res, xc, kfrac, nbands=26, diago_tol=1e-9)

    nb = 20  # occupied + low conduction spinor bands
    for ik in range(len(kfrac)):
        e_scf = np.sort(res.eigenvalues[ik].cpu().numpy())[:nb]
        e_band = np.sort(eigs[ik])[:nb]
        assert np.max(np.abs(e_scf - e_band)) < 1e-3, (
            ik, float(np.max(np.abs(e_scf - e_band))))
