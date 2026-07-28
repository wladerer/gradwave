"""Regression for meta-GGA on the non-collinear band path, band_structure_nc.

band_structure_nc now rebuilds the converged tau-dependent fields (v_r at the
converged tau_up/tau_dn, plus the 2x2 generalized-KS v_tau operator fields)
from the SCF's own converged coeffs/occupations (NCResult.occupations, the
field PR #103 originally added for the SOC stress) at the SCF's own k-mesh
(system.batch), and applies core.metagga.spinor_metagga_tau_operator per path
k-chunk exactly as the SCF's SpinorHamiltonian does. At a k-point that lies in
the SCF mesh it must reproduce that k-point's self-consistent eigenvalues --
mirrors test_bands_nc.py's SOC reproduction check, now for a meta-GGA
functional (r2SCAN) on the noncollinear path.

Uses a fully-relativistic (FR) Si pseudo: band_structure_nc's frozen-potential
rebuild unconditionally calls core.spinor_proj.build_so_projectors (the j-
resolved SOC projector path), a pre-existing property of this function (its
only prior test, test_bands_nc.py, likewise uses FR pseudos) independent of
the meta-GGA gate this test targets -- a scalar-relativistic pseudo crashes on
``beta.j`` being None regardless of the XC functional. SOC itself is not the
point here (this cell has no magnetism/canting): the point is exercising the
tau machinery end to end on the only pseudopotential family band_structure_nc
currently supports.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.r2scan import SpinR2SCAN
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import band_structure_nc, scf_noncollinear
from tests.helpers import RY, pseudo, si_fcc

pytestmark = pytest.mark.slow  # full meta-GGA spinor SCF; not a fast/standard-gate test


def test_bands_nc_metagga_reproduces_scf_spectrum_on_mesh():
    torch.set_num_threads(8)
    cell, pos = si_fcc()
    upf = parse_upf(pseudo("Si_ONCV_PBE_fr.upf"))
    # Reduced ecut/kmesh to keep this internal self-consistency guard cheap --
    # it never compares to QE, so the identity holds at any cutoff (mirrors
    # test_bands_nc.py's rationale).
    system = setup_system(cell, pos, [0, 0], [upf], ecut=18 * RY,
                          kmesh=(2, 2, 1), nbands=8, use_symmetry=False,
                          time_reversal=False)
    assert system.is_fr
    xc = NoncollinearXC(SpinR2SCAN())
    res = scf_noncollinear(system, xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                           smearing="gaussian", width=0.1, etol=1e-7,
                           rhotol=1e-6, max_iter=200, verbose=False,
                           nonmagnetic=True)
    assert res.converged

    # Rebuild the spectrum at the SCF mesh k-points with the frozen-potential
    # band solver; it must reproduce the SCF eigenvalues there.
    kfrac = np.array([sp.k_frac for sp in system.spheres], dtype=float)
    eigs = band_structure_nc(res, xc, kfrac, nbands=16, diago_tol=1e-9)

    nb = 8  # occupied (4 collinear x 2 spin) spinor bands
    for ik in range(len(kfrac)):
        e_scf = np.sort(res.eigenvalues[ik].cpu().numpy())[:nb]
        e_band = np.sort(eigs[ik])[:nb]
        assert np.max(np.abs(e_scf - e_band)) < 1e-3, (
            ik, float(np.max(np.abs(e_scf - e_band))))
