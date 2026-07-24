"""Collinear spin-polarized band structure (nspin=2 unblock).

band_structure now runs the frozen-potential Davidson once per spin channel.
Two guarantees are checked:

- ferromagnet (V↑ ≠ V↓): at an SCF-mesh k-point the per-spin band eigenvalues
  reproduce that k-point's self-consistent eigenvalues — this exercises the
  per-spin v_eff indexing, which a single channel could not.
- nonmagnetic limit (start_mag=0): both channels coincide and match the
  spin-restricted bands, and the return keeps a leading spin axis.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.bands import band_structure
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY, si_fcc, si_upf

pytestmark = pytest.mark.standard  # full spin-polarized SCF; not a fast-gate test


def test_bands_nspin2_reproduces_scf_spectrum_on_mesh():
    """Ferromagnetic fcc Ni: frozen-potential bands at the SCF mesh k-points
    reproduce the self-consistent spectrum for each spin channel."""
    torch.set_num_threads(8)
    a = 3.52
    cell = 0.5 * a * np.array([[0, 1, 1.0], [1, 0, 1], [1, 1, 0]])
    ni = parse_upf(PSEUDOS / "PD_Ni_PBE.upf")
    system = setup_system(cell, np.zeros((1, 3)), [0], [ni], ecut=45 * RY,
                          kmesh=(2, 2, 2), nbands=14, time_reversal=False)
    res = scf(system, LSDA_PW92(), smearing="gaussian", width=0.1, nspin=2,
              start_mag=[0.5], etol=1e-7, rhotol=1e-6, verbose=False, kerker=True)
    assert res.converged and res.mag_total > 0.5  # genuinely spin-split

    kfrac = np.array([sp.k_frac for sp in system.spheres], dtype=float)
    bs = band_structure(res, kfrac, nbands=14)
    assert bs.eigenvalues.shape == (2, len(kfrac), 14)

    nb = 12  # ignore the top Davidson bands (not fully converged in the solve)
    for sp in range(2):
        for ik in range(len(kfrac)):
            e_scf = np.sort(res.eigenvalues[sp, ik].cpu().numpy())[:nb]
            e_band = np.sort(bs.eigenvalues[sp, ik])[:nb]
            err = float(np.max(np.abs(e_scf - e_band)))
            assert err < 1e-3, (sp, ik, err)


def test_bands_nspin2_nonmagnetic_limit_matches_restricted():
    """nspin=2 with zero starting moment reproduces the spin-restricted bands,
    both channels identical, on a shared k-path."""
    torch.set_num_threads(4)
    cell, pos = si_fcc()
    upf = si_upf()

    def make():
        return setup_system(cell, pos, [0, 0], [upf], ecut=15 * RY,
                            kmesh=(2, 2, 2), nbands=12)

    r1 = scf(make(), LDA_PW92(), smearing="gaussian", width=0.05,
             etol=1e-9, rhotol=1e-8, verbose=False)
    r2 = scf(make(), LSDA_PW92(), smearing="gaussian", width=0.05, nspin=2,
             start_mag=[0.0, 0.0], etol=1e-9, rhotol=1e-8, verbose=False)
    assert r1.converged and r2.converged
    assert abs(r2.mag_total) < 1e-6

    kpath = np.array([[0.0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0.25, 0.25, 0.25]])
    b1 = band_structure(r1, kpath, nbands=10)
    b2 = band_structure(r2, kpath, nbands=10)
    assert b1.eigenvalues.shape == (len(kpath), 10)     # nspin=1 stays 2-D
    assert b2.eigenvalues.shape == (2, len(kpath), 10)  # nspin=2 gains spin axis
    # the two spin channels coincide in the nonmagnetic limit
    assert np.max(np.abs(b2.eigenvalues[0] - b2.eigenvalues[1])) < 1e-6
    # and match the spin-restricted bands
    assert np.max(np.abs(b2.eigenvalues[0] - b1.eigenvalues)) < 1e-4
    # the VBM reference (occupation-scale aware) agrees across the two paths
    assert abs(b2.reference - b1.reference) < 5e-3
