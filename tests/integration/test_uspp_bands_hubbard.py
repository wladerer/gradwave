"""USPP/PAW frozen-potential bands with DFT+U (the V_U-in-the-band-Hamiltonian
unblock).

``postscf.uspp_bands.bands_uspp`` rebuilds a frozen H(k) from the converged
becsum/density and diagonalizes it per requested k. ``scf.uspp._HkS`` already
applies a Hubbard term V_U = Σ|Sφ_m⟩D_mm'⟨Sφ_m'| whenever handed
``hub_sphi``/``hub_d`` — the SCF loop's own ``_solve_bands_uspp`` already builds
that pair once per SCF k — so this module now builds the same pair at each
requested BAND k (which need not lie on the SCF mesh) from the converged
``hub_occ``/``hub_sites`` the result carries, reusing
``scf.uspp_hubbard.build_uspp_hubbard``'s per-k body (factored out as
``phi_free_at_sphere``) instead of re-deriving the S-dressed atomic-orbital
projector.

Two checks, mirroring ``test_paw_stress_hubbard.py``'s pattern (one shared
Γ-only +U SCF, ``dataclasses.replace`` to probe the U=0 limit without a second
SCF):

- self-oracle: bands at the SCF's own (Γ-only) k-point reproduce the SCF
  eigenvalues for a converged +U USPP/PAW system.
- U=0 inertness: a U=0 manifold on the SAME converged state exercises the
  whole S-dressed Hubbard band path (radial SBT, Y_lm, S-dressing) but D_U
  vanishes identically, so the bands are bit-for-bit the no-+U bands.
"""

import dataclasses

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.uspp_bands import bands_uspp
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_hubbard import HubbardManifold, hubbard_sites
from tests.helpers import PSEUDOS, RY, si_fcc

pytestmark = pytest.mark.standard  # ultrasoft SCF; not a fast-gate test

# +U on the Si p manifold (unphysical, but exercises a real D_U != 0), same
# choice as test_paw_stress_hubbard.py.
MAN = [HubbardManifold(species=0, l=1, u=3.0, j=0.0)]


@pytest.fixture(scope="module")
def si_paw_hubbard():
    """One small Γ-only USPP/PAW +U SCF, shared by both tests below."""
    torch.set_num_threads(4)
    cell, pos = si_fcc()
    paw = parse_upf_paw(PSEUDOS / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    system = setup_uspp(cell, pos, [0, 0], [paw], ecut=20 * RY,
                        kmesh=(1, 1, 1), ecutrho=80 * RY, use_symmetry=False)
    res = scf_uspp(system, PBE(), smearing="gaussian", width=0.05,
                   hubbard=MAN, etol=1e-9, rhotol=1e-8, verbose=False,
                   max_iter=60)
    assert res.converged
    assert abs(float(res.energies.hubbard)) > 1e-3  # E_U genuinely nonzero
    return res


def test_uspp_bands_hubbard_reproduces_scf_spectrum(si_paw_hubbard):
    res = si_paw_hubbard
    eig = bands_uspp(res, PBE(), [[0.0, 0.0, 0.0]], nbands=res.system.nbands)
    eig = eig.cpu().numpy()
    scf_eig = res.eigenvalues[0].detach().cpu().numpy()
    assert np.max(np.abs(eig[0] - scf_eig[:res.system.nbands])) < 5e-3


def test_uspp_bands_hubbard_u0_is_inert(si_paw_hubbard):
    """A U=0 manifold runs the whole S-dressed +U band path (same converged
    hub_occ, a fresh U=0 sites list) but contributes exactly zero D_U, so the
    bands are bit-for-bit the no-+U bands on the same converged state."""
    res = si_paw_hubbard
    kpath = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]]

    b_no_u = bands_uspp(dataclasses.replace(res, hub_sites=None), PBE(), kpath,
                        nbands=res.system.nbands)
    sites_u0 = hubbard_sites(res.system,
                             [HubbardManifold(species=0, l=1, u=0.0, j=0.0)])
    b_u0 = bands_uspp(dataclasses.replace(res, hub_sites=sites_u0), PBE(), kpath,
                      nbands=res.system.nbands)
    assert torch.equal(b_no_u, b_u0)
