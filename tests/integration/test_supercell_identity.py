"""Supercell folding identity (Tier-1 metamorphic gate, docs/verification.md).

With the same ecut, the plane-wave basis of an N×1×1 supercell at Γ is
EXACTLY the union of the primitive-cell bases at the N folded k-points, and
with the supercell FFT grid pinned to N× the primitive grid the XC/Hartree
quadratures sample identical points. So

    E(supercell, folded k)  ==  N · E(primitive, full mesh)

is an identity at solver tolerance, not a convergence statement. One test
exercises k-weights, Fermi filling, Hartree G=0 ownership, the nonlocal
phases, and the density assembly at once — an end-to-end oracle that needs
no reference code. The geometry is rattled (P1) so nothing cancels by
symmetry, and eigenvalues/forces must fold too:

    eigs(supercell, Γ)   == sorted union of eigs(primitive, k_i)
    F(atom + n·a1)       == F(atom)
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.postscf.forces import forces as compute_forces
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


@pytest.fixture(autouse=True)
def _limit_threads():
    torch.set_num_threads(4)


def test_supercell_energy_eigs_forces_fold():
    a = 5.43
    lattice = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [1.45, 1.27, 1.41]])  # rattled, P1
    si = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    xc = LDA_PW92()
    kw = dict(etol=1e-10, rhotol=1e-9, diago_tol=1e-12, verbose=False)

    prim = setup_system(lattice, pos, [0, 0], [si], ecut=13 * RY, kmesh=(2, 1, 1))
    res_p = scf(prim, xc, **kw)
    assert res_p.converged

    lat2 = lattice.copy()
    lat2[0] *= 2.0
    pos2 = np.vstack([pos, pos + lattice[0]])
    n1, n2, n3 = prim.grid.shape
    sup = setup_system(lat2, pos2, [0, 0, 0, 0], [si], ecut=13 * RY,
                       kmesh=(1, 1, 1), fft_shape=(2 * n1, n2, n3),
                       nbands=2 * prim.nbands)
    res_s = scf(sup, xc, **kw)
    assert res_s.converged

    # energy per atom: identity at SCF/solver tolerance
    de = abs(float(res_s.energies.free_energy) - 2 * float(res_p.energies.free_energy)) / 4
    assert de < 2e-6, f"supercell energy identity broken: {de:.3e} eV/atom"

    # Γ supercell eigenvalues == union of primitive eigenvalues at folded k
    folded = np.sort(res_p.eigenvalues.numpy().ravel())
    gamma = np.sort(res_s.eigenvalues.numpy().ravel())
    assert np.abs(gamma - folded).max() < 5e-5

    # forces map rigidly: copies of an atom feel the copy's force
    f_p = compute_forces(res_p).numpy()
    f_s = compute_forces(res_s).numpy()
    assert np.abs(f_s - np.vstack([f_p, f_p])).max() < 1e-5


@pytest.mark.slow
def test_supercell_fold_paw_smeared():
    """The same folding identity through the USPP/PAW machinery with smeared
    occupations: per-copy becsums and one-center energies must replicate, the
    augmentation density must fold through the supercell phases, and the
    shared Fermi level must fill the folded spectrum identically. The dense
    (ecutrho) grid is pinned to 2x the primitive grid; the smooth dual grid
    needs no pinning — its v_eff restriction is exact in G-space."""
    from gradwave.core.xc.pbe import PBE
    from gradwave.postscf.paw_forces import forces_uspp
    from gradwave.pseudo.upf_paw import parse_upf_paw
    from gradwave.scf.uspp import scf_uspp, setup_uspp

    a = 5.43
    lattice = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [1.45, 1.27, 1.41]])  # rattled, P1
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    xc = PBE()
    kw = dict(smearing="gaussian", width=0.1, etol=1e-10, rhotol=1e-9,
              diago_tol=1e-12, verbose=False)

    prim = setup_uspp(lattice, pos, [0, 0], [paw], ecut=13 * RY, kmesh=(2, 1, 1))
    res_p = scf_uspp(prim, xc, **kw)
    assert res_p["converged"]

    lat2 = lattice.copy()
    lat2[0] *= 2.0
    pos2 = np.vstack([pos, pos + lattice[0]])
    n1, n2, n3 = prim.grid.shape
    sup = setup_uspp(lat2, pos2, [0, 0, 0, 0], [paw], ecut=13 * RY,
                     kmesh=(1, 1, 1), fft_shape=(2 * n1, n2, n3),
                     nbands=2 * prim.nbands)
    res_s = scf_uspp(sup, xc, **kw)
    assert res_s["converged"]

    de = abs(float(res_s["energies"].free_energy)
             - 2 * float(res_p["energies"].free_energy)) / 4
    assert de < 2e-6, f"PAW supercell energy identity broken: {de:.3e} eV/atom"

    folded = np.sort(res_p["eigenvalues"].numpy().ravel())
    gamma = np.sort(res_s["eigenvalues"].numpy().ravel())
    assert np.abs(gamma - folded).max() < 5e-5

    f_p = forces_uspp(res_p, xc).numpy()
    f_s = forces_uspp(res_s, xc).numpy()
    assert np.abs(f_s - np.vstack([f_p, f_p])).max() < 1e-5
