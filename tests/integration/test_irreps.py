"""Band irrep labels vs literature: graphene (P6/mmm, off-standard origin —
exercises the character gauge) at Γ and K.

Known assignments: Γ occupied = A1g (σ_s), A2u (π), E2g (σ doublet);
K: the Dirac pair at E_F is the 2D E″ irrep of D3h — that IS the symmetry
protection of the crossing.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.irreps import band_irreps
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

RY = 13.605693122994
A = 2.46


@pytest.fixture(scope="module")
def graphene():
    torch.set_num_threads(4)
    cell = np.array([[A, 0, 0], [-A / 2, A * np.sqrt(3) / 2, 0], [0, 0, 12.0]])
    frac = np.array([[0, 0, 0.5], [1 / 3, 2 / 3, 0.5]])
    c = parse_upf("tests/fixtures/qe/pseudos/C_ONCV_PBE-1.2.upf")
    system = setup_system(cell, frac @ cell, [0, 0], [c], ecut=20 * RY,
                          kmesh=(6, 6, 1), nbands=10, use_symmetry=True)
    res = scf(system, PBE(), smearing="gaussian", width=0.05,
              etol=1e-8, rhotol=1e-7, verbose=False)
    assert res.converged
    return res


def test_gamma_labels(graphene):
    out = band_irreps(graphene, [0, 0, 0], nbands=4)
    labels = [c.label for c in out.clusters]
    dims = [c.dim for c in out.clusters]
    assert labels[:3] == ["A1g", "A2u", "E2g"], labels
    assert dims[:3] == [1, 1, 2]
    assert all(not c.warning for c in out.clusters[:3])


def test_k_dirac_pair_is_e_doubleprime(graphene):
    # cluster_tol generous enough to fuse the σ E' doublet, whose grid-level
    # splitting at these cheap settings is a few meV
    out = band_irreps(graphene, [1 / 3, 1 / 3, 0], nbands=6, cluster_tol=8e-3)
    # the cluster at E_F must be the 2D E'' irrep, exactly degenerate
    ef = graphene.fermi
    dirac = min(out.clusters, key=lambda c: abs(np.mean(c.energies) - ef))
    assert dirac.dim == 2
    assert dirac.label == "E''", dirac.label
    # grid-level splitting at these cheap CI settings (~0.3 meV); the
    # production 40 Ry run gives 1e-11 eV
    assert abs(dirac.energies[1] - dirac.energies[0]) < 1e-3
    # σ states below: E' doublet then A1'
    labels = [c.label for c in out.clusters]
    assert "E'" in labels and "A1'" in labels
