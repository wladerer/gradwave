"""Resolvent (direct sum-over-states) Sternheimer vs the iterative CG solver.

The two solve the SAME conduction-projected linear system (H − ε_n)δψ = rhs for an
insulator, so the resolvent's one-eigh back-substitution must agree with
``cg_sternheimer`` to the CG tolerance.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.batch import BatchedHamiltonian, projectors_b
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf._response import (
    cg_sternheimer,
    resolvent_sternheimer,
    sternheimer_shift,
)
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.implicit import _occupied
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

pytestmark = pytest.mark.standard
FIX = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"


def _si_res(kmesh=(2, 2, 2)):
    si = parse_upf(str(FIX / "Si_ONCV_PBE-1.2.upf"))
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    s = setup_system(cell, pos, [0, 0], [si], ecut=20 * RY, kmesh=kmesh,
                     nbands=8, use_symmetry=False, time_reversal=False)
    return scf(s, PBE(), smearing="none", etol=1e-10, rhotol=1e-9, max_iter=120, verbose=False)


def _occ_batched(res, bk):
    nk, npw = bk.mask.shape
    nocc = _occupied(res, 0, 0)[0].shape[0]
    c_occ = torch.zeros(nk, nocc, npw, dtype=CDTYPE)
    eps = torch.zeros(nk, nocc, dtype=RDTYPE)
    for k in range(nk):
        ck, ek = _occupied(res, 0, k)
        c_occ[k, :, : ck.shape[1]] = ck
        eps[k] = ek
    return c_occ, eps


@pytest.mark.parametrize("kmesh", [(2, 2, 2), (3, 3, 3)])
def test_resolvent_matches_cg(kmesh):
    res = _si_res(kmesh)
    bk = res.system.batch
    shape = res.system.grid.shape
    h = BatchedHamiltonian(bk, shape, res.v_eff.detach(),
                           projectors_b(bk, res.system.positions).detach())
    c_occ, eps_occ = _occ_batched(res, bk)

    def p_occ(x):
        ov = torch.einsum("kng,kbg->kbn", c_occ.conj(), x)
        return torch.einsum("kbn,kng->kbg", ov, c_occ)

    torch.manual_seed(0)
    rhs = torch.randn(*c_occ.shape, dtype=CDTYPE) * bk.mask[:, None, :]
    rhs = rhs - p_occ(rhs)  # into the conduction space

    shift = sternheimer_shift(res.eigenvalues)
    cg = cg_sternheimer(h, bk, c_occ, eps_occ, rhs, torch.zeros_like(rhs), shift,
                        tol=1e-9, max_iter=400)
    rv = resolvent_sternheimer(h, bk, c_occ, eps_occ, rhs)
    rel = float((rv - cg).abs().max() / cg.abs().max())
    assert rel < 1e-6, f"resolvent vs CG rel={rel:.2e}"
