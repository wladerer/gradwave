"""Differentiable resolvent Sternheimer (Phase 2): the one-refinement-step solve
carries the correct implicit-function gradient WITHOUT differentiating the eigh.

A parameter λ enters the Hamiltonian through v_eff (v = v₀ + λ·δv); the loss is
‖δψ(λ)‖². We check ∂L/∂λ from autograd through ``solve_differentiable`` against a
central finite difference of the *true* solve (``cg_sternheimer`` at the same fixed
ground state), so this validates the degenerate-safe adjoint end to end.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.batch import BatchedHamiltonian, projectors_b
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf._response import (
    ResolventSternheimer,
    cg_sternheimer,
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
    s = setup_system(cell, pos, [0, 0], [si], ecut=16 * RY, kmesh=kmesh,
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


def test_resolvent_differentiable_matches_fd():
    res = _si_res()
    bk = res.system.batch
    shape = res.system.grid.shape
    projs = projectors_b(bk, res.system.positions).detach()
    v0 = res.v_eff.detach()
    c_occ, eps_occ = _occ_batched(res, bk)

    def p_c(x):
        ov = torch.einsum("kng,kbg->kbn", c_occ.conj(), x)
        return x - torch.einsum("kbn,kng->kbg", ov, c_occ)

    torch.manual_seed(0)
    dv = torch.randn_like(v0)
    rhs = torch.randn(*c_occ.shape, dtype=CDTYPE) * bk.mask[:, None, :]
    rhs = p_c(rhs)  # conduction space

    # factorize the fixed ground-state H once (at λ = 0)
    rsolve = ResolventSternheimer(BatchedHamiltonian(bk, shape, v0, projs),
                                  bk, c_occ, eps_occ)

    # value: at λ=0 the differentiable solve equals the CG solve
    shift = sternheimer_shift(res.eigenvalues)
    dpsi_cg = cg_sternheimer(BatchedHamiltonian(bk, shape, v0, projs), bk, c_occ,
                             eps_occ, rhs, torch.zeros_like(rhs), shift, tol=1e-10)
    with torch.no_grad():
        dpsi0 = rsolve.solve_differentiable(rhs, BatchedHamiltonian(bk, shape, v0, projs),
                                            eps_occ, c_occ)
    assert float((dpsi0 - dpsi_cg).abs().max() / dpsi_cg.abs().max()) < 1e-6

    # autograd ∂‖δψ‖²/∂λ through the differentiable solve
    lam = torch.zeros((), dtype=RDTYPE, requires_grad=True)
    h_lam = BatchedHamiltonian(bk, shape, v0 + lam * dv, projs)
    dpsi = rsolve.solve_differentiable(rhs, h_lam, eps_occ, c_occ)
    loss = (dpsi.conj() * dpsi).real.sum()
    (g,) = torch.autograd.grad(loss, lam)

    # central FD of the TRUE solve at the same fixed ground state
    def loss_true(pv):
        hp = BatchedHamiltonian(bk, shape, v0 + pv * dv, projs)
        dp = cg_sternheimer(hp, bk, c_occ, eps_occ, rhs, torch.zeros_like(rhs),
                            shift, tol=1e-10)
        return float((dp.conj() * dp).real.sum())

    eps = 1e-4
    fd = (loss_true(eps) - loss_true(-eps)) / (2 * eps)
    rel = abs(float(g) - fd) / max(abs(fd), 1e-10)
    assert rel < 2e-3, f"autograd={float(g):.6e} fd={fd:.6e} rel={rel:.2e}"
