"""SoftModeDeflate on the USPP/PAW (and +U) composite response operator.

The formalism-neutral deflation core (``solvers.deflation``) is wired to the
USPP/PAW composite screening operator ``M = K χ̃`` (``postscf.uspp_softmode``),
whose ``K = diag(K_Hxc, H_1c, K_U)`` already carries the Hartree+f_xc grid kernel,
the PAW one-center Hessian, AND the Dudarev +U second derivative — so a +U ground
state's screening operator includes its +U term with no extra kernel.

Validated on real converged USPP/PAW results (the composite χ̃ itself is
FD-validated in ``test_uspp_implicit``; here we validate the deflation wiring on
top of it):

- the composite operator is a well-defined ``LinOp`` (deterministic, right size),
- the soft subspace extracts real Ritz pairs on the composite vector,
- baseline Anderson and the deflated solve converge to the SAME composite
  solution — deflation is correct on the composite operator,
- with +U the composite vector gains the occupation-matrix block and that block
  genuinely participates in ``M`` (the +U kernel is live).

Slow tier: converged USPP/PAW SCFs plus composite Sternheimer response applies.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.learnable import LearnableX
from gradwave.postscf.uspp_softmode import (
    build_uspp_screening,
)
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_hubbard import HubbardManifold
from gradwave.solvers.deflation import (
    anderson_solve,
    deflated_solve,
    soft_subspace_from_operator,
)
from tests.helpers import RY

pytestmark = pytest.mark.slow

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
SI_CELL = 5.43 / 2.0 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _si_paw(hubbard=None):
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    pos = np.array([[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]])
    s = setup_uspp(SI_CELL, pos, [0, 0], [paw], ecut=15 * RY,
                   kmesh=(2, 2, 2), ecutrho=60 * RY)
    r = scf_uspp(s, LearnableX(), etol=1e-12, rhotol=1e-10, verbose=False,
                 max_iter=80, hubbard=hubbard)
    assert r["converged"]
    return r


def _rand_composite(sc, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(sc.ref.shape, dtype=sc.ref.dtype, generator=g)
    return v - v.mean()


def test_uspp_composite_operator_and_deflation():
    """PAW Si insulator: the composite operator is well-defined, the soft
    subspace extracts, and deflated vs baseline agree on the composite solve."""
    torch.set_num_threads(4)
    res = _si_paw()
    xc = LearnableX()
    sc = build_uspp_screening(res, xc, chi0_tol=1e-7)

    # a well-defined LinOp: deterministic in its input
    u = _rand_composite(sc, 1)
    mu_a, mu_b = sc.apply(u), sc.apply(u)
    assert torch.allclose(mu_a, mu_b, atol=1e-9)

    # soft subspace: real Ritz pairs on the composite vector, benign (< 1)
    sub = soft_subspace_from_operator(sc.apply, sc.ref, krylov=24, n_modes=3, seed=0)
    assert len(sub.values) >= 1
    assert all(v.real < 1.0 for v in sub.values), [v.real for v in sub.values]
    assert sub.max_imag < 1e-2, sub.max_imag

    # deflation correctness: baseline and deflated land on the SAME composite u
    g = torch.Generator().manual_seed(5)
    vg = torch.randn(res["rho"].shape, dtype=res["rho"].dtype, generator=g)
    vbar = sc.pack(vg - vg.mean())
    base = anderson_solve(sc.apply, vbar, beta=0.2, tol=1e-7, max_iter=200)
    defl = deflated_solve(sc.apply, vbar, sub.vectors, method="post", beta=0.2,
                          tol=1e-7, max_iter=200)
    assert base.converged and defl.converged, (base, defl)
    rel = float(torch.linalg.vector_norm(base.u - defl.u)
                / torch.linalg.vector_norm(defl.u))
    assert rel < 1e-4, rel


def test_uspp_plus_u_block_is_live_and_deflates():
    """PAW Si + U: the composite vector gains the +U occupation block, that block
    genuinely participates in M (the Dudarev kernel is live), and the deflated
    solve on the +U composite operator matches the baseline."""
    torch.set_num_threads(4)
    res = _si_paw(hubbard=[HubbardManifold(species=0, l=1, u=4.0)])
    xc = LearnableX()
    sc = build_uspp_screening(res, xc, chi0_tol=1e-7)
    sc_noU = build_uspp_screening(_si_paw(), xc, chi0_tol=1e-7)

    # the +U composite vector is strictly longer (it carries the δn block)
    assert sc.ref.numel() > sc_noU.ref.numel()

    # the +U block genuinely participates: perturb ONLY the hub block and M
    # responds there (the Dudarev k_hub kernel, plus χ̃'s δn response)
    _, _, hub0 = sc.split(torch.zeros_like(sc.ref))
    assert hub0 is not None  # +U present
    u = torch.zeros_like(sc.ref)
    g = torch.Generator().manual_seed(3)
    hub_len = sc.ref.numel() - sc_noU.ref.numel()
    u[-hub_len:] = torch.randn(hub_len, dtype=u.dtype, generator=g)
    mu = sc.apply(u)
    assert float(torch.linalg.vector_norm(mu[-hub_len:])) > 0.0

    # deflation still correct on the +U composite operator
    sub = soft_subspace_from_operator(sc.apply, sc.ref, krylov=24, n_modes=3, seed=0)
    g = torch.Generator().manual_seed(7)
    vg = torch.randn(res["rho"].shape, dtype=res["rho"].dtype, generator=g)
    vbar = sc.pack(vg - vg.mean())
    base = anderson_solve(sc.apply, vbar, beta=0.2, tol=1e-7, max_iter=200)
    defl = deflated_solve(sc.apply, vbar, sub.vectors, method="post", beta=0.2,
                          tol=1e-7, max_iter=200)
    assert base.converged and defl.converged, (base, defl)
    rel = float(torch.linalg.vector_norm(base.u - defl.u)
                / torch.linalg.vector_norm(defl.u))
    assert rel < 1e-4, rel
