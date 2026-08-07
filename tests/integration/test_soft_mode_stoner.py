"""SoftModeDeflate: the Stoner/NiO regime — deflation rescues the magnetic
response solve.

The practical payoff. On a genuinely spin-split (nspin=2) magnetic operator whose
response has a *super-critical* (gain > 1) magnetization cluster — the documented
"NiO lesson" regime where plain damping diverges and even Anderson stalls — plain
Anderson fails to converge the response solve, while deflating the cluster
converges it exactly. This is the case the whole line is for: a solve that would
not otherwise complete.

Fixed-moment (M = 2) Si is a genuine spin-split insulator (the response path is
insulator-only, so a Stoner *metal* is out until the partial-occupation χ₀ lands);
its response already carries gain > 1 magnetization modes, and fxc-scaling (the
physical exchange-enhancement / Stoner knob) drives it deeper into the instability.
Slow tier: a magnetic SCF plus many spin-channel Sternheimer response applies.
"""

from pathlib import Path

import pytest
import torch

from gradwave.core.xc.learnable import LearnableSpinX
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from gradwave.scf.soft_mode import (
    anderson_solve,
    deflated_solve,
    screening_apply,
    soft_subspace_from_operator,
)
from tests.helpers import RY, si_fcc

pytestmark = pytest.mark.slow

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
CELL, POS = si_fcc()


def _magnetic_si():
    torch.set_num_threads(4)
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(CELL, POS, [0, 0], [upf], ecut=12 * RY, kmesh=(2, 2, 2))
    xc = LearnableSpinX(kappa=0.702, mu=0.20)
    res = scf(system, xc, smearing="none", nspin=2, tot_magnetization=2.0,
              start_mag=[0.5, 0.5], etol=1e-12, rhotol=1e-9, verbose=False,
              max_iter=200)
    assert res.converged and abs(float(res.mag_total) - 2.0) < 1e-6
    return res, xc


def _mag_fraction(q):
    """Fraction of a stacked (2,*grid) mode in the magnetization channel ↑−↓."""
    charge = float((q[0] * q[0]).sum() + (q[1] * q[1]).sum())
    return float(((q[0] - q[1]) ** 2).sum() / (2.0 * charge))


def test_deflation_rescues_the_near_critical_magnetic_response():
    res, xc = _magnetic_si()
    ref = torch.stack([res.rho_spin[0], res.rho_spin[1]])
    g = torch.Generator().manual_seed(5)
    vbar = torch.randn((2, *res.rho_spin[0].shape), dtype=res.rho.dtype, generator=g)
    vbar = vbar - vbar.mean()

    # deeper into the magnetic instability: several gain>1 modes (a real
    # near-critical magnet's response looks like this)
    m = screening_apply(res, xc, fxc_scale=1.5, chi0_tol=1e-5)
    sub = soft_subspace_from_operator(m, ref, krylov=24, n_modes=6, seed=0)

    # a super-critical MAGNETIZATION cluster (the Stoner signature)
    n_gt1 = sum(1 for v in sub.values if v.real > 1.0)
    assert n_gt1 >= 3, [v.real for v in sub.values]
    assert _mag_fraction(sub.vectors[0]) > 0.5, _mag_fraction(sub.vectors[0])

    # plain Anderson cannot converge the response solve here...
    base = anderson_solve(m, vbar, tol=1e-6, max_iter=40, history=8)
    assert not base.converged, base

    # ...deflating the cluster does, exactly. Two depths land on the same
    # solution (correctness without a converged baseline to compare against).
    d4 = deflated_solve(m, vbar, sub.vectors[:4], method="post", tol=1e-6,
                        max_iter=80, history=8)
    d6 = deflated_solve(m, vbar, sub.vectors[:6], method="post", tol=1e-6,
                        max_iter=80, history=8)
    assert d4.converged and d6.converged, (d4, d6)
    assert d4.residual < 1e-6 and d6.residual < 1e-6
    assert d6.n_iter < d4.n_iter  # capturing more of the cluster helps
    agree = float(torch.linalg.vector_norm(d4.u - d6.u)
                  / torch.linalg.vector_norm(d6.u))
    assert agree < 1e-4, agree
