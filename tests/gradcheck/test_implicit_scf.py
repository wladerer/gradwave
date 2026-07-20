"""M4: implicit differentiation through the SCF fixed point.

A density-dependent loss L(ρ*) where ρ* is the CONVERGED SCF density: its
θ-gradient contains the full self-consistent response (χ₀, K_Hxc, adjoint
solve) — nothing cancels by stationarity. Validated against central finite
differences of complete SCF re-runs at θ ± h. This single comparison
exercises every piece of scf/implicit.py.
"""

from pathlib import Path

import torch

from gradwave.core.xc.learnable import LearnableX
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.implicit import density_loss_param_grads
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
CELL, POS = si_fcc()


def loss_fn(rho):
    # smooth, response-sensitive density functional (grid sum of ρ²)
    return (rho * rho).sum()


def run_scf(kappa, mu):
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    system = setup_system(CELL, POS, [0, 0], [upf], ecut=10 * RY, kmesh=(1, 1, 1))
    xc = LearnableX(kappa=kappa, mu=mu)
    res = scf(system, xc, smearing="none", etol=1e-12, rhotol=1e-11, verbose=False)
    assert res.converged
    return res, xc


def test_density_loss_gradient_vs_scf_finite_differences():
    torch.set_num_threads(4)
    k0, m0 = 0.70, 0.20
    res, xc = run_scf(k0, m0)
    loss, grads = density_loss_param_grads(res, xc, loss_fn)

    h = 2e-3
    fd = {}
    for raw_name, dk, dm in (("raw_kappa", h, 0.0), ("raw_mu", 0.0, h)):
        vals = []
        for sign in (+1, -1):
            xc_p = LearnableX(kappa=k0, mu=m0)
            with torch.no_grad():
                dict(xc_p.named_parameters())[raw_name].add_(
                    sign * (dk + dm)
                )
            upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
            system = setup_system(CELL, POS, [0, 0], [upf], ecut=10 * RY, kmesh=(1, 1, 1))
            r = scf(system, xc_p, smearing="none", etol=1e-12, rhotol=1e-11, verbose=False)
            assert r.converged
            vals.append(float(loss_fn(r.rho)))
        fd[raw_name] = (vals[0] - vals[1]) / (2 * (dk + dm))

    for name in ("raw_kappa", "raw_mu"):
        ag, ref = float(grads[name]), fd[name]
        assert abs(ag - ref) < 2e-4 * max(1.0, abs(ref)), (name, ag, ref)
