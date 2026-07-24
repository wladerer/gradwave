"""M4 NLCC regression: implicit differentiation through the SCF fixed point on a
pseudopotential WITH a non-linear core correction (NLCC).

The SCF builds v_xc at ρ + ρ_core (the partial-core charge), so the Hxc
linear-response kernel K_Hxc = ∂v_Hxc/∂ρ — and the ∂v_xc/∂θ term of the adjoint
— must be evaluated at ρ + ρ_core too. Evaluating f_xc at the bare valence
density ``res.rho`` (the pre-fix bug) leaves the implicit-diff SCF gradient ~1%
off for NLCC systems, which silently corrupts AD-through-SCF gradients used for
inverse design and learned functionals.

This oracle reconverges the SCF on a genuinely NLCC pseudo (Si_ONCV_PBE_sr.upf,
``core_correction="T"``) and compares the analytic density-loss θ-gradient
against central finite differences of complete SCF re-runs at θ ± h. It is the
test that would have caught the bug: pre-fix the relative error is ~1e-2, post-
fix it drops to the finite-difference floor.
"""

from pathlib import Path

import pytest
import torch

from gradwave.core.xc.learnable import LearnableX
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.implicit import density_loss_param_grads
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
# NLCC pseudo (partial-core correction present): this is what makes rho_core != None.
NLCC_PSEUDO = FIX / "pseudos" / "Si_ONCV_PBE_sr.upf"
CELL, POS = si_fcc()


def loss_fn(rho):
    # smooth, response-sensitive density functional (grid sum of ρ²)
    return (rho * rho).sum()


def _setup():
    upf = parse_upf(NLCC_PSEUDO)
    return setup_system(CELL, POS, [0, 0], [upf], ecut=12 * RY, kmesh=(1, 1, 1))


def run_scf(kappa, mu):
    system = _setup()
    xc = LearnableX(kappa=kappa, mu=mu)
    res = scf(system, xc, smearing="none", etol=1e-12, rhotol=1e-11, verbose=False)
    assert res.converged
    # NLCC must actually be active, else this test proves nothing.
    assert res.system.rho_core is not None
    assert float(res.system.rho_core.abs().max()) > 0.0
    return res, xc


@pytest.mark.slow
def test_nlcc_density_loss_gradient_vs_scf_finite_differences():
    """Analytic implicit-diff gradient == central-FD of the converged-SCF loss,
    on an NLCC pseudo. Fails on origin/main (f_xc at bare ρ), passes with the
    ρ + ρ_core fix."""
    torch.set_num_threads(4)
    k0, m0 = 0.70, 0.20
    res, xc = run_scf(k0, m0)
    _loss, grads = density_loss_param_grads(res, xc, loss_fn)

    h = 2e-3
    fd = {}
    for raw_name, dk, dm in (("raw_kappa", h, 0.0), ("raw_mu", 0.0, h)):
        vals = []
        for sign in (+1, -1):
            xc_p = LearnableX(kappa=k0, mu=m0)
            with torch.no_grad():
                dict(xc_p.named_parameters())[raw_name].add_(sign * (dk + dm))
            system = _setup()
            r = scf(system, xc_p, smearing="none", etol=1e-12, rhotol=1e-11,
                    verbose=False)
            assert r.converged
            vals.append(float(loss_fn(r.rho)))
        fd[raw_name] = (vals[0] - vals[1]) / (2 * (dk + dm))

    for name in ("raw_kappa", "raw_mu"):
        ag, ref = float(grads[name]), fd[name]
        rel = abs(ag - ref) / max(1.0, abs(ref))
        assert rel < 2e-4, (name, ag, ref, rel)
