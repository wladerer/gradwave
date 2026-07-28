"""M4 nspin=2 (collinear spin): implicit differentiation through the SCF fixed
point for a density-dependent loss, spin-polarized.

The nspin=2 adjoint doubles the response vector to the per-channel pair
(δρ↑, δρ↓): χ₀ is block-diagonal over spin (each channel its own occupied
bands, v_eff^σ and Sternheimer solves, degeneracy weight g = 1) while K_Hxc
keeps its cross-spin blocks (Hartree on the total δρ + f_xc^{σσ'}). Two
self-oracles:

- nonmagnetic limit: on a nonmagnetic Si insulator the spin-polarized
  gradient (LearnableSpinX started from zero moment) reproduces the
  spin-restricted (LearnableX) gradient to SCF-convergence precision. The
  per-channel f = 1 occupation bookkeeping (factor-2 spin folding) therefore
  reconstructs the nspin=1 (f = 2) result — a spin-folding factor error would
  miss by ~100%.

- genuinely magnetic (V↑ ≠ V↓): on a fixed-moment (M = 2) Si state the
  analytic nspin=2 density-loss gradient matches a central finite difference
  of the converged nspin=2 loss, exercising the two independent per-spin
  Sternheimer channels on a genuinely spin-split potential.
"""

from pathlib import Path

import pytest
import torch

from gradwave.core.xc.learnable import LearnableSpinX, LearnableX
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.implicit import density_loss_param_grads
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
CELL, POS = si_fcc()
PSEUDO = FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf"


def loss_fn(rho):
    # smooth, response-sensitive density functional (grid sum of ρ²)
    return (rho * rho).sum()


def _system():
    upf = parse_upf(PSEUDO)
    return setup_system(CELL, POS, [0, 0], [upf], ecut=12 * RY, kmesh=(2, 2, 2))


def test_nspin2_nonmagnetic_limit_matches_spin_restricted():
    """nspin=2 (start_mag=0) density-loss gradient reproduces the spin-restricted
    (nspin=1) gradient on a nonmagnetic Si insulator."""
    torch.set_num_threads(4)
    k0, m0 = 0.70, 0.20

    res1 = scf(_system(), LearnableX(kappa=k0, mu=m0), smearing="none",
               etol=1e-12, rhotol=1e-11, verbose=False)
    assert res1.converged
    _l1, g1 = density_loss_param_grads(res1, LearnableX(kappa=k0, mu=m0), loss_fn)

    res2 = scf(_system(), LearnableSpinX(kappa=k0, mu=m0), smearing="none",
               nspin=2, tot_magnetization=0.0, start_mag=[0.0, 0.0],
               etol=1e-12, rhotol=1e-11, verbose=False)
    assert res2.converged
    assert abs(float(res2.mag_total)) < 1e-6  # stayed nonmagnetic
    _l2, g2 = density_loss_param_grads(res2, LearnableSpinX(kappa=k0, mu=m0),
                                       loss_fn)

    for name in ("raw_kappa", "raw_mu"):
        a1, a2 = float(g1[name]), float(g2[name])
        assert abs(a2 - a1) < 2e-4 * max(1.0, abs(a1)), (name, a1, a2)


@pytest.mark.slow
def test_nspin2_magnetic_gradient_vs_scf_finite_differences():
    """Genuinely magnetic (fixed moment M=2, V↑≠V↓) Si: the analytic nspin=2
    density-loss gradient tracks a central finite difference of the converged
    nspin=2 loss. This exercises the two independent per-spin Sternheimer
    channels on a genuinely spin-split potential (the asymmetric case the
    nonmagnetic-limit test above cannot reach).

    The forced-moment M=2 state of Si is an EXCITED fixed-occupation
    configuration with a small (~0.03 eV) minority-channel gap at the fill
    boundary: below κ≈0.699 a band crosses that boundary and the loss(κ) is
    non-smooth, so the finite-difference is taken inside the confirmed-smooth
    window (κ₀=0.702, h=1e-3). The small gap makes χ₀ near-singular, which
    caps both the analytic (CG conduction-projection) and the finite-difference
    agreement at the few-percent level — hence the loose tolerance. The tight,
    decisive spin-folding validation is the nonmagnetic-limit test above (and,
    transitively, the USPP/PAW nspin=2 adjoint twin validated on genuinely
    magnetic O₂/NiO); this test's role is to confirm the asymmetric two-channel
    response has the right sign and magnitude on a spin-split potential."""
    torch.set_num_threads(4)
    # centre the FD in the smooth window (see docstring): κ well clear of the
    # ~0.699 band-crossing where the excited fixed-moment loss(κ) kinks.
    k0, m0, h = 0.702, 0.20, 1e-3

    def run(kappa, mu):
        r = scf(_system(), LearnableSpinX(kappa=kappa, mu=mu), smearing="none",
                nspin=2, tot_magnetization=2.0, start_mag=[0.5, 0.5],
                etol=1e-12, rhotol=1e-9, verbose=False, max_iter=200)
        assert r.converged
        return r

    res = run(k0, m0)
    assert abs(float(res.mag_total) - 2.0) < 1e-6  # moment held fixed
    _loss, grads = density_loss_param_grads(res, LearnableSpinX(kappa=k0, mu=m0),
                                            loss_fn)

    fd = {}
    for raw_name, dk, dm in (("raw_kappa", h, 0.0), ("raw_mu", 0.0, h)):
        vals = []
        for sign in (+1, -1):
            xc_p = LearnableSpinX(kappa=k0, mu=m0)
            with torch.no_grad():
                dict(xc_p.named_parameters())[raw_name].add_(sign * (dk + dm))
            r = scf(_system(), xc_p, smearing="none", nspin=2,
                    tot_magnetization=2.0, start_mag=[0.5, 0.5],
                    etol=1e-12, rhotol=1e-9, verbose=False, max_iter=200)
            assert r.converged
            vals.append(float(loss_fn(r.rho)))
        fd[raw_name] = (vals[0] - vals[1]) / (2 * (dk + dm))

    for name in ("raw_kappa", "raw_mu"):
        ag, ref = float(grads[name]), fd[name]
        # same sign and magnitude; few-percent agreement (small-gap χ₀
        # conditioning of the excited fixed-moment state, per the docstring).
        assert ag * ref > 0.0, (name, ag, ref)
        rel = abs(ag - ref) / max(1.0, abs(ref))
        assert rel < 0.1, (name, ag, ref, rel)
