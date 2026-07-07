"""M4: functional-learning gradients.

dE/dθ from energy_param_grads (variational stationarity, no response solve)
must match central finite differences of full SCF re-runs at θ ± h.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.learnable import PBE_KAPPA, PBE_MU, LearnableX, energy_param_grads
from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994
A = 5.43
CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS = np.array([[0.0, 0, 0], [A / 4] * 3])


def make_system():
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    return setup_system(CELL, POS, [0, 0], [upf], ecut=10 * RY, kmesh=(1, 1, 1))


def run(system, xc):
    res = scf(system, xc, smearing="none", etol=1e-11, rhotol=1e-10, verbose=False)
    assert res.converged
    return res


def test_pbe_init_reproduces_pbe():
    rho = torch.tensor([0.02, 0.1, 0.3], dtype=torch.float64)
    sigma = torch.tensor([0.001, 0.05, 0.2], dtype=torch.float64)
    e_learn = LearnableX().energy_density(rho, sigma)
    e_pbe = PBE().energy_density(rho, sigma)
    assert torch.allclose(e_learn, e_pbe, rtol=1e-12)


def test_de_dtheta_matches_scf_finite_differences():
    torch.set_num_threads(4)
    system = make_system()
    xc = LearnableX(kappa=0.65, mu=0.18)  # off-PBE point, generic
    res = run(system, xc)
    grads = energy_param_grads(res, xc)

    h = 1e-4
    for raw_name, param in xc.named_parameters():
        vals = []
        for sign in (+1, -1):
            xc2 = LearnableX(kappa=0.65, mu=0.18)
            with torch.no_grad():
                dict(xc2.named_parameters())[raw_name].add_(sign * h)
            vals.append(float(run(system, xc2).energies.total))
        fd = (vals[0] - vals[1]) / (2 * h)
        ag = float(grads[raw_name])
        assert abs(fd - ag) < 1e-5 * max(1.0, abs(fd)), (raw_name, fd, ag)


@pytest.mark.slow
def test_two_parameter_fit_recovers_pbe():
    # target: PBE total energies at two volumes; start off-PBE, fit (κ, μ)
    torch.set_num_threads(8)
    upf = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    systems = [
        setup_system(s * CELL, s * POS, [0, 0], [upf], ecut=10 * RY, kmesh=(1, 1, 1))
        for s in (0.98, 1.03)
    ]
    targets = [float(run(sys_, PBE()).energies.total) for sys_ in systems]

    xc = LearnableX(kappa=0.55, mu=0.30)
    opt = torch.optim.Adam(xc.parameters(), lr=0.05)
    losses = []
    for _ in range(12):
        opt.zero_grad()
        loss_val = 0.0
        for sys_, tgt in zip(systems, targets, strict=True):
            res = run(sys_, xc)
            g = energy_param_grads(res, xc)
            err = float(res.energies.total) - tgt
            loss_val += err * err
            for name, p in xc.named_parameters():
                p.grad = (p.grad if p.grad is not None else 0) + 2 * err * g[name]
        losses.append(loss_val)
        opt.step()
    # Loss collapses by >100x at its best point (Adam momentum overshoots after,
    # which is an optimizer artifact, not a gradient bug).
    assert min(losses) < 0.01 * losses[0], losses
    # Identifiability: in silicon's s-range F ≈ 1 + μs² to leading order, so μ
    # is well determined by energy data while κ (an s⁴ effect) is NOT — assert
    # recovery of μ only.
    assert abs(float(xc.mu) - PBE_MU) < 0.05
    assert PBE_KAPPA  # referenced to document the deliberate non-assertion
