"""A deployed learned multi-pole Kerker filter must not move the SCF fixed point.

`MultipoleKerkerPrecond` is a preconditioner: it reshapes the mixing path, never
the solution. Every term in f_θ(G²) = Σ wᵢ·G²/(G²+qᵢ²) carries a G² numerator, so
f_θ(0) = 0 and the pinned G=0 charge is untouched — the converged density, and
thus the free energy and eigenvalues, must match the bare-Kerker run to solver
precision no matter how the poles are placed. The filter algebra (f(0)=0, the
Kerker special case, the differentiable fit) is pinned in
`tests/unit/test_learned_precond.py`; this integration gate proves the property
survives an END-TO-END deploy through the real `scf` driver's `precond_op` hook,
the coverage the benchmark's iteration-count runs assume but never assert. The
iteration-count win itself lives in `benchmarks/bench_learned_precond.py`.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import RDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.learned_precond import MultipoleKerkerPrecond
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
AL_A = 4.05
AL_CELL = AL_A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.mark.standard
def test_learned_multipole_matches_kerker_fixed_point():
    """fcc Al: a genuinely multi-pole learned filter (spread weights/positions,
    NOT the Kerker special case) reaches the same converged free energy and
    eigenvalues as bare Kerker to solver precision — the fixed point is unchanged,
    only the path differs."""
    torch.set_num_threads(4)
    al = parse_upf(FIX / "pseudos" / "Al_ONCV_PBE-1.2.upf")

    def make_system():
        return setup_system(AL_CELL, np.array([[0.0, 0, 0]]), [0], [al],
                            ecut=20 * RY, kmesh=(4, 4, 4), nbands=10,
                            use_symmetry=True)

    common = dict(smearing="gaussian", width=0.3, etol=1e-9, rhotol=1e-8,
                  max_iter=80, verbose=False)

    # bare-Kerker reference
    rk = scf(make_system(), PBE(), mixing_alpha=0.7, kerker=True, **common)

    # a genuinely multi-pole filter: three well-separated poles with spread
    # weights — provably NOT the single-pole Kerker shape, so this test would
    # catch a precond_op path that silently no-ops or mishandles the filter.
    system = make_system()
    g2 = system.grid.g2.reshape(-1)[system.grid.dens_mask.reshape(-1)].to(RDTYPE)
    P = MultipoleKerkerPrecond.init_poles(g2, n_poles=3, q_min=0.1, q_max=3.0,
                                          requires_grad=False)
    # perturb the seed off the unit-weight log-spaced default so the filter is a
    # non-trivial multi-pole, not a disguised single pole
    P.w_raw = torch.tensor([0.6, 0.3, 0.1], dtype=RDTYPE)
    P = P.detach_()

    # self-oracle: the pinned G=0 charge mode is untouched by construction
    g0 = g2.abs() < 1e-14
    assert bool(g0.any())  # the density sphere carries the pinned G=0 component
    assert float(P.filter_vals()[g0].abs().max()) < 1e-14
    # and the filter is genuinely multi-scale (not the Kerker single pole)
    assert float(P.q2().sqrt().max() / P.q2().sqrt().min()) > 5.0

    rp = scf(make_system(), PBE(), mixing_alpha=0.7, precond_op=P, **common)

    assert rk.converged and rp.converged
    assert abs(float(rk.energies.free_energy)
               - float(rp.energies.free_energy)) < 1e-8
    assert torch.allclose(rk.eigenvalues, rp.eigenvalues, atol=1e-6)
