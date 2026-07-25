"""Fixed-basis stress for the fully-relativistic (spin-orbit) spinor path.

The stress path now threads the spinor state through: kinetic sums both
spinor components on the strained (k+G)², Hartree/local act on the total
spinor charge density, the non-collinear XC uses (ρ, m⃗) (both scaling as
1/detJ under strain, the GGA gradient picking up the strained G), and the
nonlocal term rebuilds the j-resolved spinor projectors
(core.spinor_proj.strained_so_projector_cols) on the strain graph, contracting
the DOUBLED-axis overlap through the same D matrix the SCF uses — mirroring
scf.noncollinear's build_so_projectors / nonlocal_energy assembly.

Self-oracle (the mandated stress-vs-FD check, per docs/verification.md): on a
sheared, low-symmetry GaAs cell (the SO-splitting fixture of test_soc; SOC lives
in the As/Ga p projectors, so the j-resolved nonlocal term is genuinely
exercised), (a) the ε=0 strained expression reproduces the NC-SCF total energy,
and (b) the analytic stress (autograd in ε) equals a central finite difference
of that same strained energy w.r.t. strain. The FD step is 1e-6; analytic and
central-difference agree to <1e-5 eV/Å³ on every component (the probe run gives
~1e-8), well inside the mandated ~1e-4–1e-5 relative bar.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.stress import _energy_strained, stress
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from tests.helpers import RY, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"

# GaAs zincblende, lightly sheared + atom displaced off the ideal site so the
# stress tensor is fully anisotropic (every component exercised).
_CELL0, _ = si_fcc(5.653)
_SHEAR = np.array([[1.00, 0.02, 0.015], [0.0, 0.995, 0.01], [0.0, 0.0, 1.01]])
GAAS_CELL = _CELL0 @ _SHEAR.T
GAAS_FRAC = np.array([[0.0, 0.0, 0.0], [0.26, 0.24, 0.25]])


@pytest.mark.slow
def test_stress_soc_autograd_vs_fd():
    """Fully-relativistic (SOC) GaAs: the ε=0 strained expression reproduces the
    NC-SCF total energy, and the analytic spinor stress matches a central finite
    difference of that expression w.r.t. strain — the check that exercises the
    j-resolved nonlocal projector-strain term and the both-spinor kinetic sum,
    with the non-collinear (ρ, m⃗) GGA XC strain derivative."""
    torch.set_num_threads(8)
    ga = parse_upf(FIX / "pseudos" / "Ga_ONCV_PBE_FR-1.0.upf")
    as_ = parse_upf(FIX / "pseudos" / "As_ONCV_PBE_FR-1.1.upf")
    pos = GAAS_FRAC @ GAAS_CELL
    system = setup_system(GAAS_CELL, pos, [0, 1], [ga, as_], ecut=30 * RY,
                          kmesh=(2, 2, 2), use_symmetry=False, time_reversal=False)
    xc = NoncollinearXC(SpinPBE())
    res = scf_noncollinear(system, xc, mag_vec_init=[[0, 0, 0], [0, 0, 0]],
                           smearing="gaussian", width=0.1, etol=1e-7,
                           rhotol=1e-6, max_iter=250, verbose=False)
    assert res.converged and res.system.is_fr

    # the ε=0 spinor expression must reproduce the NC-SCF total energy
    e0 = _energy_strained(res, xc, torch.zeros(3, 3, dtype=torch.float64))
    assert abs(float(e0) - float(res.energies.total)) < 5e-6, (
        float(e0), float(res.energies.total))

    sig = stress(res, xc, symmetrize=False).cpu().numpy()
    d = 1e-6

    def fd_component(i, j):
        ep = torch.zeros(3, 3, dtype=torch.float64)
        ep[i, j] = d
        return (float(_energy_strained(res, xc, ep))
                - float(_energy_strained(res, xc, -ep))) / (2 * d)

    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]:
        fd_sym = 0.5 * (fd_component(i, j) + fd_component(j, i)) / system.grid.volume
        assert abs(sig[i, j] - fd_sym) < 1e-5, (i, j, sig[i, j], fd_sym)
