"""Convention-free FD self-oracle for the BASE (PBE) PAW/USPP force & stress.

The base NC-PAW / USPP forces & stress (``postscf.paw_forces.forces_uspp``,
``postscf.paw_stress.stress_uspp``) are otherwise checked only against an
external Quantum-ESPRESSO fixture (``test_paw_derivatives_vs_qe.py``, both
SLOW). The convention-free self-oracle — analytic force = −dE/dR and analytic
stress = (1/Ω) dE/dε by central finite differences — was applied only to the
HARDER variants (r2SCAN forces in ``test_uspp_metagga_scf.py``; +U stress in
``test_paw_stress_hubbard.py``). This module closes that gap for the plain PBE
PAW path most users hit.

- forces: analytic ``forces_uspp`` vs a central FD of the *re-converged* SCF
  total energy w.r.t. each Cartesian displacement of an atom. This is the strong
  *independent* oracle — the FD probes ``scf_uspp``'s own energy accumulation
  (a different code path from the analytic backward), so a wrong augmentation /
  Pulay / S-orthogonality term would surface here even if it were shared or
  missed by the QE comparison. At fixed cell the plane-wave G-sphere does not
  change, so the FD is clean to FD-truncation. ``remove_net=False``: FD of a
  single atom's displacement measures the raw −dE/dr_{a,i}; net-force removal
  (an egg-box cleanup) would break that per-atom identity — see the NC note in
  ``test_forces_vs_qe.py::test_forces_autograd_vs_fd``.

- stress: analytic ``stress_uspp`` vs a central FD of ``_energy_strained_uspp``
  on the same converged state (the frozen-basis strain expression, mirroring
  ``test_stress_vs_qe.py::test_stress_autograd_vs_fd`` and the +U analogue).
  Straining the cell at fixed ecut re-selects the G-sphere and makes a
  re-converged-SCF strain FD basis-discontinuous, so the frozen-basis strain
  expression is the correct convention-free stress oracle here.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"

# Rattled 2-atom Si kjpaw cell (off the ideal site so every force/stress
# component is genuinely nonzero); coarse ecut/k — the derivative identities
# hold at any basis/grid. Slow tier: the force oracle needs 7 tightly-converged
# (etol 1e-11) USPP/PAW SCFs, like the r2SCAN/+U FD analogues.
SI_CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
SI_POS = np.array([[0.0, 0.0, 0.0], [1.42, 1.30, 1.38]])


def _paw():
    return parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")


@pytest.mark.slow
def test_paw_forces_match_fd_energy():
    """Base PBE PAW forces vs central FD of the re-converged SCF total energy,
    all 3 Cartesian components of the displaced atom."""
    from gradwave.postscf.paw_forces import forces_uspp

    torch.set_num_threads(6)
    paw = _paw()
    xc = PBE()

    def scf_at(p):
        s = setup_uspp(SI_CELL, p, [0, 0], [paw], ecut=20 * RY, kmesh=(1, 1, 1),
                       ecutrho=80 * RY)
        r = scf_uspp(s, xc, smearing="none", etol=1e-11, rhotol=1e-10,
                     verbose=False, max_iter=100)
        assert r["converged"]
        return r

    res = scf_at(SI_POS)
    f = forces_uspp(res, xc, remove_net=False).cpu().numpy()  # (2, 3) eV/Å

    h = 1e-3
    worst = 0.0
    for d in range(3):
        vals = []
        for sign in (+1, -1):
            p = SI_POS.copy()
            p[1, d] += sign * h
            vals.append(float(scf_at(p)["energies"].total))
        fd = -(vals[0] - vals[1]) / (2 * h)
        err = abs(fd - float(f[1, d]))
        worst = max(worst, err)
        assert err < 1e-5, (d, f[1, d], fd, err)
    print(f"\nmax |analytic − FD| PAW force = {worst:.2e} eV/Å")


@pytest.mark.slow
def test_paw_stress_match_fd_energy():
    """Base PBE PAW stress vs central FD of the frozen-basis strained energy
    (the convention-free σ = (1/Ω) dE/dε oracle), a spread of components."""
    from gradwave.postscf.paw_stress import _energy_strained_uspp, stress_uspp

    torch.set_num_threads(6)
    paw = _paw()
    xc = PBE()
    s = setup_uspp(SI_CELL, SI_POS, [0, 0], [paw], ecut=20 * RY, kmesh=(1, 1, 1),
                   ecutrho=80 * RY, use_symmetry=False)
    res = scf_uspp(s, xc, smearing="none", etol=1e-11, rhotol=1e-10,
                   verbose=False, max_iter=100)
    assert res["converged"]

    sig = stress_uspp(res, xc, symmetrize=False).cpu().numpy()  # eV/Å³
    omega = res.system.grid.volume
    d = 1e-6

    def e_strained(ep):
        return float(_energy_strained_uspp(res, xc, ep))

    def fd_component(i, j):
        ep = torch.zeros(3, 3, dtype=torch.float64)
        ep[i, j] = d
        return (e_strained(ep) - e_strained(-ep)) / (2 * d)

    worst = 0.0
    for i, j in [(0, 0), (1, 1), (0, 1), (2, 1)]:
        fd_sym = 0.5 * (fd_component(i, j) + fd_component(j, i)) / omega
        err = abs(sig[i, j] - fd_sym)
        worst = max(worst, err)
        assert err < 1e-5, (i, j, sig[i, j], fd_sym, err)
    print(f"\nmax |analytic − FD| PAW stress = {worst:.2e} eV/Å³")
