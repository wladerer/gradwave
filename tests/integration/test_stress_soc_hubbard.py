"""Fixed-basis stress with DFT+U on the fully-relativistic (spin-orbit) spinor
path — the SOC generalization of test_stress_hubbard.py's collinear +U stress
unblock (docs/gate_inventory.md, issue #147).

``postscf.stress._energy_strained_fr`` already carries the SOC nonlocal term
(j-resolved spinor projectors, test_stress_soc.py) and the collinear +U stress
already threads the Dudarev strain term through the atomic-orbital projectors
(test_stress_hubbard.py). The +U term is orthogonal to the SOC nonlocal term
(same statement PR #159 makes for the SCF/energy path), so this generalizes
the per-spin scalar occupation matrix to the 2×2 spin-block composite matrix
(core.hubbard.occupation_matrices_noncollinear) built from the SAME strained
atomic-orbital projectors and adds its Dudarev energy to e_tot — no change to
the SOC nonlocal machinery itself (``hubbard_energy_strained_nc``).

Two checks, mirroring test_stress_hubbard.py and test_stress_soc.py:

- self-oracle: on a rattled, low-symmetry Si cell (fully-relativistic ONCV
  pseudo, Hubbard U on the Si p manifold — not physically standard, but the
  check is the mathematical identity stress = (1/Ω) dE_tot/dε), the ε=0
  strained expression reproduces the NC-SCF total (E_U included), and the
  analytic +U SOC stress equals a central finite difference of that same
  total energy w.r.t. strain.
- U=0 reduces exactly to the pre-existing (no-+U) SOC stress: D = (U−J)(½−N)
  vanishes identically regardless of N, so the +U term is an exact zero
  addition to e_tot at U=0.
"""

import numpy as np
import pytest
import torch

from gradwave.core.hubbard import HubbardManifold
from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.stress import _energy_strained, stress
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from gradwave.scf.noncollinear import scf_noncollinear
from tests.helpers import PSEUDOS, RY

# Rattled, low-symmetry 2-atom Si cell (same geometry as test_stress_hubbard.py)
# so the full anisotropic stress tensor is exercised in every component.
SI_CELL = 5.43 * np.array([
    [0.000, 0.510, 0.490],
    [0.500, 0.000, 0.520],
    [0.505, 0.495, 0.000],
])
SI_FRAC = np.array([[0.010, 0.020, -0.015], [0.260, 0.240, 0.253]])
SI_POS = SI_FRAC @ SI_CELL

MAN = [HubbardManifold(species=0, l=1, u=4.0, j=0.0)]
_SCF_KW = dict(mag_vec_init=[[0, 0, 0], [0, 0, 0]], smearing="gaussian",
              width=0.05, nonmagnetic=True, verbose=False)


def _make_system():
    si = parse_upf(PSEUDOS / "Si_ONCV_PBE_fr.upf")
    return setup_system(SI_CELL, SI_POS, [0, 0], [si], ecut=24 * RY,
                        kmesh=(2, 2, 2), use_symmetry=False, time_reversal=False)


@pytest.mark.slow
def test_stress_soc_hubbard_autograd_vs_fd():
    """+U SOC stress vs central finite difference of the (+U) NC-SCF total
    energy w.r.t. strain. Also asserts the ε=0 strained expression reproduces
    the SCF total (E_U included)."""
    torch.set_num_threads(8)
    xc = NoncollinearXC(SpinPBE())
    res = scf_noncollinear(_make_system(), xc, hubbard=MAN, etol=1e-9,
                           rhotol=1e-8, max_iter=200, **_SCF_KW)
    assert res.converged and res.system.is_fr
    assert abs(float(res.energies.hubbard)) > 1e-3  # E_U is genuinely nonzero

    # the ε=0 strained expression (with +U) must reproduce the NC-SCF total
    e0 = _energy_strained(res, xc, torch.zeros(3, 3, dtype=torch.float64),
                          manifolds=MAN)
    assert abs(float(e0) - float(res.energies.total)) < 5e-6, (
        float(e0), float(res.energies.total))

    sig = stress(res, xc, symmetrize=False, manifolds=MAN).cpu().numpy()
    d = 1e-6

    def fd_component(i, j):
        ep = torch.zeros(3, 3, dtype=torch.float64)
        ep[i, j] = d
        return (float(_energy_strained(res, xc, ep, manifolds=MAN))
                - float(_energy_strained(res, xc, -ep, manifolds=MAN))) / (2 * d)

    omega = res.system.grid.volume
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]:
        fd_sym = 0.5 * (fd_component(i, j) + fd_component(j, i)) / omega
        assert abs(sig[i, j] - fd_sym) < 1e-5, (i, j, sig[i, j], fd_sym)


def _stress_no_manifold(res, xc):
    """The stress of the detached SCF state with the +U term omitted (the
    pure SOC KS strain derivative), bypassing stress()'s +U guard — mirrors
    test_stress_hubbard.py's helper of the same name."""
    dev = res.system.positions.device
    eps = torch.zeros(3, 3, dtype=torch.float64, device=dev, requires_grad=True)
    e = _energy_strained(res, xc, eps)
    (grad,) = torch.autograd.grad(e, eps)
    return (0.5 * (grad + grad.T) / res.system.grid.volume).cpu().numpy()


@pytest.mark.slow
def test_stress_soc_hubbard_u0_matches_plain_soc_stress():
    """U=0 reduces exactly to the pre-existing SOC stress: D = (U−J)(½−N)
    vanishes identically at U=0 regardless of the occupation matrix N, so the
    +U term is an exact-zero addition to e_tot — the SOC+U stress at U=0 must
    equal the plain (no-manifold) SOC stress of the SAME converged state to
    autograd round-off."""
    torch.set_num_threads(8)
    xc = NoncollinearXC(SpinPBE())
    u0_man = [HubbardManifold(species=0, l=1, u=0.0, j=0.0)]
    res = scf_noncollinear(_make_system(), xc, hubbard=u0_man, etol=1e-8,
                           rhotol=1e-7, max_iter=150, **_SCF_KW)
    assert res.converged
    assert float(res.energies.hubbard) == 0.0

    sig_plain = _stress_no_manifold(res, xc)
    sig_u0 = stress(res, xc, symmetrize=False, manifolds=u0_man).cpu().numpy()
    assert np.max(np.abs(sig_plain - sig_u0)) < 1e-10

    # asking for a +U stress without the manifolds must not silently drop E_U
    with pytest.raises(ValueError, match="DFT\\+U"):
        stress(res, xc)
