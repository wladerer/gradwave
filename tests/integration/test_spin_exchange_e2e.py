"""End-to-end spin-Hamiltonian extraction (postscf/spin_exchange.py) on a real
constrained-DFT torque rather than synthetic tensors.

The system is the same triplet O2 molecule as test_moment_config.py: two
genuinely magnetic atoms whose moments couple ferromagnetically (the triplet
ground state), so the shell-summed exchange read off the tilted-moment torque
must come out ferromagnetic. The unit suite (test_spin_exchange.py) covers the
tensor decomposition on synthetic inputs; this test exercises the full pipeline
exchange_from_atom -> decompose on three constrained SCFs (reference + two
tilts) and gates on the physics the geometry dictates:

- J_iso > 0: ferromagnetic coupling, matching the triplet ground state (the
  same physics test_config_search_finds_ferromagnet reaches by relaxation).
- D ~ 0: the antisymmetric (DMI) part needs spin-orbit coupling, absent with a
  scalar-relativistic pseudopotential, so it must vanish to numerical noise.
- isotropy in the transverse plane: with the bond and the reference axis both
  along z, the x and y tilts are equivalent by the molecule's axial symmetry,
  so the 2x2 tensor is a multiple of the identity up to noise.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.noncollinear import NoncollinearXC
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.spin_exchange import decompose, exchange_from_atom
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from tests.helpers import RY, pseudo

PSEUDO = pseudo("O_ONCV_PBE-1.2.upf")


def _o2_system(L=6.0, d=1.21):
    o = parse_upf(PSEUDO)
    cell = L * np.eye(3)
    pos = np.array([[L / 2, L / 2, L / 2 - d / 2], [L / 2, L / 2, L / 2 + d / 2]])
    return setup_system(cell, pos, [0, 0], [o, o], ecut=30 * RY, kmesh=(1, 1, 1),
                        nbands=8, time_reversal=False)


@pytest.mark.slow
def test_o2_exchange_is_ferromagnetic_isotropic_and_dmi_free():
    torch.set_num_threads(8)
    system = _o2_system()
    xc = NoncollinearXC(LSDA_PW92())
    m0 = torch.tensor([1.0, 1.0], dtype=torch.float64)

    tensors, (u, v) = exchange_from_atom(
        system, xc, j=1, m0=m0, ref_dir=(0.0, 0.0, 1.0), delta=0.08, lam=8.0,
        smearing="gaussian", width=0.1, etol=1e-9, rhotol=1e-8,
        max_iter=200, verbose=False)

    # the transverse basis spans the plane perpendicular to the bond axis
    assert abs(float(u[2])) < 1e-12 and abs(float(v[2])) < 1e-12

    J = tensors[0]                       # response of atom 0 to tilting atom 1
    j_iso, d_ref, gamma = decompose(J)

    # ferromagnetic coupling, well above the torque-difference noise
    assert j_iso > 0.05, f"J_iso = {j_iso:.4f} eV, expected ferromagnetic"
    # no SOC -> no DMI
    assert abs(d_ref) < 0.05 * j_iso, f"D = {d_ref:.5f} eV vs J = {j_iso:.4f} eV"
    # axial symmetry -> isotropic transverse tensor
    assert float(gamma.abs().max()) < 0.05 * j_iso, \
        f"anisotropic part {float(gamma.abs().max()):.5f} eV vs J = {j_iso:.4f} eV"
