"""ε∞ and Born effective charges for collinear spin (nspin=2 unblock).

``dielectric_born`` now threads the E-field DFPT per spin channel: the ∂H/∂k
Sternheimer solve runs independently for each spin (H is block-diagonal in
spin), while the self-consistent screening field u^σ couples the two channels
through the spin Hxc kernel K_Hxc^{σσ'} (Hartree on the total Δρ + f_xc^{σσ'},
the same primitive linear-response Hubbard U uses).

Self-oracle (nonmagnetic limit): on a nonmagnetic Si cell the spin-polarized
run started from zero moment (LSDA at ρ↑ = ρ↓) must reproduce the
spin-restricted (LDA) ε∞ and Born-charge tensors to SCF-convergence precision.
The per-channel f=1 occupation bookkeeping (8π/2 prefactors summed over spin)
therefore reconstructs the nspin=1 (f=2, 16π/4) result. A factor-of-two error
in the spin folding would miss by ~100%.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.dielectric import dielectric_born
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

pytestmark = pytest.mark.slow  # two full insulator SCFs + E-field DFPT

FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def test_dielectric_nspin2_matches_spin_restricted():
    """nspin=2 (start_mag=0) ε∞ and Z* reproduce the spin-restricted result on a
    nonmagnetic Si cell, to SCF-convergence precision."""
    torch.set_num_threads(8)
    a = 5.43
    si = parse_upf(PSEUDOS / "Si_ONCV_PBE_sr.upf")

    def make():
        return setup_system(a / 2 * FCC, np.array([[0.0, 0, 0], [a / 4] * 3]),
                            [0, 0], [si], ecut=12 * RY, kmesh=(2, 2, 2),
                            use_symmetry=False)

    kw = dict(cg_tol=1e-8, outer_tol=1e-6, max_outer=60)

    r1 = scf(make(), LDA_PW92(), smearing="none", etol=1e-10, rhotol=1e-9,
             verbose=False)
    assert r1.converged
    out1 = dielectric_born(r1, LDA_PW92(), **kw)

    r2 = scf(make(), LSDA_PW92(), smearing="none", nspin=2,
             start_mag=[0.0, 0.0], tot_magnetization=0.0, etol=1e-10,
             rhotol=1e-9, verbose=False)
    assert r2.converged
    assert abs(float(r2.mag_total)) < 1e-6  # stayed nonmagnetic

    out2 = dielectric_born(r2, LSDA_PW92(), **kw)

    # nonmagnetic limit: the spin-resolved tensors match the spin-restricted
    # ones. The residual is set by the loosened iterative tolerances above
    # (outer_tol 1e-6 → ε to ~1e-5); a factor-of-two spin-folding error would
    # miss ε by O(50) and Z* by O(6), so this decisively pins the threading.
    eps_err = float((out2["eps"] - out1["eps"]).abs().max())
    born_err = float((out2["born"] - out1["born"]).abs().max())
    assert eps_err < 2e-4, f"eps mismatch {eps_err}\n{out1['eps']}\n{out2['eps']}"
    assert born_err < 2e-4, (
        f"born mismatch {born_err}\n{out1['born']}\n{out2['born']}")

    # the spin path still returns a physical isotropic diagonal tensor
    eps2 = out2["eps"]
    assert float((eps2 - torch.diag(torch.diagonal(eps2))).abs().max()) < 1e-3
