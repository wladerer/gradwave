"""Newton-Krylov SCF finisher for collinear spin (nspin=2 unblock).

The composite residual/state vector doubles to the per-spin (δρ↑, δρ↓,
δbec↑, δbec↓); the exact Jacobian reuses the spin-resolved χ̃ / K_Hxc /
one-center operators in uspp_implicit._ConvergedUSPP (χ̃ block-diagonal over
spin except the shared-Fermi δμ, K_Hxc cross-spin).

Oracle — nonmagnetic limit: nspin=2 (start_mag=0) polish of a loose Al PAW
metal reaches the residual floor and its free energy matches a deep-converged
reference — the two-channel packed vector and per-spin Sternheimer batches
reconstruct the nspin=1 finisher (a factor-2 spin-folding error would miss the
residual/energy decisively).

The genuinely-magnetic (V↑≠V↓) per-spin operators the finisher calls are the
SAME _ConvergedUSPP.{apply_chi0,k_hxc_grid,hvp_onecenter} that
tests/integration/test_uspp_implicit.py validates on genuinely magnetic
systems, and the NC analogue is validated magnetically in
tests/gradcheck/test_implicit_scf_nspin2.py. A dedicated magnetic newton_polish
oracle is impractical for reasons orthogonal to spin: magnetic metals floor
their residual at smearing/occupation noise (above the finisher's tight tol),
and a molecular magnetic insulator (triplet O₂) stalls newton's inner Anderson
solve on the vacuum-cell 4π/G² gain>1 mode — the pre-existing limitation
uspp_implicit's optional Kerker preconditioner (kerker_q0) addresses but which
the finisher's inner solve does not carry.
"""

from pathlib import Path

import numpy as np
import pytest

from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.newton import newton_polish
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


@pytest.mark.slow
def test_newton_polish_nspin2_nonmagnetic_limit():
    """nspin=2 (start_mag=0) polish of a loose Al PAW metal converges to the
    residual floor and matches a deep-converged reference free energy."""
    import torch
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Al.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 4.04
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0]])
    xc = SpinPBE()

    def make():
        return setup_uspp(cell, pos, [0], [paw], ecut=20 * RY,
                          kmesh=(2, 2, 2), ecutrho=100 * RY, nbands=8)

    kw = dict(nspin=2, start_mag=[0.0], smearing="gaussian", width=0.5)
    loose = scf_uspp(make(), xc, **kw, etol=1e-4, rhotol=1e-3, verbose=False,
                     max_iter=60)
    ref = scf_uspp(make(), xc, **kw, etol=1e-12, rhotol=1e-10, verbose=False,
                   max_iter=120)
    pol = newton_polish(loose, xc, tol=5e-9)
    assert pol["converged"]
    assert abs(float(pol["mag_total"])) < 1e-5  # stayed nonmagnetic
    df = abs(float(pol["energies"].free_energy)
             - float(ref["energies"].free_energy))
    assert df < 1e-6, f"polished F off by {df:.2e} eV"
