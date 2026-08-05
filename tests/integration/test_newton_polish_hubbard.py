"""Newton-Krylov SCF finisher with DFT+U — the Hubbard occupation-matrix block.

The finisher previously raised on ``res.hub_occ`` (its packed residual/state
vector carried only the (δρ, δbec) blocks). It now gains the per-channel
Hubbard occupation-matrix block δn — packed as (Re, Im) pairs, mirroring the
USPP adjoint's ``join``/``split`` — with the raw map carrying the mixer-side n
through ``_scf_iteration`` into V_U and the exact Jacobian folding in the
Dudarev kernel K_U = −(U−J) via ``_ConvergedUSPP.{k_hub, apply_chi0}``.

Oracle (standard tier): a loose +U PAW SCF polished by ``newton_polish`` reaches
the same fixed point as a deep-converged +U reference — the free energy AND the
Hubbard occupation matrix agree. A missing or mis-scaled hub block would either
floor the residual above ``tol`` or land the finisher at a different n
(different E_U), so the joint energy+occupation check pins the whole channel.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.newton import newton_polish
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_hubbard import HubbardManifold
from tests.helpers import RY

pytestmark = pytest.mark.standard  # full +U PAW SCFs + Newton response; not a fast gate

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
A = 5.43
CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS = np.array([[0.0, 0.0, 0.0], [A / 4] * 3])
# unphysical +U on the Si p manifold, purely to make E_U (and its n response)
# genuinely nonzero — same manifold the +U position/Hessian oracle uses.
MAN = [HubbardManifold(species=0, l=1, u=3.0, j=0.0)]


def _make():
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    return setup_uspp(CELL, POS, [0, 0], [paw], ecut=15 * RY, kmesh=(2, 2, 2),
                      ecutrho=60 * RY, use_symmetry=False)


def test_newton_polish_hubbard_matches_reference():
    torch.set_num_threads(8)
    kw = dict(hubbard=MAN, verbose=False)
    loose = scf_uspp(_make(), PBE(), etol=1e-4, rhotol=1e-3, max_iter=40, **kw)
    ref = scf_uspp(_make(), PBE(), etol=1e-12, rhotol=1e-10, max_iter=150, **kw)
    assert abs(float(ref.energies.hubbard)) > 1e-2  # E_U genuinely nonzero

    pol = newton_polish(loose, PBE(), tol=5e-9)
    assert pol["converged"], pol["newton"]
    # quadratic contraction from the loose start
    assert pol["newton"][-1] < 1e-2 * pol["newton"][0], pol["newton"]

    df = abs(float(pol["energies"].free_energy)
             - float(ref["energies"].free_energy))
    assert df < 1e-6, f"polished F off by {df:.2e} eV"

    # the +U occupation matrix converges to the deep-reference's (channel 0 =
    # the nspin=1 half-occupancy channel), per site
    for a, b in zip(pol["hub_occ"][0], ref["hub_occ"][0], strict=True):
        dn = float((a - b).abs().max())
        assert dn < 1e-4, f"hub_occ off by {dn:.2e}"
