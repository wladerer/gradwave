"""The spinor (non-collinear) USPP/PAW SCF (scf/uspp_noncollinear.py).

Validation ladder, all self-referential (no external data):
1. Si, nonmagnetic: the spinor loop with m⃗ = 0 must reproduce the collinear
   nspin=1 scf_uspp free energy. Exercises the doubled coefficient axis, the
   S ⊗ 1₂ generalized Davidson, the 4-channel augmentation, and the one-center
   corrector at once. Measured: identical to all printed digits.
2. O2 PAW, magnetic: moments seeded ∥ ẑ must reproduce the collinear nspin=2
   free energy and |moment|. Measured: |ΔF| = 1e-8 eV.
3. Rotation invariance: moments ∥ x̂ must give the same energy (no SOC) — this
   is the rung that exercises the off-diagonal D↑↓ spin-flip blocks, which
   vanish in the collinear limit. Measured: |ΔF| = 1.3e-7 eV.

The O2 rungs use the known-convergent spin-O2 settings from
test_uspp_implicit.py (35/280 Ry, width 0.01 Ry, energy-criterion gating) —
25 Ry is the documented limit-cycle regime and shows ~1e-4 eV scatter in the
COLLINEAR reference itself. Gate the spinor runs on the energy tail; the
density residual floors at molecular noise. Gate the moment on |m|, not sign
(±m degenerate without SOC)."""

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_noncollinear import scf_uspp_noncollinear
from tests.helpers import PSEUDOS, RY

PSE = PSEUDOS
@pytest.mark.slow
def test_spinor_paw_si_collinear_limit():
    torch.set_num_threads(8)
    si = parse_upf_paw(f"{PSE}/Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    a = 5.43
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])

    def system():
        return setup_uspp(cell, pos, [0, 0], [si], ecut=20 * RY, kmesh=(2, 2, 2),
                          nbands=8, use_symmetry=False)

    ref = scf_uspp(system(), LDA_PW92(), smearing="gaussian", width=0.05,
                   etol=1e-8, rhotol=1e-7, verbose=False)
    nc = scf_uspp_noncollinear(system(), LSDA_PW92(), [[0, 0, 0], [0, 0, 0]],
                               smearing="gaussian", width=0.05, etol=1e-8,
                               rhotol=1e-7, verbose=False)
    assert ref["converged"] and nc["converged"]
    d = abs(float(nc["energies"].free_energy) - float(ref["energies"].free_energy))
    assert d < 5e-6, f"nonmagnetic collinear limit broken: |dF| = {d:.2e} eV"
    assert max(abs(x) for x in nc["mag_vec"]) < 1e-6


@pytest.mark.torture
def test_spinor_paw_o2_collinear_limit_and_rotation():
    torch.set_num_threads(8)
    o = parse_upf_paw(f"{PSE}/O.pbe-n-kjpaw_psl.1.0.0.UPF")
    cell = 6.0 * np.eye(3)
    pos = np.array([[3.0, 3.0, 2.40], [3.06, 3.0, 3.75]])

    def system():
        return setup_uspp(cell, pos, [0, 0], [o], ecut=35 * RY, kmesh=(1, 1, 1),
                          ecutrho=280 * RY, nbands=10, use_symmetry=False)

    ref = scf_uspp(system(), LSDA_PW92(), nspin=2, start_mag=[0.5],
                   smearing="gaussian", width=0.01 * RY, etol=3e-7,
                   criterion="energy", rhotol=1e-9, max_iter=120, verbose=False)
    assert ref["converged"] and abs(ref["mag_total"] - 2.0) < 1e-2
    f_ref = float(ref["energies"].free_energy)

    runs = {}
    for tag, d in (("z", [0, 0, 1.0]), ("x", [1.0, 0, 0])):
        r = scf_uspp_noncollinear(system(), LSDA_PW92(), [d, d],
                                  smearing="gaussian", width=0.01 * RY,
                                  etol=3e-7, rhotol=5e-4, max_iter=250,
                                  verbose=False)
        m = np.array(r["mag_vec"])
        # the moment lies along the seeded axis (± sign is degenerate)
        axis = np.argmax(np.abs(m))
        assert axis == (2 if tag == "z" else 0)
        assert abs(np.linalg.norm(m) - 2.0) < 1e-2
        runs[tag] = float(r["energies"].free_energy)

    d_col = abs(runs["z"] - f_ref)
    d_rot = abs(runs["x"] - runs["z"])
    assert d_col < 5e-6, f"magnetic collinear limit: |dF| = {d_col:.2e} eV"
    assert d_rot < 5e-6, f"rotation invariance: |dF(x,z)| = {d_rot:.2e} eV"
