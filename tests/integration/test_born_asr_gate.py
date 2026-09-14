"""Self-consistency gate: the Born-charge acoustic sum rule ΣZ* = 0 on the RAW
(pre-enforcement) Born effective charges of a small polar cell.

ΣZ* = 0 (a rigid translation of every atom carries no charge / translational
invariance of the polarization) is a convention-free identity that needs no
external number: on a polar binary the individual Z* are large and opposite
(rocksalt MgO: Z*_Mg ≈ +2, Z*_O ≈ −2) and their sum must vanish. This is the
non-circular content of the Born computation, so it is worth a cheap
STANDARD-tier gate rather than only the loose slow bound in
``test_ir_born_task.py`` (``asr_max < 0.05``, a 5% residual passes) — and it
must be read BEFORE any ``enforce_asr`` correction, which zeroes the sum by
construction (subtracting ΣZ*/N from each atom) and would mask a real residual.

The DFPT E-field route (:func:`gradwave.postscf.dielectric.dielectric_born`)
gives the raw residual for one insulator SCF + one field solve; it never
enforces the ASR, and ``use_symmetry=False`` leaves the per-atom tensors
unsymmetrized so nothing folds the residual away (measured: the symmetrized IBZ
path returns the identical ΣZ*, so symmetry is not what makes it small).

Convergence note (measured on asus, PBE/ONCV MgO): the raw |ΣZ*| is a
finite-mesh discretization quantity, not exactly zero. It is only small on a
properly sampled EVEN mesh — 4×4×4/30Ry gives 9.7e-3 |e| (Z* = +2.075/−2.085),
while a too-coarse 2×2×2 is pathological (|ΣZ*| ≈ 2.4, Z*_O ≈ −4.2) and odd
meshes miss the fcc star (3×3×3 ≈ 0.40). Across 4³–6³ the residual floats at the
1e-2 level (4³/35Ry 2.0e-2, 6³/35Ry 7.8e-2) rather than converging to machine
zero — the same k-mesh floor the slow test's loose 0.05 bound tolerates. The
gate here (4³/30Ry, < 2e-2) is ~2.5× tighter than that bound, reproducible, and
rejects the pathological under-converged regime.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.dielectric import dielectric_born
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo

A = 4.24  # PBE-ish MgO lattice constant [Angstrom]
CELL = A / 2.0 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) @ CELL  # Mg, O


@pytest.mark.standard  # one insulator SCF + one E-field DFPT solve (~10 s)
def test_born_acoustic_sum_rule_raw_residual():
    torch.set_num_threads(8)
    upfs = [parse_upf(pseudo("Mg_ONCV_PBE-1.2.upf")),
            parse_upf(pseudo("O_ONCV_PBE-1.2.upf"))]

    # use_symmetry=False: no per-atom tensor symmetrization touches the Born
    # tensors, so the returned ΣZ* is the genuinely raw residual.
    system = setup_system(CELL, POS, [0, 1], upfs, ecut=30 * RY,
                          kmesh=(4, 4, 4), use_symmetry=False)
    res = scf(system, PBE(), nspin=1, smearing="none",
              etol=1e-9, rhotol=1e-8, verbose=False)
    assert res.converged

    dres = dielectric_born(res, PBE(), cg_tol=1e-8, outer_tol=1e-6, max_outer=120)
    born = dres["born"]  # (2, 3, 3), RAW: dielectric_born never enforces the ASR
    asr = dres["asr"]    # ΣZ* over atoms (3, 3), the same raw tensor

    # sanity: the individual Z* are genuinely large and opposite, so ΣZ* = 0 is
    # a real cancellation, not 0 + 0 (which a homopolar cell like Si would give).
    z_mg = float(torch.diagonal(born[0]).mean())
    z_o = float(torch.diagonal(born[1]).mean())
    assert 1.8 < z_mg < 2.4, f"Z*_Mg off: {z_mg:.3f}"
    assert -2.4 < z_o < -1.8, f"Z*_O off: {z_o:.3f}"

    # the gate: raw |ΣZ*| (acoustic sum) must vanish, read BEFORE enforce_asr.
    # measured 9.7e-3 |e| at this converged even mesh; 2e-2 keeps ~2x margin and
    # stays ~2.5x below the slow test's loose 0.05 bound.
    asr_max = float(asr.abs().max())
    assert asr_max < 2e-2, f"raw ΣZ* residual too large: {asr_max:.3e} |e|"
