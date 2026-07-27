"""Γ-point phonon self-oracle: analytic Hvp path vs finite displacement.

Issue #141 step 2. Two *independent* routes to the same Γ dynamical matrix on
one PAW diamond-Si cell (identical ecut/kmesh/pseudo/FFT grid):

* Analytic (DFPT-equivalent): ``postscf.phonons.gamma_hessian`` — the irreducible
  Hessian column(s) from ``uspp_position.hessian_column`` (Sternheimer
  self-consistent response + a double backward through the force-energy graph,
  no SCF re-runs), symmetry-reconstructed to the full (na,3,na,3) matrix.
* Finite displacement: central difference of the analytic PAW forces
  (``paw_forces.forces_uspp``) over the six ±h displacements, folded to Γ through
  ``postscf.phonons_supercell`` on a 1×1×1 supercell.

The two share only the cm⁻¹ unit constant and the mass-weight+eigh, so agreement
cross-validates the whole analytic response machinery against brute-force
re-convergence. This is a SELF-ORACLE on a deliberately cheap discretized system
(15 Ry, 2×2×2 k): the absolute frequency here is k-under-converged (~587 vs the
physical ~520 cm⁻¹) and is NOT compared to experiment — only the two methods to
each other. Diamond glide (¼,¼,¼) forces the 20³ FFT grid (18³ breaks the group
invariance the reconstruction relies on — see ``gamma_hessian``'s guard).

Known systematic (issue #141, characterized in this PR): the two paths agree only
to ~1.6 cm⁻¹ (optical FC ~0.5%), NOT to the ~1e-5 the analytic-vs-FD *column*
check hits at a low-symmetry P1 geometry (``test_uspp_position``). The gap is:

* h-INDEPENDENT (1.551/1.551/1.554 cm⁻¹ at h=1/2/4e-3) → not FD truncation;
* ecut-INDEPENDENT (0.54/0.46/0.52 % at 15/25/35 Ry) → not basis incompleteness;
* only mildly k-dependent (0.54 % at 2×2×2 → 0.36 % at 4×4×4);
* pinned by a THIRD independent route — the total-ENERGY second difference of the
  optical displacement pattern (variational, so the exact BO-surface curvature):
  it matches the FD-of-forces Hessian to 1e-4 and sits ~0.5 % BELOW the analytic
  Hessian. So the FD path is exact here and ``hessian_column`` runs ~0.5 % high at
  this high-symmetry / band-degenerate geometry (the P1 column check has no
  degeneracies, hence its 1e-5).

The analytic path is otherwise QE-validated (SiGe ph.x DFPT, 4×4×4/45 Ry:
gradwave 419.4 vs QE 419.14, +0.06 %). Root-causing / fixing the high-symmetry
~0.5 % in ``hessian_column`` is left as #141 follow-up (out of this PR's scope).
The 2.5 cm⁻¹ optical tolerance brackets the characterized 1.6 cm⁻¹ systematic
with margin; it is a regression guard on the two paths tracking each other (it
still catches sign/factor/unit/solver breakage), not a claim of sub-cm⁻¹ parity.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.paw_forces import forces_uspp
from gradwave.postscf.phonons import gamma_frequencies, gamma_hessian
from gradwave.postscf.phonons_supercell import (
    apply_acoustic_sum_rule,
    build_supercell,
    frequencies_at_q,
    symmetrize_force_constants,
)
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
A = 5.43
CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS0 = np.array([[0.0, 0.0, 0.0], [A / 4] * 3])
MASSES = np.array([28.0855, 28.0855])  # Si
SHAPE = (20, 20, 20)  # glide-commensurate; 18³ breaks the group invariance
H_DISP = 2e-3  # Å central-difference step


def _build(pos):
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    return setup_uspp(CELL, pos, [0, 0], [paw], ecut=15 * RY, kmesh=(2, 2, 2),
                      ecutrho=60 * RY, fft_shape=SHAPE)


def _scf(pos, prev=None):
    r = scf_uspp(_build(pos), PBE(), etol=1e-12, rhotol=1e-10, verbose=False,
                 max_iter=80, start_from=prev)
    assert r["converged"]
    return r


@pytest.mark.slow
def test_gamma_phonons_hvp_matches_finite_displacement():
    torch.set_num_threads(8)
    res = _scf(POS0)

    # --- analytic Hvp path (the public Γ-phonon surface) ---
    f_hvp = np.sort(gamma_frequencies(gamma_hessian(res, PBE()), MASSES))

    # --- finite-displacement path, same cell/pseudo/grid, folded at Γ ---
    na = len(POS0)
    phi = np.zeros((na, 3, na, 3))
    for at in range(na):
        for al in range(3):
            pp, pm = POS0.copy(), POS0.copy()
            pp[at, al] += H_DISP
            pm[at, al] -= H_DISP
            fp = forces_uspp(_scf(pp, prev=res), PBE(), remove_net=False)
            fm = forces_uspp(_scf(pm, prev=res), PBE(), remove_net=False)
            phi[at, al] = -(fp - fm).detach().cpu().numpy() / (2 * H_DISP)
    scmap = build_supercell(CELL, POS0, [0, 0], (1, 1, 1))
    phi = apply_acoustic_sum_rule(symmetrize_force_constants(phi, scmap))
    f_fd = np.sort(frequencies_at_q(phi, scmap, MASSES, [0, 0, 0]))

    # acoustic modes ~0 on both (identical acoustic sum rule on both paths)
    assert np.abs(f_hvp[:3]).max() < 1.0, f"Hvp acoustic {f_hvp[:3]}"
    assert np.abs(f_fd[:3]).max() < 1.0, f"FD acoustic {f_fd[:3]}"
    # optical branch threefold-degenerate at Γ on both sides
    assert f_hvp[3:].std() < 0.1 and f_fd[3:].std() < 0.1
    # gross-error catch (NOT a physical-accuracy claim — k-under-converged here)
    assert 400.0 < f_hvp[3:].mean() < 700.0
    assert 400.0 < f_fd[3:].mean() < 700.0
    # the cross-validation: two independent methods track each other to the
    # characterized ~1.6 cm⁻¹ high-symmetry systematic (see module docstring)
    assert np.abs(f_hvp[3:] - f_fd[3:]).max() < 2.5, (
        f"optical mismatch Hvp={f_hvp[3:]} FD={f_fd[3:]}")
    # ...and the analytic side is the HIGH one (locks the sign of the systematic
    # so a future hessian_column fix that closes the gap will flag here)
    assert f_hvp[3:].mean() > f_fd[3:].mean()
