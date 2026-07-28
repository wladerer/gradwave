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

History (issue #141): as first landed, the two paths agreed only to ~1.6 cm⁻¹
(optical FC ~0.5 %), NOT to the ~1e-5 the analytic-vs-FD *column* check hits at a
low-symmetry P1 geometry (``test_uspp_position``). That gap was h-INDEPENDENT
(not FD truncation), ecut-INDEPENDENT (not basis incompleteness), and pinned by a
THIRD route — the total-ENERGY second difference of the optical pattern (the exact
BO-surface curvature) — which matched the FD-of-forces Hessian to 1e-4 and sat
~0.5 % below the analytic Hessian. It was root-caused to the degenerate-window
gauge in ``uspp_position.window_response``: at a band degeneracy the off-diagonal
S-metric coefficient was substituted with the −½⟨m|δS|n⟩ density limit, which is
correct for the density/normalization response but leaves a spurious
DISCONTINUITY in the second-derivative assembly. ``hessian_column`` now uses the
full metric coupling −⟨m|δS|n⟩ there (the continuous ε_n≠ε_m limit), which closes
the gap: the two paths agree to ~0.01 cm⁻¹ (~1e-5), matching the P1 column check
and the energy-second-difference oracle.

The analytic path is also QE-validated (SiGe ph.x DFPT, 4×4×4/45 Ry: gradwave
419.4 vs QE 419.14, +0.06 %). The 0.2 cm⁻¹ optical tolerance guards the two paths
tracking each other (catching sign/factor/unit/solver breakage) with margin over
the observed ~0.01 cm⁻¹.
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
    # the cross-validation: the two independent methods now track each other to
    # ~0.01 cm⁻¹ (~1e-5) — the degenerate-window systematic is fixed (#141, see
    # module docstring). 0.2 cm⁻¹ leaves margin for platform BLAS/FFT noise
    # while still catching any sign/factor/unit/solver regression in either path
    assert np.abs(f_hvp[3:] - f_fd[3:]).max() < 0.2, (
        f"optical mismatch Hvp={f_hvp[3:]} FD={f_fd[3:]}")
