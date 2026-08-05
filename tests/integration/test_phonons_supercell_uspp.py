"""USPP/PAW supercell finite-displacement phonons — the fold's ultrasoft/PAW
force route.

Before this change the finite-displacement force fold
(``phonons_supercell.force_constants_home``) and ``api.run_phonons`` were
norm-conserving only. Both now route the displaced-cell forces through the
augmentation-aware ``postscf.paw_forces.forces_uspp`` via the new ``force_fn``
dispatch, so an ultrasoft/PAW pseudopotential folds phonons like the NC path.

Two standard-tier oracles on a PAW diamond-Si Γ system:

* ``test_uspp_supercell_fold_matches_analytic_gamma`` — the supercell-fold Γ
  frequencies (central difference of the PAW forces, folded on a 1×1×1
  supercell through ``force_constants_home``'s new ``force_fn``) must match the
  analytic Γ-Hessian route (``postscf.phonons.gamma_hessian`` via
  ``uspp_position.hessian_column`` — the DFPT-equivalent Sternheimer response,
  no SCF re-runs). The two paths share only the mass-weight + cm⁻¹ unit
  constant. This is the same self-oracle as the slow-tier
  ``test_gamma_phonons_self_oracle`` (identical 15 Ry / 60 Ry / 20³ cell where
  the two paths agree to ~0.01 cm⁻¹) but driven through the *public* fold entry
  point that was NC-only before this change; ``remove_net=False`` matches the
  analytic route, and the acoustic modes / threefold optical degeneracy are
  checked alongside. Diamond glide (¼,¼,¼) forces the 20³ FFT grid (18³ breaks
  the group invariance the reconstruction relies on — see ``gamma_hessian``).

* ``test_run_phonons_uspp_end_to_end`` — ``api.run_phonons`` on a PAW input no
  longer raises (the USPP ``scf_uspp`` make_scf branch + ``forces_uspp`` fold
  are wired) and returns a physically-sane Γ. Cheaper coarse cell, tolerant
  absolute checks (the production defaults, incl. ``remove_net``).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms

from gradwave.api import run_phonons
from gradwave.core.xc.pbe import PBE
from gradwave.inputs import Input, KPointsParams, PhononParams, SmearingParams
from gradwave.postscf.paw_forces import forces_uspp
from gradwave.postscf.phonons import gamma_frequencies, gamma_hessian
from gradwave.postscf.phonons_supercell import (
    build_supercell,
    force_constants_home,
    frequencies_at_q,
)
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from tests.helpers import RY

pytestmark = pytest.mark.standard  # full PAW SCFs + analytic response; not a fast gate

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
PAW_NAME = "Si.pbe-n-kjpaw_psl.1.0.0.UPF"
A = 5.43
CELL = A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS0 = np.array([[0.0, 0.0, 0.0], [A / 4] * 3])
MASSES = np.array([28.0855, 28.0855])  # Si
SHAPE = (20, 20, 20)  # glide-commensurate; 18³ breaks the group invariance
H_DISP = 2e-3  # Å central-difference step


def _scf(pos, ecut, ecutrho, prev=None):
    paw = parse_upf_paw(FIX / "pseudos" / PAW_NAME)
    system = setup_uspp(CELL, pos, [0, 0], [paw], ecut=ecut, kmesh=(2, 2, 2),
                        ecutrho=ecutrho, fft_shape=SHAPE)
    r = scf_uspp(system, PBE(), etol=1e-12, rhotol=1e-10, verbose=False,
                 max_iter=100, start_from=prev)
    assert r["converged"]
    return r


def test_uspp_supercell_fold_matches_analytic_gamma():
    """Fold-vs-analytic Γ frequency cross-validation through the new USPP
    force_fn dispatch in force_constants_home (15 Ry / 60 Ry — the validated
    self-oracle regime where the two paths agree to ~0.01 cm⁻¹)."""
    torch.set_num_threads(8)
    ecut, ecutrho = 15 * RY, 60 * RY
    ref = _scf(POS0, ecut, ecutrho)

    # analytic (DFPT-equivalent) Γ frequencies
    f_an = np.sort(gamma_frequencies(gamma_hessian(ref, PBE()), MASSES))

    # supercell fold through the NEW force_fn dispatch (USPP/PAW route);
    # remove_net=False matches the analytic route (the acoustic sum rule handles
    # the residual net component either way)
    scmap = build_supercell(CELL, POS0, [0, 0], (1, 1, 1))
    phi = force_constants_home(
        lambda pos, start_from=None: _scf(pos, ecut, ecutrho, prev=start_from),
        scmap, h=H_DISP,
        force_fn=lambda r: forces_uspp(r, PBE(), remove_net=False))
    f_fd = np.sort(frequencies_at_q(phi, scmap, MASSES, [0, 0, 0]))

    # acoustic sum rule: three ~0 modes on the fold
    assert np.abs(f_fd[:3]).max() < 1.0, f"fold acoustic {f_fd[:3]}"
    # optical branch threefold degenerate at Γ
    assert f_fd[3:].std() < 0.5, f"fold optical spread {f_fd[3:]}"
    # gross-error catch (k-under-converged here — not a physical-accuracy claim)
    assert 400.0 < f_fd[3:].mean() < 700.0
    # the cross-validation: the fold (paw_forces) and the analytic response
    # (uspp_position) track each other to well under 0.5 cm⁻¹
    assert np.abs(f_fd[3:] - f_an[3:]).max() < 0.5, (
        f"fold {f_fd[3:]} vs analytic {f_an[3:]}")


def test_run_phonons_uspp_end_to_end():
    """api.run_phonons on a PAW input no longer raises (USPP route wired) and
    returns a physically-sane Γ: acoustic ≈ 0, optical threefold-degenerate."""
    torch.set_num_threads(8)
    atoms = Atoms("Si2", positions=POS0, cell=CELL, pbc=True)
    inp = Input(
        atoms=atoms, pseudo_dir=FIX / "pseudos",
        pseudo_map={"Si": PAW_NAME}, ecut=12 * RY, ecutrho=48 * RY, xc="pbe",
        kpoints=KPointsParams(mesh=(2, 2, 2)),
        smearing=SmearingParams(type="none"),
        phonons=PhononParams(supercell=(1, 1, 1), displacement=H_DISP,
                             npoints=30, dos_mesh=(0, 0, 0)))
    ph = run_phonons(inp, verbose=False)
    freqs = np.array(ph["frequencies_cm1"])
    labels, x = ph["labels"], np.array(ph["x"])
    gx = [xt for xt, lab in labels if lab in ("G", "Γ")][0]
    at_g = np.sort(freqs[int(np.argmin(np.abs(x - gx)))])
    assert np.abs(at_g[:3]).max() < 10.0            # acoustic ≈ 0
    assert at_g[3:].std() < 8.0                     # threefold degenerate
    assert 400.0 < at_g[3:].mean() < 700.0          # optical band present
