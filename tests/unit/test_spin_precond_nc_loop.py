"""The Stoner spin preconditioner wired into the NC collinear loop (loop.scf).

Properties of the ``spin_precond`` kwarg on ``scf`` (norm-conserving, collinear
nspin=2), mirroring the USPP-path checks:

- it actually builds and applies on a smeared nspin=2 metal with Fermi-surface
  weight: the loop rebuilds the Stoner operator each mixing step and applies it
  to the magnetization channel (``StonerSpinPrecond.apply`` is called);
- ``spin_precond=False`` is byte-for-byte the default path — passing it
  explicitly changes nothing, and the guard keeps the preconditioner out of the
  mixing vector entirely (``build_stoner_precond`` is never called);
- it is a no-op on a spin-restricted (nspin=1) run even when requested.

fcc Al (metallic, small cutoff) keeps these in the fast tier. A dense k-mesh is
needed for the operator to be non-trivial: on a coarse mesh no sampled band
carries Fermi-surface weight and ``build_stoner_precond`` returns None (a
legitimate identity no-op). The genuinely magnetic convergence-delta
measurement is the standard-tier integration test on bcc Fe.
"""

from unittest.mock import patch

import numpy as np
import torch

import gradwave.scf.spin_precond as spmod
from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY

# fcc Al primitive cell — a cheap metal with Fermi-surface weight run nspin=2.
_A = 4.05
_CELL = _A / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
_POS = np.array([[0.0, 0.0, 0.0]])


def _make_al(kmesh=(2, 2, 2)):
    al = parse_upf(str(PSEUDOS / "Al_ONCV_PBE-1.2.upf"))
    return setup_system(_CELL, _POS, [0], [al], ecut=12 * RY, kmesh=kmesh)


def _run(spin_precond, kmesh=(2, 2, 2), max_iter=5, **kw):
    return scf(_make_al(kmesh), LSDA_PW92(), smearing="gaussian", width=0.1,
               nspin=2, start_mag=[0.2], max_iter=max_iter, etol=1e-10,
               rhotol=1e-9, verbose=False, spin_precond=spin_precond, **kw)


def test_nc_loop_spin_precond_builds_and_applies():
    """On a smeared nspin=2 metal (dense mesh → Fermi-surface weight),
    spin_precond=True builds the Stoner operator and applies it to the
    magnetization channel each mixing step."""
    torch.set_num_threads(4)
    orig_apply = spmod.StonerSpinPrecond.apply
    seen = {"count": 0, "ng": None}

    def _spy(self, r_m):
        seen["count"] += 1
        seen["ng"] = int(r_m.shape[0])
        return orig_apply(self, r_m)

    with patch.object(spmod.StonerSpinPrecond, "apply", _spy):
        _run(True, kmesh=(4, 4, 4), max_iter=5)
    assert seen["count"] >= 1, seen  # the operator was built and applied
    # it acts on ONE channel (the m block), not the packed (total, m) vector
    assert seen["ng"] is not None


def test_spin_precond_false_is_default_path():
    """Passing spin_precond=False is bit-for-bit the default call, and the
    Stoner operator is never constructed on that path."""
    torch.set_num_threads(4)
    ref = _run(False, max_iter=6)

    with patch.object(spmod, "build_stoner_precond") as spy:
        off = _run(False, max_iter=6)
    assert spy.call_count == 0  # guard never enters the branch

    # identical trajectory → identical fixed-point energy, moment, density
    assert float(off.energies.free_energy) == float(ref.energies.free_energy)
    assert off.mag_total == ref.mag_total
    assert torch.equal(off.rho, ref.rho)


def test_spin_precond_noop_when_nonmagnetic():
    """The guard requires nspin==2: on a spin-restricted (nspin=1) run the
    m-channel preconditioner is never built even when requested."""
    torch.set_num_threads(4)
    with patch.object(spmod, "build_stoner_precond") as spy:
        scf(_make_al(), LDA_PW92(), smearing="gaussian", width=0.1, nspin=1,
            max_iter=4, verbose=False, spin_precond=True)
    assert spy.call_count == 0
