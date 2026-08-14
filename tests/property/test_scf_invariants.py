"""Fast property-based metamorphic invariants across the full SCF (Tier 1, fast).

Complements ``tests/integration/test_metamorphic_invariance.py`` (tight, fixed
fixtures) with SEEDED-RANDOM tiny low-symmetry cells at loose settings, and adds
invariants that file does not cover — spatial **inversion** (E invariant, F → −F),
**stress co-rotation** (σ → R σ Rᵀ; the integration test checks only force
co-rotation), and the **ΣF = 0** force sum rule.

Every invariant here is an *exact* identity of the theory (or has a known aliasing
floor far above the SCF tolerance), so it holds at ANY cutoff/grid/k-mesh — see
docs/verification.md. That is what lets these run on 2-atom cells at 12 Ry, Γ-few-k,
loose tol and stay in the fast gate while still having teeth: a convention/phase/sign
bug breaks them by orders of magnitude, whereas the physical inaccuracy of the tiny
cell is irrelevant to the identity. The randomization (a handful of seeds → rattled
P1 geometries) is the property-based upgrade path over a single hand-picked fixture.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.lda_pw92 import LDA_PW92
from gradwave.core.xc.spin import LSDA_PW92
from gradwave.postscf.forces import forces as compute_forces
from gradwave.postscf.stress import stress as compute_stress
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"

# Tiny + loose: the identities are exact regardless of accuracy, so this is the
# cheapest cell that still exercises the full G-sphere / projector / XC / Ewald /
# density chain. smearing="none" (Si is an insulator) keeps every SCF sub-second.
_ECUT = 12 * RY
_KMESH = (2, 1, 1)
_SCF_KW = dict(smearing="none", width=0.0, etol=1e-9, rhotol=1e-8,
               diago_tol=1e-11, verbose=False)
_SEEDS = [1, 7, 13]


@pytest.fixture(autouse=True)
def _limit_threads():
    torch.set_num_threads(4)


@pytest.fixture(scope="module")
def _si():
    return parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")


def _rattled_si(seed: int):
    """A 2-atom Si diamond primitive rattled into P1 (no residual symmetry, so
    error terms cannot cancel), returned as (lattice, positions [Å])."""
    rng = np.random.default_rng(seed)
    lattice = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0.0, 0.0], [1.357, 1.357, 1.357]])
    pos = pos + rng.normal(scale=0.12, size=(2, 3))
    return lattice, pos


def _run(lattice, pos, si, species=(0, 0)):
    res = scf(setup_system(lattice, pos, list(species), [si],
                           ecut=_ECUT, kmesh=_KMESH), LDA_PW92(), **_SCF_KW)
    assert res.converged
    return res


def _e_per_atom(res):
    return float(res.energies.free_energy) / 2.0


# --- exact geometric invariants ---------------------------------------------

@pytest.mark.parametrize("seed", _SEEDS)
def test_permutation_invariance(seed, _si):
    """Relabeling atoms leaves the Hamiltonian identical up to summation order:
    E exact to SCF tol, forces permute."""
    lattice, pos = _rattled_si(seed)
    ra = _run(lattice, pos, _si)
    rb = _run(lattice, np.ascontiguousarray(pos[::-1]), _si)
    assert abs(_e_per_atom(ra) - _e_per_atom(rb)) < 5e-8
    fa = compute_forces(ra).numpy()
    fb = compute_forces(rb).numpy()
    assert np.abs(fb - fa[::-1]).max() < 1e-6


@pytest.mark.parametrize("seed", _SEEDS)
def test_spatial_inversion(seed, _si):
    """Global inversion τ → −τ: the total-energy functional has even parity, so
    E is invariant and every force flips sign (F(−τ) = −F(τ)). Catches a sign or
    parity slip in any Cartesian term that force co-rotation alone would miss."""
    lattice, pos = _rattled_si(seed)
    ra = _run(lattice, pos, _si)
    rb = _run(lattice, -pos, _si)
    assert abs(_e_per_atom(ra) - _e_per_atom(rb)) < 5e-8
    fa = compute_forces(ra).numpy()
    fb = compute_forces(rb).numpy()
    assert np.abs(fb + fa).max() < 1e-6, "forces did not flip under inversion"


@pytest.mark.parametrize("seed", _SEEDS)
def test_rotation_energy_forces_stress(seed, _si):
    """Same crystal in a rigidly rotated Cartesian frame: E invariant, forces
    co-rotate (f → f Rᵀ), and the stress tensor co-rotates (σ → R σ Rᵀ). The
    stress covariance is the piece the integration battery does not check — a
    frame-fixed axis in the σ = (1/Ω)∂E/∂ε assembly breaks it at O(1)."""
    rng = np.random.default_rng(seed + 100)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    rot = q * np.sign(np.diag(r))[None, :]
    if np.linalg.det(rot) < 0:
        rot[:, 0] = -rot[:, 0]  # proper rotation

    lattice, pos = _rattled_si(seed)
    ra = _run(lattice, pos, _si)
    rb = _run(lattice @ rot.T, pos @ rot.T, _si)
    assert abs(_e_per_atom(ra) - _e_per_atom(rb)) < 5e-8

    fa = compute_forces(ra).numpy()
    fb = compute_forces(rb).numpy()
    assert np.abs(fb - fa @ rot.T).max() < 1e-6, "forces do not co-rotate"

    sa = compute_stress(ra, LDA_PW92()).numpy()
    sb = compute_stress(rb, LDA_PW92()).numpy()
    assert np.abs(sb - rot @ sa @ rot.T).max() < 1e-6, "stress does not co-rotate"


@pytest.mark.parametrize("seed", _SEEDS)
def test_force_sum_rule(seed, _si):
    """Translational invariance ⇒ Σ_a F_a = 0: no net force on the whole cell,
    exact to SCF tolerance (the egg-box aliasing cancels in the sum)."""
    lattice, pos = _rattled_si(seed)
    f = compute_forces(_run(lattice, pos, _si)).numpy()
    assert np.abs(f.sum(axis=0)).max() < 1e-6, f"ΣF ≠ 0: {f.sum(axis=0)}"


def test_spin_flip_symmetry(_si):
    """Collinear spin-flip: with no external field the energy functional is
    symmetric under ↑↔↓, so a fixed-moment state +M and its mirror −M share the
    same energy and have opposite total magnetization. A spin-channel-asymmetric
    bug (a factor on one channel, a swapped up/dn potential) breaks it. Uses
    fixed-moment (tot_magnetization, smearing='none') so no slow metallic loop."""
    lattice, pos = _rattled_si(1)
    fsm_kw = dict(nspin=2, smearing="none", etol=1e-9, rhotol=1e-8,
                  diago_tol=1e-11, verbose=False)

    def _fsm(m):
        sys = setup_system(lattice, pos, [0, 0], [_si], ecut=_ECUT, kmesh=_KMESH)
        return scf(sys, LSDA_PW92(), tot_magnetization=m, **fsm_kw)

    up, dn = _fsm(2.0), _fsm(-2.0)
    assert up.converged and dn.converged
    assert abs(_e_per_atom(up) - _e_per_atom(dn)) < 5e-8
    assert abs(float(up.mag_total) + float(dn.mag_total)) < 1e-6


def test_translation_invariance_to_floor(_si):
    """Rigid grid-incommensurate shift: E invariant up to the XC-quadrature
    (egg-box) aliasing floor, which is well above the SCF tol at this grid."""
    lattice, pos = _rattled_si(1)
    t = np.array([0.7137, -0.2911, 0.4302])  # grid-incommensurate
    ra = _run(lattice, pos, _si)
    rb = _run(lattice, pos + t, _si)
    # egg-box at 12 Ry Si is ~a few 1e-5 eV/atom; bound above it, not at zero
    assert abs(_e_per_atom(ra) - _e_per_atom(rb)) < 5e-5
