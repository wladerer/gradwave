"""Pure-arithmetic halves of the convergence-noise / uncertainty stack.

No SCF: the translation-shift sequence, the BM3 Monte-Carlo bootstrap
(postscf.eos.bootstrap_bm3), and the recommendation rule of
api.converge.recommend_convergence on synthetic numbers. The SCF-backed
probes live in tests/integration/test_convergence_noise.py.
"""

import numpy as np
import pytest

from gradwave.api.converge import _apply_rule
from gradwave.postscf.convergence_noise import translation_shifts
from gradwave.postscf.eos import birch_murnaghan, bootstrap_bm3

# --------------------------------------------------------------------------- #
#  translation_shifts                                                          #
# --------------------------------------------------------------------------- #


def test_translation_shifts_deterministic_and_in_range():
    shape = (18, 20, 24)
    s1 = translation_shifts(5, shape, seed=7)
    s2 = translation_shifts(5, shape, seed=7)
    assert s1.shape == (5, 3)
    assert np.array_equal(s1, s2)                       # seeded, reproducible
    # every component strictly inside one grid spacing (cell-fractional)
    upper = 1.0 / np.asarray(shape, dtype=float)
    assert np.all(s1 > 0.0) and np.all(s1 < upper[None, :])
    # probes are distinct configurations
    assert len({tuple(row) for row in np.round(s1, 12)}) == 5
    # a different seed shifts the whole sequence
    assert not np.allclose(s1, translation_shifts(5, shape, seed=8))


# --------------------------------------------------------------------------- #
#  bootstrap_bm3                                                               #
# --------------------------------------------------------------------------- #


def _synthetic_ev(n=9):
    v = np.linspace(18.0, 23.0, n)
    e = birch_murnaghan(v, -5.0, 20.4, 0.55, 4.3)
    return v, e


def test_bootstrap_recovers_central_fit_and_scales_linearly():
    """sigma(V0), sigma(B0), sigma(E0) scale linearly with the injected
    sigma_E (same seed => same standard-normal draws, so the scaling is
    draw-for-draw and nearly exact)."""
    v, e = _synthetic_ev()
    u1 = bootstrap_bm3(v, e, 1e-4, n_samples=64, seed=3)
    u2 = bootstrap_bm3(v, e, 2e-4, n_samples=64, seed=3)
    # central fit is the clean fit
    assert u1.fit.v0 == pytest.approx(20.4, abs=1e-6)
    assert u1.fit.b0 == pytest.approx(0.55, abs=1e-6)
    assert u1.n_failed == 0
    for lo, hi in [(u1.sigma_v0, u2.sigma_v0), (u1.sigma_b0, u2.sigma_b0),
                   (u1.sigma_e0, u2.sigma_e0)]:
        assert lo > 0.0
        assert hi / lo == pytest.approx(2.0, rel=0.15)
    from gradwave.constants import EV_A3_TO_GPA

    assert u1.sigma_b0_GPa == pytest.approx(u1.sigma_b0 * EV_A3_TO_GPA, rel=1e-9)


def test_bootstrap_scalar_and_per_point_sigma_agree():
    """A scalar sigma and the equivalent per-point array use identical draws."""
    v, e = _synthetic_ev()
    a = bootstrap_bm3(v, e, 5e-5, n_samples=32, seed=11)
    b = bootstrap_bm3(v, e, np.full_like(e, 5e-5), n_samples=32, seed=11)
    assert a.sigma_v0 == b.sigma_v0
    assert a.sigma_b0 == b.sigma_b0


def test_bootstrap_zero_sigma_is_noiseless():
    v, e = _synthetic_ev()
    u = bootstrap_bm3(v, e, 0.0, n_samples=16, seed=0)
    assert u.sigma_v0 < 1e-10 and u.sigma_b0 < 1e-10


def test_bootstrap_input_guards():
    v, e = _synthetic_ev()
    with pytest.raises(ValueError):
        bootstrap_bm3(v, e, -1e-4)
    with pytest.raises(ValueError):
        bootstrap_bm3(v, e, 1e-4, n_samples=1)


# --------------------------------------------------------------------------- #
#  recommendation rule                                                         #
# --------------------------------------------------------------------------- #

_RULE_KW = dict(
    ecuts=[100.0, 150.0, 200.0],
    ecut_err_at=[5e-3, 8e-4, 2e-4],
    tail_at=5e-5,
    ecut_run=400.0,
    base_ecut=150.0,
    meshes=[(2, 2, 2), (3, 3, 3)],
    kmesh_err_at=[2e-3, 3e-4],
    c_k=2e-3 * 8,   # consistent with err(N) = c N^-2 at the (2,2,2) mesh
    p=2.0,
    base_mesh=(2, 2, 2),
)


def test_rule_achievable_picks_smallest_parameters():
    ach, eff, ec, km, extrap, _notes = _apply_rule(
        target=1e-3, floor=1e-4, **_RULE_KW)
    assert ach is True
    assert eff == 1e-3
    assert ec == 150.0            # first candidate under the target
    assert km == (3, 3, 3)
    assert extrap is False


def test_rule_target_below_noise_floor_is_unachievable():
    """The paper's phase-diagram insight: a target below the statistical floor
    cannot be reached by refining ecut/k; the floor becomes the target."""
    ach, eff, ec, km, _extrap, notes = _apply_rule(
        target=1e-5, floor=5e-4, **_RULE_KW)
    assert ach is False
    assert eff == 5e-4            # the floor takes over
    assert ec == 200.0            # 2e-4 <= 5e-4
    assert km == (3, 3, 3)
    assert any("unachievable" in n for n in notes)


def test_rule_falls_back_to_reference_cutoff_and_extrapolated_mesh():
    kw = dict(_RULE_KW, ecut_err_at=[5e-3, 8e-4, 5e-4],
              kmesh_err_at=[2e-3, 1.5e-3], c_k=0.064)
    ach, eff, ec, km, extrap, notes = _apply_rule(
        target=1e-4, floor=1e-6, **kw)
    assert ach is True and eff == 1e-4
    assert ec == 400.0            # only the reference run's tail qualifies
    # 0.064 * 27^-2 = 8.8e-5 <= 1e-4: first extrapolated mesh that qualifies
    assert km == (3, 3, 3) and extrap is True
    assert any("extrapolated" in n for n in notes)
