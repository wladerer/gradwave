"""Partial (active-atom) Hessian mask in postscf.hessian.force_constants_gamma.

No SCF: a mock ``make_scf`` returns a stand-in result whose analytic ``forces``
are those of a fixed quadratic energy E = ½ τᵀ K τ, so the finite-difference
Hessian must recover K exactly. This lets the mask logic — displaced-atom
subset, returned sub-block, ASR guard — be checked bit-for-bit without a DFT
run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gradwave.postscf import hessian


class _MockRes:
    """Carries the flat force vector the patched ``forces`` will return."""

    def __init__(self, f_flat: np.ndarray):
        self._f = f_flat


def _make_field(monkeypatch, K: np.ndarray, x0: np.ndarray):
    """Wire a make_scf / forces pair for E = ½(x−x0)ᵀK(x−x0) on 3na DOFs.

    forces = −∇E = −K (x−x0), returned as a torch tensor shaped (na, 3) so the
    real ``forces`` call site (``.cpu().numpy()``) is exercised unchanged.
    """
    na = x0.shape[0]

    def make_scf(pos: np.ndarray) -> _MockRes:
        dx = (pos - x0).reshape(-1)
        f = -(K @ dx)
        return _MockRes(f.reshape(na, 3))

    def fake_forces(res: _MockRes, *a, **k) -> torch.Tensor:
        return torch.from_numpy(res._f)

    monkeypatch.setattr(hessian, "forces", fake_forces)
    return make_scf


def _random_spd(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    return A + A.T  # symmetric (a Hessian is symmetric; need not be PD here)


def test_masked_all_atoms_equals_full_bit_identical(monkeypatch):
    """active naming every atom in order reproduces the full Hessian exactly."""
    na = 4
    K = _random_spd(3 * na, seed=1)
    x0 = np.random.default_rng(2).normal(size=(na, 3))
    make_scf = _make_field(monkeypatch, K, x0)
    full = hessian.force_constants_gamma(
        make_scf, x0, h=1e-3, acoustic_sum_rule=True)
    masked_all = hessian.force_constants_gamma(
        make_scf, x0, h=1e-3, acoustic_sum_rule=True, active=list(range(na)))
    assert masked_all.shape == full.shape == (3 * na, 3 * na)
    assert np.array_equal(masked_all, full)


def test_masked_subset_is_the_correct_subblock(monkeypatch):
    """A strict active subset returns exactly the full Hessian's active-active
    block (ASR off so both are the raw second derivative)."""
    na = 5
    K = _random_spd(3 * na, seed=3)
    x0 = np.random.default_rng(4).normal(size=(na, 3))
    make_scf = _make_field(monkeypatch, K, x0)

    full = hessian.force_constants_gamma(
        make_scf, x0, h=1e-3, acoustic_sum_rule=False)
    active = [1, 3]
    masked = hessian.force_constants_gamma(
        make_scf, x0, h=1e-3, acoustic_sum_rule=False, active=active)

    dofs = np.array([3 * a + i for a in active for i in range(3)])
    sub = full[np.ix_(dofs, dofs)]
    assert masked.shape == (3 * len(active), 3 * len(active))
    assert np.array_equal(masked, sub)


def test_masked_hessian_recovers_the_quadratic_block(monkeypatch):
    """The FD partial Hessian equals the true stiffness sub-block of K."""
    na = 4
    K = _random_spd(3 * na, seed=5)
    x0 = np.random.default_rng(6).normal(size=(na, 3))
    make_scf = _make_field(monkeypatch, K, x0)
    active = [0, 2]
    masked = hessian.force_constants_gamma(
        make_scf, x0, h=1e-3, acoustic_sum_rule=False, active=active)
    dofs = np.array([3 * a + i for a in active for i in range(3)])
    assert np.allclose(masked, K[np.ix_(dofs, dofs)], atol=1e-6)


def test_partial_asr_is_ignored_with_warning(monkeypatch, caplog):
    """acoustic_sum_rule=True on a strict subset is skipped and warns; the
    returned block matches the ASR-off block."""
    na = 4
    K = _random_spd(3 * na, seed=7)
    x0 = np.random.default_rng(8).normal(size=(na, 3))
    make_scf = _make_field(monkeypatch, K, x0)
    active = [1, 2]
    with caplog.at_level("WARNING"):
        masked_asr = hessian.force_constants_gamma(
            make_scf, x0, h=1e-3, acoustic_sum_rule=True, active=active)
    masked_noasr = hessian.force_constants_gamma(
        make_scf, x0, h=1e-3, acoustic_sum_rule=False, active=active)
    assert np.array_equal(masked_asr, masked_noasr)
    assert any("acoustic_sum_rule ignored" in r.message for r in caplog.records)


def test_active_index_validation(monkeypatch):
    na = 3
    K = _random_spd(3 * na, seed=9)
    x0 = np.zeros((na, 3))
    make_scf = _make_field(monkeypatch, K, x0)
    with pytest.raises(ValueError, match="out of range"):
        hessian.force_constants_gamma(make_scf, x0, active=[0, 5])
    with pytest.raises(ValueError, match="duplicate"):
        hessian.force_constants_gamma(make_scf, x0, active=[1, 1])
    with pytest.raises(ValueError, match="non-empty"):
        hessian.force_constants_gamma(make_scf, x0, active=[])
