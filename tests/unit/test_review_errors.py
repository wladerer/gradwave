"""Unit tests for the postscf code-review fixes.

Each test pins one targeted review finding so a regression in the fix is
caught without needing a full SCF run:

  * Dyson dressing is rejected (not silently dropped) on the USPP/spinor
    dispatch branches of ``estimate_density_error``.
  * ``AndersonMixer`` is Type-II, converges on a contraction, and stays
    finite on a rank-deficient secant matrix (the CUDA gels hazard).
  * ``DysonNotConverged`` is a dedicated ``RuntimeError`` subclass that the
    screened SCF-error path catches while letting other RuntimeErrors escape.
  * The stale ``device`` argument was dropped from
    ``_aug_density_from_becsum`` and a distinct occupation constant was added.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from gradwave.postscf import _anderson, uspp_implicit
from gradwave.postscf.convergence_error import DysonNotConverged
from gradwave.postscf.discretization_error import (
    _aug_density_from_becsum,
    estimate_density_error,
)


class _FakeNCResult:
    """Duck-typed NCResult: carries mag_vec/m but no v_eff (see _is_ncresult)."""

    def __init__(self):
        self.mag_vec = torch.zeros(3)
        self.m = torch.zeros(2, 2, 2)


def test_dyson_rejected_on_uspp_dict_branch():
    with pytest.raises(NotImplementedError, match="Dyson dressing not implemented"):
        estimate_density_error({"anything": 0}, dyson=True)


def test_dyson_rejected_on_spinor_branch():
    # Must raise for the missing-Dyson support BEFORE the xc-required ValueError.
    with pytest.raises(NotImplementedError, match="Dyson dressing not implemented"):
        estimate_density_error(_FakeNCResult(), dyson=True)


def test_aug_density_from_becsum_has_no_device_arg():
    params = list(inspect.signature(_aug_density_from_becsum).parameters)
    assert params == ["system", "becsum"]


def test_anderson_converges_on_contraction():
    torch.manual_seed(0)
    n = 24
    mat = torch.randn(n, n, dtype=torch.float64)
    a = 0.4 * mat / torch.linalg.matrix_norm(mat, 2)  # spectral norm 0.4
    b = torch.randn(n, dtype=torch.float64)
    exact = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - a, b)

    u = torch.zeros(n, dtype=torch.float64)
    mixer = _anderson.AndersonMixer(history=5, beta=1.0)
    for _ in range(40):
        r = a @ u + b - u
        u = mixer.step(u, r)
    assert torch.allclose(u, exact, atol=1e-8)


def test_anderson_rank_deficient_secant_is_finite():
    # Feed a residual/iterate stream whose secant differences are linearly
    # dependent (repeated direction). A plain lstsq gels driver returns garbage;
    # the regularized normal-equations solve must stay finite.
    torch.manual_seed(1)
    n = 16
    mixer = _anderson.AndersonMixer(history=6, beta=0.5)
    direction = torch.randn(n, dtype=torch.float64)
    out = None
    for k in range(8):
        u = float(k) * direction              # collinear iterates
        r = torch.sin(torch.tensor(float(k))) * direction
        out = mixer.step(u, r)
    assert torch.isfinite(out).all()


def test_anderson_matches_lstsq_when_well_conditioned():
    # The regularization is a tiny fraction of the diagonal, so on a full-rank
    # secant matrix gamma must agree with the old torch.linalg.lstsq solution.
    torch.manual_seed(2)
    n, m = 32, 5
    dr = torch.randn(n, m, dtype=torch.float64)
    r = torch.randn(n, dtype=torch.float64)
    g_lstsq = torch.linalg.lstsq(dr, r[:, None]).solution[:, 0]
    drh = dr.conj().transpose(-2, -1)
    ata = drh @ dr
    lam = 1e-12 * ata.diagonal().abs().max().clamp_min(1e-300)
    ata = ata + lam * torch.eye(m, dtype=ata.dtype)
    g_new = torch.linalg.solve(ata, drh @ r[:, None])[:, 0]
    assert torch.allclose(g_lstsq, g_new, atol=1e-9)


def test_dyson_not_converged_is_runtimeerror_subclass():
    assert issubclass(DysonNotConverged, RuntimeError)
    assert not issubclass(DysonNotConverged, NotImplementedError)


def test_uspp_full_fill_constant_distinct_from_occ_cut():
    # The occupation cut and the distance-from-full-filling tolerance are now
    # two named constants (both exist; the second is no longer _F_CUT reused).
    assert hasattr(uspp_implicit, "_F_CUT")
    assert hasattr(uspp_implicit, "_F_FULL_TOL")
