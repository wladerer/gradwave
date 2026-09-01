"""Auto-abstain gate + build-once/reuse-across-geometry cache (M2).

These exercise the Chi0PrecondCache control flow WITHOUT running an SCF: the two
heavy calls it makes — the ρ(M) screening-eigenvalue gate
(``dominant_screening_eigenvalue``) and the subspace build
(``build_woodbury_subspace``) — are monkeypatched, so the test pins the decision
logic (engage vs abstain, decide-once, grid-invalidation) in the fast tier.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import gradwave.scf.subspace_chi0 as sc
from gradwave.scf.subspace_chi0 import (
    _CHI0_ENGAGE_RHO,
    _VAC_ABSTAIN_BELOW,
    _VAC_ENGAGE_ABOVE,
    Chi0PrecondCache,
    vacuum_fraction,
)


def _rho_with_vac(shape, vac):
    """A total-density grid whose vacuum_fraction is exactly ``vac``: a ``vac``
    fraction of points at 0 (< rel·mean), the rest at 1."""
    n = 1
    for s in shape:
        n *= s
    rho = torch.ones(n, dtype=torch.float64)
    nz = int(round(vac * n))
    rho[:nz] = 0.0
    return rho.reshape(shape)


def _fake_res(shape=(4, 4, 4), ng=17, vac=0.10):
    """Minimal stand-in for an SCFResult carrying only what the cache reads.

    ``vac`` defaults to the ambiguous band (between the pre-gate thresholds) so
    the decision routes through the ρ(M) gate the older tests pin; the pre-gate
    tests pass an explicit clear-bulk / clear-slab value."""
    mask = torch.zeros(shape, dtype=torch.bool)
    flat = mask.reshape(-1)
    flat[:ng] = True
    grid = SimpleNamespace(shape=shape, dens_mask=mask)
    system = SimpleNamespace(grid=grid)
    return SimpleNamespace(system=system, rho=_rho_with_vac(shape, vac))


def _patch_gate(monkeypatch, rho):
    monkeypatch.setattr(
        "gradwave.scf.soft_mode.dominant_screening_eigenvalue",
        lambda *a, **k: SimpleNamespace(eigenvalue=rho, n_iter=3, residual=0.0,
                                        eigenvector=None),
    )


def _patch_build(monkeypatch, op):
    monkeypatch.setattr(sc, "build_woodbury_subspace", lambda *a, **k: op)


def test_abstain_below_threshold(monkeypatch):
    # bulk-Al-like ρ(M)=0.82 < 2.0 → abstain, no operator, base precond kept.
    _patch_gate(monkeypatch, 0.82)
    built = {"n": 0}
    monkeypatch.setattr(sc, "build_woodbury_subspace",
                        lambda *a, **k: built.__setitem__("n", built["n"] + 1))
    res = _fake_res()
    cache = Chi0PrecondCache()
    cache.update(res, xc=None, nspin=1)
    assert cache.decided and not cache.engaged
    assert abs(cache.rho - 0.82) < 1e-9
    assert built["n"] == 0                     # never built below the gate
    assert cache.operator_for(res.system, 1) is None


def test_engage_above_threshold_and_reuse(monkeypatch):
    # slab-like ρ(M)=8.0 ≥ 2.0 → engage; the operator is returned on a matching
    # grid and reused (never rebuilt).
    _patch_gate(monkeypatch, 8.0)
    sentinel = object()
    calls = {"n": 0}

    def _build(*a, **k):
        calls["n"] += 1
        return sentinel

    monkeypatch.setattr(sc, "build_woodbury_subspace", _build)
    res = _fake_res(ng=17)
    cache = Chi0PrecondCache()
    cache.update(res, xc=None, nspin=1)
    assert cache.engaged and cache.precond is sentinel
    assert calls["n"] == 1
    # reuse on the SAME grid returns the frozen operator
    assert cache.operator_for(res.system, 1) is sentinel
    # a second converged SCF must NOT rebuild (decide-once)
    cache.update(_fake_res(ng=17), xc=None, nspin=1)
    assert calls["n"] == 1


def test_grid_change_invalidates(monkeypatch):
    _patch_gate(monkeypatch, 8.0)
    _patch_build(monkeypatch, object())
    res = _fake_res(shape=(4, 4, 4), ng=17)
    cache = Chi0PrecondCache()
    cache.update(res, xc=None, nspin=1)
    assert cache.operator_for(res.system, 1) is cache.precond
    # a different grid (vc-relax rebuilt the FFT box) → fall back to base precond
    other = _fake_res(shape=(6, 6, 6), ng=23)
    assert cache.operator_for(other.system, 1) is None
    # a spin-count change also invalidates
    assert cache.operator_for(res.system, 2) is None


def test_insulator_build_returns_none_abstains(monkeypatch):
    # ρ over threshold but build returns None (no Fermi-surface weight) → abstain
    _patch_gate(monkeypatch, 5.0)
    _patch_build(monkeypatch, None)
    res = _fake_res()
    cache = Chi0PrecondCache()
    cache.update(res, xc=None, nspin=1)
    assert cache.decided and not cache.engaged
    assert cache.operator_for(res.system, 1) is None


def test_threshold_constant_sits_in_the_crux_gap():
    # bulk fcc-Al ρ=0.82  <  gate  <  Al(100) slab ρ=7.89 (crux stage0)
    assert 0.82 < _CHI0_ENGAGE_RHO < 7.89


# --- zero-FFT vacuum-fraction pre-gate (GAP 3) ---

def _gate_must_not_be_called(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("ρ(M) gate paid despite a clear pre-gate decision")

    monkeypatch.setattr(
        "gradwave.scf.soft_mode.dominant_screening_eigenvalue", _boom)


def test_vacuum_fraction_measures_emptiness():
    # exactly 40% of the grid at zero density → vacuum_fraction == 0.40
    res = _fake_res(vac=0.40)
    assert abs(vacuum_fraction(res) - 0.40) < 1e-9
    # a strictly-positive-everywhere (bulk-like) density → 0
    res.rho = torch.ones_like(res.rho)
    assert vacuum_fraction(res) == 0.0


def test_pregate_abstains_on_bulk_without_paying_rho(monkeypatch):
    # clear bulk (vac below the abstain threshold): abstain at ZERO FFT, and the
    # FFT-heavy ρ(M) gate is never called.
    _gate_must_not_be_called(monkeypatch)
    built = {"n": 0}
    monkeypatch.setattr(sc, "build_woodbury_subspace",
                        lambda *a, **k: built.__setitem__("n", built["n"] + 1))
    cache = Chi0PrecondCache()
    cache.update(_fake_res(vac=_VAC_ABSTAIN_BELOW / 2), xc=None, nspin=1)
    assert cache.decided and not cache.engaged
    assert cache.rho is None          # ρ(M) never evaluated
    assert cache.gate_ffts == 0
    assert built["n"] == 0


def test_pregate_engages_on_slab_without_paying_rho(monkeypatch):
    # clear slab (vac above the engage threshold): build directly, ρ(M) skipped.
    _gate_must_not_be_called(monkeypatch)
    sentinel = object()
    monkeypatch.setattr(sc, "build_woodbury_subspace", lambda *a, **k: sentinel)
    cache = Chi0PrecondCache()
    res = _fake_res(vac=min(0.4, _VAC_ENGAGE_ABOVE * 2))
    cache.update(res, xc=None, nspin=1)
    assert cache.engaged and cache.precond is sentinel
    assert cache.rho is None          # ρ(M) never evaluated
    assert cache.operator_for(res.system, 1) is sentinel


def test_pregate_slab_but_insulator_abstains(monkeypatch):
    # high vacuum (a slab) yet the subspace carries no Fermi weight (insulator
    # slab): build returns None → abstain, still no ρ(M) paid.
    _gate_must_not_be_called(monkeypatch)
    monkeypatch.setattr(sc, "build_woodbury_subspace", lambda *a, **k: None)
    cache = Chi0PrecondCache()
    cache.update(_fake_res(vac=0.4), xc=None, nspin=1)
    assert cache.decided and not cache.engaged
    assert cache.operator_for(_fake_res().system, 1) is None
