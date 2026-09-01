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
from gradwave.scf.subspace_chi0 import _CHI0_ENGAGE_RHO, Chi0PrecondCache


def _fake_res(shape=(4, 4, 4), ng=17):
    """Minimal stand-in for an SCFResult carrying only what the cache reads."""
    mask = torch.zeros(shape, dtype=torch.bool)
    flat = mask.reshape(-1)
    flat[:ng] = True
    grid = SimpleNamespace(shape=shape, dens_mask=mask)
    system = SimpleNamespace(grid=grid)
    return SimpleNamespace(system=system)


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
