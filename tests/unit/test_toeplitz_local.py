"""Toeplitz small-cell local-potential fast path (core/batch.py).

On the wavefunction G-sphere the local term V(r)·ψ(r) is EXACTLY the convolution
out(G_i)=Σ_j V̂(G_i−G_j) c(G_j) = M @ c, so the Toeplitz GEMM path must reproduce
the FFT scatter/gather path to machine precision for ANY real v_eff (the identity
is a property of the discretized operator, not of converged physics — a random
potential is a stronger test than a physical one). These gates lock that in and
check the mode/memory gating plus the measured per-geometry verdict that decides
when the path actually runs.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import gradwave.core.batch as batch
from gradwave.core.batch import BatchedHamiltonian, projectors_b
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import setup_system
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


@pytest.fixture(autouse=True)
def _force_toeplitz():
    """Force the Toeplitz path ("on" bypasses the timing trial — a timed
    verdict is machine-dependent and would make these gates flaky) and start
    each test with a clean verdict cache. Tests that set the mode themselves
    still save/restore the current (forced) value."""
    old = batch._TOEPLITZ_MODE
    batch._TOEPLITZ_MODE = "on"
    batch._TOEP_VERDICT.clear()
    yield
    batch._TOEPLITZ_MODE = old
    batch._TOEP_VERDICT.clear()


def _si_system(kmesh=(2, 2, 2), ecut=12):
    si = parse_upf(str(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf"))
    a = 5.43
    lattice = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    # rattled off-symmetry positions so k-points carry different npw (padding)
    pos = np.array([[0.02, 0.01, -0.03], [0.24, 0.26, 0.27]])
    return setup_system(lattice, pos, [0, 0], [si], ecut=ecut * RY,
                        kmesh=kmesh, nbands=8, use_symmetry=False)


def _random_veff_H(system):
    bk = system.batch
    shape = system.grid.shape
    projs = projectors_b(bk, system.positions)
    torch.manual_seed(0)
    v_eff = torch.randn(*shape, dtype=RDTYPE)  # arbitrary real potential
    return BatchedHamiltonian(bk, shape, v_eff, projs), bk


def test_toeplitz_matches_fft_bit_exact():
    system = _si_system()
    h, bk = _random_veff_H(system)
    assert h._toep_eligible, "small cell should be Toeplitz-eligible"
    assert not bk.mask.all(), "test needs padded slots to exercise masking"

    nk, npw = bk.mask.shape
    torch.manual_seed(1)
    # random in EVERY slot, including padding: both paths must ignore padded
    # columns (FFT via the trash index, Toeplitz via the input mask)
    c = torch.randn(nk, 8, npw, dtype=CDTYPE)

    out_toep = h.apply(c)
    h._toep_eligible = False  # force the FFT path on the same operator
    out_fft = h.apply(c)

    assert torch.allclose(out_toep, out_fft, atol=1e-12, rtol=0), \
        f"max|Δ|={(out_toep - out_fft).abs().max().item():.2e}"


def test_auto_verdict_caches_and_agrees():
    """"auto" mode: the first apply runs the timing trial, caches a verdict
    per geometry signature, and BOTH verdict outcomes give the same H·c (the
    two paths are the same operator, so whichever the clock picks is right)."""
    system = _si_system()
    old = batch._TOEPLITZ_MODE
    try:
        batch._TOEPLITZ_MODE = "auto"
        h, bk = _random_veff_H(system)
        nk, npw = bk.mask.shape
        torch.manual_seed(3)
        c = torch.randn(nk, 8, npw, dtype=CDTYPE)
        out_auto = h.apply(c)
        assert len(batch._TOEP_VERDICT) == 1  # trial ran exactly once
        key, verdict = next(iter(batch._TOEP_VERDICT.items()))
        assert key == ("cpu", nk, npw, h.shape, CDTYPE)
        out_auto2 = h.apply(c)  # cached verdict, no second trial
        assert len(batch._TOEP_VERDICT) == 1
        assert torch.allclose(out_auto, out_auto2, atol=1e-12, rtol=0)
        # losing verdict must free the trial's cached tensors
        if not verdict:
            assert h._toep_idx is None and not h._toep_M_cache
    finally:
        batch._TOEPLITZ_MODE = old


def test_toeplitz_is_hermitian_matrix():
    # V real ⇒ M[i,j]=V̂(G_i−G_j) Hermitian; the assembled H must be too.
    system = _si_system(kmesh=(1, 1, 1))
    h, _ = _random_veff_H(system)
    M = h._toeplitz_M(CDTYPE)  # (nk, npw, npw)
    err = (M - M.conj().transpose(-1, -2)).abs().max() / M.abs().max()
    assert err < 1e-13, f"M not Hermitian: {err.item():.2e}"


def test_idx_cached_on_batchedk_across_rebuilds():
    """The geometry-only difference-index table is built once per BatchedK and
    shared across per-iteration Hamiltonian rebuilds (the old per-ctor build
    cost ~one FFT apply per iteration for nothing)."""
    system = _si_system(kmesh=(1, 1, 1))
    h1, bk = _random_veff_H(system)
    idx1 = h1._toeplitz_idx()
    h2, _ = _random_veff_H(system)
    assert h2._toeplitz_idx() is idx1  # same tensor object, not a rebuild


def test_memory_gate_disables_large_or_budget():
    system = _si_system()
    # tiny budget ⇒ ineligible even for a small cell
    old = batch._TOEPLITZ_M_BUDGET_BYTES
    try:
        batch._TOEPLITZ_M_BUDGET_BYTES = 1
        h, _ = _random_veff_H(system)
        assert not h._toep_eligible
    finally:
        batch._TOEPLITZ_M_BUDGET_BYTES = old


def test_disabled_mode_falls_back_to_fft():
    system = _si_system(kmesh=(1, 1, 1))
    old = batch._TOEPLITZ_MODE
    try:
        batch._TOEPLITZ_MODE = "off"
        h, bk = _random_veff_H(system)
        assert not h._toep_eligible  # gated off ⇒ pure FFT path
    finally:
        batch._TOEPLITZ_MODE = old
    # and the FFT-only operator still applies correctly vs a Toeplitz one
    h2, _ = _random_veff_H(system)
    nk, npw = bk.mask.shape
    torch.manual_seed(2)
    c = torch.randn(nk, 8, npw, dtype=CDTYPE) * bk.mask[:, None, :]
    out_on = h2.apply(c)
    h2._toep_eligible = False
    out_off = h2.apply(c)
    assert torch.allclose(out_on, out_off, atol=1e-12, rtol=0)


@pytest.mark.parametrize("smooth_shape", [(6, 6, 6)])
def test_uspp_dual_grid_not_eligible(smooth_shape):
    # v1 confines the path to the plain box; the USPP dual grid stays on FFT.
    system = _si_system(kmesh=(1, 1, 1))
    bk = system.batch
    shape = system.grid.shape
    projs = projectors_b(bk, system.positions)
    v_eff = torch.randn(*shape, dtype=RDTYPE)
    n_smooth = smooth_shape[0] * smooth_shape[1] * smooth_shape[2]
    flat = bk.flat_idx % n_smooth  # in-range stand-in smooth mapping
    v_smooth = torch.randn(*smooth_shape, dtype=RDTYPE)
    h = BatchedHamiltonian(bk, shape, v_eff, projs,
                           smooth=(smooth_shape, flat, v_smooth))
    assert not h._toep_eligible


def test_reindexed_bk_never_reuses_foreign_idx_table():
    """A BatchedK derived by dataclasses.replace with a different flat_idx (the
    k+q reindex in postscf/dfpt_q) must not apply the parent's difference-index
    table: it encodes the PARENT's spheres, so reusing it is silently wrong
    physics — the CI-caught failure mode was the q≠0 Sternheimer CG diverging
    to NaN whenever the preceding SCF's timing trial had populated the cache.
    Two layers under test: _reindex_bk drops the cache outright, and
    _toeplitz_idx revalidates a hit against its own gather_idx even when a
    sharing site hands it a foreign dict."""
    from gradwave.postscf.dfpt_q import _reindex_bk

    system = _si_system()
    h, bk = _random_veff_H(system)
    nk, npw = bk.mask.shape
    torch.manual_seed(4)
    c = torch.randn(nk, 8, npw, dtype=CDTYPE)
    h.apply(c)  # populates bk.toep_idx_cache for this grid shape

    perm = torch.arange(nk - 1, -1, -1)  # reversed k order: rows all move
    bk_r = _reindex_bk(bk, perm)
    assert bk_r.toep_idx_cache is None  # layer 1: the reindex drops the cache
    assert not torch.equal(bk_r.flat_idx, bk.flat_idx)  # rows really differ

    # layer 2: even a sharing site (the parent's dict re-attached) must not
    # let the reindexed operator trust the parent's table
    bk_r.toep_idx_cache = bk.toep_idx_cache
    h_r = BatchedHamiltonian(bk_r, system.grid.shape, h.v_eff_r,
                             projectors_b(bk_r, system.positions))
    out_toep = h_r.apply(c)
    h_r._toep_eligible = False  # force the FFT path on the same operator
    out_fft = h_r.apply(c)
    assert torch.allclose(out_toep, out_fft, atol=1e-12, rtol=0), \
        f"max|Δ|={(out_toep - out_fft).abs().max().item():.2e}"
