"""Toeplitz small-cell local-potential fast path (core/batch.py).

On the wavefunction G-sphere the local term V(r)·ψ(r) is EXACTLY the convolution
out(G_i)=Σ_j V̂(G_i−G_j) c(G_j) = M @ c, so the Toeplitz GEMM path must reproduce
the FFT scatter/gather path to machine precision for ANY real v_eff (the identity
is a property of the discretized operator, not of converged physics — a random
potential is a stronger test than a physical one). These gates lock that in and
check the memory/flag gating that keeps the path confined to small cells.
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
def _enable_toeplitz():
    """The Toeplitz local path is opt-in (default off — it regresses routine
    symmetry-on insulator SCFs); these tests exercise it, so enable it here and
    restore the default afterwards. Tests that toggle the flag themselves still
    save/restore the current (enabled) value."""
    old = batch._TOEPLITZ_LOCAL_ENABLED
    batch._TOEPLITZ_LOCAL_ENABLED = True
    yield
    batch._TOEPLITZ_LOCAL_ENABLED = old


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
    assert h._toep_idx is not None, "small cell should be Toeplitz-eligible"
    assert not bk.mask.all(), "test needs padded slots to exercise masking"

    nk, npw = bk.mask.shape
    torch.manual_seed(1)
    # random in EVERY slot, including padding: both paths must ignore padded
    # columns (FFT via the trash index, Toeplitz via the input mask)
    c = torch.randn(nk, 8, npw, dtype=CDTYPE)

    out_toep = h.apply(c)
    h._toep_idx = None  # force the FFT scatter/gather path on the same operator
    out_fft = h.apply(c)

    assert torch.allclose(out_toep, out_fft, atol=1e-12, rtol=0), \
        f"max|Δ|={(out_toep - out_fft).abs().max().item():.2e}"


def test_toeplitz_is_hermitian_matrix():
    # V real ⇒ M[i,j]=V̂(G_i−G_j) Hermitian; the assembled H must be too.
    system = _si_system(kmesh=(1, 1, 1))
    h, _ = _random_veff_H(system)
    M = h._toeplitz_M(CDTYPE)  # (nk, npw, npw)
    err = (M - M.conj().transpose(-1, -2)).abs().max() / M.abs().max()
    assert err < 1e-13, f"M not Hermitian: {err.item():.2e}"


def test_memory_gate_disables_large_or_budget():
    system = _si_system()
    # tiny budget ⇒ ineligible even for a small cell
    old = batch._TOEPLITZ_M_BUDGET_BYTES
    try:
        batch._TOEPLITZ_M_BUDGET_BYTES = 1
        h, _ = _random_veff_H(system)
        assert h._toep_idx is None
    finally:
        batch._TOEPLITZ_M_BUDGET_BYTES = old


def test_disabled_flag_falls_back_to_fft():
    system = _si_system(kmesh=(1, 1, 1))
    old = batch._TOEPLITZ_LOCAL_ENABLED
    try:
        batch._TOEPLITZ_LOCAL_ENABLED = False
        h, bk = _random_veff_H(system)
        assert h._toep_idx is None  # gated off ⇒ pure FFT path
    finally:
        batch._TOEPLITZ_LOCAL_ENABLED = old
    # and the FFT-only operator still applies correctly vs a Toeplitz one
    batch._TOEPLITZ_LOCAL_ENABLED = True
    h2, _ = _random_veff_H(system)
    nk, npw = bk.mask.shape
    torch.manual_seed(2)
    c = torch.randn(nk, 8, npw, dtype=CDTYPE) * bk.mask[:, None, :]
    out_on = h2.apply(c)
    h2._toep_idx = None
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
    assert h._toep_idx is None
