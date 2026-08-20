"""Toeplitz small-cell fast path extended to (2) the non-batched HamiltonianK
apply and (3) the meta-GGA τ-operator. Both are exact algebraic reformulations of
the FFT path, so they must reproduce it to machine precision.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import gradwave.core.batch as batch
from gradwave.core.metagga import apply_tau_toeplitz, build_tau_toeplitz, metagga_tau_operator
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.implicit import _hamiltonians
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe" / "pseudos"


@pytest.fixture(autouse=True)
def _enable_toeplitz():
    """The single-k/τ Toeplitz paths activate only when the mode is forced "on"
    (the measured auto-gate covers BatchedHamiltonian only); these tests
    exercise them, so force the mode and restore the default afterwards."""
    old = batch._TOEPLITZ_MODE
    batch._TOEPLITZ_MODE = "on"
    yield
    batch._TOEPLITZ_MODE = old


def _si_res(kmesh=(2, 2, 2)):
    si = parse_upf(str(FIX / "Si_ONCV_PBE-1.2.upf"))
    a = 5.43
    lat = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    pos = np.array([[0.0, 0, 0], [a / 4] * 3])
    sysm = setup_system(lat, pos, [0, 0], [si], ecut=18 * RY, kmesh=kmesh,
                        nbands=8, use_symmetry=False)
    return scf(sysm, PBE(), smearing="none", etol=1e-9, rhotol=1e-8,
               max_iter=100, verbose=False)


def test_hamiltoniank_toeplitz_matches_fft():
    """HamiltonianK.apply: the Toeplitz local GEMM equals the FFT scatter/gather."""
    hk = _hamiltonians(_si_res())[0]
    assert hk._toep_eligible, "small cell should be Toeplitz-eligible"
    npw = int(hk.sphere.npw)
    torch.manual_seed(0)
    c = torch.randn(8, npw, dtype=CDTYPE)
    out_toep = hk.apply(c)
    hk._toep_eligible = False          # force the FFT path on the same operator
    out_fft = hk.apply(c)
    assert torch.allclose(out_toep, out_fft, atol=1e-12, rtol=0), \
        f"max|Δ|={(out_toep - out_fft).abs().max().item():.2e}"


def test_tau_operator_toeplitz_matches_fft():
    """meta-GGA τ-operator: the weighted-Toeplitz GEMM equals the 6-FFT round-trip
    for any real v_τ (an algebraic identity, not converged physics)."""
    res = _si_res()
    bk = res.system.batch
    shape = res.system.grid.shape
    nk, npw = bk.mask.shape
    torch.manual_seed(1)
    v_tau = torch.randn(*shape, dtype=RDTYPE)
    c = torch.randn(nk, 8, npw, dtype=CDTYPE) * bk.mask[:, None, :]
    ref = metagga_tau_operator(c, v_tau, bk, shape)
    vt = build_tau_toeplitz(v_tau, bk, shape)
    assert vt is not None
    out = apply_tau_toeplitz(vt, c, bk)
    assert torch.allclose(out, ref, atol=1e-11, rtol=0), \
        f"max|Δ|={(out - ref).abs().max().item():.2e}"


def test_tau_toeplitz_memory_gate():
    res = _si_res((1, 1, 1))
    bk = res.system.batch
    shape = res.system.grid.shape
    old = batch._TOEPLITZ_M_BUDGET_BYTES
    try:
        batch._TOEPLITZ_M_BUDGET_BYTES = 1
        assert build_tau_toeplitz(torch.randn(*shape, dtype=RDTYPE), bk, shape) is None
    finally:
        batch._TOEPLITZ_M_BUDGET_BYTES = old
