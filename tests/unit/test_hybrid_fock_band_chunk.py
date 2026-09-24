"""Band-chunking the hybrid Fock apply reproduces the single-pass build.

``GammaFockExchange`` / ``MultiKFockExchange`` no longer allocate two full dense
boxes over all bands in their SCF ``fock`` hook: the ``nb`` axis is chunked via
``core.batch._dense_band_chunk`` (the shared dense-box budget), holding one band
slice plus the sphere-space output instead of 2× the full box. The ACE apply
(Σᵢ ξᵢ⟨ξᵢ|f⟩) and ``box_to_sphere_b`` are per-band linear with no cross-band
reduction, so forcing a tiny ``GRADWAVE_CPU_DENSE_BUDGET`` (multi-chunk) must
reproduce the unforced single-pass output.

There is no cross-band reduction, so the chunked and single-pass results agree
mathematically; the only difference is that the ACE matmul's shape (and, under
xdist, the BLAS thread count) changes with the chunk, reordering the GEMM's
contraction at ~1 ULP (measured max_rel ≈ 1e-16, well inside the ~1e-13 gate) —
the same floating-point class as ``density_b``'s per-chunk band sum. Asserted
with a tight ``allclose`` rather than ``torch.equal``: bit-for-bit equality holds
in a single-threaded run but not across xdist worker thread counts.

Fast-tier: exercises the exact production closures against a synthetic ACE +
minimal ``BatchedK`` — no SCF solve.
"""

from __future__ import annotations

import torch

from gradwave.core.batch import BatchedK, _dense_band_chunk
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf.exchange import ACEExchange
from gradwave.postscf.hybrid import GammaFockExchange, MultiKFockExchange

torch.manual_seed(0)

SHAPE = (8, 8, 8)
N_GRID = SHAPE[0] * SHAPE[1] * SHAPE[2]
NK, NB, NPW = 3, 6, 20
N_OCC = 4
# Forces >=2 chunks over the NB bands on CPU (see _dense_band_chunk arithmetic).
TINY_BUDGET = "5e4"


def _bk() -> BatchedK:
    """Minimal BatchedK with distinct per-k box indices for the G<->r transforms."""
    flat = torch.stack([
        torch.randperm(N_GRID)[:NPW] for _ in range(NK)
    ]).to(torch.int64)
    return BatchedK(
        npw=torch.full((NK,), NPW, dtype=torch.int64),
        mask=torch.ones(NK, NPW, dtype=torch.bool),
        flat_idx=flat,
        kpg=torch.zeros(NK, NPW, 3, dtype=RDTYPE),
        t=torch.zeros(NK, NPW, dtype=RDTYPE),
        proj_phase_free=torch.zeros(NK, 0, NPW, dtype=CDTYPE),
        proj_atom_index=torch.zeros(0, dtype=torch.int64),
        dij_full=torch.zeros((0, 0), dtype=RDTYPE),
    )


def _ace(n_occ: int) -> ACEExchange:
    xi = torch.randn(N_GRID, n_occ, dtype=CDTYPE)
    return ACEExchange(xi=xi, volume=3.7, n_r=N_GRID)


def _forced_vs_unforced(apply_delta, c, monkeypatch):
    """(unforced single-pass output, forced multi-chunk output) for one hook."""
    dev = torch.device("cpu")
    monkeypatch.delenv("GRADWAVE_CPU_DENSE_BUDGET", raising=False)
    assert _dense_band_chunk(N_GRID, NK, dev, 16) >= NB  # unforced => single pass
    ref = apply_delta(c)

    monkeypatch.setenv("GRADWAVE_CPU_DENSE_BUDGET", TINY_BUDGET)
    assert _dense_band_chunk(N_GRID, NK, dev, 16) < NB   # forced => multi-chunk
    got = apply_delta(c)
    return got, ref


def test_gamma_fock_apply_band_chunk_matches(monkeypatch):
    c = torch.randn(NK, NB, NPW, dtype=CDTYPE)
    op = GammaFockExchange(alpha=0.25)
    got, ref = _forced_vs_unforced(op._apply_for(_ace(N_OCC), _bk(), SHAPE), c, monkeypatch)
    # no cross-band reduction — mathematically identical, GEMM reorder only (~1 ULP)
    assert torch.allclose(got, ref, rtol=1e-12, atol=1e-15)


def test_multik_fock_apply_band_chunk_matches(monkeypatch):
    c = torch.randn(NK, NB, NPW, dtype=CDTYPE)
    op = MultiKFockExchange(alpha=0.25)
    # per-k ACE, including an empty-orbital k (a k with no occupied bands)
    ace_per_k = [_ace(N_OCC), _ace(0), _ace(N_OCC + 1)]
    got, ref = _forced_vs_unforced(op._apply_for(ace_per_k, _bk(), SHAPE), c, monkeypatch)
    assert torch.allclose(got, ref, rtol=1e-12, atol=1e-15)
