"""Memory-streaming SCF paths (campaign: kill the SCF memory hog).

Three contracts, all fast-tier (small tensors, no real SCF):

  * ``core.batch.projectors_b`` — the no-grad memory-light per-k path is
    BIT-IDENTICAL to the batched all-k gather (no cross-k reduction), and the
    grad path (forces / alchemical rebuild) is preserved and differentiable.
  * ``core.energies.local_pp.local_potential_g`` — the no-grad per-atom
    accumulation matches the batched einsum to ~ulp (the reduction order over
    atoms is unchanged; only the internal blocking differs), and the grad path
    stays differentiable.
  * ``scf.loop._auto_cpu_dense_budget`` + ``core.batch`` override plumbing — the
    estimator fails open (None) for small cells / non-CPU / explicit env, and
    returns a clamped budget for a large cell; the override is consulted only
    below the explicit env var.
"""

from __future__ import annotations

import pytest
import torch

from gradwave.core.batch import (
    BatchedK,
    _cpu_dense_budget_bytes,
    _dense_band_chunk,
    g_to_r_b,
    projectors_b,
    set_cpu_dense_budget_override,
)
from gradwave.dtypes import CDTYPE, RDTYPE

torch.manual_seed(0)


def _mk_bk(nk=3, nproj=5, m=7, na=4):
    """A minimal BatchedK carrying only the fields projectors_b touches."""
    kpg = torch.randn(nk, m, 3, dtype=RDTYPE)
    pf = torch.randn(nk, nproj, m, dtype=CDTYPE)
    atom_index = torch.randint(0, na, (nproj,), dtype=torch.int64)
    z_i = torch.zeros(0, dtype=torch.int64)
    return BatchedK(
        npw=torch.full((nk,), m, dtype=torch.int64),
        mask=torch.ones(nk, m, dtype=torch.bool),
        flat_idx=torch.zeros(nk, m, dtype=torch.int64),
        kpg=kpg,
        t=torch.zeros(nk, m, dtype=RDTYPE),
        proj_phase_free=pf,
        proj_atom_index=atom_index,
        dij_full=torch.zeros((nproj, nproj), dtype=RDTYPE),
    ), na, z_i


# ---------------------------------------------------------------------------
# projectors_b
# ---------------------------------------------------------------------------

def test_projectors_b_nograd_stream_is_bit_identical():
    bk, na, _ = _mk_bk()
    pos = torch.randn(na, 3, dtype=RDTYPE)
    # reference: the batched all-k gather (forced via the grad branch)
    with torch.enable_grad():
        posg = pos.clone().requires_grad_(True)
        ref = projectors_b(bk, posg).detach()
    # no-grad memory-light path
    with torch.no_grad():
        got = projectors_b(bk, pos)
    # per-k independent, same indexing/multiply → bit-identical (atol=0)
    assert torch.equal(got, ref)


def test_projectors_b_grad_path_differentiable():
    bk, na, _ = _mk_bk()
    pos = torch.randn(na, 3, dtype=RDTYPE, requires_grad=True)
    out = projectors_b(bk, pos)
    assert out.requires_grad
    out.abs().sum().backward()
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()


def test_projectors_b_empty_projectors_shortcircuits():
    bk, _, _ = _mk_bk(nproj=0)
    pos = torch.randn(4, 3, dtype=RDTYPE)
    out = projectors_b(bk, pos)
    assert out.shape[1] == 0


# ---------------------------------------------------------------------------
# g_to_r_b band-chunk (campaign: bound the dense-box construction transient)
#
# The optional band-chunk fills the returned box a band-slice at a time under a
# forced GRADWAVE_CPU_DENSE_BUDGET. g_to_r_b has NO cross-band reduction, so the
# chunked build must be BIT-IDENTICAL to the unforced single-shot build, and the
# default (no budget) path must stay differentiable.
# ---------------------------------------------------------------------------

def _mk_bk_grid(nk=2, m=13, shape=(6, 5, 4)):
    """A minimal BatchedK whose flat_idx scatters a sphere into an FFT box of
    ``shape`` (distinct in-range indices, so g_to_r_b/g_to_r_box are exercised)."""
    n = shape[0] * shape[1] * shape[2]
    assert m <= n
    flat = torch.stack([torch.randperm(n)[:m] for _ in range(nk)]).to(torch.int64)
    nproj = 0
    return BatchedK(
        npw=torch.full((nk,), m, dtype=torch.int64),
        mask=torch.ones(nk, m, dtype=torch.bool),
        flat_idx=flat,
        kpg=torch.zeros(nk, m, 3, dtype=RDTYPE),
        t=torch.zeros(nk, m, dtype=RDTYPE),
        proj_phase_free=torch.zeros(nk, nproj, m, dtype=CDTYPE),
        proj_atom_index=torch.zeros(nproj, dtype=torch.int64),
        dij_full=torch.zeros((nproj, nproj), dtype=RDTYPE),
    )


def _budget_for_chunk(target: int, n: int, nk: int, elem_bytes: int) -> float:
    """A CPU dense budget that makes ``_dense_band_chunk`` return ~``target``
    bands per chunk for this (n, nk, elem_bytes) — used to force a genuine
    multi-slice build in the tests below regardless of the tiny grid sizes."""
    return float(target * elem_bytes * n * nk)


def test_g_to_r_b_chunked_is_bit_identical():
    shape = (6, 5, 4)
    nk, nb, m = 2, 32, 13
    n = shape[0] * shape[1] * shape[2]
    bk = _mk_bk_grid(nk=nk, m=m, shape=shape)
    coeffs = torch.randn(nk, nb, m, dtype=CDTYPE)

    # reference: no budget -> single-shot historical path
    set_cpu_dense_budget_override(None)
    ref = g_to_r_b(coeffs, bk, shape)

    # forced budget sized to ~4 bands/chunk -> multi-band-slice build; must
    # engage AND match exactly
    set_cpu_dense_budget_override(_budget_for_chunk(4, n, nk, coeffs.element_size()))
    chunk = _dense_band_chunk(n, nk, coeffs.device, coeffs.element_size())
    assert 1 <= chunk < nb  # the budget really splits the band axis (>1 chunk)
    got = g_to_r_b(coeffs, bk, shape)

    assert got.shape == ref.shape
    assert torch.equal(got, ref)  # no cross-band reduction -> bit-identical


def test_g_to_r_b_complex64_chunked_bit_identical():
    shape = (5, 4, 4)
    nk, nb, m = 1, 20, 11
    n = shape[0] * shape[1] * shape[2]
    bk = _mk_bk_grid(nk=nk, m=m, shape=shape)
    coeffs = torch.randn(nk, nb, m, dtype=torch.complex64)
    set_cpu_dense_budget_override(None)
    ref = g_to_r_b(coeffs, bk, shape)
    set_cpu_dense_budget_override(_budget_for_chunk(3, n, nk, coeffs.element_size()))
    chunk = _dense_band_chunk(n, nk, coeffs.device, coeffs.element_size())
    assert 1 <= chunk < nb
    got = g_to_r_b(coeffs, bk, shape)
    assert torch.equal(got, ref)


def test_g_to_r_b_default_path_differentiable():
    shape = (4, 4, 4)
    nk, nb, m = 1, 4, 7
    bk = _mk_bk_grid(nk=nk, m=m, shape=shape)
    set_cpu_dense_budget_override(None)  # default single-shot path
    coeffs = torch.randn(nk, nb, m, dtype=CDTYPE, requires_grad=True)
    out = g_to_r_b(coeffs, bk, shape)
    assert out.requires_grad
    out.abs().pow(2).sum().backward()
    assert coeffs.grad is not None and torch.isfinite(coeffs.grad).all()


# ---------------------------------------------------------------------------
# local_potential_g
# ---------------------------------------------------------------------------

def test_local_potential_g_nograd_matches_batched_to_ulp():
    from gradwave.core.energies.local_pp import local_potential_g

    na, shape = 4, (3, 4, 5)
    pos = torch.randn(na, 3, dtype=RDTYPE)
    species_index = torch.randint(0, 2, (na,), dtype=torch.int64)
    vloc = torch.randn(2, *shape, dtype=RDTYPE)
    g_cart = torch.randn(*shape, 3, dtype=RDTYPE)
    with torch.enable_grad():
        posg = pos.clone().requires_grad_(True)
        ref = local_potential_g(posg, species_index, vloc, g_cart, 12.0).detach()
    with torch.no_grad():
        got = local_potential_g(pos, species_index, vloc, g_cart, 12.0)
    # per-atom accumulation reorders the atom sum only → agree to ~ulp
    assert torch.allclose(got, ref, rtol=0, atol=1e-12)


def test_local_potential_g_grad_path_differentiable():
    from gradwave.core.energies.local_pp import local_potential_g

    na, shape = 3, (3, 3, 3)
    pos = torch.randn(na, 3, dtype=RDTYPE, requires_grad=True)
    species_index = torch.zeros(na, dtype=torch.int64)
    vloc = torch.randn(1, *shape, dtype=RDTYPE)
    g_cart = torch.randn(*shape, 3, dtype=RDTYPE)
    v = local_potential_g(pos, species_index, vloc, g_cart, 8.0)
    assert v.requires_grad
    v.abs().sum().backward()
    assert pos.grad is not None and torch.isfinite(pos.grad).all()


# ---------------------------------------------------------------------------
# the auto CPU dense-box budget estimator + override plumbing
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("GRADWAVE_CPU_DENSE_BUDGET", raising=False)
    monkeypatch.delenv("GRADWAVE_CPU_DENSE_AUTO", raising=False)
    set_cpu_dense_budget_override(None)
    yield
    set_cpu_dense_budget_override(None)


def _est(**kw):
    from gradwave.scf.loop import _auto_cpu_dense_budget

    defaults = dict(nk=4, nb=128, ngrid=373248, max_dim_factor=4,
                    device=torch.device("cpu"))
    defaults.update(kw)
    return _auto_cpu_dense_budget(**defaults)


def test_estimator_none_on_non_cpu():
    assert _est(device=torch.device("cuda")) is None


def test_estimator_none_when_env_forces_budget(monkeypatch):
    monkeypatch.setenv("GRADWAVE_CPU_DENSE_BUDGET", "5e8")
    assert _est() is None  # explicit env wins; estimator stands down


def test_estimator_none_when_auto_disabled(monkeypatch):
    monkeypatch.setenv("GRADWAVE_CPU_DENSE_AUTO", "off")
    assert _est() is None


def test_estimator_none_for_small_cell(monkeypatch):
    # a tiny grid's unchunked box is well under the trigger fraction → unchunked
    monkeypatch.setattr("gradwave.scf.loop._available_ram_bytes", lambda: 16 << 30)
    assert _est(nk=1, nb=8, ngrid=8000) is None


def test_estimator_engages_and_clamps_for_large_cell(monkeypatch):
    from gradwave.scf.loop import _AUTO_DENSE_MAX, _AUTO_DENSE_MIN

    monkeypatch.setattr("gradwave.scf.loop._available_ram_bytes", lambda: 12 << 30)
    budget = _est(nk=4, nb=128, ngrid=373248, max_dim_factor=4)
    assert budget is not None
    assert _AUTO_DENSE_MIN <= budget <= _AUTO_DENSE_MAX


def test_estimator_none_when_ram_unknown(monkeypatch):
    monkeypatch.setattr("gradwave.scf.loop._available_ram_bytes", lambda: None)
    assert _est() is None


def test_override_resolution_order(monkeypatch):
    # override is used when the env is unset
    set_cpu_dense_budget_override(3e8)
    assert _cpu_dense_budget_bytes() == pytest.approx(3e8)
    # explicit env wins over the override
    monkeypatch.setenv("GRADWAVE_CPU_DENSE_BUDGET", "5e8")
    assert _cpu_dense_budget_bytes() == pytest.approx(5e8)
    monkeypatch.delenv("GRADWAVE_CPU_DENSE_BUDGET")
    # clearing the override restores the unchunked default (None)
    set_cpu_dense_budget_override(None)
    assert _cpu_dense_budget_bytes() is None


def test_override_rejects_nonpositive():
    with pytest.raises(ValueError, match="must be > 0"):
        set_cpu_dense_budget_override(0.0)


def test_dense_band_chunk_honors_override():
    # with no budget the CPU path is unchunked (large sentinel)
    assert _dense_band_chunk(373248, 4, torch.device("cpu"), 16) >= 1_000_000
    # a small override forces a small band chunk
    set_cpu_dense_budget_override(2e8)
    chunk = _dense_band_chunk(373248, 4, torch.device("cpu"), 16)
    assert 1 <= chunk < 1_000_000


# ---------------------------------------------------------------------------
# projector-table dedup: becp_b conj-fold (campaign: one resident projector
# representation). becp_b now folds the conjugation into a BLAS matmul instead
# of consuming a materialized/cached p.conj() table; the result must match the
# old einsum-with-resolved-conjugate contraction to bit level, and stay
# differentiable for the forces / alchemical grad paths.
# ---------------------------------------------------------------------------

def _becp_ref(p, c):
    """The pre-dedup contraction: einsum against a resolved conjugate table."""
    return torch.einsum("kpg,kbg->kbp", p.conj().resolve_conj(), c)


def test_becp_b_matmul_matches_resolved_einsum():
    from gradwave.core.batch import becp_b

    nk, nproj, m, nb = 3, 6, 11, 5
    p = torch.randn(nk, nproj, m, dtype=CDTYPE)
    c = torch.randn(nk, nb, m, dtype=CDTYPE)
    got = becp_b(p, c)
    ref = _becp_ref(p, c)
    # same contraction, conj folded into BLAS — bit-level (well under 1e-14)
    assert torch.allclose(got, ref, rtol=0.0, atol=1e-14)


def test_becp_b_ignores_passed_conj():
    """The retained p_conj argument must not change the result (it is ignored;
    the conj is folded regardless of what is passed)."""
    from gradwave.core.batch import becp_b

    nk, nproj, m, nb = 2, 4, 9, 3
    p = torch.randn(nk, nproj, m, dtype=CDTYPE)
    c = torch.randn(nk, nb, m, dtype=CDTYPE)
    a = becp_b(p, c)
    b = becp_b(p, c, p_conj=torch.zeros_like(p))  # bogus conj — must be ignored
    assert torch.equal(a, b)


def test_becp_b_complex64_matches_reference():
    from gradwave.core.batch import becp_b

    nk, nproj, m, nb = 2, 5, 8, 4
    p = torch.randn(nk, nproj, m, dtype=torch.complex64)
    c = torch.randn(nk, nb, m, dtype=torch.complex64)
    got = becp_b(p, c)
    ref = torch.einsum("kpg,kbg->kbp", p.conj().resolve_conj(), c)
    assert torch.allclose(got, ref, rtol=0.0, atol=1e-6)


def test_becp_b_grad_path_preserved():
    from gradwave.core.batch import becp_b

    nk, nproj, m, nb = 2, 3, 7, 4
    p = torch.randn(nk, nproj, m, dtype=CDTYPE, requires_grad=True)
    c = torch.randn(nk, nb, m, dtype=CDTYPE, requires_grad=True)
    becp_b(p, c).abs().pow(2).sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert c.grad is not None and torch.isfinite(c.grad).all()


def test_tables_no_resident_conj():
    """_tables returns (t, v_eff, p, dij) — four entries, no resident conj."""
    from gradwave.core.batch import BatchedHamiltonian, BatchedK

    nk, nproj, m = 2, 3, 5
    shape = (4, 4, 4)
    n = shape[0] * shape[1] * shape[2]
    bk = BatchedK(
        npw=torch.full((nk,), m, dtype=torch.int64),
        mask=torch.ones(nk, m, dtype=torch.bool),
        flat_idx=torch.arange(m).expand(nk, m).contiguous(),
        kpg=torch.randn(nk, m, 3, dtype=RDTYPE),
        t=torch.rand(nk, m, dtype=RDTYPE),
        proj_phase_free=torch.randn(nk, nproj, m, dtype=CDTYPE),
        proj_atom_index=torch.zeros(nproj, dtype=torch.int64),
        dij_full=torch.randn(nproj, nproj, dtype=RDTYPE),
    )
    p = torch.randn(nk, nproj, m, dtype=CDTYPE)
    v_eff = torch.randn(n, dtype=RDTYPE).reshape(shape)
    h = BatchedHamiltonian(bk, shape, v_eff, p)
    tab = h._tables(CDTYPE)
    assert len(tab) == 4  # (t, v_eff, p, dij) — the conj entry is gone
