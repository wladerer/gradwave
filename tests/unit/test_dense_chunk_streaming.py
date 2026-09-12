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
