"""OpenBLAS zgemm routing for the eager Davidson Rayleigh-Ritz GEMMs.

``solvers/blas_routing.zbmm`` dispatches a batched complex128 matmul to a
ctypes-loaded ``cblas_zgemm`` loop (the ``blas_zbmm`` symbol in the same
``libdavnative.so`` as the native Davidson solver) when applicable, and falls
back to ``torch.matmul`` otherwise. Same math, different kernel — the routed
result equals torch's to last-bit BLAS rounding, and ``davidson_batched``
converges to the identical eigenvalues in the identical iteration count with
routing on vs off. Tests that need the compiled ``.so`` skip cleanly when it is
absent; the env-gating and torch-fallback tests run without it.
"""

from __future__ import annotations

import pytest
import torch

import gradwave.solvers.blas_routing as br
from gradwave.solvers.blas_routing import blas_available, zbmm
from gradwave.solvers.davidson import davidson_batched

needs_blas = pytest.mark.skipif(
    not blas_available(),
    reason="native library not built (scripts/build_native_solver.sh)")

_COMBOS = [
    dict(),
    dict(conj_b_t=True),
    dict(t_a=True),
]


def _rand(*shape, seed=0):
    gen = torch.Generator().manual_seed(seed)
    re = torch.randn(*shape, generator=gen, dtype=torch.float64)
    im = torch.randn(*shape, generator=gen, dtype=torch.float64)
    return torch.complex(re, im)


def _ref(a, b, *, conj_b_t=False, t_a=False):
    op_a = a.transpose(-1, -2) if t_a else a
    op_b = b.conj().transpose(-1, -2) if conj_b_t else b
    return torch.matmul(op_a, op_b)


@pytest.mark.parametrize("kw", _COMBOS)
def test_off_is_exactly_torch(monkeypatch, kw):
    """GRADWAVE_BLAS_GEMM=off never touches the .so and is bit-identical torch."""
    monkeypatch.setenv("GRADWAVE_BLAS_GEMM", "off")
    nk, r, c, inner = 5, 7, 6, 9
    a = _rand(nk, r, inner, seed=1)
    # b shape chosen so the requested op composes: default a@b -> b is (inner,c);
    # conj_b_t a@b^H -> b is (c,inner); t_a a^T@b -> a^T is (inner,r) so b is (r,c).
    if kw.get("t_a"):
        b = _rand(nk, r, c, seed=2)
    elif kw.get("conj_b_t"):
        b = _rand(nk, c, inner, seed=2)
    else:
        b = _rand(nk, inner, c, seed=2)
    got = zbmm(a, b, **kw)
    assert torch.equal(got, _ref(a, b, **kw))


@needs_blas
@pytest.mark.parametrize("kw", _COMBOS)
def test_native_matches_torch(monkeypatch, kw):
    """The routed zgemm equals torch.matmul to last-bit rounding (rtol 1e-12)."""
    monkeypatch.setenv("GRADWAVE_BLAS_GEMM", "on")
    nk, r, c, inner = 8, 13, 11, 17
    a = _rand(nk, r, inner, seed=3)
    if kw.get("t_a"):
        b = _rand(nk, r, c, seed=4)
    elif kw.get("conj_b_t"):
        b = _rand(nk, c, inner, seed=4)
    else:
        b = _rand(nk, inner, c, seed=4)
    got = zbmm(a, b, **kw)
    ref = _ref(a, b, **kw)
    assert got.shape == ref.shape
    assert torch.allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_invalid_env_raises(monkeypatch):
    monkeypatch.setenv("GRADWAVE_BLAS_GEMM", "yes")
    with pytest.raises(ValueError, match="GRADWAVE_BLAS_GEMM"):
        zbmm(_rand(1, 2, 2), _rand(1, 2, 2))


def test_on_without_so_raises(monkeypatch):
    """'on' with no library is a hard error (unlike the silent 'auto' fallback)."""
    monkeypatch.setenv("GRADWAVE_BLAS_GEMM", "on")
    monkeypatch.setenv("GRADWAVE_NATIVE_SO", "/nonexistent/libdavnative.so")
    monkeypatch.setattr(br, "_ZBMM", None)
    monkeypatch.setattr(br, "_ZBMM_TRIED", False)
    with pytest.raises(RuntimeError, match="build_native_solver|GRADWAVE_BLAS_GEMM"):
        zbmm(_rand(1, 2, 2), _rand(1, 2, 2))


def test_auto_without_so_falls_back(monkeypatch):
    """'auto' with no library silently returns the correct torch result."""
    monkeypatch.setenv("GRADWAVE_BLAS_GEMM", "auto")
    monkeypatch.setenv("GRADWAVE_NATIVE_SO", "/nonexistent/libdavnative.so")
    monkeypatch.setattr(br, "_ZBMM", None)
    monkeypatch.setattr(br, "_ZBMM_TRIED", False)
    a, b = _rand(2, 3, 4, seed=5), _rand(2, 4, 3, seed=6)
    got = zbmm(a, b)
    assert torch.equal(got, _ref(a, b))


@needs_blas
def test_non_contiguous_and_wrong_dtype_fall_back(monkeypatch):
    """'on' still falls back (correctly) when operands are outside native scope:
    a non-contiguous view or a complex64 block routes through torch, no error."""
    monkeypatch.setenv("GRADWAVE_BLAS_GEMM", "on")
    a = _rand(2, 4, 4, seed=7)
    b = _rand(2, 4, 4, seed=8)
    # non-contiguous a (transpose view)
    got = zbmm(a.transpose(-1, -2).contiguous().transpose(-1, -2), b)
    assert torch.allclose(got, _ref(a.transpose(-1, -2).contiguous().transpose(-1, -2), b))
    # complex64 operands
    got64 = zbmm(a.to(torch.complex64), b.to(torch.complex64))
    assert got64.dtype == torch.complex64
    assert torch.allclose(got64, _ref(a.to(torch.complex64), b.to(torch.complex64)))


def _hermitian_operator(m, seed):
    gen = torch.Generator().manual_seed(seed)
    a = torch.complex(
        torch.randn(m, m, generator=gen, dtype=torch.float64),
        torch.randn(m, m, generator=gen, dtype=torch.float64))
    # add a diagonal spread so eigenvalues are well separated (stable n_iter)
    diag = torch.diag(torch.arange(m, dtype=torch.float64)).to(a.dtype)
    return a + a.conj().T + diag


def _run_davidson(hmat, x0, t, mask):
    def apply_H(c):
        return torch.einsum("ij,kbj->kbi", hmat, c)

    return davidson_batched(apply_H, x0, t, mask, tol=1e-9, max_iter=60)


@needs_blas
def test_davidson_batched_ab_identical(monkeypatch):
    """davidson_batched converges to the SAME eigenvalues (1e-10) in the SAME
    iteration count with routing on vs off — same math, different GEMM kernel."""
    nk, nb, m = 3, 4, 24
    hmat = _hermitian_operator(m, seed=11)
    gen = torch.Generator().manual_seed(12)
    x0 = torch.complex(
        torch.randn(nk, nb, m, generator=gen, dtype=torch.float64),
        torch.randn(nk, nb, m, generator=gen, dtype=torch.float64))
    t = torch.arange(m, dtype=torch.float64)[None].repeat(nk, 1)
    mask = torch.ones(nk, m, dtype=torch.bool)

    monkeypatch.setenv("GRADWAVE_BLAS_GEMM", "off")
    off = _run_davidson(hmat, x0.clone(), t, mask)
    monkeypatch.setenv("GRADWAVE_BLAS_GEMM", "on")
    on = _run_davidson(hmat, x0.clone(), t, mask)

    assert on.n_iter == off.n_iter
    assert torch.allclose(on.eigenvalues, off.eigenvalues, atol=1e-10)
    ref = torch.linalg.eigvalsh(hmat)[:nb]
    assert torch.allclose(off.eigenvalues[0], ref, atol=1e-8)
