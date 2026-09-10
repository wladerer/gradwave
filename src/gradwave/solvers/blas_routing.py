"""Route the eager Davidson Rayleigh-Ritz GEMMs through OpenBLAS zgemm.

torch's CPU complex128 matmul is MEASURED 2.6-5x slower than a plain OpenBLAS
``zgemm`` on the batched shapes the eager block Davidson runs its Rayleigh-Ritz
reductions at (asus: 210 vs 718 us at batch 8 x 754x754; 1237 vs 6354 us at
27x1260x1260). This module exposes one helper, :func:`zbmm`, that dispatches a
batched complex128 matmul to a tiny ctypes-loaded ``cblas_zgemm`` loop (the
``blas_zbmm`` symbol built into the same ``libdavnative.so`` as the native
Davidson solver) when the fast path is applicable, and falls back to
``torch.matmul`` otherwise.

Same math, different kernel: the only difference from ``torch.matmul`` is
last-bit BLAS rounding, well inside the eigensolver's tolerance. It is wired
into :mod:`gradwave.solvers.davidson`'s RR GEMMs behind ``GRADWAVE_BLAS_GEMM``:

* ``"auto"`` (default) — use the ``.so`` when loadable and the operands are
  contiguous complex128 CPU tensors NOT tracking gradients; fall back silently
  otherwise. Unlike ``davidson-native``, the fallback here is SILENT because it
  is identical-math micro-routing, not a distinct algorithm — a missing ``.so``
  costs nothing but the torch kernel. Availability is recorded once in
  :data:`_ROUTED` for diagnostics.
* ``"on"`` — require the ``.so``; raise if it cannot be loaded.
* ``"off"`` — pure torch, never touch the ``.so``.

The Davidson eigensolve runs under ``torch.no_grad``, so this routing never
sees an autograd graph; :func:`zbmm` additionally refuses the native path when
grad is enabled or any operand requires grad, so a stray grad-tracked matmul
can never be silently detached.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable

import torch

# The blas_zbmm symbol is compiled into the SAME libdavnative.so as the native
# Davidson solver (one .so, two symbols); its path resolution and build hint are
# reused from native_davidson. That module imports from davidson, and davidson
# imports this one — so the import is deferred into the functions below to break
# the cycle (davidson -> blas_routing -> native_davidson -> davidson).

_ZBMM: Callable[..., None] | None = None  # bound C function, or None if unavailable
_ZBMM_TRIED = False
# Dispatch record for diagnostics (a bench must be able to PROVE the native
# path engaged — a silent fallback makes an A/B meaningless): _N_NATIVE counts
# calls served by the .so, _N_FALLBACK calls that fell back to torch.matmul in
# auto/on mode ("off" counts in neither — it never dispatches). _ROUTED is the
# last dispatch decision (None until first use). Observability only, never
# load-bearing for correctness.
_ROUTED: bool | None = None
_N_NATIVE = 0
_N_FALLBACK = 0


def _mode() -> str:
    mode = os.environ.get("GRADWAVE_BLAS_GEMM", "auto").strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError(
            f"GRADWAVE_BLAS_GEMM must be auto|on|off, got {mode!r}")
    return mode


def _load_zbmm() -> Callable[..., None] | None:
    """Bind blas_zbmm from libdavnative.so, or None if the library is absent."""
    global _ZBMM, _ZBMM_TRIED
    if _ZBMM_TRIED:
        return _ZBMM
    _ZBMM_TRIED = True
    from gradwave.solvers.native_davidson import _so_path

    path = _so_path()
    if not path.exists():
        return None
    lib = ctypes.CDLL(str(path))
    fn = lib.blas_zbmm
    fn.restype = None
    fn.argtypes = (
        [ctypes.c_int64] * 4        # batch, M, N, K
        + [ctypes.c_int] * 3        # transa, transb, nthreads
        + [ctypes.c_int64] * 5      # lda, ldb, stride_a, stride_b, stride_c
        + [ctypes.c_void_p] * 3     # A, B, C
    )
    _ZBMM = fn
    return fn


def blas_available() -> bool:
    """Whether the blas_zbmm native symbol can be loaded (cheap after first)."""
    return _load_zbmm() is not None


def _native_ok(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether the native zgemm path applies to these operands.

    complex128 CPU 3-D batched tensors, matching batch, and NOT tracking
    gradients (the caller is under no_grad, but guard anyway so a grad-tracked
    matmul is never silently detached). Layout is checked separately per
    operand by :func:`_plan` — GEMM's lda/stride arguments express more than
    plain contiguity (last-dim slices, transpose views)."""
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return False
    return (
        a.dtype == torch.complex128
        and b.dtype == torch.complex128
        and a.device.type == "cpu"
        and b.device.type == "cpu"
        and a.dim() == 3
        and b.dim() == 3
        and a.shape[0] == b.shape[0]
    )


def _plan(x: torch.Tensor, trans: int) -> tuple[int, int, int] | None:
    """Map one (batch, R, C) operand to zgemm arguments, or None if it can't.

    Returns (trans_code, lda, batch_stride) against the operand's UNDERLYING
    storage — GEMM's leading-dimension argument expresses two layouts torch
    views produce that ``is_contiguous`` rejects:

    * direct rows (``stride(-1) == 1``): the stored matrix is (R, C) at
      lda = stride(-2). Covers contiguous tensors AND last-dim slices like the
      Ritz rotation ``u[:, :, :nb]`` (rows with a gap). trans passes through.
    * transposed storage (``stride(-2) == 1``): the stored matrix is (C, R) at
      lda = stride(-1) — exactly what ``q.transpose(-1, -2)`` views produce
      (``_orthonormalize_b`` returns these). NoTrans flips to Trans and vice
      versa; ConjTrans on transposed storage would need the nonstandard
      conj-no-trans op, so it reports None (torch fallback).

    trans: 0 = NoTrans, 1 = Trans, 2 = ConjTrans (of the LOGICAL operand).
    """
    r, c = x.shape[-2], x.shape[-1]
    if x.stride(-1) == 1 and x.stride(-2) >= max(c, 1):
        return trans, x.stride(-2), x.stride(0)
    if x.stride(-2) == 1 and x.stride(-1) >= max(r, 1):
        if trans == 2:
            return None
        return (1 - trans), x.stride(-1), x.stride(0)
    return None


def _torch_zbmm(
    a: torch.Tensor, b: torch.Tensor, *, conj_b_t: bool, t_a: bool
) -> torch.Tensor:
    """Reference torch implementation of the routed contraction."""
    op_a = a.transpose(-1, -2) if t_a else a
    op_b = b.conj().transpose(-1, -2) if conj_b_t else b
    return torch.matmul(op_a, op_b)


def zbmm(
    a: torch.Tensor, b: torch.Tensor, *, conj_b_t: bool = False, t_a: bool = False
) -> torch.Tensor:
    """Batched complex128 matmul ``op(a) @ op(b)`` routed through OpenBLAS zgemm.

    ``a``, ``b``: (batch, ., .) tensors. ``t_a`` transposes ``a`` (no conjugate);
    ``conj_b_t`` conjugate-transposes ``b``. Returns a fresh (batch, M, N) tensor
    equal to ``torch.matmul(op(a), op(b))`` up to last-bit BLAS rounding.

    ``GRADWAVE_BLAS_GEMM`` gates the native path: "off" always uses torch, "on"
    requires the ``.so`` (raises if missing), "auto" (default) uses it when
    loadable and the operands qualify, else falls back to torch silently.
    """
    global _ROUTED, _N_NATIVE, _N_FALLBACK
    mode = _mode()
    if mode == "off":
        return _torch_zbmm(a, b, conj_b_t=conj_b_t, t_a=t_a)

    fn = _load_zbmm()
    if fn is None:
        if mode == "on":
            from gradwave.solvers.native_davidson import _BUILD_HINT, _so_path

            raise RuntimeError(
                "GRADWAVE_BLAS_GEMM=on but the native library is not built at "
                f"{_so_path()} — {_BUILD_HINT}")
        _ROUTED = False
        _N_FALLBACK += 1
        return _torch_zbmm(a, b, conj_b_t=conj_b_t, t_a=t_a)

    plan_a = _plan(a, 1 if t_a else 0) if _native_ok(a, b) else None
    plan_b = _plan(b, 2 if conj_b_t else 0) if plan_a is not None else None
    if plan_a is None or plan_b is None:
        _ROUTED = False
        _N_FALLBACK += 1
        return _torch_zbmm(a, b, conj_b_t=conj_b_t, t_a=t_a)

    batch = a.shape[0]
    a0, a1 = a.shape[1], a.shape[2]
    b0, b1 = b.shape[1], b.shape[2]
    # LOGICAL op(a) is (M, K), op(b) is (K, N); the trans codes and lda in the
    # plans are already expressed against each operand's underlying storage.
    m_ = a1 if t_a else a0
    k_ = a0 if t_a else a1
    n_ = b0 if conj_b_t else b1
    kb = b1 if conj_b_t else b0
    if k_ != kb:
        raise ValueError(
            f"zbmm shape mismatch: op(a) inner {k_} != op(b) inner {kb} "
            f"(a={tuple(a.shape)}, b={tuple(b.shape)}, "
            f"t_a={t_a}, conj_b_t={conj_b_t})")
    transa, lda, stride_a = plan_a
    transb, ldb, stride_b = plan_b

    out = torch.empty(batch, m_, n_, dtype=torch.complex128)
    # BLAS threading follows torch's intra-op setting: the SCF k-parallel
    # thread pool pins torch to 1 inside its tasks, so concurrent zbmm calls
    # stay serial there (no workers x cores oversubscription); a plain serial
    # solve hands OpenBLAS torch's full thread count.
    fn(
        int(batch), int(m_), int(n_), int(k_), int(transa), int(transb),
        int(torch.get_num_threads()),
        int(lda), int(ldb), int(stride_a), int(stride_b), int(m_ * n_),
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
    )
    _ROUTED = True
    _N_NATIVE += 1
    return out
