"""Native (FFTW + CBLAS + LAPACKE) batched block Davidson — opt-in solver.

The eager batched CPU Davidson has NO intra-op thread scaling at small/medium
sizes (measured 8 threads == 1 thread: batched LAPACK loops run serially and
the many small ops between never engage parallelism), so a multi-core box runs
the eigensolve one-core-equivalent. This solver reimplements the identical
algorithm — FFT-path H-apply, Rayleigh–Ritz, Teter preconditioning, two-pass
orthonormalization + QR, restart with triangular image repair, per-k
convergence retirement — as one C call with OpenMP over k, zero framework
dispatch, FFTW_MEASURE plans (wisdom-cached), and OpenBLAS pinned serial
inside the parallel regions. Measured (experiments on the
research/native-davidson-probe branch, asus 8 threads): 3.9–6.2× the eager
solver batch-exact, 4.5–8.5× with per-k retirement, eigenvalues to ~1e-13
(batch mode) / within tol (retirement — the same rn ≤ tol per-band contract,
each k judged at its own final Rayleigh–Ritz).

Opt in with ``scf.eigensolver: davidson-native`` (Input) or
``eigensolver="davidson-native"`` (``scf.loop.scf``). The shared library is
built once per machine by ``scripts/build_native_solver.sh`` (gcc + fftw +
openblas via nix-shell) into ``~/.cache/gradwave/libdavnative.so``;
``GRADWAVE_NATIVE_SO`` overrides the path. Requesting the solver without the
library is an error with the build command in the message — never a silent
eager run.

Coverage and fallback are explicit, not silent:

* Handled natively: the plain norm-conserving ``BatchedHamiltonian`` apply
  (kinetic + local FFT term + KB nonlocal), DFT+U's second nonlocal term,
  complex128, CPU, the restart path, per-k retirement
  (``GRADWAVE_NATIVE_RETIRE`` in {"on" (default), "off"} — "off" reproduces
  the uniform-batch trajectory bit-for-bit-in-structure for A/B debugging).
* Transparent per-solve fallback to the eager ``davidson_batched`` (recorded
  in the returned diagnostics as ``fallback_reason``) when the solve is
  outside the native scope: a composed apply (hybrid Fock / meta-GGA wrap the
  bound method in a closure), the USPP dual-grid Hamiltonian, a non-CPU
  device, a low-precision block (mixed-precision draft), the fp32-expansion /
  complex64-storage / sync_free modes, a Toeplitz-forced local term, or a
  rank-deficient block the reference repairs with RNG jitter (the C kernel
  reports it and the whole solve re-runs eager — bit-identical to having
  chosen eager, at the cost of the aborted native attempt).

The eigensolve runs under ``no_grad`` (autograd re-enters through the
detached density), so swapping the solver cannot touch gradients.
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from gradwave.solvers.davidson import _resolve_max_dim_factor

if TYPE_CHECKING:
    from gradwave.solvers.registry import EigResult

logger = logging.getLogger(__name__)

_BUILD_HINT = (
    "build it once with scripts/build_native_solver.sh (gcc+fftw+openblas via "
    "nix-shell; writes ~/.cache/gradwave/libdavnative.so), or point "
    "GRADWAVE_NATIVE_SO at an existing build"
)

_lib: ctypes.CDLL | None = None
_lib_path: str | None = None


def _so_path() -> Path:
    env = os.environ.get("GRADWAVE_NATIVE_SO")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "gradwave" / "libdavnative.so"


def native_available() -> bool:
    """Whether the native library can be loaded (cheap after first call)."""
    return _load() is not None


def _load() -> ctypes.CDLL | None:
    global _lib, _lib_path
    path = str(_so_path())
    if _lib is not None and _lib_path == path:
        return _lib
    if not Path(path).exists():
        return None
    lib = ctypes.CDLL(path)
    lib.davidson_native.restype = ctypes.c_int
    lib.davidson_native.argtypes = (
        [ctypes.c_int64] * 10 + [ctypes.c_double, ctypes.c_int, ctypes.c_int]
        + [ctypes.c_void_p] * 10
        + [ctypes.c_void_p] * 4)
    _lib, _lib_path = lib, path
    return lib


def _unsupported_reason(
    apply_H: Callable[[torch.Tensor], torch.Tensor],
    X0: torch.Tensor,
    precond: torch.Tensor,
) -> str | None:
    """Why this solve cannot run natively, or None if it can."""
    from gradwave.core.batch import BatchedHamiltonian

    h = getattr(apply_H, "__self__", None)
    if not isinstance(h, BatchedHamiltonian):
        return "composed apply (hybrid Fock / meta-GGA / test operator)"
    if X0.dtype != torch.complex128:
        return f"non-fp64 block ({X0.dtype})"
    if X0.device.type != "cpu":
        return f"non-CPU device ({X0.device.type})"
    if getattr(h, "shape", None) is not None and h.gather_idx is not h.bk.flat_idx:
        return "USPP dual-grid (smooth-box) Hamiltonian"
    if not torch.equal(precond, h.bk.t):
        return "preconditioner diagonal differs from the Hamiltonian kinetic"
    if os.environ.get("GRADWAVE_FP32_EXPANSION", "off").strip().lower() != "off":
        return "GRADWAVE_FP32_EXPANSION mode"
    if (os.environ.get("GRADWAVE_SUBSPACE_STORAGE", "complex128")
            .strip().lower() != "complex128"):
        return "GRADWAVE_SUBSPACE_STORAGE=complex64 mode"
    if os.environ.get("GRADWAVE_TOEPLITZ", "auto").strip().lower() == "on":
        return "Toeplitz-forced local term (native kernel is the FFT path)"
    return None


def _retire_on() -> bool:
    mode = os.environ.get("GRADWAVE_NATIVE_RETIRE", "on").strip().lower()
    if mode not in ("on", "off"):
        raise ValueError(
            f"GRADWAVE_NATIVE_RETIRE must be on|off, got {mode!r}")
    return mode == "on"


def native_davidson_adapter(
    apply_H: Callable[[torch.Tensor], torch.Tensor],
    X0: torch.Tensor,
    precond: torch.Tensor,
    mask: torch.Tensor,
    *,
    tol: float,
    nbands: int | None = None,
    max_iter: int = 40,
    max_dim_factor: int = 4,
    **kw: Any,
) -> EigResult:
    """Registry adapter: native solve when in scope, eager fallback otherwise."""
    from gradwave.solvers.registry import EigResult, davidson_adapter

    lib = _load()
    if lib is None:
        raise RuntimeError(
            f"eigensolver 'davidson-native' requested but the native library "
            f"is not built at {_so_path()} — {_BUILD_HINT}")

    reason = _unsupported_reason(apply_H, X0, precond)
    if reason is not None:
        r = davidson_adapter(apply_H, X0, precond, mask, tol=tol,
                             nbands=nbands, max_iter=max_iter,
                             max_dim_factor=max_dim_factor, **kw)
        r.diagnostics["fallback_reason"] = reason
        return r

    h = apply_H.__self__  # type: ignore[attr-defined]  # gated above
    bk = h.bk
    nk, nb, m = X0.shape
    n1, n2, n3 = h.shape
    nproj = int(h.p.shape[1])
    hub_q = h.hub_q
    nhub = int(hub_q.shape[1]) if hub_q is not None else 0
    max_dim_factor = _resolve_max_dim_factor(
        nk, nb, m, X0.element_size(), max_dim_factor)
    max_dim = min(max_dim_factor * nb, int(mask.sum(dim=1).min()))

    x0_np = X0.contiguous().numpy()
    t_np = bk.t.contiguous().numpy()
    mask_np = mask.to(torch.uint8).contiguous().numpy()
    isc_np = h.idx_scatter.contiguous().numpy()
    iga_np = h.gather_idx.contiguous().numpy()
    veff_np = h.v_eff_r.reshape(-1).contiguous().numpy()
    p_np = h.p.contiguous().numpy()
    dij_np = bk.dij_full.to(torch.complex128).contiguous().numpy()
    hq_np = (hub_q.contiguous().numpy() if hub_q is not None
             else dij_np[:0])  # dummy pointer, nhub=0 → never read
    hdij_np = (h.hub_dij.to(torch.complex128).contiguous().numpy()
               if h.hub_dij is not None and nhub else dij_np[:0])

    eig = torch.empty(nk, nb, dtype=torch.float64)
    x = torch.empty(nk, nb, m, dtype=torch.complex128)
    rn = torch.empty(nk, nb, dtype=torch.float64)
    napply = torch.zeros(1, dtype=torch.int64)

    def ptr(a) -> ctypes.c_void_p:
        if hasattr(a, "ctypes"):
            return ctypes.c_void_p(a.ctypes.data)
        return ctypes.c_void_p(a.data_ptr())

    retire = 1 if _retire_on() else 0
    ret = lib.davidson_native(
        nk, nb, m, n1, n2, n3, nproj, nhub, max_dim, max_iter, float(tol),
        torch.get_num_threads(), retire,
        ptr(x0_np), ptr(t_np), ptr(mask_np), ptr(isc_np), ptr(iga_np),
        ptr(veff_np), ptr(p_np), ptr(dij_np), ptr(hq_np), ptr(hdij_np),
        ptr(eig), ptr(x), ptr(rn), ptr(napply))
    if ret < 0:
        # -1: rank-deficient block (reference repairs with RNG jitter — out of
        # native scope by design); -2: LAPACK failure; -3 unused; -4: nb cap.
        # All re-run eager: bit-identical to having chosen eager for this
        # solve, at the cost of the aborted native attempt.
        r = davidson_adapter(apply_H, X0, precond, mask, tol=tol,
                             nbands=nbands, max_iter=max_iter,
                             max_dim_factor=max_dim_factor, **kw)
        r.diagnostics["fallback_reason"] = f"native error {ret}"
        logger.info("davidson-native: solve fell back to eager (code %d)", ret)
        return r

    return EigResult(
        eig, x, int(ret), rn,
        {"solver": "davidson-native", "retire": bool(retire),
         "max_dim_factor": max_dim_factor, "max_iter": max_iter,
         "hit_max_iter": int(ret) >= max_iter,
         "n_apply_low": 0, "n_apply_full": int(napply.item())},
    )
