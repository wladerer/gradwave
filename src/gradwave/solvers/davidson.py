"""Block Davidson for the complex Hermitian plane-wave Hamiltonian (Layer B).

Norm-conserving pseudopotentials ⇒ standard eigenproblem, no overlap matrix.
Growing subspace with Rayleigh–Ritz via torch.linalg.eigh, Teter-
preconditioned residual expansion, band locking, restart at max_dim.
Runs entirely under torch.no_grad() — autograd must never see this.

Environment
-----------
``GRADWAVE_QR_OFFLOAD`` in {"on", "off", "auto"} (default "auto") controls the
CUDA batched-QR CPU offload (see ``_qr_offload``). "auto" gates it on a one-time
per-device fp64 microbenchmark, so the offload fires on fp64-crippled consumer
cards and stays off on datacenter fp64 hardware. "on"/"off" force it either way,
so a benchmark can toggle the path without monkeypatching. Read once at import.

``GRADWAVE_CHOLQR`` in {"on", "off"} (default "off" — measured neutral on
CPU and neutral-to-negative on RTX 3050; retest on datacenter GPUs) controls whether
``_orthonormalize_b`` uses CholQR2 (Gram GEMM → fp64 Cholesky → triangular
solve, twice) instead of the batched tall-skinny QR. CholQR2 is GEMM-shaped —
fast on every device — and removes the D2H/H2D round trip of the CUDA QR
offload entirely; a Cholesky breakdown (rank-deficient block) falls back to
the QR path for that call. "off" forces the QR path. Read once at import.

``GRADWAVE_FP32_EXPANSION`` in {"on", "off", "auto"} (default "off") controls the
fp64-certified fp32-expansion mode of ``davidson_batched``: expansion-vector
H-applies run in complex64 while every convergence-deciding quantity stays fp64
(see ``_fp32_expansion_active``). Unlike the QR knob this one is read per solve,
so a benchmark or test can A/B it inside one process. Default off: the win is
GPU-targeted (crippled-fp64 cards) and not yet measured.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from gradwave.core import opcount
from gradwave.solvers.precond import teter, teter_b

logger = logging.getLogger(__name__)


# Square subspace dim at/below which a CUDA subspace eigensolve is offloaded to
# CPU LAPACK (issue #133). The batched Davidson Rayleigh-Ritz reductions are
# tiny (nk, n, n) Hermitian solves, but on CUDA each cuSOLVER call reads a
# factorization `info` back to the host, serializing the stream; the GPU SCF is
# latency-bound on ~135 such syncs/iteration. Shipping the few-MB subspace
# matrices to the CPU, solving in LAPACK (no stream syncs), and shipping only
# the wanted eigenvectors back trades ~N host syncs for one D2H + one H2D per
# round. Measured on an RTX 3050 (issue #133): CPU+transfer eigh is break-even
# with cuSOLVER for these shapes, so the sync elimination is pure win. Capped so
# a genuinely large subspace (n^3 CPU cost) stays on the GPU.
_SUBSPACE_CPU_MAX = 256


def _offload_subspace(is_cuda: bool, n: int) -> bool:
    """Whether to run an (nk, n, n) subspace solve on CPU instead of CUDA."""
    return is_cuda and n <= _SUBSPACE_CPU_MAX


def _eigh_subspace(s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.linalg.eigh on a small subspace matrix, CPU-offloaded on CUDA.

    Returns (w, u) on s.device. Physics-neutral: LAPACK and cuSOLVER agree to
    fp64 round-off, well inside the batched-solver eigenpair contract."""
    opcount.bump("eigh")
    if _offload_subspace(s.is_cuda, s.shape[-1]):
        w, u = torch.linalg.eigh(s.cpu())
        return w.to(s.device), u.to(s.device)
    return torch.linalg.eigh(s)


# Column count at/below which a CUDA batched tall-skinny QR (the (nk, npw,
# cols) reduced QR that orthonormalizes new Davidson directions, and, on
# restart, re-orthonormalizes the Ritz block) is offloaded to CPU LAPACK. Found
# while investigating whether CUDA-graph capture could speed up a Davidson
# round (docs/manual/performance.md, "What does not help"): a graphed replay
# of the round's Rayleigh-Ritz/expansion math is bit-identical to eager at
# 1.0x -- no launch-overhead gap anywhere in the round, apply included -- but
# this QR call, isolated, was the single biggest cost in the round on an RTX
# 3050, BIGGER than the two-FFT Hamiltonian apply next to it (measured: ~3.9 ms
# vs ~2.3 ms for a diamond-C, 50 Ry, nk=8, npw=465, cols=8 round). A D2H + CPU
# LAPACK QR + H2D round trip runs that same shape in ~0.3 ms, a >10x win, for
# the same reason issue #133 CPU-offloaded eigh: cuSOLVER's batched
# geqrf/orgqr pays a fixed per-call tax that swamps a tiny problem, and the
# tensors here are a few MB. A parameter sweep (nk in 8..112, npw in
# 465..2500, cols in 8..64) shows CPU offload is a clear-to-break-even win at
# cols<=16 (worst case measured ratio 0.98x, typically 2-12x) and mixed
# (sometimes GPU-favored) above that -- capped here at the safe boundary.
# Davidson's own n_add is bounded by nb, comfortably under 16 for the systems
# in this repo's benchmark battery; wider blocks (e.g. CheFSI's buffered
# block) fall through to the GPU unchanged, same as today.
# This is the SIZE gate only. Whether the offload activates at all is a second,
# device-dependent gate (`_qr_offload_active`): the whole trick only pays off on
# a card whose fp64 units are slow enough that the D2H/H2D round trip beats them,
# which the RTX 3050 sweep above assumed but datacenter fp64 hardware breaks.
_QR_CPU_MAX_COLS = 16

# fp64/fp32 GEMM time ratio above which the CUDA batched-QR offload activates in
# "auto" mode. Two measurement sources bracket the threshold with a wide margin.
# On the RTX 3050 that PR #174 tuned, fp64 runs at ~1/64 of fp32 (docs/manual/
# performance.md), so the measured ratio is tens (~20-60) and the offload is a
# clear win. On the 4x H100 session (issue #206, benchmarks/results/h100-session)
# the same A/B measured the offload as a ~13% PENALTY -- Cr2O3 eskolaite, offload
# on 37.1 s vs off 32.7 s, identical energy to 1.8e-10 meV/atom and identical 16
# iterations -- because a datacenter fp64 GPU (ratio ~1-2) has no cuSOLVER tax to
# escape. A threshold of 8 sits an order of magnitude clear of both regimes.
_QR_OFFLOAD_PENALTY_THRESHOLD = 8.0

# One-shot escape hatch, read once at import (see module docstring). "auto" (the
# default) uses the microbenchmark gate; "on"/"off" force the offload either way.
_QR_OFFLOAD_ENV = os.environ.get("GRADWAVE_QR_OFFLOAD", "auto").strip().lower()

# fp64 penalty ratio per CUDA device index. Plain dict, GIL-guarded like the rest
# of this module -- the microbenchmark is idempotent, so a rare double-measure on
# first concurrent use is harmless.
_FP64_PENALTY: dict[int, float] = {}


def _fp64_penalty(device: torch.device) -> float:
    """Measured fp64/fp32 GEMM time ratio on a CUDA device, cached per index.

    Times a small square fp64 matmul against the same-shape fp32 matmul (warmup
    plus synchronize, a few reps, ~1 ms total). A large ratio means crippled
    fp64 (consumer card) and favors the CPU-offloaded QR; a ratio near 1 means
    real fp64 hardware that keeps the QR on-GPU. CUDA-only: the offload it gates
    is a no-op on CPU, so calling this for a non-CUDA device is a programming
    error, not a silently-handled case."""
    if device.type != "cuda":
        raise ValueError(f"_fp64_penalty is CUDA-only, got device {device!r}")
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cached = _FP64_PENALTY.get(idx)
    if cached is not None:
        return cached
    n, reps = 512, 10
    a32 = torch.randn(n, n, device=device, dtype=torch.float32)
    a64 = torch.randn(n, n, device=device, dtype=torch.float64)

    def _time(a: torch.Tensor) -> float:
        for _ in range(3):  # warm the kernel / allocator
            a @ a
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            a @ a
        torch.cuda.synchronize(device)
        return (time.perf_counter() - t0) / reps

    t32 = _time(a32)
    ratio = _time(a64) / t32 if t32 > 0 else float("inf")
    _FP64_PENALTY[idx] = ratio
    return ratio


def _qr_offload_active(device: torch.device) -> bool:
    """Whether the CUDA batched-QR CPU offload should apply on `device`.

    False on CPU (the offload is CUDA-only). On CUDA the env hatch wins in both
    directions; "auto" defers to the fp64 microbenchmark against the threshold."""
    if device.type != "cuda":
        return False
    if _QR_OFFLOAD_ENV == "on":
        return True
    if _QR_OFFLOAD_ENV == "off":
        return False
    return _fp64_penalty(device) >= _QR_OFFLOAD_PENALTY_THRESHOLD


def _qr_offload(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduced QR of x (..., rows, cols), CPU-offloaded on CUDA for small cols.

    Two gates guard the offload: the SIZE gate (`_QR_CPU_MAX_COLS`) and the
    device gate (`_qr_offload_active`, a measured fp64 penalty or the env hatch).
    Returns (q, r) on x.device. Physics-neutral: LAPACK and cuSOLVER QR agree
    to fp64 round-off, well inside the orthonormalization's own tolerance."""
    if x.is_cuda and x.shape[-1] <= _QR_CPU_MAX_COLS and _qr_offload_active(x.device):
        q, r = torch.linalg.qr(x.cpu(), mode="reduced")
        return q.to(x.device), r.to(x.device)
    return torch.linalg.qr(x, mode="reduced")


# CholQR2 escape hatch, read once at import (module docstring). Default "off":
# measured neutral on CPU (0.94-1.03x battery) and neutral-to-slightly-negative
# on RTX 3050 whole-SCF — the QR round is too thin a slice at these sizes; kept
# for a datacenter-GPU retest. When "on", CholQR2 replaces the tall-skinny QR
# wherever it succeeds, and a breakdown falls back to `_qr_offload` per call.
_CHOLQR_ENV = os.environ.get("GRADWAVE_CHOLQR", "off").strip().lower()


def _cholqr2(x: torch.Tensor) -> torch.Tensor | None:
    """Orthonormal Q spanning x (..., rows, cols) via CholQR2, or None on breakdown.

    Two rounds of Gram → Cholesky → triangular solve. Everything is a batched
    GEMM (fast on CPU and GPU alike, no cuSOLVER QR, no CPU offload round trip);
    the only non-GEMM op is a tiny (cols, cols) Cholesky. The Gram matrix and
    its factorization run in fp64 — the same "subspace reduction ALWAYS in
    fp64" contract as the Rayleigh-Ritz (issue #136) — and the factor is
    downcast to the block dtype for the solve; the second round mops up both
    the downcast and the κ(x)² conditioning of the first Gram, giving
    ‖QᴴQ − I‖ at fp64 round-off for any κ(x) ≲ 1e7 (far looser than the
    unit-normalized, projected rows that arrive here). A non-positive-definite
    Gram (rank-deficient block, κ beyond repair) reports as None so the caller
    can fall back to Householder QR, which tolerates rank deficiency.

    Physics-neutral by the same span-equivalence argument as the QR offload:
    Rayleigh-Ritz depends only on the span of the orthonormal basis, and
    CholQR2's Q spans x exactly as QR's does."""
    cd = torch.complex128
    for _ in range(2):
        gram = x.mH.to(cd) @ x.to(cd)
        lo, info = torch.linalg.cholesky_ex(gram)
        if int(info.max()) != 0:  # one scalar sync; breakdown is rare
            return None
        x = torch.linalg.solve_triangular(
            lo.mH.to(x.dtype), x, upper=True, left=False
        )
    return x


# ---------------------------------------------------------------------------
# fp64-certified fp32-expansion mode (vetted attack #4, physics-blind roadmap).
#
# The mixed-precision scheme that already exists (davidson_batched_ms and the
# SCF loop's `mixed_precision`) drafts WHOLE SOLVES in complex64 and re-polishes
# later; its measured failure mode is unconditioned fp32 noise on metals costing
# extra SCF outer iterations (docs/manual/performance.md). This mode draws the
# precision boundary INSIDE one solve instead:
#
#   fp32 (complex64):  H-applies of expansion vectors only — the O(npw) FFT
#                      work that dominates a round. The results are cast back
#                      and STORED in complex128, so they carry ~1e-7 relative
#                      error but participate in exact-arithmetic reductions.
#   fp64 (complex128): the basis and its orthonormalization (QR), the subspace
#                      matrix and Rayleigh-Ritz eigensolve (the issue-#136
#                      contract, unchanged), and — the certification — every
#                      residual norm that decides convergence: before ANY
#                      return, r = H·x − εx is recomputed for the retained
#                      Ritz block against the FULL-precision operator, and the
#                      block is refreshed by an exact nb×nb Rayleigh-Ritz.
#
# A band is therefore never declared converged by an fp32 measurement: the
# fp32-sourced residual norms only steer the expansion (which directions to
# add); the return gate sees fp64 residuals of an fp64-refreshed block.
# ---------------------------------------------------------------------------

# Certification trigger. fp32-sourced residual norms floor at roughly
# eps32·‖H‖·√npw, so at tight tol the plain `rn < tol` gate would never fire;
# certification is attempted when the measured norms drop under an estimated
# floor (eps factor × the kinetic-diagonal max, a cheap ‖H‖ proxy) or when they
# stall (an fp32 floor reads as stagnation). A spurious trigger costs one exact
# nb-wide apply and injects exact residual directions into the expansion — it
# cannot corrupt the answer, only spend time.
_FP32_CERT_FLOOR_EPS = 32 * torch.finfo(torch.float32).eps
_FP32_CERT_STALL = 0.7  # rn.max() improved by <30% over the previous round

# fp64-penalty ratio above which "auto" activates on a CUDA device — same
# measured gate (and threshold rationale) as the QR offload: the mode targets
# cards whose fp64 units are crippled enough that complex64 FFT/GEMM applies
# win big; on datacenter fp64 hardware it has nothing to sell.
_FP32_EXP_PENALTY_THRESHOLD = _QR_OFFLOAD_PENALTY_THRESHOLD


def _fp32_expansion_active(x0: torch.Tensor) -> bool:
    """Whether the fp32-expansion mode applies to a solve seeded by `x0`.

    Reads ``GRADWAVE_FP32_EXPANSION`` per call (cheap; enables in-process A/B).
    Only a complex128 block qualifies — a complex64 block is already a
    mixed-precision draft with nothing to certify. "auto" additionally requires
    CUDA with measurably crippled fp64; "on" forces the mode anywhere (CPU
    included — correct there too, just not expected to win)."""
    mode = os.environ.get("GRADWAVE_FP32_EXPANSION", "off").strip().lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError(
            f"GRADWAVE_FP32_EXPANSION must be auto|on|off, got {mode!r}")
    if mode == "off" or x0.dtype != torch.complex128:
        return False
    if mode == "on":
        return True
    return x0.is_cuda and _fp64_penalty(x0.device) >= _FP32_EXP_PENALTY_THRESHOLD


# ---------------------------------------------------------------------------
# Exact fp64 Rayleigh-Ritz stack (3M/Karatsuba complex GEMM behind a measured
# per-signature gate). The RR subspace algebra (BUILD s = v.conj()@hv.mT and the
# two COMBINE einsums x,hx) is all-complex128 and dominates the batched Davidson
# wall on cards whose fp64 ALU is crippled (RTX 3050: ~1/64 of fp32). PyTorch
# dispatches complex128 matmul to cuBLAS's 4-multiply ZGEMM; Gauss's 3-multiply
# complex product does the same contraction with 3 real fp64 GEMMs — a lossless
# 25% fp64-FLOP cut. It REGRESSES in the wide-many-k / small-block regime, so it
# ships behind a per-(device,nk,m,npw,dtype) measured A/B gate, exactly mirroring
# core/batch.py's `_use_toeplitz`: a one-time timed trial of 3M vs native ZGEMM
# on the real block, cached verdict, native fallback. Result is NOT bit-exact vs
# native (rel-err ~1e-14) but the eigensolve upcasts s to fp64 anyway and the
# convergence iteration count is unchanged (asserted in tests).
# ---------------------------------------------------------------------------

# GRADWAVE_RR3M in {"auto", "on", "off"} (default "auto"): "on" forces 3M wherever
# eligible (complex128); "off" always uses native ZGEMM; "auto" gates on the
# per-signature timed trial and is restricted to CUDA (CPU fp64 GEMM is not
# crippled, so the native path is left untouched with zero overhead there). Read
# once at import, like the toeplitz/QR knobs.
_RR3M_ENV = os.environ.get("GRADWAVE_RR3M", "auto").strip().lower()

# Adopt 3M only if t_3m < margin * t_native on the trial (a small guard band so a
# statistical tie stays on the native path). Matches _TOEP_TRIAL_MARGIN.
_RR3M_TRIAL_MARGIN = 0.95

# Per-signature verdict cache, keyed on (device.type, nk, m, npw, dtype). Plain
# dict, GIL-guarded; a rare double-trial on first concurrent use is harmless.
_RR3M_VERDICT: dict[tuple[str, int, int, int, torch.dtype], bool] = {}

# Incremental RR build (Piece 2): on a grow round the old v/hv columns are
# bit-identical, so the old block of the raw subspace matrix is carried forward
# and only the new column blocks are recomputed. Round-off exact vs a full
# rebuild (~1e-14 — the reused block is a matmul over the same rows, but a GEMM's
# tiling depends on the matrix dimensions, so a sub-block and the corner of the
# larger product differ at fp64 round-off; converged eigenvalues agree to
# round-off, iteration count can drift by a few near a flat convergence tail).
# Module flag rather than an env knob — always on in production; tests toggle it
# to A/B against the full-rebuild path.
_RR_INCREMENTAL = True

# Hermitian half-build rider (Piece 4): under the incremental build the two new
# cross blocks satisfy C' = C.conj().mT exactly by Hermiticity, so compute ONE
# cross GEMM and mirror it instead of two -> halves the incremental cross-block
# FLOPs. PROBE-GATED and measured GO on the asus RTX 3050: one GEMM + conj-
# transpose write is ~1.95x the two-GEMM path (>= the 1.6x GO threshold), because
# each cross GEMM contracts over npw on the crippled fp64 ALU while the mirror is
# a cheap transpose. Definitionally Hermitian (the BUILD symmetrization is kept as
# a cheap safety net for the corner/retained blocks). Module flag for the A/B.
_RR_HALFBUILD = True


def _gemm3(ar: torch.Tensor, ai: torch.Tensor, br: torch.Tensor,
           bi: torch.Tensor) -> torch.Tensor:
    """(ar + i·ai) @ (br + i·bi) via 3 real fp64 GEMMs (Gauss/Karatsuba).

    Cr = ar@br − ai@bi, Ci = ar@bi + ai@br, computed from three products
    t1 = ar@(br+bi), t2 = (ar+ai)@bi, t3 = (ai−ar)@br as Cr = t1−t2, Ci = t1+t3.
    Batched real matmul (cuBLAS DGEMM / MKL) on the real/imag planes; 25% fewer
    fp64 GEMM FLOPs than the native 4-multiply ZGEMM, lossless to ~1e-14."""
    t1 = ar @ (br + bi)
    t2 = (ar + ai) @ bi
    t3 = (ai - ar) @ br
    return torch.complex(t1 - t2, t1 + t3)


def _gemm3_conjl(ar: torch.Tensor, ai: torch.Tensor, br: torch.Tensor,
                 bi: torch.Tensor) -> torch.Tensor:
    """conj(ar + i·ai) @ (br + i·bi) via 3 real fp64 GEMMs (the BUILD bra is
    conjugated). Cr = ar@br + ai@bi, Ci = ar@bi − ai@br. The conjugation is
    folded into Gauss's three pre-sums at zero extra cost (no plane negation)."""
    t1 = ar @ (br + bi)
    t2 = (ar - ai) @ bi
    t3 = -(ar + ai) @ br
    return torch.complex(t1 - t2, t1 + t3)


def _planes(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Contiguous (real, imag) planes of a complex tensor — one deinterleave.

    ``view_as_real`` interleaves as [...,2]; the split planes are made
    contiguous so the downstream real GEMMs hit the fast path. Called on the
    n_add NEW columns only on a grow round (cached planes carry the old ones)."""
    zri = torch.view_as_real(z.contiguous())
    return zri[..., 0].contiguous(), zri[..., 1].contiguous()


def _rr3m_eligible(x0: torch.Tensor) -> bool:
    """Solve-level 3M eligibility (whether to maintain the real/imag plane cache).

    Only complex128 qualifies (the win targets the fp64 ALU; a complex64 draft
    runs on the fast fp32 path). "on" forces it anywhere (CPU included, for
    tests); "auto" restricts to CUDA so the CPU native path keeps zero overhead;
    "off" disables it. The per-signature timed gate below still decides 3M vs
    native each round on an eligible solve."""
    if _RR3M_ENV == "off" or x0.dtype != torch.complex128:
        return False
    if _RR3M_ENV == "on":
        return True
    return x0.is_cuda  # auto: trial per signature on CUDA only


def _rr3m_verdict(v: torch.Tensor, hv: torch.Tensor, vr: torch.Tensor,
                  vi: torch.Tensor, hvr: torch.Tensor, hvi: torch.Tensor) -> bool:
    """Per-signature timed A/B: 3M BUILD vs native ZGEMM, cached (mirrors
    ``_use_toeplitz``). "on" forces True. The BUILD is the dominant (~2/3) RR
    GEMM, so its verdict also governs the two COMBINE GEMMs — they share the
    fp64-ALU-roofline regime and flip together. The trial costs one extra BUILD
    per (device, nk, m, npw, dtype) signature per process."""
    if _RR3M_ENV == "on":
        return True
    nk, m, npw = v.shape
    key = (v.device.type, nk, m, npw, v.dtype)
    verdict = _RR3M_VERDICT.get(key)
    if verdict is None:
        brt, bit = hvr.transpose(-1, -2), hvi.transpose(-1, -2)
        is_cuda = v.is_cuda

        def timed(f: Callable[[], torch.Tensor]) -> float:
            f()  # warm
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            f()
            if is_cuda:
                torch.cuda.synchronize()
            return time.perf_counter() - t0

        t_nat = timed(lambda: torch.matmul(v.conj(), hv.mT))
        t_3m = timed(lambda: _gemm3_conjl(vr, vi, brt, bit))
        verdict = t_3m < _RR3M_TRIAL_MARGIN * t_nat
        _RR3M_VERDICT[key] = verdict
    return verdict


def _certify_fp64(
    h_apply: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact refresh of a retained Ritz block against the full-precision operator.

    x: (nk, nb, m), rows orthonormal (they come out of an fp64 QR basis and an
    fp64 eigh rotation, so orthonormality holds to fp64 round-off regardless of
    how H was applied). Applies the FULL-precision H once, re-solves the small
    nb×nb Rayleigh-Ritz problem, and returns the refreshed
    (eig, x, hx, r, rn) — eigenvalues, rotated Ritz vectors and images, and the
    residuals/norms every convergence decision must be made from."""
    hx = h_apply(x)
    s = torch.matmul(x.conj(), hx.mT)
    s = 0.5 * (s + s.conj().transpose(-1, -2))
    w, u = _eigh_subspace(s)
    eig = w.real.to(x.real.dtype)
    u = u.to(x.dtype)
    x = _combine(u, x)
    hx = _combine(u, hx)
    r = hx - eig[..., None] * x
    rn = torch.linalg.norm(r, dim=-1).real
    return eig, x, hx, r, rn


@dataclass
class DavidsonResult:
    eigenvalues: torch.Tensor  # (nb,) ascending [eV]
    eigenvectors: torch.Tensor  # (nb, npw) rows orthonormal
    n_iter: int
    residual_norms: torch.Tensor  # (nb,)


def _orthonormalize(v: torch.Tensor, against: torch.Tensor | None = None) -> torch.Tensor:
    """Project out `against`, then QR-orthonormalize rows; drop null rows."""
    if against is not None and against.shape[0]:
        v = v - (v @ against.conj().T) @ against
        v = v - (v @ against.conj().T) @ against  # second pass for stability
    q, r = torch.linalg.qr(v.T, mode="reduced")
    keep = r.diagonal().abs() > 1e-10
    return q.T[keep]


@torch.no_grad()
def davidson(
    h_apply: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,  # (nb, npw) initial guess, rows ~orthonormal
    t_g: torch.Tensor,  # (npw,) kinetic diagonal for the preconditioner
    tol: float = 1e-9,
    max_iter: int = 40,
    max_dim_factor: int = 4,
) -> DavidsonResult:
    nb, npw = x0.shape
    max_dim = min(max_dim_factor * nb, npw)

    v = _orthonormalize(x0.clone())
    if v.shape[0] < nb:  # degenerate guess — pad with random
        gen = torch.Generator(device="cpu").manual_seed(1234)
        pad = torch.randn(nb - v.shape[0], npw, generator=gen, dtype=torch.float64) + 1j * (
            torch.randn(nb - v.shape[0], npw, generator=gen, dtype=torch.float64)
        )
        v = torch.cat([v, _orthonormalize(pad.to(x0.dtype).to(x0.device), against=v)])
    hv = h_apply(v)

    eig = torch.zeros(nb, dtype=torch.float64, device=x0.device)
    x = v[:nb]
    res_norms = torch.full((nb,), float("inf"), dtype=torch.float64, device=x0.device)

    for it in range(1, max_iter + 1):
        # Rayleigh–Ritz on the current subspace. Convention here: s conjugates
        # hv, so s = conj(⟨v_i|H|v_j⟩) — the complex conjugate of the true
        # subspace matrix (equal to it after Hermitian symmetrization, so the
        # eigenvalues are unchanged). The eigenvectors u come out conjugated to
        # match, so the Ritz combination conjugates u below. (davidson_batched
        # uses the opposite pair: s conjugates v, u is used unconjugated.)
        s = v @ hv.conj().T  # (m, m) — rows of v are the basis
        s = 0.5 * (s + s.conj().T)
        opcount.bump("eigh")
        w, u = torch.linalg.eigh(s)
        eig = w[:nb].real
        x = u[:, :nb].T.conj() @ v  # (nb, npw) Ritz vectors
        hx = u[:, :nb].T.conj() @ hv

        r = hx - eig[:, None] * x
        res_norms = torch.linalg.norm(r, dim=1).real
        unconverged = res_norms > tol
        if not bool(unconverged.any()):
            return DavidsonResult(eig, x, it, res_norms)

        # precondition unconverged residuals and expand
        t_band = torch.einsum("bg,g,bg->b", x.conj(), t_g.to(x.dtype), x).real
        t = teter(r[unconverged], t_g, t_band[unconverged])
        t = _orthonormalize(t, against=v)
        if t.shape[0] == 0:
            return DavidsonResult(eig, x, it, res_norms)

        if v.shape[0] + t.shape[0] > max_dim:
            # restart: collapse to Ritz vectors, keep new directions
            v = _orthonormalize(torch.cat([x, t]))
            hv = h_apply(v)
        else:
            v = torch.cat([v, t])
            hv = torch.cat([hv, h_apply(t)])

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Davidson hit max_iter=%d with %d/%d bands unconverged "
            "(max res=%.3e > tol=%.1e)", max_iter, int((res_norms > tol).sum()),
            eig.shape[0], float(res_norms.max()), tol)
    return DavidsonResult(eig, x, max_iter, res_norms)


# ---------------------------------------------------------------------------
# k-batched variant: all k-points advance together with uniform subspace size,
# so every step is one batched tensor op (batched FFT H-applies, batched QR,
# batched eigh). No locking — converged bands ride along; the cost of a step
# is set by the batch anyway, and uniformity is what buys the throughput.
# ---------------------------------------------------------------------------


def _orthonormalize_b(
    v: torch.Tensor, mask: torch.Tensor, against: torch.Tensor | None = None,
    jitter: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rows of v (nk, j, npw) → orthonormal per k; padded slots stay zero.

    For rank-deficient input, QR's surplus columns are arbitrary orthonormal
    complements that may LEAK INTO PADDED SLOTS (spurious near-zero Ritz
    values). Rows that are (near-)zero AFTER projection get a deterministic
    masked jitter (then re-projection) so the input is full-rank INSIDE the
    masked complement space; healthy rows stay bit-exact — a blanket jitter
    would put a noise floor under the SCF density residual.

    jitter: optional pre-generated (nk, ≥j, npw) noise for the sync-free
    path — the `bool(any())` shortcut below is a host sync every call.
    Rows arrive unit-normalized there, so a BLANKET 1e-10 relative
    jitter folded in before the single projection rank-repairs zero rows
    (QR normalizes the surviving 1e-10 direction) while perturbing
    healthy rows at 1e-10, far below any solver tolerance; the
    conditional path's second projection never runs.
    """

    def project(x):
        if against is not None and against.shape[1]:
            for _ in range(2):  # two passes for stability
                x = x - (x @ against.conj().transpose(-1, -2)) @ against
        return x

    def orthonormalize(x):
        """Row-orthonormalize x: CholQR2 first (GEMM-shaped, every device),
        Householder QR on breakdown (rank-deficient blocks need QR's arbitrary
        orthonormal complements for the padded slots to stay consistent)."""
        if _CHOLQR_ENV != "off":
            q = _cholqr2(x.transpose(-1, -2))
            if q is not None:
                return q.transpose(-1, -2)
        q, _ = _qr_offload(x.transpose(-1, -2))
        return q.transpose(-1, -2)

    if jitter is not None:
        v = v + 1e-10 * jitter[:, : v.shape[1]]
    v = project(v * mask[:, None, :])
    if jitter is not None:
        return orthonormalize(v)
    row_norm = torch.linalg.norm(v, dim=-1, keepdim=True).real
    degenerate = row_norm < 1e-8
    if bool(degenerate.any()):
        gen = torch.Generator(device="cpu").manual_seed(v.shape[1] + 7919)
        noise = torch.randn(*v.shape, 2, generator=gen, dtype=torch.float64)
        jit = torch.view_as_complex(noise).to(v.device).to(v.dtype)
        v = project((v + degenerate * jit) * mask[:, None, :])
    return orthonormalize(v)


def _rr(q: torch.Tensor, hq: torch.Tensor, nw: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rayleigh-Ritz on an orthonormal basis `q` with precomputed images `hq`.

    q, hq: (nk, dim, m). davidson_batched convention: s = <q_i|H|q_j> built with
    the conjugate on the bra, u used UNCONJUGATED in the combination. Returns the
    nw lowest Ritz values (nk, nw) and the (nk, dim, nw) coefficient block c whose
    columns combine `q` (or `hq`) into the Ritz vectors (or their images) via
    ``_combine(c, q)``. Shared with CheFSI (davidson_batched keeps its own s
    built from lazy-conj views to avoid a large-nk memory spike).
    """
    s = torch.einsum("kig,kjg->kij", q.conj(), hq)
    # Subspace reduction ALWAYS in fp64 (issue #136), same contract as
    # davidson_batched's inline RR: an fp32 eigh of an ill-conditioned subspace
    # matrix in the complex64 draft phase yields inaccurate Ritz rotations.
    # Upcast only the small (nk, m, m) matrix and the eigensolve; returns are
    # downcast so the caller's basis and Ritz combination stay in the block
    # dtype. (davidson_batched keeps its matmul-built s for memory, but the
    # fp64 contract is now identical in both.)
    s = (0.5 * (s + s.conj().transpose(-1, -2))).to(torch.complex128)
    w, u = _eigh_subspace(s)
    return w[:, :nw].real.to(q.real.dtype), u[:, :, :nw].to(q.dtype)


def _combine(c: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """Ritz combination: (nk, dim, nw) coeffs @ (nk, dim, m) basis -> (nk, nw, m)."""
    return torch.einsum("kja,kjg->kag", c, block)


def _buffered_block(x0: torch.Tensor, n_buffer: int | None) -> tuple[torch.Tensor, int, int]:
    """Widen the initial block with `n_buffer` random buffer bands.

    CheFSI carries extra bands above the requested nb so the top wanted band
    does not stall on the block/filter edge (a real failure mode,
    not a bug); only the lowest nb are gated and returned. Returns
    (x0w, n_buffer, nw): `n_buffer` resolved from its default when None,
    nw = nb + n_buffer, and x0w = x0 padded with deterministic random bands
    (bit-identical to the former inline copies — same seed, same construction).
    """
    nk, nb, m = x0.shape
    if n_buffer is None:
        n_buffer = min(max(2, (nb + 7) // 8), m - nb)
    nw = nb + n_buffer  # working block width (gated set is the lowest nb of these)
    if not n_buffer:
        return x0, n_buffer, nw
    gen = torch.Generator(device="cpu").manual_seed(nb + 104729)
    pad = torch.view_as_complex(
        torch.randn(nk, n_buffer, m, 2, generator=gen, dtype=torch.float64)
    ).to(x0.device).to(x0.dtype)
    return torch.cat([x0, pad], dim=1), n_buffer, nw


@dataclass
class BatchedDavidsonResult:
    eigenvalues: torch.Tensor  # (nk, nb) ascending
    eigenvectors: torch.Tensor  # (nk, nb, npw_max), padded slots zero
    n_iter: int
    residual_norms: torch.Tensor  # (nk, nb)
    # Apply bookkeeping in band-vector units (nk·rows per H call). "low" counts
    # applies routed through the fp32-expansion cast (complex64 compute, result
    # stored back in the block dtype); "full" counts block-dtype applies —
    # certification/refresh applies in fp32-expansion mode, every apply
    # otherwise. low/(low+full) bounds the fraction of apply work the mode can
    # accelerate. Defaults keep positional construction by CheFSI valid.
    n_apply_low: int = 0
    n_apply_full: int = 0


@torch.no_grad()
def davidson_batched(
    h_apply: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,  # (nk, nb, npw_max), padded slots zero
    t: torch.Tensor,  # (nk, npw_max) kinetic diagonal, 0 in padding
    mask: torch.Tensor,  # (nk, npw_max) bool
    tol: float = 1e-9,
    max_iter: int = 40,
    max_dim_factor: int = 4,
    sync_free: bool = False,
    history_out: list[tuple[int, torch.Tensor, torch.Tensor]] | None = None,
) -> BatchedDavidsonResult:
    """Batched block Davidson over all k at once (the workhorse solver).

    All k-points advance together with a uniform subspace size, so every step
    is one batched tensor op (see the section comment above). The remainder of
    this docstring documents the ``sync_free`` flag and its measured verdict.

    sync_free removes every per-round host readback: convergence stats
    travel through a non-blocking copy into pinned memory and are judged
    one round late via a CUDA event query that never blocks; worst case
    is one extra expansion round whose Rayleigh–Ritz solution is strictly
    better. MEASURED VERDICT (RTX 3050, C 50 Ry, 2026-07-15): still
    slower than the synchronous path at 4³ AND 6³ (0.85 vs 0.73 s/it;
    2.37 vs 2.16) — the binding constraint at these sizes is the eager-
    mode host ISSUING dozens of small kernels per round, not the syncs
    riding on it, and the delayed expansion count does extra H-apply
    work. Default off.

    UPDATE (2026-07-28): the CUDA-graphs round capture this docstring used to
    recommend as "the real fix" was tried and does NOT help — see
    docs/manual/performance.md, "What does not help". A graphed replay of a
    whole round's post-eigh math (this sync_free skeleton is exactly the
    branch-free substrate a capture needs) reproduced eager output bit-for-bit
    but ran at 1.0x eager speed; isolating the pure-GPU remainder (eigh and QR
    excluded) still showed no gap. Kernels in this loop are already
    back-to-back on this hardware, extending the same finding an earlier probe
    made for the Hamiltonian apply alone. The actual outlier the same
    investigation surfaced was `torch.linalg.qr` on the (nk, npw, cols)
    expansion-direction shape, which cuSOLVER runs ~10x slower than a CPU
    LAPACK round trip for cols this small — bigger than the apply itself. That
    is fixed directly (`_qr_offload`, used by `_orthonormalize_b`), the same
    CPU-offload pattern issue #133 used for the subspace eigh; it needs no
    flag and is not conditioned on sync_free."""
    nk, nb, m = x0.shape
    max_dim = min(max_dim_factor * nb, int(mask.sum(dim=1).min()))
    rdtype = x0.real.dtype  # float32 in the mixed-precision draft phase, else float64

    # fp64-certified fp32-expansion mode (see the module-level note). Only the
    # synchronous gate supports it: sync_free's delayed convergence readback
    # cannot interleave the certification recompute, and sync_free is a measured
    # loss anyway (docstring below), so the combination is simply not taken.
    # Requires a dtype-polymorphic h_apply (BatchedHamiltonian.apply is; a test
    # operator must cast its matrix to c.dtype).
    use_fp32_exp = (not sync_free) and _fp32_expansion_active(x0)
    n_apply_low = 0
    n_apply_full = 0

    def _apply_full(z: torch.Tensor) -> torch.Tensor:
        nonlocal n_apply_full
        n_apply_full += z.shape[0] * z.shape[1]
        return h_apply(z)

    def _apply_exp(z: torch.Tensor) -> torch.Tensor:
        """Expansion-vector apply: complex64 in fp32-expansion mode, else full."""
        if not use_fp32_exp:
            return _apply_full(z)
        nonlocal n_apply_low
        n_apply_low += z.shape[0] * z.shape[1]
        return h_apply(z.to(torch.complex64)).to(z.dtype)

    # Certification trigger state: the estimated fp32 residual-norm floor
    # (kinetic-diagonal max as the ‖H‖ proxy) and the previous round's MEASURED
    # max norm for the stall test. Both are heuristics for WHEN to pay an exact
    # nb-wide apply; correctness never depends on them (every return path below
    # re-measures in fp64).
    cert_floor = _FP32_CERT_FLOOR_EPS * max(float(t.max()), 1.0) if use_fp32_exp else 0.0
    prev_rnmax = float("inf")

    jitter = None
    ev = flag_host = pending_stats = None
    use_event = sync_free and x0.is_cuda
    if sync_free:
        gen = torch.Generator(device="cpu").manual_seed(nb + 7919)
        noise = torch.randn(nk, nb + 1, m, 2, generator=gen,
                            dtype=torch.float64)
        jitter = torch.view_as_complex(noise).to(x0.device).to(x0.dtype)
        if use_event:
            ev = torch.cuda.Event()
            flag_host = torch.zeros(2, dtype=torch.float64).pin_memory()
        else:
            # CPU twin of the delayed-check algorithm (for testing it
            # without a GPU); reads are free here
            flag_host = torch.zeros(2, dtype=torch.float64)
    n_add_cur = nb
    pending = False

    v = _orthonormalize_b(x0, mask, jitter=jitter)
    hv = _apply_exp(v)
    # 3M complex-GEMM plane cache (Piece 1). Maintained only on an eligible solve
    # (complex128; CUDA under "auto"); one deinterleave of the NEW columns per
    # grow round carries the old-column planes forward, so a single view_as_real
    # feeds all three RR GEMMs (BUILD + two COMBINE). Native path leaves these
    # None and pays nothing.
    rr3m_on = _rr3m_eligible(x0)
    vr, vi = _planes(v) if rr3m_on else (None, None)
    hvr, hvi = _planes(hv) if rr3m_on else (None, None)
    eig = torch.zeros(nk, nb, dtype=rdtype, device=x0.device)
    x = v[:, :nb]
    rn = torch.full((nk, nb), float("inf"), dtype=rdtype, device=x0.device)
    # Carried raw (un-symmetrized) subspace matrix for the incremental RR build
    # (Piece 2). None forces a full rebuild; the grow branch sets it to the
    # extended matrix, restart/handoff reset it to None.
    s_carry: torch.Tensor | None = None

    for it in range(1, max_iter + 1):
        # Convention here is the opposite of single-k davidson()'s: s conjugates
        # v, so s = ⟨v_i|H|v_j⟩ is the true subspace matrix directly and u is
        # used UNCONJUGATED in the Ritz combination below. (Single-k conjugates
        # hv instead and then conjugates u; both are correct — the two just
        # differ by an overall complex conjugation that cancels.)
        # matmul on the lazy-conj + transpose VIEWS: same contraction as
        # einsum("kig,kjg->kij", v.conj(), hv) without materializing a
        # conj copy of the whole subspace — that transient peaks right
        # before restart and was the A100 large-nk memory spike
        # Rayleigh-Ritz subspace reduction ALWAYS in fp64 (issue #136): the
        # (nk, m, m) matrix and its eigh are tiny, but an fp32 eigh of a
        # supercell insulator block — whose orbital condition number can outrun
        # the ~1e7 fp32 budget (Higham & Mary 2022) — yields inaccurate Ritz
        # rotations in the complex64 draft. Upcast only the small matrix and the
        # eigensolve; the long v/hv basis and the Ritz combination stay
        # complex64. A no-op in the fp64 polish, mirroring the generalized
        # reduction in scf/uspp_batch.py.
        # 3M/Karatsuba complex GEMM (Piece 1) behind the per-signature measured
        # gate: the BUILD s = conj(v)@hv.mT and the two COMBINE contractions run
        # as 3 real fp64 GEMMs on the cached planes when the trial says 3M wins
        # for this (device, nk, m, npw, dtype); native ZGEMM otherwise (the
        # wide-many-k / small-block fallback). Verdict shared across all three.
        use3m = False
        if rr3m_on:
            assert vr is not None and vi is not None
            assert hvr is not None and hvi is not None
            use3m = _rr3m_verdict(v, hv, vr, vi, hvr, hvi)
        # Incremental RR build (Piece 2): on a grow round the old columns of v/hv
        # are bit-identical, so the old (m_old x m_old) block of the RAW subspace
        # matrix is unchanged; the grow branch below carries it forward in
        # `s_carry` and recomputes only the new column blocks. A full rebuild
        # runs on the first round and after every restart / fp32 handoff (where
        # the basis is replaced wholesale), gated by 3M exactly as before.
        if s_carry is not None and _RR_INCREMENTAL:
            s_raw = s_carry
        elif use3m:
            assert vr is not None and vi is not None
            assert hvr is not None and hvi is not None
            s_raw = _gemm3_conjl(vr, vi, hvr.transpose(-1, -2), hvi.transpose(-1, -2))
        else:
            s_raw = torch.matmul(v.conj(), hv.mT)
        s = (0.5 * (s_raw + s_raw.conj().transpose(-1, -2))).to(torch.complex128)
        w, u = _eigh_subspace(s)
        u = u[:, :, :nb].to(v.dtype)  # rotation back to the block dtype
        eig = w[:, :nb].real.to(rdtype)
        if use3m:
            assert vr is not None and vi is not None
            assert hvr is not None and hvi is not None
            ur, ui = _planes(u)  # small (nk, m, nb); rebuilt every round
            urt, uit = ur.transpose(-1, -2), ui.transpose(-1, -2)
            x = _gemm3(urt, uit, vr, vi)
            hx = _gemm3(urt, uit, hvr, hvi)
        else:
            x = torch.einsum("kja,kjg->kag", u, v)
            hx = torch.einsum("kja,kjg->kag", u, hv)

        r = hx - eig[..., None] * x
        rn = torch.linalg.norm(r, dim=-1).real
        if use_fp32_exp:
            # Certification: the rn just measured is fp32-sourced (hv carries
            # the cast-apply error), so it may only STEER; when it suggests
            # convergence (under max(tol, estimated fp32 floor)) or stalls at
            # its noise floor, re-measure the retained block against the
            # full-precision operator. The certified (eig, x, hx, r, rn)
            # replace the fp32-sourced ones: the return gate below then judges
            # fp64 residuals of an fp64-refreshed block, and a failed
            # certification feeds exact residual directions to the expansion.
            rnmax = float(rn.max())
            stalled = it > 1 and rnmax > _FP32_CERT_STALL * prev_rnmax
            prev_rnmax = rnmax
            if rnmax < max(tol, cert_floor) or stalled:
                eig, x, hx, r, rn = _certify_fp64(_apply_full, x)
                if float(rn.max()) >= tol:
                    # fp64 handoff on the first FAILED certification. The Ritz
                    # rotation inherits the fp32 apply error of hv (the fp64
                    # eigh is exact, but its input matrix is not), so the TRUE
                    # residual of an fp32-imaged subspace floors near
                    # eps32·‖H‖ regardless of how the reduction is done —
                    # once fp32 rounds stop reaching tol, more of them only
                    # thrash at that floor (measured: repeated cert+restart
                    # cycles cost MORE fp64 applies than handing off at once).
                    # Collapse to the certified block with its EXACT images
                    # and finish the solve with full-precision expansion
                    # applies — a draft→polish schedule INSIDE one solve whose
                    # boundary is an fp64 certification, never an fp32
                    # convergence claim.
                    v, hv = x, hx
                    use_fp32_exp = False
                    s_carry = None  # basis replaced — force a full RR rebuild
                    # v/hv just shrank to the nb-column certified block; the grow
                    # branch below reuses `s_raw` as its top-left, so it must now
                    # match the NEW basis (the BUILD above built it for the old,
                    # wider basis). Recompute for the nb block (small, native).
                    s_raw = torch.matmul(v.conj(), hv.mT)
                    if rr3m_on:  # basis replaced wholesale — rebuild planes
                        vr, vi = _planes(v)
                        hvr, hvi = _planes(hv)
        if history_out is not None:  # opt-in per-band convergence telemetry (off by default)
            history_out.append((it, rn.detach().to("cpu").clone(),
                                eig.detach().to("cpu").clone()))
        if not sync_free:
            if float(rn.max()) < tol:
                return BatchedDavidsonResult(eig, x, it, rn,
                                             n_apply_low, n_apply_full)
            # expand with the worst unconverged residuals only — uniform
            # count across k (max over k of the per-k unconverged tally)
            # keeps batching
            n_add = int((rn > tol).sum(dim=1).max())
        else:
            # judge the stats copy launched in an earlier round; query()
            # never blocks, and the returned (eig, x) are the CURRENT
            # round's — at least one refinement past the converged one.
            # ev/flag_host are only None when sync_free is False (the
            # `else` branch we are in implies sync_free, which is what set
            # them); the asserts are pure type narrowing for ty, not new
            # runtime checks on the default (non-sync_free) hot path.
            ready = not use_event
            if pending and use_event:
                assert ev is not None
                ready = ev.query()
            if pending and ready:
                assert flag_host is not None
                if float(flag_host[0]) < tol:
                    return BatchedDavidsonResult(eig, x, it, rn,
                                                 n_apply_low, n_apply_full)
                n_add_cur = max(1, min(nb, int(flag_host[1])))
                pending = False
            if not pending:
                assert flag_host is not None
                pending_stats = torch.stack(
                    [rn.max().to(torch.float64),
                     (rn > tol).sum(dim=1).max().to(torch.float64)])
                flag_host.copy_(pending_stats, non_blocking=use_event)
                if use_event:
                    assert ev is not None
                    ev.record()
                pending = True
            n_add = n_add_cur
        sel = torch.argsort(rn, dim=1, descending=True)[:, :n_add]  # (nk, n_add)
        r_sel = torch.gather(r, 1, sel[..., None].expand(-1, -1, m))

        t_band = torch.einsum("kbg,kg,kbg->kb", x.conj(), t.to(x.dtype), x).real
        tb_sel = torch.gather(t_band, 1, sel)
        d = teter_b(r_sel, t, tb_sel)
        # unit-normalize rows BEFORE ortho: near-converged residuals are
        # tiny but their DIRECTIONS are the information; below the 1e-8
        # threshold _orthonormalize_b replaces them with rank-safety
        # jitter — random directions that waste the whole expansion round.
        # The USPP batched solver learned this first; measured here the
        # jitter fired on most rounds (251 H-applies for 9 SCF solves,
        # 1.4 s of torch.randn in a 21 s profile).
        dn = torch.linalg.norm(d, dim=-1, keepdim=True).real
        d = torch.where(dn > 1e-300, d / dn.clamp_min(1e-300), d)

        if v.shape[1] + n_add > max_dim:
            # Restart reusing hx (no H re-application) — but the Ritz block
            # accumulates orthonormality drift across restarts, which at tight
            # tolerances corrupts the Rayleigh–Ritz projection (observed as a
            # ~1 eV energy jump on CUDA). Kill the drift with a QR of x and
            # transform hx by the same triangular factor: x_old = Rᵀ·x_new ⇒
            # hx_new = (Rᵀ)⁻¹·hx_old. Cost: one (nb × nb) triangular solve.
            q, rmat = _qr_offload(x.transpose(-1, -2))
            x_orth = q.transpose(-1, -2)
            hx_orth = torch.linalg.solve_triangular(
                rmat.transpose(-1, -2), hx, upper=False
            )
            d = _orthonormalize_b(d, mask, against=x_orth, jitter=jitter)
            hd = _apply_exp(d)
            if _RR_INCREMENTAL:
                # Piece 3: the retained subspace block is exactly diag(eig) in the
                # pre-QR Ritz basis (x are Ritz vectors, so <x_a|H|x_b>=eig_a δ_ab
                # to round-off); under the drift-killing QR factor R (rmat) it
                # becomes s_ret = R^-H diag(eig) R^-1 — two tiny batched triangular
                # solves, NO npw contraction. On the fp64-crippled 3050 this beats
                # the fat retained-block GEMM by 2.4x-340x (measured probe sweep),
                # so the restart round is incremental too: only the d-involving
                # cross blocks pay an npw GEMM. Carry the result as s_carry for the
                # next round's BUILD instead of forcing a full rebuild.
                rmat_c = rmat.to(torch.complex128)
                diag_e = torch.diag_embed(eig.to(torch.complex128))
                z = torch.linalg.solve_triangular(
                    rmat_c, diag_e, upper=True, left=False)
                s_ret = torch.linalg.solve_triangular(
                    rmat_c.mH, z, upper=False, left=True).to(x.dtype)
                # cross blocks with the new directions d (npw GEMMs; d is new)
                r_tr = torch.matmul(x_orth.conj(), hd.mT)
                r_co = torch.matmul(d.conj(), hd.mT)
                # Piece 4: bottom-left = top-right^H by Hermiticity (one GEMM saved)
                r_bl = (r_tr.conj().transpose(-1, -2).contiguous() if _RR_HALFBUILD
                        else torch.matmul(d.conj(), hx_orth.mT))
                s_carry = torch.cat([torch.cat([s_ret, r_tr], dim=2),
                                     torch.cat([r_bl, r_co], dim=2)], dim=1)
            else:
                s_carry = None  # A/B seam: full RR rebuild next round
            v = torch.cat([x_orth, d], dim=1)
            hv = torch.cat([hx_orth, hd], dim=1)
            if rr3m_on:  # basis rebuilt at restart — recompute planes fully
                vr, vi = _planes(v)
                hvr, hvi = _planes(hv)
        else:
            d = _orthonormalize_b(d, mask, against=v, jitter=jitter)
            hd = _apply_exp(d)
            # Incremental RR build (Piece 2): recompute only the new column blocks
            # of the raw subspace matrix (old block is `s_raw`, carried forward).
            # top-right = conj(v_old)@hd.mT, bottom-left = conj(d)@hv_old.mT,
            # corner = conj(d)@hd.mT (v/hv still hold the OLD columns here).
            dr = di = hdr = hdi = None
            if rr3m_on:
                dr, di = _planes(d)
                hdr, hdi = _planes(hd)
            if use3m:
                assert vr is not None and vi is not None
                assert hvr is not None and hvi is not None
                assert dr is not None and di is not None
                assert hdr is not None and hdi is not None
                hdrt, hdit = hdr.transpose(-1, -2), hdi.transpose(-1, -2)
                s_tr = _gemm3_conjl(vr, vi, hdrt, hdit)
                s_co = _gemm3_conjl(dr, di, hdrt, hdit)
                # Piece 4: bottom-left = top-right^H by Hermiticity (one GEMM saved)
                s_bl = (s_tr.conj().transpose(-1, -2).contiguous() if _RR_HALFBUILD
                        else _gemm3_conjl(dr, di, hvr.transpose(-1, -2),
                                          hvi.transpose(-1, -2)))
            else:
                s_tr = torch.matmul(v.conj(), hd.mT)
                s_co = torch.matmul(d.conj(), hd.mT)
                s_bl = (s_tr.conj().transpose(-1, -2).contiguous() if _RR_HALFBUILD
                        else torch.matmul(d.conj(), hv.mT))
            s_carry = torch.cat([torch.cat([s_raw, s_tr], dim=2),
                                 torch.cat([s_bl, s_co], dim=2)], dim=1)
            if not _RR_INCREMENTAL:
                s_carry = None  # A/B seam: force a full rebuild next round
            v = torch.cat([v, d], dim=1)
            hv = torch.cat([hv, hd], dim=1)
            if rr3m_on:  # extend planes with the n_add new columns
                assert vr is not None and vi is not None
                assert hvr is not None and hvi is not None
                assert dr is not None and di is not None
                assert hdr is not None and hdi is not None
                vr, vi = torch.cat([vr, dr], dim=1), torch.cat([vi, di], dim=1)
                hvr = torch.cat([hvr, hdr], dim=1)
                hvi = torch.cat([hvi, hdi], dim=1)

    if use_fp32_exp:
        # max_iter exit: the returned block must still be certified — one final
        # full-precision apply refreshes the Ritz pairs and re-measures rn.
        eig, x, hx, r, rn = _certify_fp64(_apply_full, x)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "batched Davidson hit max_iter=%d: %d band·k unconverged "
            "(max res=%.3e > tol=%.1e)", max_iter, int((rn > tol).sum()),
            float(rn.max()), tol)
    return BatchedDavidsonResult(eig, x, max_iter, rn, n_apply_low, n_apply_full)


@torch.no_grad()
def davidson_batched_ms(
    h_apply: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    t: torch.Tensor,
    mask: torch.Tensor,
    tol: float = 1e-9,
    max_iter: int = 40,
    max_dim_factor: int = 4,
    crossover: float = 1e-5,
    mixed_precision: bool = True,
) -> BatchedDavidsonResult:
    """Two-stage Davidson: a fast low-precision draft to `crossover`, then a
    full-precision polish to `tol` warm-started from it.

    The draft runs in the dtype of `x0` cast down to complex64 (and `t` to
    float32) so a dtype-polymorphic H apply computes in fp32 throughout — the
    regime where a GeForce is 8–32× faster than fp64. The polish re-solves in
    the original precision, so the returned eigenpairs are full-precision
    regardless of the draft. Skipped (single fp64 solve) when mixed_precision
    is off, x0 is already low precision, or crossover ≥ tol."""
    from gradwave.dtypes import real_of
    from gradwave.solvers._ms import LOW, mixed_precision_solve

    def solve(x, t_, tl):
        return davidson_batched(h_apply, x, t_, mask, tol=tl, max_iter=max_iter,
                                max_dim_factor=max_dim_factor)

    return mixed_precision_solve(
        x0, tol, crossover, mixed_precision,
        full=lambda: solve(x0, t, tol),
        make_stages=lambda: (
            lambda xlo: solve(xlo, t.to(real_of(LOW)), crossover),
            lambda x1: solve(x1, t, tol),
        ),
    )
