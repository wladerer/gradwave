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
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

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
# in this repo's benchmark battery; wider blocks (LOBPCG's stacked [X,W,P],
# CheFSI's buffered block) fall through to the GPU unchanged, same as today.
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

    if jitter is not None:
        v = v + 1e-10 * jitter[:, : v.shape[1]]
    v = project(v * mask[:, None, :])
    if jitter is not None:
        q, _ = _qr_offload(v.transpose(-1, -2))
        return q.transpose(-1, -2)
    row_norm = torch.linalg.norm(v, dim=-1, keepdim=True).real
    degenerate = row_norm < 1e-8
    if bool(degenerate.any()):
        gen = torch.Generator(device="cpu").manual_seed(v.shape[1] + 7919)
        noise = torch.randn(*v.shape, 2, generator=gen, dtype=torch.float64)
        jit = torch.view_as_complex(noise).to(v.device).to(v.dtype)
        v = project((v + degenerate * jit) * mask[:, None, :])
    q, _ = _qr_offload(v.transpose(-1, -2))
    return q.transpose(-1, -2)


def _rr(q: torch.Tensor, hq: torch.Tensor, nw: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rayleigh-Ritz on an orthonormal basis `q` with precomputed images `hq`.

    q, hq: (nk, dim, m). davidson_batched convention: s = <q_i|H|q_j> built with
    the conjugate on the bra, u used UNCONJUGATED in the combination. Returns the
    nw lowest Ritz values (nk, nw) and the (nk, dim, nw) coefficient block c whose
    columns combine `q` (or `hq`) into the Ritz vectors (or their images) via
    ``_combine(c, q)``. Shared by CheFSI and LOBPCG (davidson_batched keeps its
    own s built from lazy-conj views to avoid a large-nk memory spike).
    """
    s = torch.einsum("kig,kjg->kij", q.conj(), hq)
    s = 0.5 * (s + s.conj().transpose(-1, -2))
    w, u = _eigh_subspace(s)
    return w[:, :nw].real, u[:, :, :nw]


def _combine(c: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """Ritz combination: (nk, dim, nw) coeffs @ (nk, dim, m) basis -> (nk, nw, m)."""
    return torch.einsum("kja,kjg->kag", c, block)


def _buffered_block(x0: torch.Tensor, n_buffer: int | None) -> tuple[torch.Tensor, int, int]:
    """Widen the initial block with `n_buffer` random buffer bands.

    CheFSI and LOBPCG both carry extra bands above the requested nb so the top
    wanted band does not stall on the block/filter edge (a real failure mode,
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
) -> BatchedDavidsonResult:
    """sync_free removes every per-round host readback: convergence stats
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
    hv = h_apply(v)
    eig = torch.zeros(nk, nb, dtype=rdtype, device=x0.device)
    x = v[:, :nb]
    rn = torch.full((nk, nb), float("inf"), dtype=rdtype, device=x0.device)

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
        s = torch.matmul(v.conj(), hv.mT)
        s = (0.5 * (s + s.conj().transpose(-1, -2))).to(torch.complex128)
        w, u = _eigh_subspace(s)
        u = u[:, :, :nb].to(v.dtype)  # rotation back to the block dtype
        eig = w[:, :nb].real.to(rdtype)
        x = torch.einsum("kja,kjg->kag", u, v)
        hx = torch.einsum("kja,kjg->kag", u, hv)

        r = hx - eig[..., None] * x
        rn = torch.linalg.norm(r, dim=-1).real
        if not sync_free:
            if float(rn.max()) < tol:
                return BatchedDavidsonResult(eig, x, it, rn)
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
                    return BatchedDavidsonResult(eig, x, it, rn)
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
            v = torch.cat([x_orth, d], dim=1)
            hv = torch.cat([hx_orth, h_apply(d)], dim=1)
        else:
            d = _orthonormalize_b(d, mask, against=v, jitter=jitter)
            v = torch.cat([v, d], dim=1)
            hv = torch.cat([hv, h_apply(d)], dim=1)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "batched Davidson hit max_iter=%d: %d band·k unconverged "
            "(max res=%.3e > tol=%.1e)", max_iter, int((rn > tol).sum()),
            float(rn.max()), tol)
    return BatchedDavidsonResult(eig, x, max_iter, rn)


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
