"""Block LOBPCG for the complex Hermitian plane-wave Hamiltonian (Layer B).

Locally Optimal Block Preconditioned Conjugate Gradient (Knyazev 2001). The
alternative to block Davidson with a FIXED subspace: each round works in
span[X, W, P] where X is the current Ritz block, W the Teter-preconditioned
residual, and P the previous search direction, so the Rayleigh-Ritz `eigh` is a
constant (nk, 3*nw, 3*nw) — no growing subspace, no restarts. The wanted band
count is small, so that small predictable eigh can beat the growing Davidson
eigh (21-29% of SCF runtime in profiles) for the lowest states.

Norm-conserving pseudopotentials => standard eigenproblem, no overlap matrix.
Batched over k-points exactly like ``davidson_batched``: all k advance together
with uniform subspace width so every step is one batched tensor op (batched FFT
H-applies, batched QR, batched eigh).

Numerics — the two things LOBPCG gets wrong if implemented naively:

1. Basis conditioning. As bands converge, P collapses onto span[X, W] and the
   [X, W, P] Gram matrix goes singular; the Rayleigh-Ritz `eigh` (which assumes
   an orthonormal basis) then returns garbage and the iterates blow up. Measured
   on a real Al metal: incrementally orthonormalizing W against X and then P
   against [X, W] let the Gram condition number climb 1 -> 1e5 -> 1e18 over a
   handful of rounds and ‖X‖ diverge. Fixed here by ONE JOINT orthonormalization
   of the whole stacked [X, W, P] per round via ``_orthonormalize_b`` (a QR whose
   masked-jitter path also drops null directions and refills them with random
   orthonormal complements, uniformly across the batch), so the basis handed to
   `eigh` is orthonormal by construction — Gram condition number pinned at 1.

   Because the joint QR mixes the blocks, H is re-applied to the whole orthonormal
   basis (~3*nw applies/round). The residual, though, reuses the EXACT image
   HX = combine(HQ) recovered from that same apply (H is linear), so no separate
   apply for X.

2. The top requested band stalls at the block edge (a real LOBPCG/CheFSI failure
   mode, not a bug): with no band above it to define the gap, the nb-th band's
   residual plateaus. Carrying `n_buffer` extra bands and gating only the nb that
   are returned fixes it — on the Al metal, buffer 0 never converges the top band
   while buffer 4 converges the wanted 10 in ~22 rounds (fewer than Davidson).

No per-band soft locking: converged bands ride along in the fixed block (as in
``davidson_batched``), keeping the batch uniform; near-converged residuals are
unit-normalized before orthogonalization so their DIRECTION survives while a
truly null W row falls to the jitter guard. The returned nb Ritz pairs are
variational over a subspace that always contains the current X, so the lowest nb
eigenvalues match a dense eigh / Davidson to the requested tolerance (~1e-13 on
both an Al metal and the synthetic operators in the unit tests).

Runs entirely under torch.no_grad() — autograd must never see this.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from gradwave.solvers.davidson import (
    BatchedDavidsonResult,
    _buffered_block,
    _combine,
    _orthonormalize_b,
    _rr,
)
from gradwave.solvers.precond import teter_b

logger = logging.getLogger(__name__)


@torch.no_grad()
def lobpcg_batched(
    h_apply: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,  # (nk, nb, npw_max), padded slots zero
    t: torch.Tensor,  # (nk, npw_max) kinetic diagonal, 0 in padding
    mask: torch.Tensor,  # (nk, npw_max) bool
    tol: float = 1e-9,
    max_iter: int = 60,
    n_buffer: int | None = None,
) -> BatchedDavidsonResult:
    """Converge the nb lowest eigenpairs of a FIXED Hermitian H by block LOBPCG.

    Drop-in for ``davidson_batched``: identical signature and result type
    (returns the nb requested bands), so an SCF loop can select it per iteration.
    `t` is the kinetic diagonal for the Teter preconditioner (as in Davidson).

    Each round: form the preconditioned residual W, JOINTLY orthonormalize
    [X, W, P] to a fixed (nk, <=3*nw, m) basis, one batched Rayleigh-Ritz eigh,
    take the lowest nw; only the nb returned bands are gated for convergence.
    `n_buffer` carries nw-nb extra bands so the nb-th does not stall at the block
    edge (default a small nb-dependent value; pass 0 to disable).
    """
    nk, nb, m = x0.shape
    rdtype = x0.real.dtype
    tc = t.to(x0.dtype)  # complex kinetic diagonal for the t_band contraction
    x0w, n_buffer, nw = _buffered_block(x0, n_buffer)

    # X orthonormal + its image; rotate to the Ritz basis of its own span so the
    # first residual is well defined (per-band Ritz values, not raw guesses).
    x = _orthonormalize_b(x0w, mask)
    hx = h_apply(x)
    eig, c = _rr(x, hx, nw)
    x = _combine(c, x)
    hx = _combine(c, hx)

    p = None
    rn = torch.full((nk, nw), float("inf"), dtype=rdtype, device=x0.device)

    for it in range(1, max_iter + 1):
        r = hx - eig[..., None] * x
        rn = torch.linalg.norm(r, dim=-1).real
        # gate only the nb returned bands; buffer bands may stay loose
        if float(rn[:, :nb].max()) < tol:
            return BatchedDavidsonResult(eig[:, :nb], x[:, :nb], it, rn[:, :nb])

        # Teter-preconditioned residual. Unit-normalize rows BEFORE ortho: a
        # near-converged residual is tiny but its DIRECTION carries the update;
        # below _orthonormalize_b's null threshold it is replaced by jitter.
        t_band = torch.einsum("kbg,kg,kbg->kb", x.conj(), tc, x).real
        w = teter_b(r, t, t_band)
        wn = torch.linalg.norm(w, dim=-1, keepdim=True).real
        w = torch.where(wn > 1e-300, w / wn.clamp_min(1e-300), w)

        # ONE joint orthonormalization of the whole basis (the conditioning
        # guard). H is then re-applied to the orthonormal Q; the residual reuses
        # the exact image HX = combine(HQ) below, so X needs no separate apply.
        stack = torch.cat([x, w], dim=1) if p is None else torch.cat([x, w, p], dim=1)
        q = _orthonormalize_b(stack, mask)
        hq = h_apply(q)

        eig, c = _rr(q, hq, nw)
        x = _combine(c, q)
        hx = _combine(c, hq)
        # new search direction P from the W/P block only (the "conjugate" part
        # not already spanned by the current X); re-orthonormalized next round.
        p = _combine(c[:, nw:, :], q[:, nw:, :])

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "batched LOBPCG hit max_iter=%d: %d band·k unconverged "
            "(max res=%.3e > tol=%.1e)", max_iter, int((rn[:, :nb] > tol).sum()),
            float(rn[:, :nb].max()), tol)
    return BatchedDavidsonResult(eig[:, :nb], x[:, :nb], max_iter, rn[:, :nb])


@torch.no_grad()
def lobpcg_batched_ms(
    h_apply: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    t: torch.Tensor,
    mask: torch.Tensor,
    tol: float = 1e-9,
    max_iter: int = 60,
    n_buffer: int | None = None,
    crossover: float = 1e-5,
    mixed_precision: bool = True,
) -> BatchedDavidsonResult:
    """Two-stage LOBPCG: a low-precision draft to `crossover`, then a full-
    precision polish warm-started from it. Mirrors ``davidson_batched_ms`` /
    ``chebyshev_filtered_batched_ms``.

    The draft runs in the dtype of `x0` cast to complex64 (and `t` to float32);
    the polish re-solves in the original precision, so the returned eigenpairs
    are full precision regardless of the draft. Skipped (single full-precision
    solve) when mixed_precision is off, x0 is already low precision, or
    crossover >= tol.
    """
    from gradwave.dtypes import real_of
    from gradwave.solvers._ms import LOW, mixed_precision_solve

    def solve(x, t_, tl):
        return lobpcg_batched(h_apply, x, t_, mask, tol=tl, max_iter=max_iter,
                              n_buffer=n_buffer)

    return mixed_precision_solve(
        x0, tol, crossover, mixed_precision,
        full=lambda: solve(x0, t, tol),
        make_stages=lambda: (
            lambda xlo: solve(xlo, t.to(real_of(LOW)), crossover),
            lambda x1: solve(x1, t, tol),
        ),
    )
