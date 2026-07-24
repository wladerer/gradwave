"""Block Davidson for the complex Hermitian plane-wave Hamiltonian (Layer B).

Norm-conserving pseudopotentials ⇒ standard eigenproblem, no overlap matrix.
Growing subspace with Rayleigh–Ritz via torch.linalg.eigh, Teter-
preconditioned residual expansion, band locking, restart at max_dim.
Runs entirely under torch.no_grad() — autograd must never see this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from gradwave.solvers.precond import teter, teter_b

logger = logging.getLogger(__name__)


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
    h_apply,
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
            "(max res=%.3e > tol=%.1e)", max_iter, int(unconverged.sum()),
            eig.shape[0], float(res_norms.max()), tol)
    return DavidsonResult(eig, x, max_iter, res_norms)


# ---------------------------------------------------------------------------
# k-batched variant: all k-points advance together with uniform subspace size,
# so every step is one batched tensor op (batched FFT H-applies, batched QR,
# batched eigh). Uniform leading-band locking (see davidson_batched docstring)
# deflates converged low bands at restarts with a k-uniform count, so the eigh
# shrinks while every k keeps the same subspace size that buys the throughput.
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
        q, _ = torch.linalg.qr(v.transpose(-1, -2), mode="reduced")
        return q.transpose(-1, -2)
    row_norm = torch.linalg.norm(v, dim=-1, keepdim=True).real
    degenerate = row_norm < 1e-8
    if bool(degenerate.any()):
        gen = torch.Generator(device="cpu").manual_seed(v.shape[1] + 7919)
        noise = torch.randn(*v.shape, 2, generator=gen, dtype=torch.float64)
        jit = torch.view_as_complex(noise).to(v.device).to(v.dtype)
        v = project((v + degenerate * jit) * mask[:, None, :])
    q, _ = torch.linalg.qr(v.transpose(-1, -2), mode="reduced")
    return q.transpose(-1, -2)


@dataclass
class BatchedDavidsonResult:
    eigenvalues: torch.Tensor  # (nk, nb) ascending
    eigenvectors: torch.Tensor  # (nk, nb, npw_max), padded slots zero
    n_iter: int
    residual_norms: torch.Tensor  # (nk, nb)


@torch.no_grad()
def davidson_batched(
    h_apply,
    x0: torch.Tensor,  # (nk, nb, npw_max), padded slots zero
    t: torch.Tensor,  # (nk, npw_max) kinetic diagonal, 0 in padding
    mask: torch.Tensor,  # (nk, npw_max) bool
    tol: float = 1e-9,
    max_iter: int = 40,
    max_dim_factor: int = 4,
    sync_free: bool = False,
    lock: bool = True,
) -> BatchedDavidsonResult:
    """LOCKING (lock=True, default): at each subspace restart the lowest L
    bands whose residual is below `tol` at EVERY k-point are deflated out of the
    Rayleigh–Ritz eigh — frozen as an (nk, L, npw) block kept only to
    orthogonalize new directions. The eigh then runs on the (nb−L)+expansion
    active block instead of the full subspace, shrinking the #1 SCF kernel
    (dense batched `eigh`, 21–29% of runtime, worst under SOC). L is uniform
    across k (min over k of the leading contiguous converged count) so every k
    keeps the SAME subspace size and the batching that buys the throughput.
    Locking is gated to the natural (max_dim) restart, so restart frequency and
    the trajectory up to the first restart are exactly the baseline's;
    deflation only makes later eighs smaller. A band locks only after the driver
    would already call it converged, so the converged eigenvalues are unchanged
    (see PR: a locked Ritz pair has eigenvalue error ≤ ρ²/gap ≤ tol²/gap, and
    deflating it perturbs the remaining Ritz values by the same order — ~1e-18
    at tol=1e-9). Disabled on the sync_free path (which deliberately avoids the
    per-round host sync locking needs); set lock=False to recover the exact
    pre-locking numerics.

    sync_free removes every per-round host readback: convergence stats
    travel through a non-blocking copy into pinned memory and are judged
    one round late via a CUDA event query that never blocks; worst case
    is one extra expansion round whose Rayleigh–Ritz solution is strictly
    better. MEASURED VERDICT (RTX 3050, C 50 Ry, 2026-07-15): still
    slower than the synchronous path at 4³ AND 6³ (0.85 vs 0.73 s/it;
    2.37 vs 2.16) — the binding constraint at these sizes is the eager-
    mode host ISSUING dozens of small kernels per round, not the syncs
    riding on it, and the delayed expansion count does extra H-apply
    work. Default off; the path is kept as the substrate for a CUDA-
    graphs round capture, which is the real fix."""
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

    use_lock = lock and not sync_free

    v = _orthonormalize_b(x0, mask, jitter=jitter)
    hv = h_apply(v)
    eig = torch.zeros(nk, nb, dtype=rdtype, device=x0.device)
    x = v[:, :nb]
    rn = torch.full((nk, nb), float("inf"), dtype=rdtype, device=x0.device)

    # Deflated ("locked") leading block: bands whose residual fell below tol at
    # every k. Frozen and excluded from the active eigh but retained to keep new
    # directions orthogonal to them. Empty (L=0) unless use_lock; when L stays 0
    # every operation below reduces exactly to the pre-locking code path.
    x_lock = x0.new_zeros(nk, 0, m)
    hx_lock = x0.new_zeros(nk, 0, m)
    eig_lock = torch.zeros(nk, 0, dtype=rdtype, device=x0.device)
    rn_lock = torch.zeros(nk, 0, dtype=rdtype, device=x0.device)
    L = 0

    for it in range(1, max_iter + 1):
        na = nb - L  # active bands sought from the deflated subspace
        # Convention here is the opposite of single-k davidson()'s: s conjugates
        # v, so s = ⟨v_i|H|v_j⟩ is the true subspace matrix directly and u is
        # used UNCONJUGATED in the Ritz combination below. (Single-k conjugates
        # hv instead and then conjugates u; both are correct — the two just
        # differ by an overall complex conjugation that cancels.)
        # matmul on the lazy-conj + transpose VIEWS: same contraction as
        # einsum("kig,kjg->kij", v.conj(), hv) without materializing a
        # conj copy of the whole subspace — that transient peaks right
        # before restart and was the A100 large-nk memory spike
        # Rayleigh–Ritz on the ACTIVE (deflated) subspace only. With L==0 this
        # is the full subspace, identical to the pre-locking path.
        s = torch.matmul(v.conj(), hv.mT)
        s = 0.5 * (s + s.conj().transpose(-1, -2))
        w, u = torch.linalg.eigh(s)
        eig_act = w[:, :na].real
        x_act = torch.einsum("kja,kjg->kag", u[:, :, :na], v)
        hx_act = torch.einsum("kja,kjg->kag", u[:, :, :na], hv)

        r = hx_act - eig_act[..., None] * x_act
        rn_act = torch.linalg.norm(r, dim=-1).real

        # Full nb-band result = locked block (lowest, already <tol) ++ active.
        # Ascending by construction: locked bands are the lowest converged
        # bands and the active eigh targets the next-lowest above them.
        eig = torch.cat([eig_lock, eig_act], dim=1)
        x = torch.cat([x_lock, x_act], dim=1)
        rn = torch.cat([rn_lock, rn_act], dim=1)

        if not sync_free:
            if float(rn_act.max()) < tol:
                return BatchedDavidsonResult(eig, x, it, rn)
            # expand with the worst unconverged residuals only — uniform
            # count across k (max over k of the per-k unconverged tally)
            # keeps batching
            n_add = int((rn_act > tol).sum(dim=1).max())
        else:
            # judge the stats copy launched in an earlier round; query()
            # never blocks, and the returned (eig, x) are the CURRENT
            # round's — at least one refinement past the converged one
            if pending and (not use_event or ev.query()):
                if float(flag_host[0]) < tol:
                    return BatchedDavidsonResult(eig, x, it, rn)
                n_add_cur = max(1, min(nb, int(flag_host[1])))
                pending = False
            if not pending:
                pending_stats = torch.stack(
                    [rn_act.max().to(torch.float64),
                     (rn_act > tol).sum(dim=1).max().to(torch.float64)])
                flag_host.copy_(pending_stats, non_blocking=use_event)
                if use_event:
                    ev.record()
                pending = True
            n_add = n_add_cur
        # Expansion selects among ACTIVE bands; locked bands (residual <tol) have
        # the smallest residuals and are never picked by the descending sort.
        sel = torch.argsort(rn_act, dim=1, descending=True)[:, :n_add]
        r_sel = torch.gather(r, 1, sel[..., None].expand(-1, -1, m))

        t_band = torch.einsum(
            "kbg,kg,kbg->kb", x_act.conj(), t.to(x_act.dtype), x_act).real
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
            # Restart. Locking is gated to this natural (max_dim) restart so the
            # restart frequency and the pre-first-restart trajectory are exactly
            # the baseline's — deflation only makes the POST-restart subspace
            # (and thus every later eigh) smaller. Lock the leading contiguous
            # converged active bands, uniform across k (min over k) so every k
            # keeps the same subspace size: cumprod of the 0/1 converged mask
            # along bands is 1 over the leading all-converged prefix and 0
            # after, so its sum is that prefix length. We already returned above
            # if EVERY active band converged, so new_lock < na and na stays ≥1.
            new_lock = 0
            if use_lock:
                conv = (rn_act < tol).to(torch.int8)
                new_lock = int(torch.cumprod(conv, dim=1).sum(dim=1).min().item())
            x_rem, hx_rem = x_act, hx_act
            if new_lock > 0:
                x_lock = torch.cat([x_lock, x_act[:, :new_lock]], dim=1)
                hx_lock = torch.cat([hx_lock, hx_act[:, :new_lock]], dim=1)
                eig_lock = torch.cat([eig_lock, eig_act[:, :new_lock]], dim=1)
                rn_lock = torch.cat([rn_lock, rn_act[:, :new_lock]], dim=1)
                L += new_lock
                x_rem, hx_rem = x_act[:, new_lock:], hx_act[:, new_lock:]

            # Restart reusing hx (no H re-application) — but the Ritz block
            # accumulates orthonormality drift across restarts, which at tight
            # tolerances corrupts the Rayleigh–Ritz projection (observed as a
            # ~1 eV energy jump on CUDA). Kill the drift with a QR of x and
            # transform hx by the same triangular factor: x_old = Rᵀ·x_new ⇒
            # hx_new = (Rᵀ)⁻¹·hx_old. Cost: one (nb × nb) triangular solve.
            # When bands are locked, first project the remaining Ritz block off
            # the locked block; applying the SAME linear map to hx keeps
            # hx = H·x with no re-application, so orthogonality to the deflated
            # space is exact going into the next eigh.
            if L > 0:
                for _ in range(2):
                    c = torch.matmul(x_rem, x_lock.conj().mT)
                    x_rem = x_rem - torch.matmul(c, x_lock)
                    hx_rem = hx_rem - torch.matmul(c, hx_lock)
            q, rmat = torch.linalg.qr(x_rem.transpose(-1, -2), mode="reduced")
            x_orth = q.transpose(-1, -2)
            hx_orth = torch.linalg.solve_triangular(
                rmat.transpose(-1, -2), hx_rem, upper=False
            )
            against = torch.cat([x_lock, x_orth], dim=1) if L > 0 else x_orth
            d = _orthonormalize_b(d, mask, against=against, jitter=jitter)
            v = torch.cat([x_orth, d], dim=1)
            hv = torch.cat([hx_orth, h_apply(d)], dim=1)
        else:
            against = torch.cat([x_lock, v], dim=1) if L > 0 else v
            d = _orthonormalize_b(d, mask, against=against, jitter=jitter)
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
    h_apply,
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
