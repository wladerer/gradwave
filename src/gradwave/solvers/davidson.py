"""Block Davidson for the complex Hermitian plane-wave Hamiltonian (Layer B).

Norm-conserving pseudopotentials ⇒ standard eigenproblem, no overlap matrix.
Growing subspace with Rayleigh–Ritz via torch.linalg.eigh, Teter-
preconditioned residual expansion, band locking, restart at max_dim.
Runs entirely under torch.no_grad() — autograd must never see this.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gradwave.solvers.precond import teter, teter_b


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
        # Rayleigh–Ritz on the current subspace
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
    sync_free: bool | None = None,  # None → on for CUDA inputs
) -> BatchedDavidsonResult:
    """sync_free avoids every per-round host readback (the pipeline-drain
    bottleneck on GPUs at small sizes): convergence stats travel through a
    non-blocking copy into pinned memory and are judged one round late via
    a CUDA event query that never blocks. Worst case is one extra
    expansion round after convergence, whose Rayleigh–Ritz solution is
    strictly better. CPU behavior is bit-identical to the synchronous
    path."""
    nk, nb, m = x0.shape
    max_dim = min(max_dim_factor * nb, int(mask.sum(dim=1).min()))
    rdtype = x0.real.dtype  # float32 in the mixed-precision draft phase, else float64
    if sync_free is None:
        sync_free = x0.is_cuda

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
        s = torch.einsum("kig,kjg->kij", v.conj(), hv)
        s = 0.5 * (s + s.conj().transpose(-1, -2))
        w, u = torch.linalg.eigh(s)
        eig = w[:, :nb].real
        x = torch.einsum("kja,kjg->kag", u[:, :, :nb], v)
        hx = torch.einsum("kja,kjg->kag", u[:, :, :nb], hv)

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
            # round's — at least one refinement past the converged one
            if pending and (not use_event or ev.query()):
                if float(flag_host[0]) < tol:
                    return BatchedDavidsonResult(eig, x, it, rn)
                n_add_cur = max(1, min(nb, int(flag_host[1])))
                pending = False
            if not pending:
                pending_stats = torch.stack(
                    [rn.max().to(torch.float64),
                     (rn > tol).sum(dim=1).max().to(torch.float64)])
                flag_host.copy_(pending_stats, non_blocking=use_event)
                if use_event:
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
            q, rmat = torch.linalg.qr(x.transpose(-1, -2), mode="reduced")
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

    low = torch.complex64
    if (not mixed_precision) or x0.dtype == low or crossover >= tol:
        return davidson_batched(h_apply, x0, t, mask, tol=tol, max_iter=max_iter,
                                max_dim_factor=max_dim_factor)
    hi_dtype = x0.dtype
    draft = davidson_batched(
        h_apply, x0.to(low), t.to(real_of(low)), mask,
        tol=crossover, max_iter=max_iter, max_dim_factor=max_dim_factor,
    )
    x1 = draft.eigenvectors.to(hi_dtype)
    return davidson_batched(h_apply, x1, t, mask, tol=tol, max_iter=max_iter,
                            max_dim_factor=max_dim_factor)
