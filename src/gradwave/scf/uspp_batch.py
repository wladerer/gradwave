"""k-batched USPP/PAW eigensolve (Layer B fast path).

Batches the generalized problem H|ψ⟩ = εS|ψ⟩ over all k at once using the
NC padded-batch machinery (core/batch.py): H reuses BatchedHamiltonian with
the per-iteration screened D swapped in; S = 1 + Σ q|β⟩⟨β| is one becp
contraction. davidson_gen_batched mirrors davidson_batched (uniform
subspace across k, worst-residual expansion, QR-restart reusing cached
applies) and the per-k davidson_gen reduction (standard-orthonormal basis,
batched Cholesky L⁻¹HL⁻† — NOT L⁻¹HL⁻¹, the complex-eigensolver trap).

Correctness contract: identical eigenpairs to the per-k davidson_gen path
at the same tolerance (validated on the Si USPP/PAW fast test).
"""

from __future__ import annotations

import dataclasses

import torch

from gradwave.core.batch import BatchedHamiltonian, BatchedK, becp_b
from gradwave.solvers.davidson import _orthonormalize_b, teter_b


class BatchedHS:
    """H and S applies for all k at once, fixed v_eff and screened D.
    hub_sphi/hub_d: optional DFT+U term (S-dressed orbital projectors +
    Dudarev D, already conj-transposed for the apply convention)."""

    def __init__(self, bk: BatchedK, shape, v_eff_r, p, dscr, q_full,
                 hub_sphi=None, hub_d=None, smooth=None):
        self.bk = bk
        self.p = p  # (nk, nproj, npw_max)
        self.q = q_full.to(p.dtype)
        self.ham = BatchedHamiltonian(
            dataclasses.replace(bk, dij_full=dscr), shape, v_eff_r, p,
            hub_q=hub_sphi, hub_dij=hub_d, smooth=smooth,
        )
        self.t = bk.t
        self._pq_cache: dict = {}  # cdtype → (p, p_conj, q) for mixed precision

    def _pq(self, cdtype):
        cached = self._pq_cache.get(cdtype)
        if cached is None:
            p = self.p.to(cdtype)
            # conjugate cached: S is applied every Davidson round and a fresh
            # p.conj() there re-materializes the whole projector table
            cached = (p, p.conj().resolve_conj(), self.q.to(cdtype))
            self._pq_cache[cdtype] = cached
        return cached

    def h(self, c):
        return self.ham.apply(c)  # dtype-follows c (BatchedHamiltonian._tables)

    def s(self, c):
        p, p_conj, q = self._pq(c.dtype)
        b = becp_b(p, c, p_conj=p_conj)
        return (c + torch.einsum("kbp,pq,kqg->kbg", b, q, p)
                ) * self.bk.mask[:, None, :]


def davidson_gen_batched(hs: BatchedHS, x0: torch.Tensor, nbands: int,
                         tol: float, max_iter: int = 60,
                         max_dim_factor: int = 4):
    """Block Davidson for H x = ε S x over (nk, ·, npw_max) padded blocks.

    Returns (eig (nk, nbands), x (nk, nbands, npw_max)); x is Ritz-rotated
    from a standard-orthonormal basis (not S-normalized — callers normalize).
    """
    nk, nb0, m = x0.shape
    mask = hs.bk.mask
    max_dim = min(int(hs.bk.npw.min()),
                  max(max_dim_factor * nbands, nbands + 24))

    def _contract(x_r, hx_r, sx_r):
        """QR-restart: re-orthonormalize the Ritz block in the standard
        metric, rotating the cached applies with the triangular factor."""
        q, rmat = torch.linalg.qr(x_r.transpose(-1, -2), mode="reduced")
        rt = rmat.transpose(-1, -2)
        return (q.transpose(-1, -2).contiguous(),
                torch.linalg.solve_triangular(rt, hx_r, upper=False),
                torch.linalg.solve_triangular(rt, sx_r, upper=False))

    def _subspace(v_, hv_, sv_):
        # subspace algebra ALWAYS in fp64: the generalized reduction's
        # Cholesky of a near-singular S is exactly where fp32 produces
        # garbage rotations; the matrices are (nk, nsub, nsub)-tiny, so
        # upcasting costs nothing while the applies stay low-precision.
        # matmul on the lazy-conj + transpose views — same contraction as
        # einsum("kig,kjg->kij", v.conj(), ·) without materializing a conj
        # copy of the whole subspace (the large-nk memory spike)
        vc = v_.conj()
        h_sub = torch.matmul(vc, hv_.mT).to(torch.complex128)
        s_sub = torch.matmul(vc, sv_.mT).to(torch.complex128)
        return (0.5 * (h_sub + h_sub.conj().transpose(-1, -2)),
                0.5 * (s_sub + s_sub.conj().transpose(-1, -2)))

    v = _orthonormalize_b(x0, mask)
    hv, sv = hs.h(v), hs.s(v)
    eig = x = hx = sx = None
    for _ in range(max_iter):
        # at low ecut the USPP S can be INDEFINITE on the truncated sphere
        # (negative q eigenvalues; QE errors out the same way) — an orthonormal
        # basis touching those directions makes vSv† non-PD, and the reduction
        # then yields garbage rotations (spurious below-minimum states). The
        # Cholesky factorization is exactly the non-PD detector, since
        # cholesky_ex reports info>0 the moment vSv† loses positive-definiteness.
        # Drop the
        # OLDEST subspace entries while it fails (keeps the newest curvature;
        # contracting to the Ritz block instead KEEPS the contaminated
        # directions and cycles forever). A prior guard also computed a full
        # linalg.cond every round, but probing a low-ecut Si PAW SCF (8-12 Ry)
        # showed the overlap tips into non-PD (info>0) long before its condition
        # number nears the 1e14 trip (max observed ~9e7), so the cond SVD never
        # fired independently. The factorization catch is the whole guard, as
        # in the per-k path.
        while True:
            h_sub, s_sub = _subspace(v, hv, sv)
            ell, info = torch.linalg.cholesky_ex(s_sub)
            bad = int(info.max()) > 0
            if not bad or v.shape[1] <= nbands + 1:
                break
            v, hv, sv = v[:, 1:], hv[:, 1:], sv[:, 1:]
        if int(info.max()) > 0:
            eye = torch.eye(s_sub.shape[-1], dtype=s_sub.dtype,
                            device=s_sub.device)
            ell = torch.linalg.cholesky(s_sub + 1e-10 * eye)
        a = torch.linalg.solve_triangular(ell, h_sub, upper=False)  # L⁻¹H
        a = torch.linalg.solve_triangular(
            ell, a.conj().transpose(-1, -2), upper=False
        ).conj().transpose(-1, -2)  # (L⁻¹H)L⁻†
        w, u = torch.linalg.eigh(0.5 * (a + a.conj().transpose(-1, -2)))
        u = torch.linalg.solve_triangular(ell.conj().transpose(-1, -2), u,
                                          upper=True)
        eig = w[:, :nbands].real
        u_r = u[:, :, :nbands].transpose(-1, -2).to(x0.dtype)  # (nk, nb, nsub)
        x = torch.einsum("kbj,kjg->kbg", u_r, v)
        hx = torch.einsum("kbj,kjg->kbg", u_r, hv)
        sx = torch.einsum("kbj,kjg->kbg", u_r, sv)

        r = hx - eig[..., None].to(x0.dtype) * sx
        rn = torch.linalg.norm(r, dim=-1).real
        if float(rn.max()) < tol:
            return eig, x

        # uniform expansion count across k (batching invariant): the worst
        # unconverged residuals everywhere, max tally over k
        n_add = int((rn > tol).sum(dim=1).max())
        sel = torch.argsort(rn, dim=1, descending=True)[:, :n_add]
        r_sel = torch.gather(r, 1, sel[..., None].expand(-1, -1, m))
        # TPA scale = band kinetic expectation (POSITIVE — like the NC batched
        # solver, per teter's contract). The per-k path passes eigenvalues and
        # survives their negativity because its plain QR renormalizes the
        # ~1e-14 preconditioned rows; _orthonormalize_b instead declares rows
        # < 1e-8 degenerate and REPLACES them with jitter — random directions,
        # no convergence for any band with eps <= 0.
        t_band = torch.einsum("kbg,kg,kbg->kb", x.conj(), hs.t.to(x.dtype),
                              x).real
        tb_sel = torch.gather(t_band, 1, sel)
        # cast t to the residual's real dtype — fp64 tables would silently
        # promote a complex64 residual back to complex128
        d = teter_b(r_sel, hs.t.to(r_sel.real.dtype), tb_sel)
        # unit-normalize rows BEFORE ortho: near-converged residuals are tiny
        # (~1e-9) but their directions are the information; below the 1e-8
        # threshold _orthonormalize_b would replace them with rank-safety
        # jitter, flooring the SCF density residual near 1e-8 (per-k avoids
        # this because plain QR rescales). Truly zero rows still jitter.
        dn = torch.linalg.norm(d, dim=-1, keepdim=True).real
        d = torch.where(dn > 1e-300, d / dn.clamp_min(1e-300), d)

        if v.shape[1] + n_add > max_dim:
            # restart from the Ritz block re-orthonormalized in the STANDARD
            # metric (S-orthonormal-as-is drifts toward linear dependence and
            # the overlap Cholesky then yields spurious below-minimum states)
            v, hv, sv = _contract(x, hx, sx)
        d = _orthonormalize_b(d, mask, against=v)
        v = torch.cat([v, d], dim=1)
        hv = torch.cat([hv, hs.h(d)], dim=1)
        sv = torch.cat([sv, hs.s(d)], dim=1)
    return eig, x
