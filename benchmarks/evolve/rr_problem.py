"""The Davidson Rayleigh-Ritz step as an evolve ``Problem``.

Target kernel: ``solvers.davidson._rr(q, hq, nw)`` -- a separable pure function.
``q, hq`` are band-major ``(nk, m, npw)`` (m = subspace width, npw = basis dim);
it forms ``s = <q_i|H|q_j> = einsum('kig,kjg->kij', q.conj(), hq)``, symmetrises,
upcasts the small ``(nk, m, m)`` matrix to complex128, eighs, and returns the
``nw`` lowest Ritz values ``(nk, nw)`` and the coefficient block ``(nk, m, nw)``.

Workload: a synthetic batched Hermitian operator (the diag(1..100)+random-Hermitian
form the CheFSI/fp32-expansion tests use), an orthonormal basis ``q`` (so
``q q^H = I_m``), and its images ``hq = H q``. Sized to a "supercell RR" regime
(wide subspace, large npw) so the step costs measurable time -- at Si-primitive
dims the RR is microseconds and any speedup is unmeasurable, which is itself the
honest reason the historical RR win needed the native large-dim kernel.

Oracle (gauge/degeneracy robust, no external H needed beyond the frozen q, hq):
a candidate's ``(w, c)`` must (1) reproduce the baseline eigenvalues, (2) yield
orthonormal Ritz vectors ``R = combine(c, q)``, and (3) diagonalise the operator on
that span: ``R^H H R = diag(w)`` (via ``combine(c, hq)``). Any implementation that
is algebraically the same RR passes to fp64 round-off; anything that changes the
answer fails, regardless of how fast it is.
"""

from __future__ import annotations

import torch

from gradwave.solvers.davidson import _rr as _baseline_rr

# Gate tolerance: arithmetic-reordering variants of one fp64 Hermitian eigensolve
# agree to ~1e-13; 1e-9 leaves generous head-room while still rejecting any real
# change of the computed subspace.
ORACLE_TOL = 1e-9


def _combine(c: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """Trusted Ritz combination used only by the oracle: (nk,m,nw) x (nk,m,npw)
    -> (nk,nw,npw). Fixed here so candidates are free to implement _rr however
    they like without touching the correctness check."""
    return torch.einsum("kja,kjg->kag", c, block)


class RRProblem:
    """Frozen Rayleigh-Ritz workload + baseline-anchored correctness oracle."""

    name = "davidson_rr"

    def __init__(
        self,
        *,
        nk: int = 8,
        npw: int = 4000,
        m: int = 120,
        nw: int = 32,
        seed: int = 0,
        dtype: torch.dtype = torch.complex128,
    ) -> None:
        if not (nw <= m <= npw):
            raise ValueError(f"need nw <= m <= npw, got nw={nw} m={m} npw={npw}")
        self.nk, self.npw, self.m, self.nw = nk, npw, m, nw
        self.dtype = dtype
        self._build(seed)

    def _build(self, seed: int) -> None:
        torch.manual_seed(seed)
        nk, npw, m = self.nk, self.npw, self.m
        diag = torch.linspace(1.0, 100.0, npw, dtype=torch.float64)
        h = torch.empty(nk, npw, npw, dtype=torch.complex128)
        for k in range(nk):
            a = torch.randn(npw, npw, dtype=torch.complex128)
            h[k] = 0.5 * (a + a.conj().T) + torch.diag(diag.to(torch.complex128))
            h[k] = 0.5 * (h[k] + h[k].conj().T)  # kill round-off asymmetry

        # Orthonormal band-major basis q (nk, m, npw) with q q^H = I_m, and hq = H q.
        a = torch.randn(nk, npw, m, dtype=torch.complex128)
        qcols, _ = torch.linalg.qr(a)  # (nk, npw, m), columns orthonormal
        q = qcols.mH.contiguous()  # (nk, m, npw), rows orthonormal
        hq = torch.einsum("kpn,kin->kip", h, q).contiguous()

        self._q = q.to(self.dtype)
        self._hq = hq.to(self.dtype)

        # Baseline reference the oracle anchors to.
        w_ref, _ = _baseline_rr(self._q, self._hq, self.nw)
        self._w_ref = w_ref.to(torch.float64)
        self._w_scale = float(self._w_ref.abs().max()) + 1e-30

    def workload(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        return (self._q, self._hq, self.nw)

    def oracle(self, out: tuple[torch.Tensor, torch.Tensor]) -> tuple[bool, float]:
        w, c = out
        w = w.to(torch.float64)
        if w.shape != self._w_ref.shape:
            return False, float("inf")

        # (1) eigenvalues reproduce the baseline (relative).
        err_eig = float((w - self._w_ref).abs().max()) / self._w_scale

        # (2) & (3) reconstruct the Ritz vectors and check span + diagonalisation.
        ritz = _combine(c, self._q)  # (nk, nw, npw)
        hritz = _combine(c, self._hq)  # H applied to the Ritz vectors
        nw = self.nw
        eye = torch.eye(nw, dtype=ritz.dtype)
        gram = torch.einsum("kag,kbg->kab", ritz.conj(), ritz)  # R^H R, want I
        err_orth = float((gram - eye).abs().max())
        proj = torch.einsum("kag,kbg->kab", ritz.conj(), hritz)  # R^H H R, want diag(w)
        want = torch.diag_embed(w.to(proj.dtype))
        err_diag = float((proj - want).abs().max())

        max_err = max(err_eig, err_orth, err_diag)
        return (max_err < ORACLE_TOL), max_err
