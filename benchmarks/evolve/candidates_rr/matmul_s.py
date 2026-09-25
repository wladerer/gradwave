"""Build the subspace matrix with matmul on conj/transpose views instead of
einsum -- the exact contraction davidson_batched's inline RR uses. Same math;
tests whether einsum dispatch overhead is measurable at this size."""

import torch

from gradwave.solvers.davidson import _eigh_subspace


def rr(q, hq, nw):
    s = torch.matmul(q.conj(), hq.mT)  # <q_i|H|q_j>, no materialised conj copy
    s = (0.5 * (s + s.conj().transpose(-1, -2))).to(torch.complex128)
    w, u = _eigh_subspace(s)
    return w[:, :nw].real.to(q.real.dtype), u[:, :, :nw].to(q.dtype)
