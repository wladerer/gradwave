"""Upcast the subspace matrix to complex128 only when it is not already fp64.

The issue-#136 fp64 contract matters in the complex64 draft phase; in the default
fp64 path ``.to(complex128)`` is already a no-op in torch, so this should measure
~1.0x -- a deliberate negative control proving the loop reports "no win" honestly
rather than inventing one."""

import torch

from gradwave.solvers.davidson import _eigh_subspace


def rr(q, hq, nw):
    s = torch.matmul(q.conj(), hq.mT)
    s = 0.5 * (s + s.conj().transpose(-1, -2))
    if s.dtype != torch.complex128:
        s = s.to(torch.complex128)
    w, u = _eigh_subspace(s)
    return w[:, :nw].real.to(q.real.dtype), u[:, :, :nw].to(q.dtype)
