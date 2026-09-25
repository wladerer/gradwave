"""Skip the explicit Hermitian symmetrisation and let eigh(UPLO='L') read only
the lower triangle.

``s = Q^H H Q`` is Hermitian up to round-off, and torch.linalg.eigh (UPLO='L' by
default, as _eigh_subspace uses) already ignores the upper triangle and assumes
the matrix Hermitian. So the baseline's ``0.5*(s + s^H)`` add + temporary are
redundant here: dropping them changes the result only at round-off (the discarded
upper triangle differed from the lower by ~1e-13), which the oracle admits. A
genuine candidate-win to test, not a control."""

import torch

from gradwave.solvers.davidson import _eigh_subspace


def rr(q, hq, nw):
    s = torch.matmul(q.conj(), hq.mT).to(torch.complex128)
    w, u = _eigh_subspace(s)  # UPLO='L': symmetrises implicitly from the lower triangle
    return w[:, :nw].real.to(q.real.dtype), u[:, :, :nw].to(q.dtype)
