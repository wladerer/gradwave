"""A uniform block-eigensolver interface + registry (Layer B).

The SCF loop diagonalizes the norm-conserving standard problem H x = ε x once
per spin channel per iteration. Historically it dispatched on a hard-coded
``eigensolver`` string ("davidson" | "chebyshev"). This module replaces that
with a thin registry so any block eigensolver — the existing batched Davidson
and CheFSI, and later LOBPCG, subset Rayleigh-Ritz, an adaptive scheduler —
can register under a name and be selected by ``scf(..., eigensolver=name)``.

The uniform adapter signature is

    solve(apply_H, X0, precond, mask, *, tol, nbands, **kw) -> EigResult

with
    apply_H : (nk, nb, npw) -> (nk, nb, npw)   batched Hamiltonian apply
    X0      : (nk, nbands, npw) complex        initial band block, padded 0
    precond : (nk, npw) real                   kinetic diagonal T (the datum the
                                               Teter/plane-wave preconditioner is
                                               built from; solvers that don't
                                               precondition — CheFSI — ignore it)
    mask    : (nk, npw) bool                    valid plane waves per k
    tol     : float                            residual-norm convergence gate
    nbands  : int                              # wanted lowest eigenpairs

returning

    EigResult(eigenvalues (nk, nbands), eigenvectors (nk, nbands, npw),
              n_iter, residual_norms (nk, nbands), diagnostics: dict)

The adapter keeps the (eigvals, eigvecs, n_iter, residuals) core identical to the
existing ``BatchedDavidsonResult`` so nothing downstream changes; ``diagnostics``
is the extension slot for a solver to report per-solve knobs (degree, buffer
width, restart count, ...) that the benchmark battery and the future adaptive
scheduler read.

Stays low-level per the import contract: this module imports only sibling
solver modules, never gradwave.scf / gradwave.postscf.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Protocol

import torch


class EigResult(NamedTuple):
    """Uniform return type for every registered block eigensolver."""

    eigenvalues: torch.Tensor  # (nk, nbands) ascending
    eigenvectors: torch.Tensor  # (nk, nbands, npw), padded slots zero
    n_iter: int
    residual_norms: torch.Tensor  # (nk, nbands)
    diagnostics: dict[str, Any]


class BlockEigensolver(Protocol):
    """Any callable with the uniform adapter signature is a solver."""

    def __call__(
        self,
        apply_H: Callable[[torch.Tensor], torch.Tensor],
        X0: torch.Tensor,
        precond: torch.Tensor,
        mask: torch.Tensor,
        *,
        tol: float,
        nbands: int,
        **kw: Any,
    ) -> EigResult: ...


_REGISTRY: dict[str, BlockEigensolver] = {}


def register(name: str, solver: BlockEigensolver, *, overwrite: bool = False) -> None:
    """Register `solver` under `name`. A later worker (LOBPCG, subset-RR, an
    adaptive scheduler) calls this at import time to make its solver selectable
    via ``scf(eigensolver=name)`` and visible to the benchmark battery."""
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"eigensolver {name!r} already registered; pass overwrite=True to replace"
        )
    _REGISTRY[name] = solver


def get(name: str) -> BlockEigensolver:
    """The registered solver, or a ValueError listing what is available."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown eigensolver {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def available() -> list[str]:
    """Names of all registered solvers (sorted, stable)."""
    return sorted(_REGISTRY)


def is_registered(name: str) -> bool:
    return name in _REGISTRY


# ---------------------------------------------------------------------------
# Adapters for the two existing solvers. Each wraps the native entry point,
# translating (precond, mask) into that solver's argument order and repackaging
# its BatchedDavidsonResult into an EigResult with a small diagnostics dict.
# The native defaults (max_iter=40, max_dim_factor=4, degree=10, ...) are the
# solvers' own; the SCF path passes only tol, so behaviour is unchanged.
# ---------------------------------------------------------------------------


def davidson_adapter(
    apply_H, X0, precond, mask, *, tol, nbands=None, max_iter=40,
    max_dim_factor=4, **kw,
) -> EigResult:
    """Batched block Davidson — the baseline. `precond` is the kinetic diagonal
    T the Teter preconditioner uses; `nbands` is implied by X0 (kept for
    signature parity)."""
    from gradwave.solvers.davidson import davidson_batched

    r = davidson_batched(
        apply_H, X0, precond, mask, tol=tol, max_iter=max_iter,
        max_dim_factor=max_dim_factor,
    )
    return EigResult(
        r.eigenvalues, r.eigenvectors, r.n_iter, r.residual_norms,
        {"solver": "davidson", "max_dim_factor": max_dim_factor,
         "max_iter": max_iter, "hit_max_iter": r.n_iter >= max_iter},
    )


def chebyshev_adapter(
    apply_H, X0, precond, mask, *, tol, nbands=None, max_iter=40, degree=10,
    n_lanczos=6, n_buffer=None, bounds=None, **kw,
) -> EigResult:
    """Chebyshev-filtered subspace iteration (CheFSI). Ignores `precond` (the
    filter needs no preconditioner); reports its degree/buffer in diagnostics."""
    from gradwave.solvers.chebyshev import chebyshev_filtered_batched

    r = chebyshev_filtered_batched(
        apply_H, X0, precond, mask, tol=tol, max_iter=max_iter, degree=degree,
        n_lanczos=n_lanczos, n_buffer=n_buffer, bounds=bounds,
    )
    return EigResult(
        r.eigenvalues, r.eigenvectors, r.n_iter, r.residual_norms,
        {"solver": "chebyshev", "degree": degree, "n_lanczos": n_lanczos,
         "n_buffer": n_buffer, "max_iter": max_iter,
         "hit_max_iter": r.n_iter >= max_iter},
    )


register("davidson", davidson_adapter)
register("chebyshev", chebyshev_adapter)
