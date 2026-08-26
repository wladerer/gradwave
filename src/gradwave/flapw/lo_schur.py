"""In-context local-orbital conditioning via the overlap Schur complement.

A LAPW local orbital can pass the per-LO norm check (``scf._build_lo``'s ``cond_tol``,
which tests redundancy only against its *own* sphere's ``{u, u̇}``) yet still be
near-linearly-dependent on the **full** APW + other-LO basis at a given k and potential.
That in-context redundancy is what the scalar check cannot see, and it is the source of
the ill-conditioned extended overlap ``S`` that makes the canonical ``S^{-1/2}`` drop
directions unpredictably (or NaN). It has cost real debugging time in this campaign — the
confined l=1 LO's ~85% redundancy needed a bespoke probe to pin down.

The overlap Schur complement isolates exactly this, exactly. Partition the extended overlap

    S = [[S_aa, S_al],
         [S_la, S_ll]]          (``npw`` APW rows first, then ``nlo`` LO rows)

The Schur complement ``S_ll_eff = S_ll − S_la S_aa⁻¹ S_al`` is the LO Gram matrix with the
APW space projected **out**: ``S_ll_eff[i,j] = ⟨P⊥φ_i | P⊥φ_j⟩`` with ``P⊥ = 1 − P_APW``.
It is EXACT — ``S`` is energy-independent, so this side carries none of the energy-dependent
approximation the *Hamiltonian* downfold would. The generalized eigenvalues of
``(S_ll_eff, S_ll)`` are the fraction of each LO direction's norm that survives projection
out of the APW span — a per-direction ``resid_frac`` in ``[0, 1]``; a value near 0 is an LO
direction the APW basis already spans (redundant — the conditioning culprit), attributable
to a specific ``(atom, species, l, m)`` LO.

This is a diagnostic (``O(npw²·nlo)`` via one dense solve against ``S_aa``), meant for the
LO-conditioning debug path, not the hot inner loop.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["lo_labels", "lo_overlap_schur", "lo_resid_fracs", "lo_conditioning_report"]


def lo_labels(
    atoms_cart: list[tuple[Any, str]], lodat: dict[str, list[Any]]
) -> list[tuple[Any, ...]]:
    """``(atom_index, species, l, m)`` per LO row, in the secular build order.

    Matches ``scf._solve_secular``'s extension exactly: atoms outer, ``lodat[key]`` LOs in
    list order, ``m = −l..l`` innermost.
    """
    labels: list[tuple[Any, ...]] = []
    for ai, (_tau, key) in enumerate(atoms_cart):
        for lo in lodat.get(key, []):
            ell = int(lo["l"])
            labels.extend((ai, key, ell, m) for m in range(-ell, ell + 1))
    return labels


def lo_overlap_schur(S: np.ndarray, npw: int) -> np.ndarray:
    """``S_ll_eff = S_ll − S_la S_aa⁻¹ S_al`` (the LO overlap with the APW span projected out).

    ``S`` is the assembled extended overlap ``(npw+nlo, npw+nlo)``; ``S_aa`` (the APW overlap
    ``I + augmentation``) is Hermitian positive-definite, so the solve is well-conditioned.
    Returns the Hermitised ``(nlo, nlo)`` block.
    """
    if npw >= S.shape[0]:
        raise ValueError(f"no LO block: npw={npw} >= dim={S.shape[0]}")
    S_aa = np.ascontiguousarray(S[:npw, :npw])
    S_al = np.ascontiguousarray(S[:npw, npw:])
    S_ll = S[npw:, npw:]
    x = np.linalg.solve(S_aa, S_al)                 # S_aa⁻¹ S_al, (npw, nlo)
    eff = S_ll - S_al.conj().T @ x
    return 0.5 * (eff + eff.conj().T)


def lo_resid_fracs(S: np.ndarray, npw: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-LO-direction surviving-norm fractions ``λ ∈ [0, 1]`` (ascending) and their vectors.

    Solves the small generalized problem ``S_ll_eff x = λ S_ll x``. ``λ`` near 0 marks an LO
    direction already spanned by the APW basis (redundant); ``λ`` near 1 is a genuinely
    distinct direction. Returns ``(lam, vecs)`` with ``vecs`` the LO-basis eigenvectors.
    """
    from scipy.linalg import eigh
    eff = lo_overlap_schur(S, npw)
    S_ll = np.ascontiguousarray(S[npw:, npw:])
    S_ll = 0.5 * (S_ll + S_ll.conj().T)
    lam, vecs = eigh(eff, S_ll)                      # generalized, ascending real λ
    return np.real(lam), vecs


def lo_conditioning_report(S: np.ndarray, npw: int, labels: list[Any] | None = None,
                           tol: float = 0.05) -> tuple[list[tuple[float, Any]], str]:
    """Localize redundant LO directions. Returns ``(redundant, report)``.

    ``redundant`` = ``[(resid_frac, label), ...]`` for every direction below ``tol``, each
    attributed to the LO (``labels[j]``, or its row index if ``labels`` is None) carrying the
    largest weight in that direction. ``report`` is a per-direction text table.
    """
    lam, vecs = lo_resid_fracs(S, npw)
    lines, redundant = [], []
    for i, frac in enumerate(lam):
        j = int(np.argmax(np.abs(vecs[:, i])))
        label = labels[j] if labels is not None else j
        lines.append(f"  resid_frac={frac:.3e}  <- {label}")
        if frac < tol:
            redundant.append((float(frac), label))
    return redundant, "\n".join(lines)
