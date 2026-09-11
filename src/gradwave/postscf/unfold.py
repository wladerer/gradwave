"""Supercell band-structure unfolding (Popescu–Zunger effective band structure).

Given a converged *supercell* SCF and an integer supercell matrix ``M`` relating
the primitive cell to the supercell (``A_super = M @ A_prim``, rows = lattice
vectors), this maps a band path defined on the **primitive** cell back onto the
supercell eigenstates and assigns each supercell band a Bloch spectral weight
``P_{K,m}(k) ∈ [0, 1]`` — the fraction of the supercell state ``|K m⟩`` that
carries the primitive Bloch character of the primitive wavevector ``k``. A
perfect (defect-free) supercell yields weights that are 0 or 1 and reproduces
the primitive band structure exactly; symmetry breaking (a defect, a
displacement, an alloy) fractionalizes the weight, which is the effective band
structure (EBS).

Conventions (derived once, used throughout):

- ``A_super = M @ A_prim`` with ``M`` an integer 3×3 matrix (``det M = N`` =
  number of primitive cells in the supercell). The *primitive* cell is therefore
  ``A_prim = M⁻¹ @ A_super`` — the band path must be built on this cell, not on
  the supercell (that is the whole point of unfolding).
- Reciprocal cells (``B = 2π (A⁻¹)ᵀ``) satisfy ``B_super = M⁻ᵀ @ B_prim``, so a
  primitive-fractional wavevector ``k`` maps to the supercell-fractional image
  ``K = k @ Mᵀ`` (row-vector convention), reduced into the supercell BZ modulo 1.
- A supercell plane wave with Miller index ``G`` (integer, on the K-sphere)
  carries primitive-fractional coordinate ``(K + G) @ M⁻ᵀ``; it belongs to the
  residue class of primitive ``k`` iff ``(K + G) @ M⁻ᵀ − k ∈ ℤ³``. Written with
  the integer wraparound offset ``n`` from reducing ``K`` (``K_reduced =
  k @ Mᵀ − n``), the test collapses to the exact integer condition
  ``(G − n) @ M⁻ᵀ ∈ ℤ³``.

The weight is then ``P_{K,m}(k) = Σ_{G ∈ class} |c_{K,m,G}|²`` over the
normalized supercell eigenvector, which we obtain from the same frozen-V_eff
Davidson path as ``band_structure`` (``postscf.bands._diagonalize_at_kpts``) —
only the unique reduced-K set is diagonalized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ase.atoms import Atoms

from gradwave.postscf.bands import _bands_reference_energy, _diagonalize_at_kpts
from gradwave.scf.loop import SCFResult


def as_supercell_matrix(M) -> np.ndarray:
    """Normalize the user's supercell spec to an integer 3×3 array.

    Accepts a diagonal shorthand ``[2, 2, 2]`` (→ ``diag(2, 2, 2)``) or a full
    3×3 nested list/array. Entries must be (integer-valued) and the matrix
    non-singular."""
    arr = np.asarray(M)
    if arr.ndim == 1:
        if arr.shape != (3,):
            raise ValueError(
                f"diagonal supercell shorthand must have 3 entries, got {arr.shape}")
        arr = np.diag(arr)
    if arr.shape != (3, 3):
        raise ValueError(f"supercell matrix must be 3×3 (or a 3-vector), got {arr.shape}")
    rounded = np.rint(arr)
    if not np.allclose(arr, rounded, atol=1e-8):
        raise ValueError(f"supercell matrix must be integer-valued, got {arr.tolist()}")
    Mi = rounded.astype(np.int64)
    if int(round(np.linalg.det(Mi))) == 0:
        raise ValueError("supercell matrix is singular (det = 0)")
    return Mi


def primitive_cell(supercell: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Primitive lattice ``A_prim = M⁻¹ @ A_super`` (rows = lattice vectors)."""
    return np.linalg.solve(M.astype(float), np.asarray(supercell, dtype=float))


@dataclass
class FoldMap:
    """The primitive-path → unique-supercell-K fold with the exact preselection.

    ``kpts_prim`` (nkpath, 3) are the primitive-fractional path points; each maps
    to a reduced supercell-fractional image ``kpts_super[i]`` = ``unique_K[image[i]]``.
    ``offset[i]`` is the integer wraparound ``n`` (``k @ Mᵀ − K_reduced``) needed
    by the residue mask. Only ``unique_K`` (nK ≤ nkpath) is diagonalized."""

    kpts_prim: np.ndarray       # (nkpath, 3) primitive fractional
    kpts_super: np.ndarray      # (nkpath, 3) reduced supercell fractional
    offset: np.ndarray          # (nkpath, 3) int64 wraparound offsets
    unique_K: np.ndarray        # (nK, 3) unique reduced supercell fractional
    image: np.ndarray           # (nkpath,) int64 index into unique_K


def fold_map(M: np.ndarray, kpts_prim: np.ndarray, tol: float = 1e-8) -> FoldMap:
    """Fold primitive-fractional path points to unique supercell BZ images.

    ``K = k @ Mᵀ`` reduced into ``[-0.5, 0.5)`` per component; the integer part
    removed is the wraparound offset ``n``. Path points sharing a reduced ``K``
    (to ``tol``) collapse to one unique image — the dedup that makes unfolding
    cheap."""
    k = np.asarray(kpts_prim, dtype=float)
    raw = k @ M.T.astype(float)               # supercell-fractional (unreduced)
    n = np.rint(raw).astype(np.int64)         # integer wraparound offset
    red = raw - n                             # reduced into [-0.5, 0.5)

    unique_K: list[np.ndarray] = []
    image = np.empty(len(k), dtype=np.int64)
    for i, K in enumerate(red):
        for j, U in enumerate(unique_K):
            if np.allclose(K, U, atol=tol):
                image[i] = j
                break
        else:
            image[i] = len(unique_K)
            unique_K.append(K)
    return FoldMap(
        kpts_prim=k, kpts_super=red, offset=n,
        unique_K=np.asarray(unique_K, dtype=float), image=image)


def residue_mask(
    miller: torch.Tensor,      # (npw, 3) int64 supercell Miller indices on the K-sphere
    offset: np.ndarray,        # (3,) int64 wraparound offset n for this path point
    Minv_T: np.ndarray,        # (3, 3) M⁻ᵀ
    tol: float = 1e-6,
) -> torch.Tensor:
    """Boolean mask over the K-sphere: which G belong to primitive k's class.

    ``G`` belongs iff ``(G − n) @ M⁻ᵀ`` is an integer vector (see module
    docstring for the derivation of this exact integer form)."""
    G = miller.detach().cpu().numpy().astype(np.int64)
    t = (G - offset[None, :]) @ Minv_T          # (npw, 3), rational entries
    is_int = np.all(np.abs(t - np.rint(t)) < tol, axis=1)
    return torch.as_tensor(is_int, dtype=torch.bool, device=miller.device)


@dataclass
class UnfoldedBands:
    """Effective band structure: primitive path, supercell eigenvalues at the
    folded images, and the per-band Bloch spectral weights."""

    kpts_prim: np.ndarray       # (nkpath, 3) primitive fractional path
    kpts_super: np.ndarray      # (nkpath, 3) reduced supercell fractional images
    eigenvalues: np.ndarray     # (nkpath, nbands) [eV]; leading spin axis if nspin=2
    weights: np.ndarray         # same shape as eigenvalues; P ∈ [0, 1]
    reference: float            # Fermi level or VBM [eV]
    unique_K: np.ndarray        # (nK, 3) unique reduced supercell k that were diagonalized
    # raw supercell eigenvalues at unique_K; (nK, nb) or (nspin, nK, nb)
    folded_eigenvalues: np.ndarray
    image: np.ndarray           # (nkpath,) index of each path point into unique_K
    labels: list[tuple[int, str]] | None = None
    x: np.ndarray | None = None


@torch.no_grad()
def unfold_bands(
    res: SCFResult,
    M,
    path: str = "",
    npoints: int = 120,
    nbands: int | None = None,
    diago_tol: float = 1e-9,
    verbose: bool = False,
) -> UnfoldedBands:
    """Unfold a converged supercell SCF onto a primitive-cell band path.

    ``M`` is the integer supercell matrix (``A_super = M @ A_prim``; diagonal
    shorthand ``[2, 2, 2]`` accepted). The band path (``path`` / ``npoints``) is
    generated on the **primitive** cell ``M⁻¹ @ res.system.atoms.cell``."""
    Mi = as_supercell_matrix(M)
    Minv_T = np.linalg.inv(Mi.astype(float)).T

    # primitive cell + band path (ASE) on it — NOT on the supercell.
    super_cell = np.asarray(res.system.grid.cell, dtype=float)
    prim = primitive_cell(super_cell, Mi)
    prim_atoms = Atoms(cell=prim, pbc=True)
    bp = prim_atoms.cell.bandpath(path=path or None, npoints=npoints)
    x, xticks, xlabels = bp.get_linear_kpoint_axis()
    labels = list(zip(xticks.tolist(), list(xlabels), strict=True))

    fm = fold_map(Mi, bp.kpts)
    nbands = nbands or res.system.nbands
    nspin = getattr(res, "nspin", 1)
    nK = len(fm.unique_K)

    folded = np.empty((nspin, nK, nbands))
    # weights indexed by the UNIQUE K and by which path residue class each band
    # belongs to. We accumulate per unique-K, then scatter to the path below.
    # w_by_K[sp][uk] holds, per band, the weight for that unique K's own residue
    # class — but distinct path points can fold to the same K with DIFFERENT
    # residue classes, so we must key the weight on the path point, not on K.
    # Diagonalize the unique K set once; recompute the (cheap, integer) mask per
    # path point sharing that K.
    coeffs_by_K: dict[int, list[torch.Tensor]] = {sp: [None] * nK for sp in range(nspin)}  # type: ignore[misc]
    millers_by_K: list[torch.Tensor] = [None] * nK  # type: ignore[list-item]

    for lo, hi, sp, _nspin, spheres, out in _diagonalize_at_kpts(
        res, fm.unique_K, nbands, diago_tol, verbose):
        folded[sp, lo:hi] = out.eigenvalues.cpu().numpy()
        for j in range(hi - lo):
            uk = lo + j
            npw = spheres[j].npw
            # (nb, npw) coefficients on this K-sphere (drop the padded tail)
            coeffs_by_K[sp][uk] = out.eigenvectors[j, :, :npw]
            if sp == 0:
                millers_by_K[uk] = spheres[j].miller

    eigenvalues = np.empty((nspin, len(fm.kpts_prim), nbands))
    weights = np.empty((nspin, len(fm.kpts_prim), nbands))
    for ip in range(len(fm.kpts_prim)):
        uk = int(fm.image[ip])
        mask = residue_mask(millers_by_K[uk], fm.offset[ip], Minv_T)
        for sp in range(nspin):
            eigenvalues[sp, ip] = folded[sp, uk]
            c = coeffs_by_K[sp][uk]              # (nb, npw) complex
            w = (c.abs() ** 2 * mask[None, :]).sum(dim=1)
            weights[sp, ip] = w.cpu().numpy()

    reference = _bands_reference_energy(res)
    if nspin == 1:
        eigenvalues = eigenvalues[0]
        weights = weights[0]
        folded_out = folded[0]
    else:
        folded_out = folded
    return UnfoldedBands(
        kpts_prim=fm.kpts_prim, kpts_super=fm.kpts_super,
        eigenvalues=eigenvalues, weights=weights, reference=reference,
        unique_K=fm.unique_K, folded_eigenvalues=folded_out, image=fm.image,
        labels=labels, x=np.asarray(x),
    )
