"""Γ-point Hessian by symmetry: irreducible displacement columns.

Under a space-group operation {W|w} with Cartesian rotation S and atom
permutation a → g(a), the Γ dynamical matrix obeys

    H[g(b), g(a)] = S H[b, a] Sᵀ            (3×3 Cartesian blocks)

so a computed column c_b = H[b, a]·e implies the column S·c_{g⁻¹(b′)}
for the displacement S·e at atom g(a). Once three linearly independent
directions have accumulated at an atom, its full 3×3 blocks follow by a
least-squares solve against the direction matrix — the same relation
ph.x exploits through irreducible representations. Diamond Si needs one
column instead of six; zincblende two.

hessian_column (postscf/uspp_position) provides the columns; this module
only selects and reconstructs, so it is exact for whatever symmetry
spglib finds at symprec.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.symmetry import SpaceGroup, find_spacegroup

_RANK_TOL = 1e-8

# ω[cm⁻¹] = _SQRT_EV_AMU_ANG2_TO_CM1 · sign(λ)·√|λ| for the mass-weighted
# eigenvalues λ [eV/(amu·Å²)]. Derived from the shared plane-wave kinetic
# prefactor HBAR2_2M = ħ²/2mₑ [eV·Å²]:  (ħω)² = 2·HBAR2_2M·(mₑ/u)·λ  [eV²],
# so ħω[eV] = √(2·HBAR2_2M·mₑ/u)·√λ, then eV→cm⁻¹ via e/(hc). The mₑ/u ratio
# and e/(hc) are CODATA-2018 (not in gradwave.constants, which only fixes the
# eV/Å unit system). Value matches the older explicit-SI form to ~13 digits.
_ME_OVER_U = 5.48579909065e-4        # electron mass / atomic mass unit
_EV_TO_CM1 = 8065.54393734921        # 1 eV in cm⁻¹  = e/(h·c)
_SQRT_EV_AMU_ANG2_TO_CM1 = math.sqrt(2.0 * HBAR2_2M * _ME_OVER_U) * _EV_TO_CM1


class HessianSymmetry:
    """Irreducible (atom, axis) displacements for a Γ Hessian and the
    reconstruction of the full matrix from their computed columns."""

    def __init__(self, cell, positions, species_of_atom, symprec: float = 1e-6):
        cell = np.asarray(cell, dtype=float)
        pos = np.asarray(positions, dtype=float)
        frac = pos @ np.linalg.inv(cell)
        self.na = len(frac)
        self.sg: SpaceGroup = find_spacegroup(cell, frac, species_of_atom,
                                              symprec=symprec)
        a_t = cell.T  # columns = lattice vectors
        self.s_cart = np.array(
            [a_t @ w @ np.linalg.inv(a_t) for w in self.sg.rotations])
        self.displacements = self._select()

    def _spread(self, disp):
        """Per-atom accumulated (direction, column-source) sets implied by
        the displacement list under the group. Returns dirs[a] = list of
        (3,) unit directions reachable at atom a."""
        dirs = [[] for _ in range(self.na)]
        for a, alpha in disp:
            e = np.zeros(3)
            e[alpha] = 1.0
            for s, amap in zip(self.s_cart, self.sg.atom_map, strict=True):
                dirs[amap[a]].append(s @ e)
        return dirs

    def _select(self):
        disp: list[tuple[int, int]] = []
        for a in range(self.na):
            for alpha in range(3):
                dirs = self._spread(disp)[a]
                stack = (np.array(dirs).reshape(-1, 3) if dirs
                         else np.zeros((0, 3)))
                if np.linalg.matrix_rank(stack, tol=_RANK_TOL) == 3:
                    break
                new = self._spread(disp + [(a, alpha)])[a]
                if (np.linalg.matrix_rank(np.array(new).reshape(-1, 3),
                                          tol=_RANK_TOL)
                        > np.linalg.matrix_rank(stack, tol=_RANK_TOL)):
                    disp.append((a, alpha))
        # every atom must reach full rank (guaranteed: worst case all 3
        # axes of every atom get added)
        for a, dirs in enumerate(self._spread(disp)):
            if np.linalg.matrix_rank(np.array(dirs).reshape(-1, 3),
                                     tol=_RANK_TOL) != 3:
                raise RuntimeError(f"displacement selection failed at atom {a}")
        return disp

    def reconstruct(self, cols) -> np.ndarray:
        """Full (na, 3, na, 3) Hessian from the computed columns.

        cols: sequence matching self.displacements; cols[i] is the (na, 3)
        column H[b, :, a_i, α_i] (hessian_column's return). Redundant
        symmetry images enter one least-squares solve per atom, which also
        averages numerical noise over the group.
        """
        cols = [np.asarray(
            c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else c,
            dtype=float).reshape(self.na, 3) for c in cols]
        if len(cols) != len(self.displacements):
            raise ValueError("cols must match self.displacements")
        acc_d = [[] for _ in range(self.na)]
        acc_c = [[] for _ in range(self.na)]
        for (a, alpha), c in zip(self.displacements, cols, strict=True):
            e = np.zeros(3)
            e[alpha] = 1.0
            for s, amap in zip(self.s_cart, self.sg.atom_map, strict=True):
                cp = np.zeros_like(c)
                cp[amap] = c @ s.T  # c′_{g(b)} = S c_b
                acc_d[amap[a]].append(s @ e)
                acc_c[amap[a]].append(cp)
        h_full = np.zeros((self.na, 3, self.na, 3))
        for a in range(self.na):
            e_mat = np.stack(acc_d[a], axis=1)  # (3, m)
            c_mat = np.stack(acc_c[a], axis=2)  # (na, 3, m)
            h_full[:, :, a, :] = c_mat @ np.linalg.pinv(e_mat)
        return h_full


def gamma_hessian(res: dict, xc, *, response_kw=None,
                  verbose: bool = False) -> np.ndarray:
    """(na, 3, na, 3) analytic Γ Hessian [eV/Å²]: irreducible columns via
    hessian_column, symmetry reconstruction, transpose symmetrization and
    the acoustic sum rule."""
    from gradwave.postscf.uspp_position import hessian_column

    system = res["system"]
    pos = system.positions.detach().cpu().numpy()
    cell = np.asarray(system.grid.cell, dtype=float)
    hs = HessianSymmetry(cell, pos, list(system.species_of_atom))
    if verbose:
        print(f"irreducible displacements ({hs.sg.international}): "
              f"{hs.displacements} of {3 * hs.na}")
    # the XC grid quadrature is invariant only under whole-grid-spacing
    # translations, so non-symmorphic translations must land on grid
    # points or the reconstruction inherits the eggbox anisotropy
    # (measured on Si 15/60: 18³ breaks group invariance at 2.6e-2,
    # 20³ restores it)
    dims = np.asarray(system.grid.shape)
    off = hs.sg.translations * dims[None, :]
    if not np.allclose(off, np.round(off), atol=1e-6):
        import warnings

        warnings.warn(
            "FFT grid incommensurate with non-symmorphic translations — "
            "symmetry-reconstructed Hessian columns inherit the XC eggbox "
            "anisotropy; choose fft_shape with dims divisible by the "
            "translation denominators", stacklevel=2)
    cols = []
    for a, alpha in hs.displacements:
        cols.append(hessian_column(res, xc, a, alpha,
                                   response_kw=response_kw, verbose=verbose))
    h_full = hs.reconstruct(cols)
    na = hs.na
    h2 = h_full.reshape(3 * na, 3 * na)
    h2 = 0.5 * (h2 + h2.T)
    hblk = h2.reshape(na, 3, na, 3)
    for a in range(na):
        hblk[a, :, a, :] -= hblk[a].sum(axis=1)  # acoustic sum rule
    h2 = hblk.reshape(3 * na, 3 * na)
    return (0.5 * (h2 + h2.T)).reshape(na, 3, na, 3)


def gamma_frequencies(hess: np.ndarray, masses_amu) -> np.ndarray:
    """Frequencies [cm⁻¹] (negative = imaginary) from an (na,3,na,3)
    Hessian [eV/Å²] and per-atom masses [amu]."""
    m = np.asarray(masses_amu, dtype=float)
    na = len(m)
    d = hess / np.sqrt(m[:, None, None, None] * m[None, None, :, None])
    w2 = np.linalg.eigvalsh(d.reshape(3 * na, 3 * na))
    return np.sign(w2) * _SQRT_EV_AMU_ANG2_TO_CM1 * np.sqrt(np.abs(w2))
