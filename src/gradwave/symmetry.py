"""Space-group symmetry: IBZ k-reduction and G-space density symmetrization.

Conventions (derived once, tested by the full-mesh-vs-IBZ equality test):

spglib returns operations {W|w} acting on FRACTIONAL positions, x' = W x + w
(column vectors). With cell rows a_i and reciprocal rows b_i (a_i·b_j=2πδ):

- Cartesian rotation:      S = Aᵀ W A⁻ᵀ
- reciprocal vector G=Bᵀm: SᵀG  ↔  Miller m' = Wᵀ m         (integer, exact)
- k in fractional coords:  S k  ↔  k' = W⁻ᵀ k
- phase:                   G·t = 2π m·w   (t = Aᵀ w Cartesian translation)

Density invariance ρ(g⁻¹r) = ρ(r) gives, per operation,

    ρ_sym(m) = (1/N_op) Σ_op e^{−2πi m·w_op} ρ(W_opᵀ m)

The non-symmorphic phases matter immediately: silicon (diamond, Fd-3̄m) has
glide operations with w = (¼,¼,¼)-type translations.

IBZ reduction requires the k-mesh to be invariant under the group. Unshifted
Γ-centered Monkhorst–Pack meshes always are (W integer ⇒ W⁻ᵀ maps m/n grid
onto itself); shifted meshes may not be — callers fall back to time-reversal
reduction there. Similarly the FFT box must be closed under m → Wᵀm mod n,
which cubic-equal dims guarantee; setup enforces equal dims when symmetry
is on and the check fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib
import torch


@dataclass(frozen=True)
class SpaceGroup:
    rotations: np.ndarray  # (nops, 3, 3) int, fractional-coordinate W
    translations: np.ndarray  # (nops, 3) fractional w
    atom_map: np.ndarray  # (nops, na) int — op sends atom a onto atom_map[op, a]
    international: str
    origin_shift: np.ndarray = None  # spglib standard-origin shift (fractional)

    @property
    def n_ops(self) -> int:
        return len(self.rotations)


def find_spacegroup(
    cell: np.ndarray,
    frac_positions: np.ndarray,
    species_of_atom: list[int],
    symprec: float = 1e-6,
) -> SpaceGroup:
    cell = np.asarray(cell, dtype=float)
    frac = np.asarray(frac_positions, dtype=float) % 1.0
    numbers = np.asarray(species_of_atom, dtype=int)
    ds = spglib.get_symmetry_dataset((cell, frac, numbers), symprec=symprec)

    rots_all = np.asarray(ds.rotations, dtype=np.int64)
    trans_all = np.asarray(ds.translations, dtype=np.float64)

    # Supercells carry pure lattice translations: spglib returns every
    # (rotation × centering) combination — up to 48·N ops whose symmetrizer
    # maps would be gigabytes for large cells (observed: 1536 ops → 9 GB →
    # OOM for a 64-atom Si supercell). Keep one representative translation
    # per unique rotation (QE does the same); this preserves the point-group
    # physics and drops only the enforcement of sub-supercell periodicity.
    seen: dict[bytes, int] = {}
    keep = []
    for i, w_mat in enumerate(rots_all):
        key = w_mat.tobytes()
        if key not in seen:
            seen[key] = i
            keep.append(i)
    rots = rots_all[keep]
    trans = trans_all[keep]

    # atom permutations: op sends atom a to the site matching W x_a + w
    na = len(frac)
    atom_map = np.empty((len(rots), na), dtype=np.int64)
    for iop, (w_mat, w_vec) in enumerate(zip(rots, trans, strict=True)):
        for a in range(na):
            target = (w_mat @ frac[a] + w_vec) % 1.0
            delta = (frac - target + 0.5) % 1.0 - 0.5
            dist = np.linalg.norm(delta @ cell, axis=1)
            b = int(np.argmin(dist))
            if dist[b] > 1e-5 or numbers[b] != numbers[a]:
                raise RuntimeError("symmetry atom mapping failed — inconsistent spglib result")
            atom_map[iop, a] = b

    return SpaceGroup(
        rotations=rots, translations=trans, atom_map=atom_map,
        international=ds.international,
        origin_shift=np.asarray(ds.origin_shift, dtype=float),
    )


def coupled_axis_groups(sg: SpaceGroup) -> list[tuple[int, ...]]:
    """Group the three lattice axes that the point-group rotations actually
    mix, as `equal_dims` for `build_fft_grid`. The FFT box must be closed under
    m → Wᵀm, so coupled axes need equal dimensions — but only coupled ones. A
    slab's vacuum axis is independent of the in-plane pair, so it stays its own
    group; equalizing all three (a blanket cubic box) would blow the slab grid
    up by the vacuum-to-in-plane ratio (e.g. an Al(100) slab at ecutrho=120 Ry
    becomes 105³ instead of ~19×19×105, a ~30× over-allocation)."""
    coupled = np.zeros((3, 3), dtype=bool)
    for w in sg.rotations:
        coupled |= np.asarray(w) != 0
    coupled |= coupled.T
    groups, seen = [], set()
    for i in range(3):
        if i in seen:
            continue
        group, frontier = {i}, {i}
        while frontier:
            j = frontier.pop()
            for k in range(3):
                if coupled[j, k] and k not in group:
                    group.add(k)
                    frontier.add(k)
        seen |= group
        groups.append(tuple(sorted(group)))
    return groups


def reduce_mesh(mesh, shift, sg: SpaceGroup, time_reversal: bool = True):
    """IBZ reduction of a Γ-centered MP mesh. Returns (k_frac (nk,3), weights).

    Only valid for unshifted meshes (asserted by the caller); orbits are taken
    under {W⁻ᵀ} and optionally time reversal.
    """
    mesh = np.asarray(mesh, dtype=np.int64)
    grids = [np.arange(n) for n in mesh]
    mm = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)  # integer m, k=m/n

    inv_rots_t = np.array([np.round(np.linalg.inv(w).T).astype(np.int64) for w in sg.rotations])

    def key_of(m_int):
        return tuple(m_int % mesh)

    index = {key_of(m): i for i, m in enumerate(mm)}
    n_full = len(mm)
    owner = -np.ones(n_full, dtype=np.int64)
    reps, weights = [], []
    for i, m in enumerate(mm):
        if owner[i] >= 0:
            continue
        orbit = set()
        for w_t in inv_rots_t:
            im = w_t @ m
            orbit.add(index[key_of(im)])
            if time_reversal:
                orbit.add(index[key_of(-im)])
        rep = len(reps)
        for j in orbit:
            owner[j] = rep
        reps.append(i)
        weights.append(len(orbit) / n_full)

    kfrac = mm[reps] / mesh
    kfrac = -((-kfrac + 0.5) % 1.0 - 0.5)  # fold to (-1/2, 1/2]
    w = np.array(weights)
    assert abs(w.sum() - 1.0) < 1e-12
    return kfrac, w


class RhoSymmetrizer:
    """Precomputed G-space symmetrization maps for a fixed FFT box.

    dens_mask restricts to the density sphere, where the Miller map is exact:
    at the box Nyquist boundary, folding Wᵀm mod n misidentifies G-vectors
    (phases differ by e^{iπ n·w} for non-symmorphic ops). Physical densities
    are zero there; masking makes the operator exactly idempotent.
    """

    def __init__(self, shape, sg: SpaceGroup, dens_mask=None):
        n1, n2, n3 = shape
        dims = np.array([n1, n2, n3])
        millers = np.stack(
            np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n).astype(np.int64) for n in shape],
                        indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)

        idx_maps, phases = [], []
        for w_mat, w_vec in zip(sg.rotations, sg.translations, strict=True):
            mprime = millers @ w_mat  # rows: (Wᵀ m)ᵀ = mᵀ W
            # box closure check: mapping must be a bijection mod dims
            folded = mprime % dims
            flat = folded[:, 0] * (n2 * n3) + folded[:, 1] * n3 + folded[:, 2]
            idx_maps.append(flat)
            phases.append(np.exp(-2j * np.pi * (millers @ w_vec)))
        idx = np.stack(idx_maps)  # (nops, N)
        # bijection sanity (fails if the box is not closed under the group)
        for row in idx:
            if len(np.unique(row)) != row.shape[0]:
                raise ValueError(
                    "FFT box not closed under the space group — use equal grid dims"
                )
        self.idx = torch.as_tensor(idx, dtype=torch.int64)
        self.phase = torch.as_tensor(np.stack(phases), dtype=torch.complex128)
        self.shape = tuple(shape)
        if dens_mask is not None:
            self.mask = dens_mask.reshape(-1).clone()
        else:
            self.mask = torch.ones(n1 * n2 * n3, dtype=torch.bool)

    def to(self, device) -> "RhoSymmetrizer":
        new = object.__new__(RhoSymmetrizer)
        new.idx = self.idx.to(device)
        new.phase = self.phase.to(device)
        new.mask = self.mask.to(device)
        new.shape = self.shape
        return new

    def apply(self, rho_g_box: torch.Tensor) -> torch.Tensor:
        """Symmetrize ρ(G) on the dense box: (n1,n2,n3) complex → same."""
        flat = rho_g_box.reshape(-1) * self.mask
        acc = (self.phase * flat[self.idx]).mean(dim=0) * self.mask
        return acc.reshape(self.shape)


def symmetrize_forces(forces: torch.Tensor, sg: SpaceGroup, cell: np.ndarray) -> torch.Tensor:
    """Project forces onto the symmetry-invariant subspace.

    F_a ← (1/N) Σ_op Sᵀ F_{map(op,a)}, with S the Cartesian rotation of op.
    """
    a_t = np.asarray(cell, dtype=float).T
    dev = forces.device
    acc = torch.zeros_like(forces)
    f = forces.detach()
    for w_mat, amap in zip(sg.rotations, sg.atom_map, strict=True):
        s = a_t @ w_mat @ np.linalg.inv(a_t)
        s_t = torch.as_tensor(s.T, dtype=forces.dtype, device=dev)
        acc = acc + f[torch.as_tensor(amap.copy(), device=dev)] @ s_t.T
    return acc / sg.n_ops
