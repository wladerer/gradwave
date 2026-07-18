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


def _k_ops(rotations) -> list[np.ndarray]:
    """Reciprocal-space integer action of fractional rotations: k' = W⁻ᵀ k."""
    return [np.round(np.linalg.inv(w).T).astype(np.int64) for w in rotations]


def _orbit_reduce(mesh, ops_t):
    """Fold a Γ-centered MP mesh into orbits under integer k-space ops.

    ops_t is a list of (3,3) integer matrices acting on the mesh integers m
    (k = m/n). Returns (k_frac (nk,3) in (-1/2,1/2], weights summing to 1).
    """
    mesh = np.asarray(mesh, dtype=np.int64)
    grids = [np.arange(n) for n in mesh]
    mm = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)  # integer m, k=m/n

    def key_of(m_int):
        return tuple(m_int % mesh)

    index = {key_of(m): i for i, m in enumerate(mm)}
    n_full = len(mm)
    owner = -np.ones(n_full, dtype=np.int64)
    reps, weights = [], []
    for i, m in enumerate(mm):
        if owner[i] >= 0:
            continue
        orbit = {index[key_of(w_t @ m)] for w_t in ops_t}
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


def reduce_mesh(mesh, shift, sg: SpaceGroup, time_reversal: bool = True):
    """IBZ reduction of a Γ-centered MP mesh. Returns (k_frac (nk,3), weights).

    Only valid for unshifted meshes (asserted by the caller); orbits are taken
    under {W⁻ᵀ} and optionally time reversal.
    """
    ops_t = _k_ops(sg.rotations)
    if time_reversal:
        ops_t = ops_t + [-w for w in ops_t]
    return _orbit_reduce(mesh, ops_t)


@dataclass(frozen=True)
class MagneticGroup:
    """Shubnikov magnetic space group of a (possibly non-collinear) moment set.

    `unitary` ops leave the moments invariant and act exactly like an ordinary
    SpaceGroup (drop-in for RhoSymmetrizer/BecsumSymmetrizer). The anti-unitary
    set holds ops that reverse every moment and therefore survive only combined
    with time reversal (g·T); they act on k as −W⁻ᵀ and add a −1 to any axial
    (m⃗) channel. With all moments zero this is the grey group: every op appears
    in both sets, and the magnetic k-fold reduces to reduce_mesh(..., TR=True).
    """

    unitary: SpaceGroup
    anti_rotations: np.ndarray  # (n_anti, 3, 3) int fractional W
    anti_translations: np.ndarray  # (n_anti, 3) fractional w
    anti_atom_map: np.ndarray  # (n_anti, na)

    @property
    def n_unitary(self) -> int:
        return self.unitary.n_ops

    @property
    def n_anti(self) -> int:
        return len(self.anti_rotations)

    def combined(self) -> SpaceGroup:
        """Unitary + anti-unitary spatial parts as one SpaceGroup (in that
        order — axial factors index ops ≥ n_unitary as the anti set)."""
        return SpaceGroup(
            rotations=np.concatenate([self.unitary.rotations, self.anti_rotations]),
            translations=np.concatenate([self.unitary.translations, self.anti_translations]),
            atom_map=np.concatenate([self.unitary.atom_map, self.anti_atom_map]),
            international=self.unitary.international,
            origin_shift=self.unitary.origin_shift,
        )


def magnetic_spacegroup(
    sg: SpaceGroup, magmoms, cell: np.ndarray, tol: float = 1e-5
) -> MagneticGroup:
    """Filter the paramagnetic group by its action on the atomic moments.

    Moments are axial vectors: an op with fractional rotation W (Cartesian
    S = Aᵀ W A⁻ᵀ) sends m⃗_a on atom a to det(S)·S·m⃗_a on atom map(op, a).
    Ops with m⃗' = m⃗ everywhere are unitary; m⃗' = −m⃗ everywhere survive as
    anti-unitary g·T; anything else is dropped (they'd relate *different*
    magnetic configurations). Cross-checked against spglib's
    get_magnetic_symmetry in tests — this filter inherits find_spacegroup's
    dedup and atom mapping instead of re-deriving them.
    """
    m = np.atleast_2d(np.asarray(magmoms, dtype=float))
    a_t = np.asarray(cell, dtype=float).T
    a_t_inv = np.linalg.inv(a_t)
    scale = max(1.0, float(np.abs(m).max()))
    keep_u, keep_a = [], []
    for iop, w_mat in enumerate(sg.rotations):
        s = a_t @ w_mat @ a_t_inv
        r_ax = np.linalg.det(s) * s  # axial (pseudo-vector) action
        m_img = m @ r_ax.T  # det(S)·S·m⃗_a, per atom
        m_tgt = m[sg.atom_map[iop]]  # moments at the image sites
        if np.abs(m_img - m_tgt).max() < tol * scale:
            keep_u.append(iop)
        if np.abs(m_img + m_tgt).max() < tol * scale:
            keep_a.append(iop)
    unitary = SpaceGroup(
        rotations=sg.rotations[keep_u],
        translations=sg.translations[keep_u],
        atom_map=sg.atom_map[keep_u],
        international=sg.international,
        origin_shift=sg.origin_shift,
    )
    return MagneticGroup(
        unitary=unitary,
        anti_rotations=sg.rotations[keep_a],
        anti_translations=sg.translations[keep_a],
        anti_atom_map=sg.atom_map[keep_a],
    )


def reduce_mesh_magnetic(mesh, shift, mg: MagneticGroup):
    """Magnetic-IBZ reduction of a Γ-centered MP mesh under a Shubnikov group.

    Unitary ops act on k as W⁻ᵀ; anti-unitary ops (g·T) as −W⁻ᵀ (time reversal
    sends k → −k). Zero moments (grey group) reproduce
    reduce_mesh(..., time_reversal=True) exactly. Returns (k_frac, weights).
    """
    ops_t = _k_ops(mg.unitary.rotations)
    ops_t += [-w for w in _k_ops(mg.anti_rotations)]
    return _orbit_reduce(mesh, ops_t)


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


class MagneticSymmetrizer:
    """G-space symmetrization of (ρ, m⃗) under a magnetic (Shubnikov) group.

    The spatial part is a RhoSymmetrizer over the COMBINED op list (unitary
    then anti-unitary): ρ and m⃗ are real fields, so time reversal itself acts
    trivially on their G-space maps and only the spatial parts of the
    anti-unitary ops fold charge. The m⃗ channels additionally mix through the
    axial 3×3  s_T·det(S)·S  per op, with s_T = −1 on the anti-unitary set
    (T reverses magnetization). Both ρ and m⃗ are thus constrained by the FULL
    magnetic group — the anti-unitary half is not lost by working in the
    magnetic IBZ of reduce_mesh_magnetic.
    """

    def __init__(self, shape, mg: MagneticGroup, cell: np.ndarray, dens_mask=None):
        combined = mg.combined()
        self.rho_sym = RhoSymmetrizer(shape, combined, dens_mask=dens_mask)
        a_t = np.asarray(cell, dtype=float).T
        a_t_inv = np.linalg.inv(a_t)
        ax = []
        for iop, w_mat in enumerate(combined.rotations):
            s = a_t @ w_mat @ a_t_inv
            r_ax = np.linalg.det(s) * s
            if iop >= mg.n_unitary:
                r_ax = -r_ax  # s_T: time reversal flips m⃗
            ax.append(r_ax)
        self.axial = torch.as_tensor(np.stack(ax), dtype=torch.float64)
        self.shape = tuple(shape)

    def to(self, device) -> "MagneticSymmetrizer":
        new = object.__new__(MagneticSymmetrizer)
        new.rho_sym = self.rho_sym.to(device)
        new.axial = self.axial.to(device)
        new.shape = self.shape
        return new

    def apply(self, rho_g_box: torch.Tensor) -> torch.Tensor:
        """Symmetrize ρ(G) on the dense box: (n1,n2,n3) complex → same."""
        return self.rho_sym.apply(rho_g_box)

    def apply_m(self, m_g_box: torch.Tensor) -> torch.Tensor:
        """Symmetrize m⃗(G): (3, n1,n2,n3) complex → same.

        m_α(G) ← (1/N) Σ_op  ax[op]_{αβ} · e^{−2πi m·w_op} · m_β(W_opᵀ G).
        """
        rs = self.rho_sym
        flat = m_g_box.reshape(3, -1) * rs.mask
        gathered = flat[:, rs.idx]  # (3, nops, N)
        mixed = torch.einsum("oab,bon->aon", self.axial.to(flat.dtype), gathered)
        acc = (rs.phase * mixed).mean(dim=1) * rs.mask
        return acc.reshape(3, *self.shape)


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
