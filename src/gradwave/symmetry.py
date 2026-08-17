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
from typing import cast

import numpy as np
import spglib
import torch
from spglib import SpgCell


@dataclass(frozen=True)
class SpaceGroup:
    rotations: np.ndarray  # (nops, 3, 3) int, fractional-coordinate W
    translations: np.ndarray  # (nops, 3) fractional w
    atom_map: np.ndarray  # (nops, na) int — op sends atom a onto atom_map[op, a]
    international: str
    origin_shift: np.ndarray | None = None  # spglib standard-origin shift (fractional)

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
    # spglib's own SpgCell type alias is a tuple[Lattice, Positions, Numbers] of
    # plain Sequences; spglib itself casts at this exact seam internally (spg.py)
    # rather than widen the stub, since the numpy arrays built just above satisfy
    # the runtime contract (row-iterable) but not the stub's nested-Sequence shape.
    ds = spglib.get_symmetry_dataset(cast(SpgCell, (cell, frac, numbers)), symprec=symprec)
    if ds is None:
        raise ValueError(
            f"spglib.get_symmetry_dataset returned None (symprec={symprec}); "
            f"check the cell/positions for a degenerate or ill-conditioned "
            f"lattice (cell=\n{cell})"
        )

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


def _k_ops(rotations: np.ndarray) -> list[np.ndarray]:
    """Reciprocal-space integer action of fractional rotations: k' = W⁻ᵀ k."""
    return [np.round(np.linalg.inv(w).T).astype(np.int64) for w in rotations]


def _orbit_reduce(
    mesh: tuple[int, int, int], ops_t: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a Γ-centered MP mesh into orbits under integer k-space ops.

    ops_t is a list of (3,3) integer matrices acting on the mesh integers m
    (k = m/n). Returns (k_frac (nk,3) in (-1/2,1/2], weights summing to 1).
    """
    mesh_arr = np.asarray(mesh, dtype=np.int64)
    grids = [np.arange(n) for n in mesh_arr]
    mm = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)  # integer m, k=m/n

    def key_of(m_int: np.ndarray) -> tuple[int, ...]:
        return tuple(m_int % mesh_arr)

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

    kfrac = mm[reps] / mesh_arr
    kfrac = -((-kfrac + 0.5) % 1.0 - 0.5)  # fold to (-1/2, 1/2]
    w = np.array(weights)
    assert abs(w.sum() - 1.0) < 1e-12
    return kfrac, w


def reduce_mesh(
    mesh: tuple[int, int, int],
    shift: tuple[int, int, int],
    sg: SpaceGroup,
    time_reversal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """IBZ reduction of a Γ-centered MP mesh. Returns (k_frac (nk,3), weights).

    Only valid for unshifted meshes; orbits are taken under {W⁻ᵀ} and
    optionally time reversal.
    """
    if np.any(shift):
        raise NotImplementedError(
            "shifted meshes not supported here; caller must reduce unshifted"
        )
    ops_t = _k_ops(sg.rotations)
    if time_reversal:
        ops_t = ops_t + [-w for w in ops_t]
    return _orbit_reduce(mesh, ops_t)


def _fold_bz(q: np.ndarray) -> np.ndarray:
    """Fold a fractional wavevector into the (-1/2, 1/2] cell (per component)."""
    return -((-np.asarray(q, float) + 0.5) % 1.0 - 0.5)


def little_cogroup(
    q_frac: np.ndarray, sg: SpaceGroup, tol: float = 1e-6
) -> tuple[SpaceGroup, np.ndarray]:
    """The little co-group of q: the point-group operations whose reciprocal
    action fixes q modulo a reciprocal-lattice vector, W⁻ᵀq ≡ q (mod 1).

    Returns ``(SpaceGroup of those ops, g0)`` where ``g0[i]`` is the integer
    umklapp W⁻ᵀq − q of op i (needed by a q-dependent field symmetrizer to fold
    G-vectors of the q-modulated response back onto the box). At q=Γ this is the
    full ``sg`` with g0 all zero. This is the group that symmetrizes a
    perturbation (and its response) of wavevector q; the k-points of the DFPT
    sum reduce under it.
    """
    q = np.asarray(q_frac, dtype=float)
    keep: list[int] = []
    g0s: list[np.ndarray] = []
    for i, w in enumerate(sg.rotations):
        w_inv_t = np.round(np.linalg.inv(w).T).astype(np.int64)
        g0 = w_inv_t @ q - q
        if np.max(np.abs(g0 - np.round(g0))) <= tol:
            keep.append(i)
            g0s.append(np.round(g0).astype(np.int64))
    lg = SpaceGroup(
        rotations=sg.rotations[keep], translations=sg.translations[keep],
        atom_map=sg.atom_map[keep], international=sg.international,
        origin_shift=sg.origin_shift)
    return lg, (np.stack(g0s) if g0s else np.zeros((0, 3), dtype=np.int64))


def star_of_q(
    q_frac: np.ndarray, sg: SpaceGroup, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """The star of q: the distinct images {W⁻ᵀ q} (mod 1), folded to (-1/2, 1/2].

    Returns ``(qs (s,3), rep_ops (s,))`` — the star members and, for each, the
    index into ``sg.rotations`` of one operation carrying q onto it. By
    orbit–stabilizer ``len(star) * little_cogroup(q).n_ops == sg.n_ops``; the
    full response over the BZ is reconstructed from the one representative q by
    generating its star.
    """
    q = np.asarray(q_frac, dtype=float)
    stars: list[np.ndarray] = []
    reps: list[int] = []
    for i, w in enumerate(sg.rotations):
        w_inv_t = np.round(np.linalg.inv(w).T).astype(np.int64)
        qi = _fold_bz(w_inv_t @ q)
        if not any(np.max(np.abs(_fold_bz(qi - s))) < tol for s in stars):
            stars.append(qi)
            reps.append(i)
    return np.stack(stars), np.asarray(reps, dtype=np.int64)


def little_group_ibz(
    mesh: tuple[int, int, int], q_frac: np.ndarray, sg: SpaceGroup,
    time_reversal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """IBZ reduction of a Γ-centered MP mesh under the little group of q.

    The DFPT k-sum at wavevector q is invariant under the little co-group of q
    (it fixes q, so it maps the k↔k+q pairing onto itself); the k-points reduce
    to that group's IBZ. Time reversal is added only when it too fixes q
    (−q ≡ q, i.e. q at Γ or a TR-invariant zone-boundary point). Reduces to
    ``reduce_mesh`` at q=Γ.
    """
    lg, _ = little_cogroup(q_frac, sg)
    ops_t = _k_ops(lg.rotations)
    if time_reversal and np.max(np.abs(_fold_bz(2.0 * np.asarray(q_frac, float)))) < 1e-6:
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
    sg: SpaceGroup,
    magmoms: list[list[float | int]] | np.ndarray,
    cell: np.ndarray,
    tol: float = 1e-5,
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


def reduce_mesh_magnetic(
    mesh: tuple[int, int, int],
    shift: tuple[int, int, int],
    mg: MagneticGroup,
    time_reversal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnetic-IBZ reduction of a Γ-centered MP mesh under a Shubnikov group.

    Unitary ops act on k as W⁻ᵀ; anti-unitary ops (g·T) as −W⁻ᵀ (the g·T time
    reversal sends k → −k). Zero moments (grey group) reproduce
    reduce_mesh(..., time_reversal=True) exactly. Returns (k_frac, weights).

    ``time_reversal`` adds the PLAIN k → −k fold on top of the magnetic group.
    This is the ORDINARY (non-spin-flipping) time reversal of a COLLINEAR
    (ρ↑, ρ↓) system: each spin channel has a real Hamiltonian H_σ(−k) = H_σ(k)*,
    so ε_σ(−k) = ε_σ(k) and n_σ(−k) = n_σ(k) per channel — folding −k onto k
    just doubles its weight, with no spin swap, so the collinear moment is
    preserved. It is NOT valid for the spinor (ρ, m⃗) path, where time reversal
    flips m⃗ (that reversal is already carried by the anti-unitary g·T ops), so
    it defaults off and only the collinear caller (collinear_magnetic=True)
    turns it on.

    Why it matters: the magnetic group only encodes k → −k when some op's
    ROTATION part already inverts k. bcc-Cr AFM gets it for free — its
    sublattice-swap anti-op is a pure LATTICE TRANSLATION (W = I, so −W⁻ᵀ = −I).
    Corundum R-3̄c AFMs (hematite, eskolaite) do NOT: their swap op is
    INVERSION·T (W = −I, so −W⁻ᵀ = +I, a trivial k-action), so the −k fold is
    absent from the magnetic group and must be added here. Without it hematite
    folds a 4×4×4 mesh to 20 and eskolaite to 16; with it both reach 13, the
    Quantum ESPRESSO count, and the folded SCF still matches the full-mesh free
    energy because the plain-TR reference (monkhorst_pack time_reversal=True)
    already assumes the same per-channel k → −k.
    """
    if np.any(shift):
        raise NotImplementedError(
            "shifted meshes not supported here; caller must reduce unshifted"
        )
    ops_t = _k_ops(mg.unitary.rotations)
    ops_t += [-w for w in _k_ops(mg.anti_rotations)]
    if time_reversal:
        ops_t += [-w for w in ops_t]  # plain per-channel k → −k (collinear)
    return _orbit_reduce(mesh, ops_t)


class RhoSymmetrizer:
    """Precomputed G-space symmetrization maps for a fixed FFT box.

    dens_mask restricts to the density sphere, where the Miller map is exact:
    at the box Nyquist boundary, folding Wᵀm mod n misidentifies G-vectors
    (phases differ by e^{iπ n·w} for non-symmorphic ops). Physical densities
    are zero there; masking makes the operator exactly idempotent.
    """

    def __init__(
        self, shape: tuple[int, int, int], sg: SpaceGroup, dens_mask: torch.Tensor | None = None
    ) -> None:
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

    def to(self, device: torch.device | str) -> RhoSymmetrizer:
        new = object.__new__(RhoSymmetrizer)
        new.idx = self.idx.to(device)
        new.phase = self.phase.to(device)
        new.mask = self.mask.to(device)
        new.shape = self.shape
        return new

    def with_mask(self, dens_mask: torch.Tensor) -> RhoSymmetrizer:
        """Shallow copy sharing the idx/phase maps with a fresh density-sphere
        mask. The maps are the expensive part (n_ops serial passes over the
        dense box) and depend only on (shape, ops); the mask is the one
        cell-dependent piece (strain moves the ecutrho sphere), so a memoized
        instance is re-dressed with the current grid's mask on reuse."""
        new = object.__new__(RhoSymmetrizer)
        new.idx = self.idx
        new.phase = self.phase
        new.shape = self.shape
        if dens_mask is not None:
            new.mask = dens_mask.reshape(-1).clone()
        else:
            n1, n2, n3 = self.shape
            new.mask = torch.ones(n1 * n2 * n3, dtype=torch.bool)
        return new

    def apply(self, rho_g_box: torch.Tensor) -> torch.Tensor:
        """Symmetrize ρ(G) on the dense box: (n1,n2,n3) complex → same."""
        flat = rho_g_box.reshape(-1) * self.mask
        acc = (self.phase * flat[self.idx]).mean(dim=0) * self.mask
        return acc.reshape(self.shape)


class QFieldSymmetrizer:
    """G-space symmetrization of a scalar field of crystal wavevector q.

    A response of wavevector q is stored on the dense FFT box as coefficients
    c(G) of the plane waves e^{i(q+G)·r} (the periodic part of
    δρ_q(r) = e^{iq·r} Σ_G c(G) e^{iG·r}). Only the **little co-group of q** (the
    ops with W⁻ᵀq ≡ q mod 1) symmetrizes it. Averaging over that group,
    ``(P c)(m) = (1/N) Σ_g e^{−2πi (q+m)·w_g} c(Wᵀ(m − g0_g))`` with g0 = W⁻ᵀq − q,
    is the projector onto the q-symmetric subspace — the group-average identity
    that folds the IBZ(of q) response to the full-BZ response, and the direct
    generalization of ``RhoSymmetrizer`` (recovered at q=Γ, g0=0, w-phase only).

    Two things differ from the q=Γ scalar case and are handled here:

    * **q-shifted mask.** The little-group action preserves |q+G|, not |G|, so
      the invariant band-limited set is the sphere centred at −q,
      {G : |q+G| ≤ G_cut}, not the ordinary density sphere. Using the latter
      drops a boundary shell for a nonzero umklapp and the operator stops being a
      projector. The mask is rebuilt as {|q+G| ≤ G_cut} from ``g2``/``dens_mask``.
    * **Projective small reps.** At a non-symmorphic zone-boundary q the small
      representation is projective (non-trivial factor system) and the plain
      average is not idempotent. Construction verifies idempotence on a probe and
      raises rather than fold with a wrong operator (see
      docs/design/little-group-star-unfold.md — use the full mesh at such q).
    """

    def __init__(
        self, shape: tuple[int, int, int], q_frac: np.ndarray, sg: SpaceGroup,
        cell: np.ndarray, g2: torch.Tensor, dens_mask: torch.Tensor,
    ) -> None:
        lg, g0 = little_cogroup(q_frac, sg)
        q = np.asarray(q_frac, dtype=float)
        n1, n2, n3 = shape
        dims = np.array([n1, n2, n3])
        millers = np.stack(
            np.meshgrid(*[np.fft.fftfreq(n, 1.0 / n).astype(np.int64) for n in shape],
                        indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        # q-shifted band-limit: {|q+G| ≤ G_cut(density)}, the set the little group
        # preserves. G_cut² is the max |G|² carried by the ordinary density mask.
        b = 2.0 * np.pi * np.linalg.inv(np.asarray(cell, float)).T   # reciprocal rows
        g_cart = millers @ b
        q_cart = q @ b
        g2_flat = g2.reshape(-1).cpu().numpy()
        gmax2 = float(g2_flat[dens_mask.reshape(-1).cpu().numpy()].max())
        qpg2 = np.einsum("ij,ij->i", g_cart + q_cart, g_cart + q_cart)
        mask_q = torch.as_tensor(qpg2 <= gmax2 * (1 + 1e-9), dtype=torch.bool)

        idx_maps, phases = [], []
        for w_mat, w_vec, g in zip(lg.rotations, lg.translations, g0, strict=True):
            source = (millers - g) @ w_mat            # Wᵀ(m − g0)  (gather source)
            folded = source % dims
            flat = folded[:, 0] * (n2 * n3) + folded[:, 1] * n3 + folded[:, 2]
            idx_maps.append(flat)
            phases.append(np.exp(-2j * np.pi * ((q + millers) @ w_vec)))
        self.idx = torch.as_tensor(np.stack(idx_maps), dtype=torch.int64)   # (nops,N)
        self.phase = torch.as_tensor(np.stack(phases), dtype=torch.complex128)
        self.shape = tuple(shape)
        self.mask = mask_q
        # projector check: idempotent iff the small rep is ordinary.
        gen = torch.Generator().manual_seed(0)
        probe = (torch.randn(n1 * n2 * n3, generator=gen, dtype=torch.float64)
                 + 1j * torch.randn(n1 * n2 * n3, generator=gen, dtype=torch.float64)
                 ).reshape(shape)
        p1 = self.apply(probe)
        p2 = self.apply(p1)
        scale = float(p1.abs().max())
        if scale > 1e-12 and float((p2 - p1).abs().max()) > 1e-8 * scale:
            raise NotImplementedError(
                "QFieldSymmetrizer: projective small representation at this q "
                "(non-symmorphic zone boundary) — the little-co-group average is "
                "not a projector here. Use the full (unreduced) mesh at this q; "
                "see docs/design/little-group-star-unfold.md.")

    def to(self, device: torch.device | str) -> QFieldSymmetrizer:
        new = object.__new__(QFieldSymmetrizer)
        new.idx = self.idx.to(device)
        new.phase = self.phase.to(device)
        new.mask = self.mask.to(device)
        new.shape = self.shape
        return new

    def apply(self, c_g_box: torch.Tensor) -> torch.Tensor:
        """Project a wavevector-q field c(G) onto the q-symmetric subspace:
        (n1,n2,n3) complex → same. Idempotent (a little-co-group average)."""
        flat = c_g_box.reshape(-1) * self.mask
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

    def __init__(
        self,
        shape: tuple[int, int, int],
        mg: MagneticGroup,
        cell: np.ndarray,
        dens_mask: torch.Tensor | None = None,
    ) -> None:
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

    def to(self, device: torch.device | str) -> MagneticSymmetrizer:
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


class CollinearMagneticSymmetrizer:
    """G-space symmetrization of a collinear (ρ↑, ρ↓) pair under a magnetic
    (Shubnikov) group — the nspin=2 FM/AFM analogue of MagneticSymmetrizer.

    A collinear spin density is a T-EVEN charge ρ = ρ↑+ρ↓ plus a T-ODD scalar
    magnetization m = ρ↑−ρ↓ (aligned on the magnetic axis). Under a UNITARY op
    both channels fold with the bare spatial op (RhoSymmetrizer); under an
    ANTI-UNITARY op g·T the spatial fold carries a SPIN-CHANNEL SWAP, because T
    reverses m (ρ↑ ↔ ρ↓). Equivalently:

        ρ_sym = combined_group.average(ρ↑+ρ↓)          (all ops, sign +1)
        m_sym = combined_group.average(ρ↑−ρ↓)  with −1 on every anti-unitary op

    and ρ↑,↓ = (ρ_sym ± m_sym)/2. This reconstructs the full-BZ collinear density
    from the magnetic-IBZ (reduce_mesh_magnetic) representatives, so the folded
    SCF converges to the same fixed point as the unreduced use_symmetry=False run.

    The ±1 axial sign is exact only for a genuinely COLLINEAR moment set (every
    op's Cartesian axial action sends the magnetic axis n̂ to ±n̂); the
    constructor asserts this — a non-collinear set must use MagneticSymmetrizer
    (scf_noncollinear) instead. The grey group (m ≡ 0) makes every op both
    unitary and anti-unitary; the swap then averages ρ↑ and ρ↓ together, which is
    correct for a non-spin-polarized cell.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        mg: MagneticGroup,
        cell: np.ndarray,
        magmoms: list[list[float | int]] | np.ndarray,
        dens_mask: torch.Tensor | None = None,
    ) -> None:
        combined = mg.combined()
        self.rho_sym = RhoSymmetrizer(shape, combined, dens_mask=dens_mask)
        # collinear magnetic axis n̂ from the moment set (first nonzero moment)
        m = np.atleast_2d(np.asarray(magmoms, dtype=float))
        norms = np.linalg.norm(m, axis=1)
        nz = np.flatnonzero(norms > 1e-8)
        axis = m[nz[0]] / norms[nz[0]] if len(nz) else np.array([0.0, 0.0, 1.0])
        # every moment must be parallel/antiparallel to n̂ (collinearity)
        if len(nz) and np.abs(np.cross(m[nz], axis)).max() > 1e-6 * norms[nz].max():
            raise ValueError(
                "CollinearMagneticSymmetrizer requires collinear moments; use "
                "MagneticSymmetrizer (scf_noncollinear) for a non-collinear set")
        a_t = np.asarray(cell, dtype=float).T
        a_t_inv = np.linalg.inv(a_t)
        signs = []
        for iop, w_mat in enumerate(combined.rotations):
            s = a_t @ w_mat @ a_t_inv
            r_ax = np.linalg.det(s) * s  # axial (pseudo-vector) spatial action
            # the m-field sign is s_T · (axial action on n̂): s_T = −1 on the
            # anti-unitary half (T reverses m), and the SPATIAL op must send n̂
            # to ±n̂ (else the moment set is non-collinear). The sublattice swap
            # that makes an op anti-unitary is carried by the spatial fold's
            # atom permutation + phase, not by this scalar sign.
            v = r_ax @ axis
            proj = float(v @ axis)
            if not np.allclose(v, proj * axis, atol=1e-8) or abs(abs(proj) - 1) > 1e-6:
                raise ValueError(
                    "magnetic op does not map the moment axis to ±itself — the "
                    "moment set is non-collinear; use MagneticSymmetrizer instead")
            s_t = -1.0 if iop >= mg.n_unitary else 1.0
            signs.append(s_t * round(proj))
        self.sign = torch.as_tensor(signs, dtype=torch.float64)
        self.shape = tuple(shape)

    def to(self, device: torch.device | str) -> CollinearMagneticSymmetrizer:
        new = object.__new__(CollinearMagneticSymmetrizer)
        new.rho_sym = self.rho_sym.to(device)
        new.sign = self.sign.to(device)
        new.shape = self.shape
        return new

    def apply(self, rho_g_box: torch.Tensor) -> torch.Tensor:
        """Charge-only symmetrization (T-even) over the full magnetic group.

        A convenience for callers that symmetrize a single scalar (e.g. the
        total density); the spin channels use apply_pair."""
        return self.rho_sym.apply(rho_g_box)

    def apply_pair(
        self, rho_up_box: torch.Tensor, rho_dn_box: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetrize a collinear (ρ↑, ρ↓) pair on the dense box: each
        (n1,n2,n3) complex → same. Anti-unitary ops swap the two channels."""
        rs = self.rho_sym
        tot = (rho_up_box + rho_dn_box).reshape(-1) * rs.mask
        mag = (rho_up_box - rho_dn_box).reshape(-1) * rs.mask
        tot_s = (rs.phase * tot[rs.idx]).mean(dim=0) * rs.mask
        sgn = self.sign.to(mag.dtype).reshape(-1, 1)
        mag_s = (rs.phase * sgn * mag[rs.idx]).mean(dim=0) * rs.mask
        up = (0.5 * (tot_s + mag_s)).reshape(self.shape)
        dn = (0.5 * (tot_s - mag_s)).reshape(self.shape)
        return up, dn


class VectorFieldSymmetrizer:
    """G-space symmetrization of a real POLAR vector field V_α(r) (α=x,y,z).

    The response of a scalar (the density) to a Cartesian perturbation that
    transforms as a vector — Δρ_α(r) under an applied E-field E_α — is a polar
    vector field: the three components mix through the Cartesian rotation
    S = Aᵀ W A⁻ᵀ of each space-group operation, exactly as MagneticSymmetrizer
    folds the axial m⃗ EXCEPT there is no det(S) factor (Δρ is a proper vector,
    m⃗ a pseudovector). The scalar G-fold (Miller map W⁻ᵀ, the non-symmorphic
    phase e^{−2πi m·w}, and the density-sphere mask) is reused verbatim from a
    RhoSymmetrizer built on the same group and box:

        V_sym,α(G) = (1/N_op) Σ_op S_op[α,β] · e^{−2πi m·w_op} · V_β(W_opᵀ G).

    Averaging the group-transform g·V over the closed group projects V onto the
    field-symmetric subspace; folding the E-field density response this way (in
    place of a naive per-component scalar symmetrization, which would wrongly
    treat the response as totally symmetric) reconstructs the full-BZ vector
    response from the IBZ representatives — the correctness statement behind
    running the dielectric/Born DFPT with IBZ symmetry reduction.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        sg: SpaceGroup,
        cell: np.ndarray,
        dens_mask: torch.Tensor | None = None,
    ) -> None:
        self.rho_sym = RhoSymmetrizer(shape, sg, dens_mask=dens_mask)
        a_t = np.asarray(cell, dtype=float).T
        a_t_inv = np.linalg.inv(a_t)
        rot = np.stack([a_t @ np.asarray(w, dtype=float) @ a_t_inv
                        for w in sg.rotations])  # polar Cartesian S per op
        self.rot = torch.as_tensor(rot, dtype=torch.float64)
        self.shape = tuple(shape)

    def to(self, device: torch.device) -> VectorFieldSymmetrizer:
        new = object.__new__(VectorFieldSymmetrizer)
        new.rho_sym = self.rho_sym.to(device)
        new.rot = self.rot.to(device)
        new.shape = self.shape
        return new

    def apply(self, v_g_box: torch.Tensor) -> torch.Tensor:
        """Symmetrize V(G) on the dense box: (3, n1,n2,n3) complex → same."""
        rs = self.rho_sym
        flat = v_g_box.reshape(3, -1) * rs.mask
        gathered = flat[:, rs.idx]  # (3, nops, N)
        mixed = torch.einsum("oab,bon->aon", self.rot.to(flat.dtype), gathered)
        acc = (rs.phase * mixed).mean(dim=1) * rs.mask
        return acc.reshape(3, *self.shape)


def symmetrize_tensor(t: torch.Tensor, sg: SpaceGroup, cell: np.ndarray) -> torch.Tensor:
    """Project a Cartesian rank-2 tensor onto the point-group-invariant subspace.

    T ← (1/N_op) Σ_op S_op T S_opᵀ, with S = Aᵀ W A⁻ᵀ. Used to symmetrize the
    dielectric tensor ε∞ accumulated over the IBZ: the star sum of the per-k
    contributions ⟨ξ^α|Δψ^β⟩ (each rotating as S·M·Sᵀ) is exactly this average
    scaled by the IBZ weights (closed group ⇒ Sᵀ...S gives the same result)."""
    a_t = np.asarray(cell, dtype=float).T
    a_t_inv = np.linalg.inv(a_t)
    dev = t.device
    acc = torch.zeros_like(t)
    for w in sg.rotations:
        s = torch.as_tensor(a_t @ np.asarray(w, dtype=float) @ a_t_inv,
                            dtype=t.dtype, device=dev)
        acc = acc + s @ t @ s.T
    return acc / sg.n_ops


def symmetrize_atom_tensor(z: torch.Tensor, sg: SpaceGroup, cell: np.ndarray) -> torch.Tensor:
    """Project a per-atom Cartesian rank-2 tensor (e.g. Born charges Z*_{s,αβ})
    onto the symmetric subspace, permuting atoms by the space-group action.

    Z[s] ← (1/N_op) Σ_op S_opᵀ Z[atom_map[op,s]] S_op — the two-index analogue
    of symmetrize_forces (same S = Aᵀ W A⁻ᵀ, same Sᵀ + atom_map convention), so
    the IBZ star sum of the Born tensor is reconstructed consistently with the
    force symmetrization already used by the code."""
    a_t = np.asarray(cell, dtype=float).T
    a_t_inv = np.linalg.inv(a_t)
    dev = z.device
    acc = torch.zeros_like(z)
    for w, amap in zip(sg.rotations, sg.atom_map, strict=True):
        s = torch.as_tensor(a_t @ np.asarray(w, dtype=float) @ a_t_inv,
                            dtype=z.dtype, device=dev)
        zt = z[torch.as_tensor(amap.copy(), device=dev)]  # (na,3,3) permuted
        acc = acc + torch.einsum("ij,ajk,kl->ail", s.T, zt, s)
    return acc / sg.n_ops


def symmetrize_forces(forces: torch.Tensor, sg: SpaceGroup, cell: np.ndarray) -> torch.Tensor:
    """Project forces onto the symmetry-invariant subspace.

    F_a ← (1/N) Σ_op Sᵀ F_{map(op,a)}, with S the Cartesian rotation of op.
    """
    a_t = np.asarray(cell, dtype=float).T
    dev = forces.device
    acc = torch.zeros_like(forces)
    for w_mat, amap in zip(sg.rotations, sg.atom_map, strict=True):
        s = a_t @ w_mat @ np.linalg.inv(a_t)
        s_t = torch.as_tensor(s.T, dtype=forces.dtype, device=dev)
        acc = acc + forces[torch.as_tensor(amap.copy(), device=dev)] @ s_t.T
    return acc / sg.n_ops
