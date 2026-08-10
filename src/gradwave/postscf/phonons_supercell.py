"""Supercell finite-displacement phonons — dispersion and DOS on a q-mesh.

The frozen-phonon (phonopy-style) method: build an N×N×N supercell of the
primitive cell, displace atoms, collect the real-space force constants
Φ_μν(R) = ∂²E/∂τ_{0μ}∂τ_{Rν} from the ground-state forces, Fourier-interpolate
to the dynamical matrix D(q) = Σ_R Φ_μν(R)/√(M_μ M_ν) · e^{iq·R}, and diagonalize
at any q. Unlike the analytic Γ path (`postscf.phonons`, DFPT-scoped), this needs
only ground-state forces (`postscf.forces`), so it runs for any q on a plain SCF.

Key cost point: because the force constants have the periodicity of the
PRIMITIVE lattice, only the N_prim atoms of the home cell need to be displaced
(their forces on every supercell image fill Φ_μν(R) for all R). So the SCF count
is 6·N_prim — INDEPENDENT of supercell size (Si 2×2×2 → 12 SCFs, not 96). This
is the Born–von-Kármán translational reduction; it is built into
`force_constants_home` by construction.

Units follow the package (eV, Å, amu); frequencies come out in cm⁻¹ (negative =
imaginary), sharing the exact conversion constant with the Γ path so the two
cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.postscf.phonons import _SQRT_EV_AMU_ANG2_TO_CM1
from gradwave.symmetry import SpaceGroup, find_spacegroup


@dataclass(frozen=True)
class SupercellMap:
    """Deterministic primitive→supercell layout and the (μ, R) label of each
    supercell site. Site `s` is ordered as `s = t·N_prim + μ` where `t` indexes
    the translation cell (0,0,0) first, so the home-cell atoms are sites
    0…N_prim-1."""

    supercell: tuple[int, int, int]  # (n1, n2, n3)
    cell_prim: np.ndarray       # (3,3) primitive cell, rows = lattice vectors [Å]
    cell_super: np.ndarray      # (3,3) supercell cell [Å]
    positions_super: np.ndarray  # (N_sc, 3) Cartesian [Å]
    species_super: list[int]    # (N_sc,) species-of-atom indices
    mu_of_site: np.ndarray      # (N_sc,) int — primitive basis atom of each site
    rint_of_site: np.ndarray    # (N_sc, 3) int — integer lattice translation R
    n_prim: int
    n_sc: int

    @property
    def home_sites(self) -> np.ndarray:
        """Site indices of the home (R=0) cell, one per primitive atom."""
        return np.arange(self.n_prim)


def build_supercell(
    cell: np.ndarray, positions: np.ndarray, species_of_atom: list[int],
    supercell: tuple[int, int, int],
) -> SupercellMap:
    """Tile a primitive cell into a diagonal (n1,n2,n3) supercell, tracking the
    (primitive-atom μ, lattice-translation R) label of every supercell site.

    `cell` rows are the primitive lattice vectors [Å]; `positions` are Cartesian
    [Å]; `species_of_atom` are indices into the pseudopotential list.
    """
    cell = np.asarray(cell, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n1, n2, n3 = (int(x) for x in supercell)
    if min(n1, n2, n3) < 1:
        raise ValueError(f"supercell must be positive integers, got {supercell}")
    n_prim = len(positions)
    cell_super = np.diag([n1, n2, n3]) @ cell

    pos, spec, mu, rint = [], [], [], []
    for m0 in range(n1):
        for m1 in range(n2):
            for m2 in range(n3):
                r_cart = np.array([m0, m1, m2], dtype=float) @ cell
                for a in range(n_prim):
                    pos.append(positions[a] + r_cart)
                    spec.append(species_of_atom[a])
                    mu.append(a)
                    rint.append((m0, m1, m2))
    return SupercellMap(
        supercell=(n1, n2, n3), cell_prim=cell, cell_super=cell_super,
        positions_super=np.array(pos), species_super=spec,
        mu_of_site=np.array(mu, dtype=int), rint_of_site=np.array(rint, dtype=int),
        n_prim=n_prim, n_sc=n_prim * n1 * n2 * n3)


def _site_lookup(scmap: SupercellMap) -> dict[tuple[tuple[int, int, int], int], int]:
    """(integer R, primitive atom μ) → supercell site index."""
    return {(tuple(scmap.rint_of_site[s]), int(scmap.mu_of_site[s])): s
            for s in range(scmap.n_sc)}


def symmetrize_force_constants(phi_home: np.ndarray,
                               scmap: SupercellMap) -> np.ndarray:
    """Enforce the physical symmetry Φ_μν(R) = Φ_νμ(−R)ᵀ, broken slightly by
    finite-difference noise, by averaging each block with its (−R) partner.
    Improves the acoustic modes and the accuracy of D(q) at every q."""
    n = np.array(scmap.supercell)
    look = _site_lookup(scmap)
    out = np.empty_like(phi_home)
    for s in range(scmap.n_sc):
        r = scmap.rint_of_site[s]
        nu = int(scmap.mu_of_site[s])
        neg_r = tuple((-r) % n)
        for mu in range(scmap.n_prim):
            s2 = look[(neg_r, mu)]
            out[mu, :, s, :] = 0.5 * (phi_home[mu, :, s, :]
                                      + phi_home[nu, :, s2, :].T)
    return out


def apply_acoustic_sum_rule(phi_home: np.ndarray) -> np.ndarray:
    """Enforce Σ_{s} Φ_home[μ,i,s,j] = 0 by correcting each μ's self block
    (the R=0, ν=μ term at site μ). Guarantees three exactly-zero acoustic modes
    at q=0. `phi_home` is (N_prim, 3, N_sc, 3)."""
    phi = phi_home.copy()
    n_prim = phi.shape[0]
    resid = phi.sum(axis=2)  # (N_prim, 3, 3) — Σ over supercell sites
    for a in range(n_prim):
        phi[a, :, a, :] -= resid[a]  # site a is the home-cell copy of atom a
    return phi


_RANK_TOL = 1e-8


class SupercellDisplacementSymmetry:
    """Point-group reduction of the finite-displacement set for the supercell
    force constants Φ_home (N_prim, 3, N_sc, 3).

    The FD path in `force_constants_home` displaces every home atom along every
    axis — 6·N_prim SCFs. But the crystal point group relates those columns:
    under a space-group op {W|w} with Cartesian rotation S and the full
    supercell site-permutation P (the primitive `atom_map` composed with the
    integer lattice translation W·R, reduced mod the diagonal supercell), the
    force constants obey the two-index covariance

        Φ[P(s₁), P(s₂)] = S · Φ[s₁, s₂] · Sᵀ            (3×3 Cartesian blocks).

    So the column computed for the displacement (home atom a, axis α) fills the
    column for the displacement (home atom g(a), direction S·eα) by rotating the
    3-vectors and permuting the sites — the *force-completion* transform. This is
    the finite-displacement analogue of `postscf.phonons.HessianSymmetry`: the
    irreducible-direction selection is byte-identical (it depends only on the
    per-atom direction orbits, i.e. on `s_cart`/`atom_map`), and the extra work
    is the full N_sc site permutation plus the home-cell re-centering translation
    that maps the image column back onto Φ_home.

    Only ops that are genuine symmetries of the *diagonal* supercell are kept: an
    op whose site permutation is not a bijection (a rotation that mixes lattice
    axes of unequal repeat count, e.g. a cubic op on a 2×1×1 box) is dropped. If
    the surviving ops cannot span all three directions at some atom, the caller
    must fall back to the full displacement set (`can_reduce` reports this).

    Exactness caveat (shared with `HessianSymmetry`): the reconstruction is exact
    only if the underlying force evaluations respect the point group. For a
    symmorphic monatomic metal on a symmetry-commensurate FFT grid this holds to
    round-off; a non-symmorphic translation that does not land on grid points
    leaves the XC-quadrature eggbox anisotropy in the rotated columns.
    """

    def __init__(self, scmap: SupercellMap, symprec: float = 1e-6) -> None:
        self.scmap = scmap
        n_prim = scmap.n_prim
        cell = np.asarray(scmap.cell_prim, dtype=float)
        pos_home = np.asarray(scmap.positions_super[:n_prim], dtype=float)
        species_home = list(scmap.species_super[:n_prim])
        frac = pos_home @ np.linalg.inv(cell)
        self.na = n_prim
        self.sg: SpaceGroup = find_spacegroup(cell, frac, species_home,
                                              symprec=symprec)
        a_t = cell.T  # columns = lattice vectors
        s_cart_all = np.array(
            [a_t @ w @ np.linalg.inv(a_t) for w in self.sg.rotations])

        # Dense (μ, r0, r1, r2) → site lookup for the diagonal supercell.
        n1, n2, n3 = scmap.supercell
        self._nsuper = np.array([n1, n2, n3], dtype=int)
        self._site_of = np.full((n_prim, n1, n2, n3), -1, dtype=int)
        for s in range(scmap.n_sc):
            r = scmap.rint_of_site[s] % self._nsuper
            self._site_of[scmap.mu_of_site[s], r[0], r[1], r[2]] = s

        # Per-op supercell site permutation, kept only when it is a bijection
        # (i.e. the op is a symmetry of the diagonal box). `mu_image[op]` and
        # `r_image[op]` give P(s) split into (primitive atom, lattice R mod n).
        keep: list[int] = []
        self.mu_image: list[np.ndarray] = []
        self.r_image: list[np.ndarray] = []
        for iop, w_mat in enumerate(self.sg.rotations):
            amap = self.sg.atom_map[iop]
            w_vec = self.sg.translations[iop]
            # L_μ: integer lattice shift so that W·x_μ + w = x_{g(μ)} + L_μ
            lshift = np.rint(
                (frac @ w_mat.T + w_vec[None, :]) - frac[amap]).astype(int)
            mu_s = scmap.mu_of_site           # (N_sc,)
            r_s = scmap.rint_of_site          # (N_sc, 3)
            mu_img = amap[mu_s]
            r_img = (lshift[mu_s] + r_s @ w_mat.T) % self._nsuper
            s_img = self._site_of[mu_img, r_img[:, 0], r_img[:, 1], r_img[:, 2]]
            if np.unique(s_img).size != scmap.n_sc:
                continue  # not a bijection → not a supercell symmetry, drop
            keep.append(iop)
            self.mu_image.append(mu_img)
            self.r_image.append(r_img)
        self.s_cart = s_cart_all[keep]
        self.atom_map = self.sg.atom_map[keep]
        self.displacements, self.can_reduce = self._select()

    def _spread(self, disp: list[tuple[int, int]]) -> list[list[np.ndarray]]:
        dirs: list[list[np.ndarray]] = [[] for _ in range(self.na)]
        for a, alpha in disp:
            e = np.zeros(3)
            e[alpha] = 1.0
            for s, amap in zip(self.s_cart, self.atom_map, strict=True):
                dirs[amap[a]].append(s @ e)
        return dirs

    def _select(self) -> tuple[list[tuple[int, int]], bool]:
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
        can_reduce = all(
            np.linalg.matrix_rank(np.array(dirs).reshape(-1, 3), tol=_RANK_TOL) == 3
            for dirs in self._spread(disp))
        return disp, can_reduce

    def reconstruct(self, cols: list[np.ndarray]) -> np.ndarray:
        """Full raw Φ_home (N_prim, 3, N_sc, 3) from the irreducible columns.

        `cols[i]` is the (N_sc, 3) column Φ_home[a_i, α_i, :, :] for the i-th
        entry of `self.displacements` (as `force_constants_home` builds it:
        −∂F_{s,j}/∂τ_{a,α}). Each computed column is spread over the group by the
        force-completion transform and the per-atom orbits enter one least-
        squares solve — identical in spirit to `HessianSymmetry.reconstruct`,
        with the added site permutation. Linear in the columns, so autograd flows
        through it when the columns are tensors converted upstream."""
        cols = [np.asarray(
            c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else c,
            dtype=float).reshape(self.scmap.n_sc, 3) for c in cols]
        if len(cols) != len(self.displacements):
            raise ValueError("cols must match self.displacements")
        n_sc = self.scmap.n_sc
        acc_d: list[list[np.ndarray]] = [[] for _ in range(self.na)]
        acc_c: list[list[np.ndarray]] = [[] for _ in range(self.na)]
        for (a, alpha), c in zip(self.displacements, cols, strict=True):
            e = np.zeros(3)
            e[alpha] = 1.0
            for op in range(len(self.s_cart)):
                s = self.s_cart[op]
                amap = self.atom_map[op]
                mu_img = self.mu_image[op]
                r_img = self.r_image[op]
                # re-center: op sends home atom a to site (g(a), R_a); translate
                # the whole image by −R_a so g(a) lands back in the home cell.
                r_a = r_img[a]
                r_out = (r_img - r_a) % self._nsuper
                s_out = self._site_of[mu_img, r_out[:, 0], r_out[:, 1], r_out[:, 2]]
                cprime = np.zeros((n_sc, 3))
                cprime[s_out] = c @ s.T  # rotate the output 3-vectors, permute sites
                acc_d[amap[a]].append(s @ e)
                acc_c[amap[a]].append(cprime)
        phi = np.zeros((self.na, 3, n_sc, 3))
        for b in range(self.na):
            e_mat = np.stack(acc_d[b], axis=1)                 # (3, m)
            c_mat = np.stack(acc_c[b], axis=2)                 # (N_sc, 3, m)
            pinv = np.linalg.pinv(e_mat)                       # (m, 3)
            # Φ_home[b, i, s, j] = Σ_k c_mat[s, j, k] · pinv[k, i]
            phi[b] = np.einsum("sjk,ki->isj", c_mat, pinv)
        return phi


def force_constants_home(make_scf, scmap: SupercellMap, h: float = 0.01,
                         acoustic_sum_rule: bool = True, warm_start: bool = True,
                         xc: XCFunctional | SpinXC | None = None,
                         force_fn: Callable[[Any], Any] | None = None,
                         use_displacement_symmetry: bool = False,
                         verbose: bool = False) -> np.ndarray:
    """FD force constants Φ_home (N_prim, 3, N_sc, 3) [eV/Å²].

    Displaces ONLY the N_prim home-cell atoms (±h) and reads the analytic force
    on every supercell site — the translational reduction that makes the SCF
    count 6·N_prim regardless of supercell size. `make_scf(positions,
    start_from=None)` must return a converged SCF result. Each displaced SCF
    warm-starts from the undisplaced reference when `warm_start`.

    The force route is formalism-agnostic through `force_fn`: when None (the
    default) the norm-conserving `postscf.forces.forces(res, xc=xc)` path is
    used; pass `force_fn=lambda res: forces_uspp(res, xc)` to fold an
    ultrasoft/PAW run (the displacement count logic is identical either way,
    only the analytic force differs by pseudopotential kind). `force_fn` must
    return a torch tensor of shape (N_sc, 3) [eV/Å].

    `xc` is forwarded to the default `postscf.forces.forces`; it is required
    only when the pseudos carry an NLCC core charge (the core-correction force
    term), and is ignored for valence-only species. It is unused when a
    `force_fn` is supplied (the caller closes over its own xc there).

    With `use_displacement_symmetry=True` only the point-group-irreducible
    (home atom, axis) displacements are actually run through `make_scf`; the
    remaining columns of Φ are filled by the crystal group action (see
    `SupercellDisplacementSymmetry`). This is exact for a symmorphic crystal on
    a symmetry-commensurate grid (fcc Al / bcc Fe: 6→2 SCFs) and is byte-
    identical to the full path when the reduction cannot be safely derived
    (low symmetry, or a non-diagonal-compatible op set), in which case it
    transparently falls back to the full 6·N_prim displacement set. Default
    False → the exact 6·N_prim path, byte-for-byte unchanged."""
    from gradwave.postscf.forces import forces

    if force_fn is None:
        def force_fn(res: Any) -> Any:
            return forces(res, xc=xc)

    pos0 = scmap.positions_super.copy()
    ref = make_scf(pos0) if warm_start else None
    n_prim, n_sc = scmap.n_prim, scmap.n_sc

    def _column(a: int, i: int) -> np.ndarray:
        """Raw FD column Φ_home[a, i, :, :] = −∂F_{:,:}/∂τ_{a,i} [eV/Å²]."""
        pp, pm = pos0.copy(), pos0.copy()
        pp[a, i] += h
        pm[a, i] -= h
        fp = force_fn(make_scf(pp, start_from=ref)).detach().cpu().numpy()
        fm = force_fn(make_scf(pm, start_from=ref)).detach().cpu().numpy()
        if verbose:
            print(f"  displaced home atom {a} axis {i}", flush=True)
        return -(fp - fm) / (2.0 * h)

    sym = None
    if use_displacement_symmetry:
        sym = SupercellDisplacementSymmetry(scmap)
        if not sym.can_reduce:
            if verbose:
                print("  displacement symmetry: reduction unsafe → full set",
                      flush=True)
            sym = None

    if sym is not None:
        if verbose:
            print(f"  displacement symmetry ({sym.sg.international}): "
                  f"{len(sym.displacements)} irreducible of {3 * n_prim} "
                  f"→ {2 * len(sym.displacements)} force SCFs", flush=True)
        cols = [_column(a, i) for a, i in sym.displacements]
        phi = sym.reconstruct(cols)
    else:
        phi = np.zeros((n_prim, 3, n_sc, 3))
        for a in range(n_prim):  # home-cell atoms are sites 0…N_prim-1
            for i in range(3):
                # Φ_{(a,i),(s,j)} = −∂F_{s,j}/∂τ_{a,i}
                phi[a, i, :, :] = _column(a, i)

    phi = symmetrize_force_constants(phi, scmap)
    if acoustic_sum_rule:
        phi = apply_acoustic_sum_rule(phi)
    return phi


def displacement_list(
    scmap: SupercellMap, h: float
) -> list[tuple[int, int, int, np.ndarray]]:
    """The 6·N_prim central-FD displacements as ``(a, i, sign, positions)``.

    Displaces ONLY the N_prim home-cell atoms (sites 0…N_prim-1) by ``sign·h``
    along axis ``i``; ``positions`` is the full (N_sc, 3) displaced supercell
    geometry [Å]. This is the parallel-dispatch counterpart of the inner loop in
    :func:`force_constants_home` — the two enumerate the same displacements in
    the same order, so a serial run and a SeedPool run see an identical spoke
    set. Reassemble the force constants from the per-spoke forces with
    :func:`force_constants_from_forces`."""
    pos0 = scmap.positions_super
    tasks: list[tuple[int, int, int, np.ndarray]] = []
    for a in range(scmap.n_prim):
        for i in range(3):
            for sign in (+1, -1):
                p = pos0.copy()
                p[a, i] += sign * h
                tasks.append((a, i, sign, p))
    return tasks


def force_constants_from_forces(
    force_map: dict[tuple[int, int, int], np.ndarray],
    scmap: SupercellMap,
    h: float,
    acoustic_sum_rule: bool = True,
) -> np.ndarray:
    """Assemble Φ_home (N_prim, 3, N_sc, 3) [eV/Å²] from precomputed spoke forces.

    ``force_map[(a, i, sign)]`` is the analytic force (N_sc, 3) [eV/Å] on every
    supercell site when home atom ``a`` is displaced by ``sign·h`` along axis
    ``i`` (the spokes enumerated by :func:`displacement_list`). The central
    difference, symmetrization and acoustic-sum-rule steps are byte-identical to
    :func:`force_constants_home`'s — only the SCF/force evaluation was hoisted
    out (into parallel workers)."""
    n_prim, n_sc = scmap.n_prim, scmap.n_sc
    phi = np.zeros((n_prim, 3, n_sc, 3))
    for a in range(n_prim):
        for i in range(3):
            fp = np.asarray(force_map[(a, i, +1)])
            fm = np.asarray(force_map[(a, i, -1)])
            # Φ_{(a,i),(s,j)} = −∂F_{s,j}/∂τ_{a,i}
            phi[a, i, :, :] = -(fp - fm) / (2.0 * h)
    phi = symmetrize_force_constants(phi, scmap)
    if acoustic_sum_rule:
        phi = apply_acoustic_sum_rule(phi)
    return phi


def phonon_dos(phi_home, scmap, masses_amu, q_mesh, weights,
               width: float = 6.0, npoints: int = 600):
    """Gaussian-broadened phonon DOS over a q-mesh. `width` in cm⁻¹. Returns
    (frequency_grid [cm⁻¹], dos)."""
    from gradwave.postscf.pdos import _broaden, spectral_grid

    freqs = dispersion(phi_home, scmap, masses_amu, q_mesh)  # (nq, 3N)
    e = freqs.reshape(-1)
    w = np.repeat(np.asarray(weights, dtype=float), freqs.shape[1])
    _window, grid = spectral_grid(e, width, npoints)
    return grid, _broaden(grid, e, w, width)


def dynamical_matrix(phi_home: np.ndarray, scmap: SupercellMap,
                     masses_amu: np.ndarray, q_frac: Sequence[int | float]) -> np.ndarray:
    """Mass-weighted dynamical matrix D(q) (3N_prim, 3N_prim) complex-Hermitian.

    D_{μi,νj}(q) = (1/√(M_μ M_ν)) Σ_R Φ_{μν}(R)[i,j] e^{2πi q·R}, with the sum
    over every supercell site (each carries one (ν, R)). `q_frac` is in
    fractional coordinates of the primitive reciprocal lattice; `masses_amu` are
    the N_prim primitive-atom masses [amu]."""
    n = scmap.n_prim
    m = np.asarray(masses_amu, dtype=float)
    q = np.asarray(q_frac, dtype=float)
    d = np.zeros((n, 3, n, 3), dtype=complex)
    phases = np.exp(2j * np.pi * (scmap.rint_of_site @ q))  # (N_sc,)
    for s in range(scmap.n_sc):
        nu = scmap.mu_of_site[s]
        d[:, :, nu, :] += phi_home[:, :, s, :] * phases[s]
    msqrt = np.sqrt(m)
    d /= msqrt[:, None, None, None] * msqrt[None, None, :, None]
    d = d.reshape(3 * n, 3 * n)
    return 0.5 * (d + d.conj().T)


def frequencies_at_q(
    phi_home: np.ndarray, scmap: SupercellMap, masses_amu: np.ndarray,
    q_frac: Sequence[int | float],
) -> np.ndarray:
    """Phonon frequencies at one q [cm⁻¹], sorted; negative = imaginary."""
    d = dynamical_matrix(phi_home, scmap, masses_amu, q_frac)
    w2 = np.linalg.eigvalsh(d)  # real (Hermitian), ascending
    return np.sign(w2) * _SQRT_EV_AMU_ANG2_TO_CM1 * np.sqrt(np.abs(w2))


def dispersion(phi_home, scmap, masses_amu, q_frac_list) -> np.ndarray:
    """Frequencies [cm⁻¹] along a list of q-points → (n_q, 3N_prim)."""
    return np.array([frequencies_at_q(phi_home, scmap, masses_amu, q)
                     for q in q_frac_list])
