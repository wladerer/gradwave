"""Elastic constants by finite difference of the analytic stress tensor.

C_ij = ∂σ_i/∂ε_j (Voigt 6×6) is obtained by applying each of the six symmetric
strains ±h, re-converging the SCF, and central-differencing gradwave's analytic
stress ``postscf.stress.stress`` (norm-conserving) / ``paw_stress.stress_uspp``
(USPP/PAW). gradwave's stress is +(1/Ω)∂E/∂ε, so C comes out positive-definite
for a mechanically stable crystal with no sign flip.

This module is the pure-numerics half (Voigt bookkeeping, the FD driver over a
caller-supplied ``stress_at_strain`` closure, and the Voigt–Reuss–Hill
polycrystalline averages); the SCF/strain plumbing lives in ``api.run_elastic``.

Whether C comes out clamped-ion or relaxed-ion is decided by the closure, not
by this module. With fractional coordinates held fixed at each strain (only the
electrons re-relaxed) the result is the CLAMPED-ION tensor: exact for any
constant with no symmetry-allowed internal displacement — the bulk modulus of
any crystal, and every constant of rocksalt (MgO, NaCl) where the atoms stay on
their special positions under strain — but NOT for shear constants of the
diamond/zincblende structure (Si, C, GaAs), where a shear strain induces an
internal sublattice shift (the Kleinman internal parameter) and the clamped-ion
C44 overestimates the fully-relaxed value (PBE Si: clamped-ion C44 ≈ 98 GPa vs
relaxed ≈ 76). If the closure instead relaxes the internal coordinates at every
strained cell before returning the stress, the same driver yields the
RELAXED-ION tensor. ``api.run_elastic`` provides both via ``elastic.mode``
("clamped" | "relaxed").

Units: stress in eV/Å³, so C is eV/Å³ internally and reported in GPa.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gradwave.constants import EV_A3_TO_GPA
from gradwave.symmetry import (
    SpaceGroup,
    coupled_axis_groups,
    find_spacegroup,
)

_RANK_TOL = 1e-8

# Voigt index (0-based) → tensor (i, j) component, engineering-shear convention:
# ε_voigt = [ε_xx, ε_yy, ε_zz, 2ε_yz, 2ε_xz, 2ε_xy], σ_voigt = [σ_xx, …, σ_xy].
_VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def voigt_strain_tensor(j: int, h: float) -> np.ndarray:
    """Symmetric 3×3 strain for a unit Voigt perturbation ``h`` in direction j.

    Engineering-shear convention: a shear Voigt component ε_4 = 2·ε_yz, so the
    off-diagonal tensor entry is h/2 (setting ε_voigt exactly to h·e_j)."""
    eps = np.zeros((3, 3))
    a, b = _VOIGT[j]
    if a == b:
        eps[a, b] = h
    else:
        eps[a, b] = eps[b, a] = 0.5 * h
    return eps


def stress_to_voigt(sigma: np.ndarray) -> np.ndarray:
    """(3,3) stress → 6-vector [xx, yy, zz, yz, xz, xy]."""
    s = np.asarray(sigma, dtype=float)
    return np.array([s[i, j] for i, j in _VOIGT])


def _stress_voigt_to_tensor(v: np.ndarray) -> np.ndarray:
    """6-vector Voigt stress → symmetric (3,3); off-diagonals carry the value
    directly (no engineering factor, unlike the strain convention)."""
    sig = np.zeros((3, 3))
    for m, (i, j) in enumerate(_VOIGT):
        sig[i, j] = sig[j, i] = v[m]
    return sig


def _strain_tensor_to_voigt(eps: np.ndarray) -> np.ndarray:
    """Symmetric (3,3) strain → engineering 6-vector [ε_xx,…, 2ε_yz, 2ε_xz,
    2ε_xy] (the inverse of :func:`voigt_strain_tensor`, off-diagonals doubled)."""
    e = np.asarray(eps, dtype=float)
    return np.array([e[i, j] if i == j else 2.0 * e[i, j] for i, j in _VOIGT])


def elastic_tensor(
    stress_at_strain: Callable[[np.ndarray], np.ndarray],
    h: float = 0.005,
    symmetry: ElasticStrainSymmetry | None = None,
) -> np.ndarray:
    """Clamped-ion stiffness C (6×6) [GPa] by central FD of the stress.

    ``stress_at_strain(eps)`` takes a symmetric 3×3 strain tensor, applies it to
    the cell, re-converges the SCF and returns the analytic stress (3,3) in
    eV/Å³. Column j is C[:, j] = ∂σ/∂ε_j; the result is symmetrized
    (C = ½(C+Cᵀ)) and converted to GPa.

    When ``symmetry`` is supplied and it can safely reduce (``can_reduce``), only
    the Laue-point-group-irreducible Voigt strains are actually run through
    ``stress_at_strain``; the remaining stress columns — and thus the full C —
    are reconstructed from the crystal group action
    (:class:`ElasticStrainSymmetry`). This is exact for a crystal whose stress
    evaluations respect the point group (a symmetry-commensurate FFT box) and is
    byte-identical to the full 6-strain path when the reduction is unsafe
    (``symmetry=None`` or ``not symmetry.can_reduce``): cubic metals run 2
    irreducible strains instead of 6.
    """
    if symmetry is not None and symmetry.can_reduce:
        cols = []
        for j in symmetry.strains:
            sp = stress_to_voigt(stress_at_strain(voigt_strain_tensor(j, +h)))
            sm = stress_to_voigt(stress_at_strain(voigt_strain_tensor(j, -h)))
            cols.append((sp - sm) / (2.0 * h))
        c = symmetry.reconstruct(cols)
    else:
        c = np.zeros((6, 6))
        for j in range(6):
            sp = stress_to_voigt(stress_at_strain(voigt_strain_tensor(j, +h)))
            sm = stress_to_voigt(stress_at_strain(voigt_strain_tensor(j, -h)))
            c[:, j] = (sp - sm) / (2.0 * h)
    c = 0.5 * (c + c.T)
    return c * EV_A3_TO_GPA


def _cart_rotations(sg: SpaceGroup, cell: np.ndarray) -> np.ndarray:
    """Cartesian rotations S = AᵀW A⁻ᵀ for every space-group op (rows of ``cell``
    are the lattice vectors), the same convention as ``symmetry.symmetrize_tensor``
    / ``symmetrize_forces``. Returns (n_op, 3, 3)."""
    a_t = np.asarray(cell, dtype=float).T
    a_t_inv = np.linalg.inv(a_t)
    return np.array([a_t @ np.asarray(w, dtype=float) @ a_t_inv
                     for w in sg.rotations])


def symmetrize_elastic_tensor(c4: np.ndarray, sg: SpaceGroup,
                              cell: np.ndarray) -> np.ndarray:
    """Project a rank-4 Cartesian tensor onto the point-group-invariant subspace.

    C_ijkl ← (1/N_op) Σ_op S_ip S_jq S_kr S_ls C_pqrs, with S = AᵀW A⁻ᵀ — the
    rank-4 generalization of ``symmetry.symmetrize_tensor`` (rank-2 ε∞/stress).
    Enforces the crystal point-group symmetry of the stiffness exactly; combined
    with the minor/major symmetries carried by the Voigt round-trip it yields the
    fully symmetry-reduced elastic tensor (e.g. the cubic C11/C12/C44 pattern).
    Linear in ``c4`` (a fixed group average), so autograd flows through it when
    ``c4`` is kept as a differentiable array upstream."""
    s_all = _cart_rotations(sg, cell)
    c4 = np.asarray(c4, dtype=float)
    acc = np.zeros((3, 3, 3, 3))
    for s in s_all:
        acc += np.einsum("ip,jq,kr,ls,pqrs->ijkl", s, s, s, s, c4)
    return acc / len(s_all)


def voigt_to_tensor(c6: np.ndarray) -> np.ndarray:
    """6×6 Voigt stiffness → full rank-4 C_ijkl (3,3,3,3).

    Stiffness needs NO engineering factor (unlike compliance): σ_ij = C_ijkl ε_kl
    in tensor form maps term-for-term to σ_m = C[m,n] ε̃_n once the factor-2 on
    the engineering shear strain is absorbed by summing the (kl) and (lk) tensor
    slots. So C_ijkl = C_voigt[m,n] with (ij)→m, (kl)→n, filled over both minor
    symmetries."""
    c6 = np.asarray(c6, dtype=float)
    c4 = np.zeros((3, 3, 3, 3))
    for m, (i, j) in enumerate(_VOIGT):
        for n, (k, l) in enumerate(_VOIGT):
            val = c6[m, n]
            for a, b in {(i, j), (j, i)}:
                for p, q in {(k, l), (l, k)}:
                    c4[a, b, p, q] = val
    return c4


def tensor_to_voigt(c4: np.ndarray) -> np.ndarray:
    """Full rank-4 C_ijkl (3,3,3,3) → 6×6 Voigt stiffness (inverse of
    :func:`voigt_to_tensor`, no engineering factor for the stiffness)."""
    c4 = np.asarray(c4, dtype=float)
    c6 = np.zeros((6, 6))
    for m, (i, j) in enumerate(_VOIGT):
        for n, (k, l) in enumerate(_VOIGT):
            c6[m, n] = c4[i, j, k, l]
    return c6


class ElasticStrainSymmetry:
    """Laue-point-group reduction of the 6 Voigt strains for the stiffness FD.

    ``elastic_tensor`` applies all six ±Voigt strains and reads the stress column
    C[:, j] = ∂σ/∂ε_j. But the crystal point group relates those columns: a strain
    ε transforms as ε → S ε Sᵀ and the stiffness is invariant (C_ijkl = S_ip S_jq
    S_kr S_ls C_pqrs), so the stress response to the rotated strain is the rotated
    response — no extra SCF. This is the strain analogue of
    ``postscf.phonons_supercell.SupercellDisplacementSymmetry``: ``_select`` picks
    a minimal set of Voigt directions whose group orbit spans the full 6-D space
    of symmetric strains, and ``reconstruct`` fills the remaining columns (and thus
    all of C) by one least-squares solve over the orbit — identical in spirit to
    the phonon force-completion, in 6-D Voigt space instead of per-atom 3-D.

    Only ops that are genuine symmetries of the pinned (anisotropic) FFT box are
    used: an op that couples two FFT axes of unequal length would break the
    quadrature symmetry the reconstruction relies on, so if the box is not
    isotropic enough for the point group the reduction is refused (``can_reduce``
    → False) and the caller falls back to the full 6-strain set. For a cubic
    metal the FFT box comes out n1=n2=n3 and the reduction is safe (6 → 2 Voigt
    strains: one axial + one shear).

    Exactness caveat (shared with the phonon path): the reconstruction is exact
    only where the underlying stress evaluations respect the point group — for a
    symmorphic crystal on a symmetry-commensurate grid this holds to the SCF
    symmetry floor; a grid that breaks the group leaves the XC-quadrature eggbox
    anisotropy in the rotated columns, which is why the FFT-box guard exists.
    """

    def __init__(self, cell: np.ndarray, frac_positions: np.ndarray,
                 species_of_atom: list[int],
                 fft_shape: tuple[int, ...] | None = None,
                 symprec: float = 1e-6) -> None:
        cell = np.asarray(cell, dtype=float)
        self.sg: SpaceGroup = find_spacegroup(cell, frac_positions,
                                              species_of_atom, symprec=symprec)
        self.grid_ok = _grid_compatible(self.sg, fft_shape)
        s_all = _cart_rotations(self.sg, cell)
        # Per-op 6×6 Voigt representations of ε → S ε Sᵀ (engineering strain) and
        # σ → S σ Sᵀ (stress, no engineering factor). Built numerically from the
        # tensor transform so the two conventions can never drift.
        self.m_strain = np.array([self._voigt_strain_rep(s) for s in s_all])
        self.m_stress = np.array([self._voigt_stress_rep(s) for s in s_all])
        self.strains, self._spans = self._select()
        # Reduce only when the orbit spans all 6 strains, the FFT box respects the
        # group, AND we actually save work (< 6 strains). Trivial groups select
        # all 6 and gain nothing — treat that as "cannot reduce".
        self.can_reduce = bool(self._spans and self.grid_ok
                               and len(self.strains) < 6)

    @staticmethod
    def _voigt_strain_rep(s: np.ndarray) -> np.ndarray:
        m = np.zeros((6, 6))
        for j in range(6):
            eps = voigt_strain_tensor(j, 1.0)          # engineering unit strain
            m[:, j] = _strain_tensor_to_voigt(s @ eps @ s.T)
        return m

    @staticmethod
    def _voigt_stress_rep(s: np.ndarray) -> np.ndarray:
        m = np.zeros((6, 6))
        for j in range(6):
            e = np.zeros(6)
            e[j] = 1.0
            sig = _stress_voigt_to_tensor(e)
            m[:, j] = stress_to_voigt(s @ sig @ s.T)
        return m

    def _orbit(self, chosen: list[int]) -> np.ndarray:
        """Every Voigt strain 6-vector reachable from ``chosen`` by the group."""
        rows: list[np.ndarray] = []
        for j in chosen:
            e = np.zeros(6)
            e[j] = 1.0
            for m in self.m_strain:
                rows.append(m @ e)
        return np.array(rows).reshape(-1, 6) if rows else np.zeros((0, 6))

    def _select(self) -> tuple[list[int], bool]:
        chosen: list[int] = []
        for j in range(6):
            r = np.linalg.matrix_rank(self._orbit(chosen), tol=_RANK_TOL)
            if r == 6:
                break
            if np.linalg.matrix_rank(self._orbit(chosen + [j]),
                                     tol=_RANK_TOL) > r:
                chosen.append(j)
        spans = np.linalg.matrix_rank(self._orbit(chosen), tol=_RANK_TOL) == 6
        return chosen, bool(spans)

    def reconstruct(self, cols: list[np.ndarray]) -> np.ndarray:
        """Full 6×6 Voigt stiffness (eV/Å³ units) from the irreducible columns.

        ``cols[i]`` is the measured stress-response 6-vector C[:, strains[i]]. Each
        column is spread over the group — the strain direction rotates by
        ``m_strain`` and its stress response by ``m_stress`` — and the whole orbit
        enters one least-squares solve C = Σ' pinv(E'), where E' stacks the
        rotated strain directions and Σ' the rotated stress responses. Linear in
        the columns (mirrors ``SupercellDisplacementSymmetry.reconstruct``)."""
        if len(cols) != len(self.strains):
            raise ValueError("cols must match self.strains")
        e_cols: list[np.ndarray] = []
        s_cols: list[np.ndarray] = []
        for j, c in zip(self.strains, cols, strict=True):
            e = np.zeros(6)
            e[j] = 1.0
            c = np.asarray(c, dtype=float).reshape(6)
            for me, ms in zip(self.m_strain, self.m_stress, strict=True):
                e_cols.append(me @ e)
                s_cols.append(ms @ c)
        e_mat = np.array(e_cols).T                 # (6, m) rotated strains
        s_mat = np.array(s_cols).T                 # (6, m) rotated stresses
        return s_mat @ np.linalg.pinv(e_mat)       # C @ E = Σ  →  C = Σ pinv(E)


def _grid_compatible(sg: SpaceGroup,
                     fft_shape: tuple[int, ...] | None) -> bool:
    """True when the FFT box respects the full point group: every set of
    lattice/FFT axes the rotations couple (``symmetry.coupled_axis_groups``) has a
    single common dimension. ``None`` means "grid unknown" → assume compatible
    (the reduction is still gated on the symmetry span). For cubic all three axes
    couple, so it demands n1=n2=n3; a slab or anisotropic box fails and the
    caller keeps the full strain set."""
    if fft_shape is None:
        return True
    for group in coupled_axis_groups(sg):
        if len({int(fft_shape[a]) for a in group}) != 1:
            return False
    return True


@dataclass(frozen=True)
class Moduli:
    """Voigt–Reuss–Hill polycrystalline averages [GPa] (Poisson dimensionless)."""

    bulk_voigt: float
    bulk_reuss: float
    bulk_hill: float
    shear_voigt: float
    shear_reuss: float
    shear_hill: float
    young: float          # Hill Young's modulus
    poisson: float        # Hill Poisson ratio


def moduli_from_cij(c: np.ndarray) -> Moduli:
    """Voigt–Reuss–Hill averages from a 6×6 stiffness [GPa].

    Voigt (uniform strain) and Reuss (uniform stress, via the compliance
    S = C⁻¹) bounds, Hill = their mean; Young's modulus and Poisson ratio from
    the Hill K and G."""
    c = np.asarray(c, dtype=float)
    tr3 = c[0, 0] + c[1, 1] + c[2, 2]
    off3 = c[0, 1] + c[0, 2] + c[1, 2]
    sh3 = c[3, 3] + c[4, 4] + c[5, 5]
    kv = (tr3 + 2.0 * off3) / 9.0
    gv = (tr3 - off3 + 3.0 * sh3) / 15.0

    s = np.linalg.inv(c)
    str3 = s[0, 0] + s[1, 1] + s[2, 2]
    sof3 = s[0, 1] + s[0, 2] + s[1, 2]
    ssh3 = s[3, 3] + s[4, 4] + s[5, 5]
    kr = 1.0 / (str3 + 2.0 * sof3)
    gr = 15.0 / (4.0 * str3 - 4.0 * sof3 + 3.0 * ssh3)

    kh, gh = 0.5 * (kv + kr), 0.5 * (gv + gr)
    young = 9.0 * kh * gh / (3.0 * kh + gh)
    poisson = (3.0 * kh - 2.0 * gh) / (2.0 * (3.0 * kh + gh))
    return Moduli(bulk_voigt=kv, bulk_reuss=kr, bulk_hill=kh,
                  shear_voigt=gv, shear_reuss=gr, shear_hill=gh,
                  young=young, poisson=poisson)


def compliance_tensor(c: np.ndarray) -> np.ndarray:
    """Full 4-index compliance S_ijkl (3,3,3,3) from a 6×6 stiffness [any unit].

    Invert C to the Voigt compliance S = C⁻¹, then unfold with the compliance
    factor convention. Engineering shear strains carry a factor 2, so the
    off-diagonal Voigt compliances absorb it: S_ijkl = s[m,n]·f with f = 1 for
    two normal Voigt indices, 1/2 when exactly one index is a shear (3,4,5), and
    1/4 when both are. The unfolded tensor is symmetric under i↔j, k↔l and
    pair↔pair so all equivalent (i,j,k,l) slots get the same value."""
    s = np.linalg.inv(np.asarray(c, dtype=float))
    st = np.zeros((3, 3, 3, 3))
    fac = lambda m: 1.0 if m < 3 else 0.5  # noqa: E731
    for m, (i, j) in enumerate(_VOIGT):
        for n, (k, l) in enumerate(_VOIGT):
            val = s[m, n] * fac(m) * fac(n)
            for a, b in {(i, j), (j, i)}:
                for p, q in {(k, l), (l, k)}:
                    st[a, b, p, q] = val
    return st


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def directional_poisson(c: np.ndarray, n: np.ndarray, m: np.ndarray) -> float:
    """Poisson ratio for uniaxial stress along ``n``, transverse strain along ``m``.

    Pull along unit vector n and read the strain along a perpendicular unit
    vector m. ν(n,m) = −(S_ijkl n_i n_j m_k m_l)/(S_pqrs n_p n_q n_r n_s). A
    negative value is auxetic (the solid widens transverse to a pull). n and m
    are normalized here and should be orthogonal."""
    st = compliance_tensor(c)
    return _poisson_from_tensor(st, _unit(n), _unit(m))


def _poisson_from_tensor(st: np.ndarray, n: np.ndarray, m: np.ndarray) -> float:
    num = np.einsum("ijkl,i,j,k,l->", st, n, n, m, m)
    den = np.einsum("ijkl,i,j,k,l->", st, n, n, n, n)
    return float(-num / den)


def _fibonacci_sphere(n_dir: int) -> np.ndarray:
    """``n_dir`` roughly-equidistant unit vectors on the sphere (N,3)."""
    i = np.arange(n_dir) + 0.5
    z = 1.0 - 2.0 * i / n_dir
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i
    return np.column_stack((r * np.cos(theta), r * np.sin(theta), z))


def min_directional_poisson(
    c: np.ndarray, n_dir: int = 2000, n_trans: int = 180
) -> tuple[float, np.ndarray, np.ndarray]:
    """Minimum directional Poisson ratio over loading/transverse directions.

    Scan ``n_dir`` loading directions on the unit sphere and, for each, ``n_trans``
    transverse directions in the plane ⟂ n. The global minimum below zero is the
    auxetic indicator. Returns ``(nu_min, n_hat, m_hat)`` with the achieving unit
    vectors."""
    st = compliance_tensor(c)
    ndirs = _fibonacci_sphere(n_dir)  # (N,3)

    # per-loading orthonormal basis of the transverse plane
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (n_dir, 1))
    near_x = np.abs(ndirs[:, 0]) > 0.9
    ref[near_x] = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ndirs, ref)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(ndirs, e1)

    phi = np.linspace(0.0, np.pi, n_trans, endpoint=False)
    # M[a,p,:] = cos φ_p e1[a] + sin φ_p e2[a]
    mdirs = np.cos(phi)[None, :, None] * e1[:, None, :] \
        + np.sin(phi)[None, :, None] * e2[:, None, :]

    den = np.einsum("ijkl,ai,aj,ak,al->a", st, ndirs, ndirs, ndirs, ndirs)
    num = np.einsum("ijkl,ai,aj,apk,apl->ap", st, ndirs, ndirs, mdirs, mdirs)
    nu = -num / den[:, None]

    a, p = np.unravel_index(np.argmin(nu), nu.shape)
    return float(nu[a, p]), ndirs[a].copy(), mdirs[a, p].copy()


def is_mechanically_stable(c: np.ndarray) -> bool:
    """Born stability: the 6×6 stiffness is positive-definite (all eigenvalues
    > 0). A symmetric, weakly-negative eigenvalue from FD noise is treated as
    unstable — tighten the SCF/strain if a stable crystal trips this."""
    return bool(np.all(np.linalg.eigvalsh(np.asarray(c, dtype=float)) > 0.0))
