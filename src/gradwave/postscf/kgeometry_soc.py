"""Spinor (SOC) dense Bloch Hamiltonian + the mirror operator on plane waves
(milestone 3, fully-relativistic norm-conserving formalism).

``BlochHKSpinor`` doubles ``kgeometry.BlochHK`` to the spinor space: for a
converged ``scf_noncollinear`` run it freezes the effective fields
(V(r), B⃗(r)) rebuilt from (ρ, m⃗) exactly as ``scf.noncollinear.
band_structure_nc`` does (Hartree + XC via ``core.xc.noncollinear.
vxc_and_bxc`` + local PP; LDA/GGA only, no meta-GGA τ operator), and exposes
the dense (2npw × 2npw) H(k) as a differentiable function of continuous
Cartesian k:

- kinetic and the scalar local V(G−G′) act per spin component (⊗ I₂); the
  magnetic part adds the 2×2 blocks V ± B_z and B_x ∓ iB_y (Toeplitz gathers
  of the box FFTs, same difference-index as the scalar builder);
- the SOC nonlocal term uses the j-resolved spinor KB projectors of
  ``core.spinor_proj`` (spin spherical harmonics = Clebsch–Gordan weights ×
  complex Y_lm), with the radial F_{l,j}(|k+G|) evaluated by the same
  differentiable ``jl_t`` contraction as the scalar module, so the whole
  H(k) stays traceable by forward- and reverse-mode autograd. The spinor
  coefficient layout is [↑ (npw), ↓ (npw)], matching the spinor SCF.

``mirror_matrix`` represents a (possibly glide) mirror {W|w} of the space
group on the spinor plane-wave basis at a mirror-invariant k:

    (O ψ)(m′) = e^{−2πi (k+m′)·w} ψ(m),  m′ = W⁻ᵀ m + g₀   (orbital part —
    exactly ``postscf.irreps._rep_matrix``'s convention: a Miller permutation
    × translation phase, with g₀ = W⁻ᵀk − k the integer umklapp),

tensored with the spin-½ rotation −i σ·n̂ (n̂ the Cartesian mirror normal),
so M² = −1 and the eigenvalues in the occupied space are ±i. Mirror-sector
band splitting and the sector Chern numbers (mirror Chern) run through
``kgeometry_topo.MirrorSectorStates`` / ``chern_fhs``.

``SpinorBlochLinkStates`` is the spinor link-overlap provider (ℤ₂ via
``kgeometry_topo.z2_invariant``, sector FHS, Wilson loops): same per-folded-k
sphere caching and absolute-Miller re-indexing as the scalar provider, with
the contraction over both spin blocks.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

from gradwave.constants import HBAR2_2M, MINUS_I_POW
from gradwave.core.energies.hartree import hartree_potential_g
from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.core.spinor_proj import _cg, complex_ylm, so_dij, so_projector_channels
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_gsphere, reciprocal_cell
from gradwave.postscf.kgeometry import (
    _Q2_ZERO,
    _beta_channels,
    _BetaChannel,
    _toeplitz_index,
)
from gradwave.postscf.kgeometry_topo import (
    _as_np,
    _fold,
    _on_graph,
    _pack_miller,
    mirror_sector_split,
)
from gradwave.pseudo.radial_torch import jl_t

if TYPE_CHECKING:
    from gradwave.core.xc.noncollinear import NoncollinearXC
    from gradwave.scf.noncollinear import NCResult

_PAULI = (
    torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE),
    torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE),
    torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE),
)


def nc_potential_fields(res: NCResult, xc: NoncollinearXC) -> tuple[Tensor, Tensor | None]:
    """(V(r), B⃗(r) or None) [eV] frozen from a converged noncollinear SCF —
    the ``band_structure_nc`` reconstruction (Hartree + XC + local PP), on CPU.

    B⃗ is None for a nonmagnetic (m⃗ ≡ 0) run. Meta-GGA functionals are out of
    scope here (their τ operator is not a local potential)."""
    if getattr(xc, "needs_tau", False):
        raise NotImplementedError("meta-GGA τ operator not supported in the dense spinor H")
    system = res.system
    grid = system.grid
    from gradwave.core.xc.noncollinear import vxc_and_bxc

    rho_g_box = r_to_g(res.rho.to(CDTYPE))
    v_h = g_to_r_box(hartree_potential_g(rho_g_box, grid.g2), real=True)
    v_xc, b_xc, _ = vxc_and_bxc(xc, res.rho, res.m, grid, rho_core=system.rho_core)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume)
    v_r = v_h + v_xc + g_to_r_box(vloc_g, real=True)
    nonmagnetic = float(res.m.abs().max()) < 1e-12
    b_vec = None if nonmagnetic else b_xc
    return (
        v_r.detach().cpu(),
        None if b_vec is None else b_vec.detach().cpu(),
    )


@dataclass(frozen=True)
class BlochHKSpinor:
    """Dense differentiable spinor H(k) at a fixed converged (V, B⃗) + SOC
    projectors. Coefficient layout [↑ (npw), ↓ (npw)]; build via
    :meth:`from_nc_scf`, evaluate via :meth:`h` at Cartesian k [Å⁻¹]."""

    b: Tensor  # (3,3) reciprocal-cell rows [Å⁻¹]
    k_ref_cart: Tensor  # (3,)
    miller: Tensor  # (npw, 3) int64
    g_cart: Tensor  # (npw, 3) [Å⁻¹]
    v_uu: Tensor  # (npw, npw) complex: (V + B_z)^(G−G′)
    v_dd: Tensor  # (npw, npw) complex: (V − B_z)^(G−G′)
    v_ud: Tensor | None  # (npw, npw) complex: (B_x − iB_y)^(G−G′); None if B⃗ ≡ 0
    tau: Tensor  # (na, 3) [Å]
    dij_so: Tensor  # (nproj_so, nproj_so) [eV]
    col_meta: tuple[tuple[int, int, int, float, float], ...]  # (atom, chan, l, j, mj)
    channels: tuple[_BetaChannel, ...]
    lmax: int
    volume: float

    @property
    def npw(self) -> int:
        return int(self.g_cart.shape[0])

    @classmethod
    def from_nc_scf(
        cls,
        res: NCResult,
        xc: NoncollinearXC,
        k_frac: Sequence[float] | np.ndarray,
        *,
        ecut: float | None = None,
        fields: tuple[Tensor, Tensor | None] | None = None,
    ) -> BlochHKSpinor:
        """Freeze a converged fully-relativistic ``scf_noncollinear`` run.

        ``fields`` short-circuits the (V, B⃗) reconstruction — pass
        :func:`nc_potential_fields`'s output when building many spheres from
        one result (the link-overlap provider does)."""
        system = res.system
        grid = system.grid
        if not system.is_fr:
            raise ValueError("BlochHKSpinor needs fully-relativistic (j-resolved) pseudos")
        sphere = build_gsphere(grid, ecut or system.ecut, np.asarray(k_frac, dtype=float))
        v_r, b_vec = fields if fields is not None else nc_potential_fields(res, xc)

        idx = _toeplitz_index(sphere.miller, grid.shape)

        def toep(field: Tensor) -> Tensor:
            return r_to_g(field.to(CDTYPE)).reshape(-1)[idx]

        if b_vec is None:
            v_uu = toep(v_r)
            v_dd, v_ud = v_uu, None
        else:
            v_uu = toep(v_r + b_vec[2])
            v_dd = toep(v_r - b_vec[2])
            v_ud = toep(torch.complex(b_vec[0], -b_vec[1]))

        channels, chan_of = _beta_channels(system.upfs)
        raw_meta, lmax = so_projector_channels(system)
        col_meta = tuple((a, chan_of[sp][i], l, j, mj) for a, sp, i, l, j, mj in raw_meta)

        return cls(
            b=torch.as_tensor(reciprocal_cell(grid.cell), dtype=RDTYPE),
            k_ref_cart=sphere.k_cart.to(RDTYPE),
            miller=sphere.miller,
            g_cart=sphere.kpg.to(RDTYPE) - sphere.k_cart.to(RDTYPE),
            v_uu=v_uu,
            v_dd=v_dd,
            v_ud=v_ud,
            tau=system.positions.detach().cpu().to(RDTYPE),
            dij_so=so_dij(system, raw_meta),
            col_meta=col_meta,
            channels=channels,
            lmax=lmax,
            volume=grid.volume,
        )

    def k_cart(self, k_frac: Sequence[float] | np.ndarray | Tensor) -> Tensor:
        """Cartesian k [Å⁻¹] from fractional coordinates (stays on the graph)."""
        kf = (
            k_frac.to(RDTYPE)
            if isinstance(k_frac, Tensor)
            else torch.as_tensor(np.asarray(k_frac, dtype=float), dtype=RDTYPE)
        )
        return kf @ self.b

    def spinor_projectors(self, k_cart: Tensor) -> Tensor:
        """j-resolved KB projectors q (nproj_so, 2·npw), differentiable in k."""
        kpg = self.g_cart + k_cart
        q2 = (kpg * kpg).sum(-1)
        cusp = q2 < _Q2_ZERO
        q = torch.sqrt(torch.where(cusp, torch.ones_like(q2), q2))
        q = torch.where(cusp, torch.zeros_like(q), q)

        fvals = [ch.gw @ jl_t(ch.l, q[None, :] * ch.r[:, None]) for ch in self.channels]
        ylm_c = complex_ylm(self.lmax, kpg)  # (npw, (lmax+1)²) complex
        phase_arg = kpg @ self.tau.T
        phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))

        zero = torch.zeros(self.npw, dtype=CDTYPE, device=kpg.device)
        cols = []
        for a, chan, l, j, mj in self.col_meta:
            c_up, m_up, c_dn, m_dn = _cg(l, j, mj)
            pref = (4.0 * math.pi / math.sqrt(self.volume)) * MINUS_I_POW[l]
            base = pref * fvals[chan].to(CDTYPE) * phases[:, a]
            qu = base * (c_up * ylm_c[:, l * l + l + m_up]) if m_up is not None else zero
            qd = base * (c_dn * ylm_c[:, l * l + l + m_dn]) if m_dn is not None else zero
            cols.append(torch.cat([qu, qd]))
        if not cols:
            return torch.zeros((0, 2 * self.npw), dtype=CDTYPE, device=kpg.device)
        return torch.stack(cols, dim=0)

    def h(self, k_cart: Tensor) -> Tensor:
        """Dense Hermitian spinor H(k) (2npw, 2npw) [eV], differentiable in k."""
        kpg = self.g_cart + k_cart
        kin = torch.diag_embed((HBAR2_2M * (kpg * kpg).sum(-1)).to(CDTYPE))
        huu = self.v_uu + kin
        hdd = self.v_dd + kin
        if self.v_ud is None:
            off = torch.zeros_like(huu)
            hmat = torch.cat(
                [torch.cat([huu, off], dim=1), torch.cat([off, hdd], dim=1)], dim=0
            )
        else:
            hmat = torch.cat(
                [
                    torch.cat([huu, self.v_ud], dim=1),
                    torch.cat([self.v_ud.mH, hdd], dim=1),
                ],
                dim=0,
            )
        if self.dij_so.shape[0]:
            q = self.spinor_projectors(k_cart)
            hmat = hmat + q.mT @ (self.dij_so.to(CDTYPE) @ q.conj())
        return hmat


# --------------------------------------------------------------------------- #
# mirror operator on the spinor plane-wave basis                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MirrorOp:
    """One space-group mirror {W|w}: fractional data + Cartesian normal."""

    w_frac: np.ndarray  # (3,3) int
    t_frac: np.ndarray  # (3,) fractional translation (glides: ≠ 0)
    normal_cart: np.ndarray  # (3,) unit mirror normal


def find_mirror_ops(
    cell: np.ndarray,
    frac_positions: np.ndarray,
    species_of_atom: list[int],
    symprec: float = 1e-5,
) -> list[MirrorOp]:
    """All mirror/glide operations of the space group, with their Cartesian
    plane normals. A mirror is det W = −1 with Cartesian trace +1 (eigenvalues
    +1, +1, −1) — the trace test excludes the inversion, which also has
    det = −1 and W² = 1. Positions are FRACTIONAL
    (``symmetry.find_spacegroup``'s convention)."""
    from gradwave.symmetry import find_spacegroup

    sg = find_spacegroup(np.asarray(cell, float), frac_positions, species_of_atom, symprec)
    a = np.asarray(cell, dtype=float)
    out = []
    for w, t in zip(sg.rotations, sg.translations, strict=True):
        if round(float(np.linalg.det(w))) != -1:
            continue
        r_col = a.T @ w @ np.linalg.inv(a.T)  # Cartesian column action
        if round(float(np.trace(r_col))) != 1:
            continue
        vals, vecs = np.linalg.eigh((r_col + r_col.T) / 2.0)
        normal = vecs[:, int(np.argmin(vals))]  # eigenvalue −1 direction
        out.append(MirrorOp(w_frac=np.asarray(w, np.int64), t_frac=np.asarray(t, float),
                            normal_cart=normal / np.linalg.norm(normal)))
    return out


def mirror_matrix(hk: BlochHKSpinor, op: MirrorOp, k_frac: np.ndarray) -> Tensor:
    """Dense (2npw, 2npw) spinor mirror at a mirror-invariant k (W⁻ᵀk ≡ k mod 1).

    Orbital part per ``postscf.irreps._rep_matrix``: c′(m′) = e^{−2πi(k+m′)·w} c(m)
    with m′ = W⁻ᵀm + g₀; spin part −i σ·n̂ (so M² = −1)."""
    k_frac = np.asarray(k_frac, dtype=float)
    wk = np.round(np.linalg.inv(op.w_frac).T).astype(np.int64)
    g0 = wk @ k_frac - k_frac
    if np.max(np.abs(g0 - np.round(g0))) > 1e-8:
        raise ValueError(f"k {k_frac} is not invariant under this mirror")
    g0 = np.round(g0).astype(np.int64)

    m = hk.miller.numpy()
    mprime = m @ wk.T + g0
    index = {tuple(row): i for i, row in enumerate(m)}
    try:
        perm = np.array([index[tuple(row)] for row in mprime])
    except KeyError as e:  # pragma: no cover — sphere is closed for invariant k
        raise RuntimeError("mirror image of a sphere G left the sphere") from e
    phase = np.exp(-2j * np.pi * ((k_frac + mprime) @ op.t_frac))
    m0 = torch.zeros((hk.npw, hk.npw), dtype=CDTYPE)
    m0[torch.as_tensor(perm), torch.arange(hk.npw)] = torch.as_tensor(phase, dtype=CDTYPE)

    n = op.normal_cart
    spin = -1j * (n[0] * _PAULI[0] + n[1] * _PAULI[1] + n[2] * _PAULI[2])
    return torch.kron(spin, m0)


# --------------------------------------------------------------------------- #
# spinor link-overlap provider (ℤ₂ / sector FHS / Wilson loops)               #
# --------------------------------------------------------------------------- #


class SpinorBlochLinkStates:
    """Spinor link overlaps from a converged FR ``scf_noncollinear`` run.

    Same per-folded-k sphere caching and absolute-Miller re-indexing as
    ``kgeometry_topo.BlochLinkStates``, contracted over both spin blocks.
    With ``mirror``/``sector`` set (+1 → +i, −1 → −i), states are projected
    onto one mirror sector at each k (all ks must then lie in the
    mirror-invariant plane): the sector-resolved provider for mirror Chern.
    """

    def __init__(
        self,
        res: NCResult,
        xc: NoncollinearXC,
        bands: Sequence[int],
        *,
        ecut: float | None = None,
        mirror: MirrorOp | None = None,
        sector: int | None = None,
    ) -> None:
        if (mirror is None) != (sector is None):
            raise ValueError("mirror and sector must be passed together")
        self.res = res
        self.xc = xc
        self.bands = list(bands)
        self.ecut = ecut
        self.mirror = mirror
        self.sector = sector
        self._fields = nc_potential_fields(res, xc)
        self._hk: dict[tuple[float, ...], BlochHKSpinor] = {}
        self._mirror_mat: dict[tuple[float, ...], Tensor] = {}
        self._states: dict[tuple[float, ...], Tensor] = {}
        self._gcache: dict[tuple[float, ...], Tensor] | None = None

    @contextmanager
    def graph_scope(self) -> Iterator[None]:
        self._gcache = {}
        try:
            yield
        finally:
            self._gcache = None

    def _hk_at(self, kf: np.ndarray) -> BlochHKSpinor:
        key = tuple(np.round(kf, 9))
        if key not in self._hk:
            self._hk[key] = BlochHKSpinor.from_nc_scf(
                self.res, self.xc, kf, ecut=self.ecut, fields=self._fields
            )
        return self._hk[key]

    def _sector_project(self, hk: BlochHKSpinor, kf: np.ndarray, u: Tensor) -> Tensor:
        assert self.mirror is not None and self.sector is not None
        key = tuple(np.round(kf, 9))
        m_op = self._mirror_mat.get(key)
        if m_op is None:
            m_op = mirror_matrix(hk, self.mirror, kf)
            self._mirror_mat[key] = m_op
        plus, minus = mirror_sector_split(u, m_op)
        return plus if self.sector > 0 else minus

    def _build_states(self, hk: BlochHKSpinor, kf: np.ndarray, kt: Tensor) -> Tensor:
        u = torch.linalg.eigh(hk.h(hk.k_cart(kt))).eigenvectors[:, self.bands]
        if self.mirror is not None:
            u = self._sector_project(hk, kf, u)
        return u

    def _entry(self, k: object) -> tuple[np.ndarray, Tensor]:
        knp = _as_np(k)
        kf, shift = _fold(knp)
        hk = self._hk_at(kf)
        keys = _pack_miller(hk.miller.numpy() - shift[None, :])
        gkey = tuple(np.round(kf, 9))
        if _on_graph(k):
            if self._gcache is not None and gkey in self._gcache:
                return keys, self._gcache[gkey]
            kt = (k if isinstance(k, Tensor) else torch.as_tensor(k, dtype=RDTYPE)).to(RDTYPE)
            u = self._build_states(hk, kf, kt - torch.as_tensor(shift, dtype=RDTYPE))
            if self._gcache is not None:
                self._gcache[gkey] = u
            return keys, u
        if gkey not in self._states:
            with torch.no_grad():
                self._states[gkey] = self._build_states(
                    hk, kf, torch.as_tensor(kf, dtype=RDTYPE)
                )
        return keys, self._states[gkey]

    def overlap(self, k1: object, k2: object) -> Tensor:
        keys1, u1 = self._entry(k1)
        keys2, u2 = self._entry(k2)
        _, i1, i2 = np.intersect1d(keys1, keys2, return_indices=True)
        n1, n2 = len(keys1), len(keys2)
        rows1 = torch.as_tensor(np.concatenate([i1, i1 + n1]), dtype=torch.int64)
        rows2 = torch.as_tensor(np.concatenate([i2, i2 + n2]), dtype=torch.int64)
        return u1[rows1].mH @ u2[rows2]
