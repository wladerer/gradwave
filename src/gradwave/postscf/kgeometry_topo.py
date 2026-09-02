"""Quantized band topology from link overlaps: FHS Chern numbers, Wilson
loops / Wannier charge centers (milestone 2, scalar-relativistic).

Everything here is built from ONE primitive: the band-group link overlap
M_mn(k1, k2) = ⟨u_m(k1)|u_n(k2)⟩. Two providers implement it:

- :class:`ModelLinkStates` — an explicit dense H(k) callable (tight-binding /
  toy models), periodic in the fractional k it is given, so wraparound links
  are exact by construction.
- :class:`BlochLinkStates` — plane-wave states from a converged SCF via
  ``kgeometry.BlochHK``, one G-sphere per (folded) k. A k outside [0,1)³ is
  folded, k = k_fold + n with integer n, and the coefficients are re-labelled
  by the Miller shift: c_k(μ) = c_{k_fold}(m) with μ = m − n. Overlaps then
  contract over ABSOLUTE Miller labels (intersection of the two spheres),
  which imposes the periodic gauge ψ_{k+B} = ψ_k exactly — the BZ-boundary
  embedding is an integer re-index, zero tolerance (tested).

On top of the overlap:

- :func:`chern_fhs` — Fukui–Hatsugai–Suzuki plaquette flux on a 2D slice of
  the BZ: link variables U = det M / |det M|, plaquette flux
  F = arg(U₁U₂U₃⁻¹U₄⁻¹), C = ΣF/2π. Because every link phase cancels between
  its two adjacent plaquettes, ΣF is 2π × integer to machine precision on
  ANY mesh (the residual only detects |F| ≥ π undersampling).
- :func:`wilson_loop` / :func:`wcc_flow` — path-ordered products of
  (SVD-unitarized) link matrices along a closed k-line; eigenphases/2π are
  the Wannier charge centers x̄_n(k_perp), and the winding of arg det W
  across one k_perp period recovers the Chern number (cross-check vs FHS;
  sign convention verified against FHS on the QWZ model).
- :func:`wcc_gap` — the largest circular gap in the Wannier spectrum of one
  loop, kept DIFFERENTIABLE in the loop origin (the "distance to a
  topological transition" observable): raw (non-unitarized) overlaps, so no
  SVD backward; ``torch.linalg.eigvals`` + angle/sort/max are all on the
  graph. Valid where the Wilson eigenvalues are simple.
- :func:`weyl_chirality` — total Berry flux/2π of a band group through a
  small cube around a k-point (open FHS patches over the 6 faces, outward
  oriented; shared-edge links cancel so the sum is again exactly quantized).

Chern sign convention: ``chern_fhs`` counts flux positive for the (e1, e2)
orientation, C = (1/2π)∮ F with F = arg(U_{e1}U_{e2}U_{e1}⁻¹U_{e2}⁻¹) on
plaquettes ordered e1-then-e2. ``wcc_flow(e_loop=e1, e_perp=e2)`` reproduces
the same integer with the same sign.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
from torch import Tensor

from gradwave.dtypes import RDTYPE
from gradwave.postscf.kgeometry import BlochHK, HFun

if TYPE_CHECKING:  # typing only — this module stays off the scf runtime graph
    from gradwave.scf.loop import SCFResult

KVec = "np.ndarray | Sequence[float] | Tensor"

# Miller indices are packed into one int64 key: |m_i| < _MKEY/2 always holds
# for any realistic sphere (grids.build_gsphere raises far earlier).
_MKEY = 1024

# a normalized link |det M| below this means the band group is not isolated
# across the link (or the mesh is far too coarse) — the phase is meaningless.
_DET_TOL = 1e-3


class LinkStates(Protocol):
    """Provider of band-group link overlaps M_mn(k1,k2) = ⟨u_m(k1)|u_n(k2)⟩."""

    def overlap(self, k1: object, k2: object) -> Tensor: ...

    def graph_scope(self) -> AbstractContextManager[None]:
        """Context manager: cache autograd-carrying states for one closed-loop
        computation, so a loop's endpoint reuses the START's eigh node and the
        eigenvector gauge cancels exactly on the graph (torch's eigh backward
        rejects gauge-dependent losses otherwise). States cached inside the
        scope are dropped on exit — they hold a stale graph after backward."""
        ...


def _as_np(k: object) -> np.ndarray:
    if isinstance(k, Tensor):
        return k.detach().cpu().numpy().astype(float)
    return np.asarray(k, dtype=float)


def _fold(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """k = k_fold + n with k_fold ∈ [0,1) and integer n."""
    n = np.floor(k)
    return k - n, n.astype(np.int64)


def _on_graph(k: object) -> bool:
    return isinstance(k, Tensor) and (k.requires_grad or k.grad_fn is not None)


class ModelLinkStates:
    """Link overlaps for an explicit dense H(k) (k in fractional/torus coords).

    ``periodic=True`` folds k componentwise into [0,1) before evaluating —
    H(k) must be exactly periodic under k → k+1 per component (Bloch models
    in cell-periodic convention). ``periodic=False`` evaluates H at k as
    given (open k-space patches, e.g. around a Weyl node). States are cached
    per rounded k; a k carrying autograd history bypasses the cache and
    keeps the graph.
    """

    def __init__(self, h_fn: HFun, bands: Sequence[int], *, periodic: bool = True) -> None:
        self.h_fn = h_fn
        self.bands = list(bands)
        self.periodic = periodic
        self._cache: dict[tuple[float, ...], Tensor] = {}
        self._gcache: dict[tuple[float, ...], Tensor] | None = None

    @contextmanager
    def graph_scope(self) -> Iterator[None]:
        self._gcache = {}
        try:
            yield
        finally:
            self._gcache = None

    def _states(self, k: object) -> Tensor:
        if _on_graph(k):
            kt = k if isinstance(k, Tensor) else torch.as_tensor(k, dtype=RDTYPE)
            if self.periodic:
                kt = kt - torch.floor(kt.detach())
            key = tuple(np.round(kt.detach().cpu().numpy().astype(float), 9))
            if self._gcache is not None and key in self._gcache:
                return self._gcache[key]
            u = torch.linalg.eigh(self.h_fn(kt)).eigenvectors[:, self.bands]
            if self._gcache is not None:
                self._gcache[key] = u
            return u
        kf = _as_np(k)
        if self.periodic:
            kf, _ = _fold(kf)
        key = tuple(np.round(kf, 9))
        if key not in self._cache:
            with torch.no_grad():
                u = torch.linalg.eigh(self.h_fn(torch.as_tensor(kf, dtype=RDTYPE))).eigenvectors
            self._cache[key] = u[:, self.bands]
        return self._cache[key]

    def overlap(self, k1: object, k2: object) -> Tensor:
        return self._states(k1).mH @ self._states(k2)


def _pack_miller(m: np.ndarray) -> np.ndarray:
    """(n,3) int Miller triples → unique int64 keys."""
    if np.abs(m).max(initial=0) >= _MKEY // 2:
        raise ValueError("Miller index exceeds packing range")
    return (m[:, 0] + _MKEY // 2) * _MKEY * _MKEY + (m[:, 1] + _MKEY // 2) * _MKEY + (
        m[:, 2] + _MKEY // 2
    )


class BlochLinkStates:
    """Link overlaps for plane-wave states of a converged (NC) SCF.

    One ``BlochHK`` (own G-sphere) per folded k, cached; eigenvectors cached
    per rounded k. Coefficients are labelled by ABSOLUTE Miller index
    μ = m − n (n the fold shift), and overlaps contract over the sphere
    intersection — the periodic-gauge BZ-boundary embedding. A k with
    autograd history reuses the cached (constant) ``BlochHK`` but rebuilds
    the eigenvectors on the graph.
    """

    def __init__(
        self,
        res: SCFResult,
        bands: Sequence[int],
        *,
        ecut: float | None = None,
        spin: int = 0,
    ) -> None:
        self.res = res
        self.bands = list(bands)
        self.ecut = ecut
        self.spin = spin
        self._hk: dict[tuple[float, ...], BlochHK] = {}
        self._states: dict[tuple[float, ...], tuple[np.ndarray, Tensor]] = {}
        self._gcache: dict[tuple[float, ...], Tensor] | None = None

    @contextmanager
    def graph_scope(self) -> Iterator[None]:
        self._gcache = {}
        try:
            yield
        finally:
            self._gcache = None

    def _hk_at(self, kf: np.ndarray) -> BlochHK:
        key = tuple(np.round(kf, 9))
        if key not in self._hk:
            self._hk[key] = BlochHK.from_scf(self.res, kf, ecut=self.ecut, spin=self.spin)
        return self._hk[key]

    def _entry(self, k: object) -> tuple[np.ndarray, Tensor]:
        knp = _as_np(k)
        kf, shift = _fold(knp)
        hk = self._hk_at(kf)
        keys = _pack_miller(hk.miller.numpy() - shift[None, :])
        if _on_graph(k):
            gkey = tuple(np.round(kf, 9))
            if self._gcache is not None and gkey in self._gcache:
                return keys, self._gcache[gkey]
            kt = (k if isinstance(k, Tensor) else torch.as_tensor(k, dtype=RDTYPE)).to(RDTYPE)
            kt_fold = kt - torch.as_tensor(shift, dtype=RDTYPE)
            u = torch.linalg.eigh(hk.h(hk.k_cart(kt_fold))).eigenvectors[:, self.bands]
            if self._gcache is not None:
                self._gcache[gkey] = u
            return keys, u
        skey = tuple(np.round(kf, 9))
        if skey not in self._states:
            with torch.no_grad():
                u = torch.linalg.eigh(hk.h(hk.k_cart(kf))).eigenvectors
            self._states[skey] = (_pack_miller(hk.miller.numpy()), u[:, self.bands])
        _, u = self._states[skey]
        return keys, u

    def overlap(self, k1: object, k2: object) -> Tensor:
        keys1, u1 = self._entry(k1)
        keys2, u2 = self._entry(k2)
        _, i1, i2 = np.intersect1d(keys1, keys2, return_indices=True)
        return u1[torch.as_tensor(i1, dtype=torch.int64)].mH @ u2[
            torch.as_tensor(i2, dtype=torch.int64)
        ]


# --------------------------------------------------------------------------- #
# Fukui–Hatsugai–Suzuki Chern number                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChernResult:
    chern: int
    residual: float  # |ΣF/2π − chern|: floating-error scale unless undersampled
    fluxes: np.ndarray  # (n, n) per-plaquette Berry flux [rad]
    min_link: float  # smallest raw |det M| seen (isolation margin; ≥ _DET_TOL by construction)


def _link_det(provider: LinkStates, k1: np.ndarray, k2: np.ndarray) -> tuple[complex, float]:
    """The unit-modulus link variable ``det M / |det M|`` and the raw magnitude
    ``|det M|`` (an isolation margin — small means the band group is nearly
    degenerate with a neighbour across the link)."""
    d = complex(torch.linalg.det(provider.overlap(k1, k2)).item())
    a = abs(d)
    if a < _DET_TOL:
        raise ValueError(
            f"link |det M| = {a:.2e} < {_DET_TOL}: band group not isolated across "
            "the link (or mesh far too coarse) — its phase is meaningless"
        )
    return d / a, a


def chern_fhs(
    provider: LinkStates,
    n: int = 8,
    *,
    e1: Sequence[float] | np.ndarray,
    e2: Sequence[float] | np.ndarray,
    origin: Sequence[float] | np.ndarray,
) -> ChernResult:
    """FHS Chern number of the band group on the periodic slice
    k(s,t) = origin + s·e1 + t·e2, s,t ∈ [0,1), on an n×n mesh.

    e1/e2 must be full periods (integer fractional vectors for Bloch
    providers). The result is exactly quantized by construction; ``residual``
    ≫ 1e-10 means plaquette fluxes hit ±π (refine n).
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    origin = np.asarray(origin, dtype=float)

    def kat(i: int, j: int) -> np.ndarray:
        return origin + (i / n) * e1 + (j / n) * e2

    with torch.no_grad():
        # links; periodicity of the providers makes index-(n) ≡ index-0 exact,
        # so modulo indexing into the link arrays is legitimate
        ux = np.empty((n, n), dtype=complex)
        uy = np.empty((n, n), dtype=complex)
        min_link = float("inf")
        for i in range(n):
            for j in range(n):
                ux[i, j], ax = _link_det(provider, kat(i, j), kat(i + 1, j))
                uy[i, j], ay = _link_det(provider, kat(i, j), kat(i, j + 1))
                min_link = min(min_link, ax, ay)
        # orientation: conjugate loop, so that C agrees with the Berry-curvature
        # integral (1/2π)∫Ω of kgeometry.qgt (Ω = −2 Im Q) and with the WCC
        # winding of wcc_flow — verified on the QWZ model in the tests
        fluxes = -np.angle(
            ux
            * np.roll(uy, -1, axis=0)
            * np.conj(np.roll(ux, -1, axis=1))
            * np.conj(uy)
        )
    total = float(fluxes.sum()) / (2.0 * np.pi)
    chern = round(total)
    return ChernResult(
        chern=chern, residual=abs(total - chern), fluxes=fluxes, min_link=min_link
    )


# --------------------------------------------------------------------------- #
# Wilson loops / Wannier charge centers                                       #
# --------------------------------------------------------------------------- #


def _unitarize(m: Tensor) -> Tensor:
    u, _, vh = torch.linalg.svd(m)
    return u @ vh


def wilson_loop(
    provider: LinkStates,
    start: object,
    e_loop: Sequence[float] | np.ndarray,
    n: int,
    *,
    unitarize: bool = True,
) -> Tensor:
    """Path-ordered product of link matrices around the closed line
    start → start + e_loop in n steps (the closing link wraps via the
    provider's periodic embedding). ``start`` may carry autograd history;
    pass ``unitarize=False`` on the differentiable path (SVD backward is
    ill-conditioned at degenerate singular values, and the raw product's
    eigenphases converge to the same WCCs).
    """
    el = np.asarray(e_loop, dtype=float)
    if isinstance(start, Tensor):
        pts: list[object] = [
            start + torch.as_tensor((i / n) * el, dtype=RDTYPE) for i in range(n + 1)
        ]
    else:
        s0 = np.asarray(start, dtype=float)
        pts = [s0 + (i / n) * el for i in range(n + 1)]
    scope = provider.graph_scope() if _on_graph(start) else nullcontext()
    with scope:
        w: Tensor | None = None
        for i in range(n):
            m = provider.overlap(pts[i], pts[i + 1])
            if unitarize:
                m = _unitarize(m)
            w = m if w is None else w @ m
    assert w is not None
    return w


def wilson_wcc(w: Tensor) -> Tensor:
    """Wannier charge centers x̄_n ∈ [0,1) (sorted) from a Wilson loop."""
    lam = torch.linalg.eigvals(w)
    return torch.sort(torch.remainder(torch.angle(lam) / (2.0 * np.pi), 1.0)).values


@dataclass(frozen=True)
class WCCFlow:
    k_perp: np.ndarray  # (n_perp,) fractional positions along e_perp
    wcc: np.ndarray  # (n_perp, nb) Wannier charge centers ∈ [0,1)
    chern: int  # winding of arg det W across the k_perp cycle
    residual: float  # |winding/2π − chern|


def wcc_flow(
    provider: LinkStates,
    *,
    e_loop: Sequence[float] | np.ndarray,
    e_perp: Sequence[float] | np.ndarray,
    origin: Sequence[float] | np.ndarray,
    n_loop: int = 8,
    n_perp: int = 8,
) -> WCCFlow:
    """WCC spectrum x̄_n(k_perp) across one period of e_perp, plus the Chern
    number read off as the winding of arg det W(k_perp) (same sign convention
    as :func:`chern_fhs` with (e1, e2) = (e_loop, e_perp))."""
    origin = np.asarray(origin, dtype=float)
    e_perp = np.asarray(e_perp, dtype=float)
    with torch.no_grad():
        wccs, thetas = [], []
        for j in range(n_perp):
            w = wilson_loop(provider, origin + (j / n_perp) * e_perp, e_loop, n_loop)
            wccs.append(wilson_wcc(w).numpy())
            thetas.append(float(torch.angle(torch.linalg.det(w)).item()))
    th = np.asarray(thetas)
    dth = np.diff(np.append(th, th[0]))  # closed cycle
    dth = (dth + np.pi) % (2.0 * np.pi) - np.pi  # principal branch per step
    total = float(dth.sum()) / (2.0 * np.pi)
    chern = round(total)
    return WCCFlow(
        k_perp=np.arange(n_perp) / n_perp,
        wcc=np.asarray(wccs),
        chern=chern,
        residual=abs(total - chern),
    )


def wcc_gap(
    provider: LinkStates,
    start: Tensor,
    e_loop: Sequence[float] | np.ndarray,
    n_loop: int = 8,
) -> Tensor:
    """Largest circular gap in the Wannier spectrum of the loop at ``start``
    — differentiable in ``start`` (the distance-to-transition observable).

    Uses raw (non-unitarized) link products; valid where the Wilson
    eigenvalues are simple (torch.linalg.eigvals backward needs that).
    """
    w = wilson_loop(provider, start, e_loop, n_loop, unitarize=False)
    lam = torch.linalg.eigvals(w)
    x = torch.sort(torch.remainder(torch.angle(lam) / (2.0 * np.pi), 1.0)).values
    if x.shape[0] == 1:
        return torch.ones_like(x[0])  # one Wannier center: the gap is the full circle
    gaps = torch.cat([x[1:] - x[:-1], (x[0] + 1.0 - x[-1]).reshape(1)])
    return gaps.max()


# --------------------------------------------------------------------------- #
# ℤ₂ invariant (Soluyanov–Vanderbilt WCC crossing parity)                     #
# --------------------------------------------------------------------------- #


def _largest_gap_center(x: np.ndarray) -> float:
    """Midpoint of the largest circular gap of a WCC set on [0,1)."""
    xs = np.sort(x)
    gaps = np.diff(np.append(xs, xs[0] + 1.0))
    i = int(np.argmax(gaps))
    return float((xs[i] + gaps[i] / 2.0) % 1.0)


@dataclass(frozen=True)
class Z2Result:
    z2: int  # crossings mod 2
    crossings: int  # WCC crossings of the largest-gap line over the half cycle
    k_perp: np.ndarray  # (n_perp+1,) fractional e_perp positions, 0 … ½
    wcc: np.ndarray  # (n_perp+1, nb)
    gap_center: np.ndarray  # (n_perp+1,) largest-gap midpoints


def z2_invariant(
    provider: LinkStates,
    *,
    e_loop: Sequence[float] | np.ndarray,
    e_perp: Sequence[float] | np.ndarray,
    origin: Sequence[float] | np.ndarray,
    n_loop: int = 8,
    n_perp: int = 8,
) -> Z2Result:
    """Soluyanov–Vanderbilt ℤ₂: WCC flow over HALF the e_perp cycle (k_perp
    from 0 to ½, both TRIM lines included) and the parity of the number of
    WCC crossings of the largest-gap line between consecutive loops.

    The band group must be a Kramers-closed (even) set of occupied bands of a
    time-reversal-symmetric H, and ``origin`` must put k_perp = 0 and ½ on
    TRIM lines of the slice. The crossing parity is arc-choice independent
    exactly because nb is even.
    """
    origin = np.asarray(origin, dtype=float)
    e_perp = np.asarray(e_perp, dtype=float)
    with torch.no_grad():
        wccs = []
        for j in range(n_perp + 1):
            w = wilson_loop(
                provider, origin + (j / (2.0 * n_perp)) * e_perp, e_loop, n_loop
            )
            wccs.append(wilson_wcc(w).numpy())
    wcc = np.asarray(wccs)
    gaps = np.array([_largest_gap_center(x) for x in wcc])
    crossings = 0
    for j in range(n_perp):
        arc = (gaps[j + 1] - gaps[j]) % 1.0
        rel = (wcc[j + 1] - gaps[j]) % 1.0
        crossings += int(np.sum((rel > 0.0) & (rel < arc)))
    return Z2Result(
        z2=crossings % 2,
        crossings=crossings,
        k_perp=np.arange(n_perp + 1) / (2.0 * n_perp),
        wcc=wcc,
        gap_center=gaps,
    )


# --------------------------------------------------------------------------- #
# mirror sectors (mirror Chern)                                               #
# --------------------------------------------------------------------------- #


def mirror_sector_split(u: Tensor, m_op: Tensor, tol: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Split a band-group basis u (dim, nb) into (+i, −i) mirror sectors.

    Requires [H, M] = 0 on the group (mirror-invariant k): S = u†Mu is then
    unitary with eigenvalues ±i (spinful mirror, M² = −1), so −iS is
    Hermitian with eigenvalues ±1. Returns (u₊, u₋); raises if any eigenvalue
    strays from ±1 by > tol (the group is not mirror-closed)."""
    s = u.mH @ m_op.to(u.dtype) @ u
    a = -1j * s
    a = (a + a.mH) / 2.0
    vals, w = torch.linalg.eigh(a)
    dev = (vals.abs() - 1.0).abs().max().item()
    if dev > tol:
        raise ValueError(
            f"band group is not mirror-closed: sector eigenvalues deviate from ±1 by {dev:.2e}"
        )
    return u @ w[:, vals > 0], u @ w[:, vals < 0]


class MirrorSectorStates:
    """Model link states projected onto one mirror sector (±i).

    For mirror Chern on a mirror-invariant slice of a k-periodic model whose
    mirror representation ``m_op`` (a constant matrix with M² = −1) commutes
    with H(k) on that slice: C_m = (C₊ − C₋)/2 with C_± = :func:`chern_fhs`
    of the ``sector = +1`` / ``−1`` providers. (The plane-wave analogue is
    ``kgeometry_soc.SpinorBlochLinkStates`` with ``mirror``/``sector`` set.)
    """

    def __init__(
        self,
        h_fn: HFun,
        m_op: Tensor,
        bands: Sequence[int],
        sector: int,
        *,
        periodic: bool = True,
    ) -> None:
        self._base = ModelLinkStates(h_fn, bands, periodic=periodic)
        self.m_op = m_op
        self.sector = sector

    def graph_scope(self) -> AbstractContextManager[None]:
        return self._base.graph_scope()

    def _states(self, k: object) -> Tensor:
        u = self._base._states(k)
        plus, minus = mirror_sector_split(u, self.m_op)
        return plus if self.sector > 0 else minus

    def overlap(self, k1: object, k2: object) -> Tensor:
        return self._states(k1).mH @ self._states(k2)


# --------------------------------------------------------------------------- #
# Weyl-node chirality (open patches over a closed box)                        #
# --------------------------------------------------------------------------- #


def _patch_flux(
    provider: LinkStates, origin: np.ndarray, ea: np.ndarray, eb: np.ndarray, n: int
) -> float:
    """Σ plaquette flux over the OPEN patch origin + s·ea + t·eb, s,t ∈ [0,1]."""
    ux = np.empty((n, n + 1), dtype=complex)
    uy = np.empty((n + 1, n), dtype=complex)

    def kat(i: int, j: int) -> np.ndarray:
        return origin + (i / n) * ea + (j / n) * eb

    with torch.no_grad():
        for i in range(n):
            for j in range(n + 1):
                ux[i, j], _ = _link_det(provider, kat(i, j), kat(i + 1, j))
        for i in range(n + 1):
            for j in range(n):
                uy[i, j], _ = _link_det(provider, kat(i, j), kat(i, j + 1))
    # same conjugated orientation as chern_fhs (see the comment there)
    f = -np.angle(ux[:, :-1] * uy[1:, :] * np.conj(ux[:, 1:]) * np.conj(uy[:-1, :]))
    return float(f.sum())


def weyl_chirality(
    h_fn: HFun,
    bands: Sequence[int],
    center: Sequence[float] | np.ndarray,
    half_width: float,
    n: int = 6,
) -> tuple[int, float]:
    """Chirality (enclosed Berry charge) of a band group: total flux/2π
    through the surface of the cube ``center ± half_width`` in model k-space.

    Shared-edge link phases cancel between adjacent faces, so the sum is
    exactly quantized (same telescoping as FHS); returns (integer, residual).
    """
    c = np.asarray(center, dtype=float)
    provider = ModelLinkStates(h_fn, bands, periodic=False)
    e = np.eye(3) * (2.0 * half_width)
    total = 0.0
    for ax in range(3):
        eb, ec = e[(ax + 1) % 3], e[(ax + 2) % 3]
        lo = c - half_width * np.ones(3)
        hi = lo + e[ax]
        # outward orientation: (eb, ec) on the + face, swapped on the − face
        total += _patch_flux(provider, hi, eb, ec, n)
        total += _patch_flux(provider, lo, ec, eb, n)
    q = total / (2.0 * np.pi)
    return round(q), abs(q - round(q))
