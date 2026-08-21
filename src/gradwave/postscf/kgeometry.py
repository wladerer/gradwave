"""k-space band geometry at a fixed converged potential (autograd ∂/∂k).

Milestone 1 of the QGT/GIPAW track: a dense Bloch Hamiltonian H(k) for the
norm-conserving formalism that is a differentiable torch function of the
CONTINUOUS Cartesian k, built once from a converged SCF (frozen v_eff(r) and
KB projectors), plus the gauge-invariant quantum geometric tensor of a band
group by forward-mode autograd through ``torch.linalg.eigh``.

Construction (all conventions inherited from the normative modules):

- kinetic:   T(k)_GG = HBAR2_2M |k+G|²  (diagonal; constants.HBAR2_2M)
- local:     V_GG' = V̂_eff(G−G') — the exact dense form of the FFT apply,
  via the same Miller difference-index (Toeplitz) logic as
  ``core.hamiltonian.HamiltonianK._toeplitz_M``; constant in k.
- nonlocal:  p_i(k+G) = (4π/√Ω)(−i)^l Y_lm(k+G^) F_i(|k+G|) e^{−i(k+G)·τ},
  H_NL = Σ_ij D_ij |p_i⟩⟨p_j| (core/hamiltonian.py phase conventions).
  F_i is evaluated by a DIRECT spherical Bessel contraction in the smooth
  form [F_l(q)/q^l]·S_lm(k+G): scaled kernel ``pseudo.radial_torch.
  jl_scaled_x2_t`` (a function of q², even/analytic at 0) × solid harmonics
  (``ylm_all(..., solid=True)``), plain tensor math traceable by both
  reverse- and forward-mode autograd (``pseudo.radial_torch.sbt_t`` is a
  custom Function without a jvp, so forward-mode cannot pass through it; at
  dense-H scale, npw·nmesh is tiny, so tracing the kernel is free). This
  matches the SCF's splined tables to ~1e-9 (the spline is fitted to the
  same transform), and — because nothing takes a √ or normalizes a direction
  — H(k) and the projectors are smooth in k straight through k+G = 0 (the
  earlier cusp-row masking is gone; the velocity apply's FD cross-check at Γ
  pinned it).

The G-sphere is FIXED at a reference k (basis of ``BlochHK.from_scf``'s
``k_frac``), so H(k) is a smooth function of k on a fixed basis — the
variational quality of that basis degrades away from the reference k, so for
mesh work build one ``BlochHK`` per mesh point and evaluate at its own k.

Quantum geometric tensor, projector form (gauge invariant, degeneracy-safe
WITHIN the chosen band group):

    Q_μν(k) = Tr[∂_μP (1−P) ∂_νP],   P = Σ_{n∈group} |u_nk⟩⟨u_nk|
    g_μν = Re Q_μν  (Fubini–Study metric, Å²),  Ω_μν = −2 Im Q_μν  (Berry
    curvature, Å²).

``qgt`` differentiates P(k) by forward-mode AD through ``eigh`` (three dual
passes). torch's eigh tangent divides by eigenvalue differences, so it needs
every eigenvalue COUPLING TO the group to be simple — evaluate at generic k,
or where only states outside the group are mutually degenerate. ``qgt_sos``
assembles the same tensor from first-order perturbation theory using only
∂H/∂k (autograd through the H build, no eigh differentiation); it tolerates
internal degeneracies of the group and needs only a gap between the group
and its complement — use it at high-symmetry k.

(The spinor builder in kgeometry_soc still uses the normalized-Ylm form with
the k+G = 0 row guard — its derivatives at exactly Γ retain the M1 cusp
caveat until it adopts the solid-harmonic assembly.)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor

from gradwave.constants import HBAR2_2M, MINUS_I_POW
from gradwave.core.fftbox import r_to_g
from gradwave.core.ylm import ylm_all
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import GSphere, build_gsphere, reciprocal_cell
from gradwave.pseudo.radial_torch import jl_scaled_x2_t, simpson_weights
from gradwave.pseudo.upf import UPFData

if TYPE_CHECKING:  # leaf module: no runtime scf import (import-linter layering)
    from gradwave.scf.loop import SCFResult, System

HFun = Callable[[Tensor], Tensor]
"""A dense Bloch Hamiltonian: Cartesian k (3,) real → (n, n) Hermitian complex."""

# |k+G|² below this [Å⁻²] counts as the k+G = 0 cusp row (see module docstring).
_Q2_ZERO = 1e-24


@dataclass(frozen=True)
class _BetaChannel:
    """One radial KB channel: l and the frozen quadrature for F(q) = Σ_r gw·j_l(qr).

    ``gw_solid`` = gw·r^l pairs with the scaled kernel j_l(qr)/(qr)^l:
    F_l(q)/q^l = Σ_r gw_solid · jl_scaled_x2(q²r²) — the smooth-at-q=0 form
    the projector build uses (with solid harmonics carrying the q^l)."""

    l: int
    r: Tensor  # (nmesh,) radial mesh [bohr-scaled as parsed]
    gw: Tensor  # (nmesh,) (r·β)(r)·r · simpson weights — contract against j_l(qr)
    gw_solid: Tensor  # (nmesh,) gw · r^l — contract against j_l(qr)/(qr)^l
    r2: Tensor  # (nmesh,) r² (the scaled kernel's argument is q²·r²)


def _beta_channels(
    upfs: Sequence[UPFData],
) -> tuple[tuple[_BetaChannel, ...], list[list[int]]]:
    """Unique (species, radial channel) quadratures + the species→channel map.

    Shared by the scalar builder below and the spinor one (kgeometry_soc)."""
    channels: list[_BetaChannel] = []
    chan_of: list[list[int]] = []
    for upf in upfs:
        idxs = []
        for beta in upf.betas:
            nr = beta.cutoff_idx
            r = torch.as_tensor(upf.r[:nr], dtype=RDTYPE)
            w = torch.as_tensor(simpson_weights(upf.rab[:nr]), dtype=RDTYPE)
            gw = torch.as_tensor(beta.rbeta, dtype=RDTYPE) * r * w
            idxs.append(len(channels))
            channels.append(
                _BetaChannel(l=beta.l, r=r, gw=gw, gw_solid=gw * r**beta.l, r2=r * r)
            )
        chan_of.append(idxs)
    return tuple(channels), chan_of


def _toeplitz_index(miller: Tensor, shape: tuple[int, int, int]) -> Tensor:
    """(npw, npw) flat FFT-box index of G_i − G_j (the dense local-term gather)."""
    n1, n2, n3 = shape
    nbox = torch.tensor((n1, n2, n3), device=miller.device)
    diff = (miller[:, None, :] - miller[None, :, :]) % nbox
    return diff[..., 0] * (n2 * n3) + diff[..., 1] * n3 + diff[..., 2]


@dataclass(frozen=True)
class KBProjectors:
    """Traceable KB projector assembly on one fixed G-sphere.

    ``p(k_cart)`` rebuilds the full projector matrix at a continuous k —
    radial F via the ``jl_t`` contraction, Ylm, and position phases all on
    the autograd graph (forward or reverse mode). ``p_and_dp`` adds the
    analytic k-derivatives by forward-mode dual passes: the matrix-free
    velocity apply and the dense H(k) share exactly this object.
    """

    g_cart: Tensor  # (npw, 3) Cartesian G of the sphere [Å⁻¹]
    tau: Tensor  # (na, 3) atom positions [Å]
    dij_full: Tensor  # (nproj_tot, nproj_tot) m-expanded D matrix [eV]
    col_atom: Tensor  # (nproj_tot,) atom of each projector column
    col_chan: Tensor  # (nproj_tot,) radial-channel index into `channels`
    col_lm: Tensor  # (nproj_tot,) index into ylm_all's (l,m) ordering
    channels: tuple[_BetaChannel, ...]  # unique (species, radial channel) pairs
    lmax: int
    volume: float

    @property
    def npw(self) -> int:
        return int(self.g_cart.shape[0])

    @classmethod
    def for_sphere(
        cls, system: System, sphere: GSphere, template: KBProjectors | None = None
    ) -> KBProjectors:
        """Build for one ``grids.GSphere`` of an NC System.

        Everything except ``g_cart`` is k-independent (the radial quadratures,
        the m-expanded D matrix, the column maps), so a per-k loop should build
        one instance and pass it back as ``template`` for the rest — the
        rebuild of ``_beta_channels`` + the dij blocks per k was a measured
        constructor hotspot of the NMR pipeline's ``VelocityApply`` /
        ``_NLPairVelocity``."""
        if template is not None:
            return replace(
                template,
                g_cart=sphere.kpg.to(RDTYPE) - sphere.k_cart.to(RDTYPE),
            )
        channels, chan_of = _beta_channels(system.upfs)
        col_atom, col_chan, col_lm = [], [], []
        blocks: list[Tensor] = []
        for a, s in enumerate(system.species_of_atom):
            upf = system.upfs[s]
            ls = [beta.l for beta in upf.betas]
            offs = np.cumsum([0, *(2 * l + 1 for l in ls)])
            for i, l in enumerate(ls):
                for mc in range(2 * l + 1):
                    col_atom.append(a)
                    col_chan.append(chan_of[s][i])
                    col_lm.append(l * l + mc)  # ylm_all is dense in l²..(l+1)²−1
            block = torch.zeros((offs[-1], offs[-1]), dtype=RDTYPE)
            dij = torch.as_tensor(upf.dij, dtype=RDTYPE)
            for i, li in enumerate(ls):
                for j, lj in enumerate(ls):
                    if li != lj:
                        continue
                    for mc in range(2 * li + 1):
                        block[offs[i] + mc, offs[j] + mc] = dij[i, j]
            blocks.append(block)
        grid = system.grid
        return cls(
            g_cart=sphere.kpg.to(RDTYPE) - sphere.k_cart.to(RDTYPE),
            tau=system.positions.detach().cpu().to(RDTYPE),
            dij_full=(
                torch.block_diag(*blocks)
                if blocks
                else torch.zeros((0, 0), dtype=RDTYPE)
            ),
            col_atom=torch.tensor(col_atom, dtype=torch.int64),
            col_chan=torch.tensor(col_chan, dtype=torch.int64),
            col_lm=torch.tensor(col_lm, dtype=torch.int64),
            channels=channels,
            lmax=max((c.l for c in channels), default=0),
            volume=grid.volume,
        )

    def p(self, k_cart: Tensor) -> Tensor:
        """KB projectors p (nproj_tot, npw) at continuous k, differentiable in k.

        Assembled in the manifestly smooth form [F_l(q)/q^l]·S_lm(k+G) with
        S_lm the solid harmonic and the scaled Bessel kernel a function of
        q² — no √ or normalization anywhere, so the columns (and their
        autograd derivatives, forward or reverse) are exact straight through
        k+G = 0. Values coincide with the F_l·Y_lm form for q > 0."""
        kpg = self.g_cart + k_cart  # (npw, 3)
        q2 = (kpg * kpg).sum(-1)

        fvals = [
            ch.gw_solid @ jl_scaled_x2_t(ch.l, q2[None, :] * ch.r2[:, None])
            for ch in self.channels
        ]  # F_l(q)/q^l, smooth in q²
        y = ylm_all(self.lmax, kpg, solid=True)  # (npw, (lmax+1)²): q^l·Y_lm
        phase_arg = kpg @ self.tau.T  # (npw, na)
        phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))

        cols = []
        for c in range(int(self.col_atom.shape[0])):
            chan = self.channels[int(self.col_chan[c])]
            pref = (4.0 * math.pi / math.sqrt(self.volume)) * MINUS_I_POW[chan.l]
            radial_angular = fvals[int(self.col_chan[c])] * y[:, int(self.col_lm[c])]
            cols.append(pref * radial_angular.to(CDTYPE) * phases[:, int(self.col_atom[c])])
        if not cols:
            return torch.zeros((0, self.npw), dtype=CDTYPE, device=kpg.device)
        return torch.stack(cols, dim=0)

    def p_and_dp(self, k_cart: Tensor) -> tuple[Tensor, list[Tensor]]:
        """(p, [∂p/∂k_x, ∂p/∂k_y, ∂p/∂k_z]) at k — the derivatives by
        forward-mode dual passes through the traceable build. Composes with
        reverse-mode graphs on the frozen tensors (positions enter the
        phases), so downstream ∂/∂params stays reachable."""
        k = k_cart.to(RDTYPE)
        p0: Tensor | None = None
        dp = []
        for mu in range(3):
            tangent = torch.zeros(3, dtype=RDTYPE, device=k.device)
            tangent[mu] = 1.0
            with fwAD.dual_level():
                pd = self.p(fwAD.make_dual(k, tangent))
                prim, tang = fwAD.unpack_dual(pd)
            p0 = prim
            dp.append(tang if tang is not None else torch.zeros_like(prim))
        assert p0 is not None
        return p0, dp


@dataclass(frozen=True)
class BlochHK:
    """Dense differentiable H(k) at fixed converged potential (NC formalism).

    Build with :meth:`from_scf`; call :meth:`h` with a Cartesian k [Å⁻¹]
    (any tensor on the autograd graph — forward or reverse mode). The
    G-sphere (basis) is frozen at the reference k of the build.
    """

    b: Tensor  # (3,3) reciprocal-cell rows [Å⁻¹]
    k_ref_cart: Tensor  # (3,) the reference (sphere-defining) k [Å⁻¹]
    miller: Tensor  # (npw, 3) int64 Miller indices of the fixed sphere
    g_cart: Tensor  # (npw, 3) Cartesian G of the fixed sphere [Å⁻¹]
    vmat: Tensor  # (npw, npw) complex V̂_eff(G−G') [eV]
    tau: Tensor  # (na, 3) atom positions [Å]
    dij_full: Tensor  # (nproj_tot, nproj_tot) m-expanded D matrix [eV]
    col_atom: Tensor  # (nproj_tot,) atom of each projector column
    col_chan: Tensor  # (nproj_tot,) radial-channel index into `channels`
    col_lm: Tensor  # (nproj_tot,) index into ylm_all's (l,m) ordering
    channels: tuple[_BetaChannel, ...]  # unique (species, radial channel) pairs
    lmax: int
    volume: float

    @property
    def npw(self) -> int:
        return int(self.g_cart.shape[0])

    @classmethod
    def from_scf(
        cls,
        res: SCFResult,
        k_frac: Sequence[float] | np.ndarray,
        *,
        spin: int = 0,
        ecut: float | None = None,
    ) -> BlochHK:
        """Freeze the converged potential/projectors of ``res`` into an H(k).

        ``k_frac`` fixes the G-sphere (fractional, in ``res``'s cell);
        ``spin`` selects the v_eff channel when nspin = 2; ``ecut`` [eV]
        defaults to the SCF's own cutoff.
        """
        system = res.system
        grid = system.grid
        sphere = build_gsphere(grid, ecut or system.ecut, np.asarray(k_frac, dtype=float))

        # Frozen-potential analysis on CPU: at dense-H scale (npw ~ 10²–10³)
        # everything is tiny, and the SCF tensors may live on CUDA.
        v = res.v_eff if res.v_eff.dim() == 3 else res.v_eff[spin]
        vhat = r_to_g(v.detach().cpu().to(CDTYPE)).reshape(-1)
        vmat = vhat[_toeplitz_index(sphere.miller, grid.shape)]

        kb = KBProjectors.for_sphere(system, sphere)
        return cls(
            b=torch.as_tensor(reciprocal_cell(grid.cell), dtype=RDTYPE),
            k_ref_cart=sphere.k_cart.to(RDTYPE),
            miller=sphere.miller,
            g_cart=kb.g_cart,
            vmat=vmat,
            tau=kb.tau,
            dij_full=kb.dij_full,
            col_atom=kb.col_atom,
            col_chan=kb.col_chan,
            col_lm=kb.col_lm,
            channels=kb.channels,
            lmax=kb.lmax,
            volume=kb.volume,
        )

    def k_cart(self, k_frac: Sequence[float] | np.ndarray | Tensor) -> Tensor:
        """Cartesian k [Å⁻¹] from fractional coordinates (stays on the graph)."""
        kf = (
            k_frac.to(RDTYPE)
            if isinstance(k_frac, Tensor)
            else torch.as_tensor(np.asarray(k_frac, dtype=float), dtype=RDTYPE)
        )
        return kf @ self.b

    def projectors(self, k_cart: Tensor) -> Tensor:
        """KB projectors p (nproj_tot, npw) at continuous k, differentiable in k."""
        return KBProjectors(
            g_cart=self.g_cart, tau=self.tau, dij_full=self.dij_full,
            col_atom=self.col_atom, col_chan=self.col_chan, col_lm=self.col_lm,
            channels=self.channels, lmax=self.lmax, volume=self.volume,
        ).p(k_cart)

    def h(self, k_cart: Tensor) -> Tensor:
        """Dense Hermitian H(k) (npw, npw) [eV], differentiable in Cartesian k."""
        kpg = self.g_cart + k_cart
        kin = HBAR2_2M * (kpg * kpg).sum(-1)
        hmat = self.vmat + torch.diag_embed(kin.to(CDTYPE))
        if self.dij_full.shape[0]:
            p = self.projectors(k_cart)
            # H_NL[G,G'] = Σ_ij p_i(G) D_ij p_j(G')*  (D real symmetric → Hermitian)
            hmat = hmat + p.mT @ (self.dij_full.to(CDTYPE) @ p.conj())
        return hmat


def band_projector(h_fn: HFun, k_cart: Tensor, bands: Sequence[int] | Tensor) -> Tensor:
    """P(k) = Σ_{n∈bands} |u_n⟩⟨u_n| from a dense eigh — gauge invariant."""
    _, u = torch.linalg.eigh(h_fn(k_cart))
    ug = u[:, torch.as_tensor(bands, dtype=torch.int64, device=u.device)]
    return ug @ ug.mH


def qgt(h_fn: HFun, k_cart: Tensor, bands: Sequence[int] | Tensor) -> Tensor:
    """Quantum geometric tensor Q_μν = Tr[∂_μP (1−P) ∂_νP] of a band group.

    Forward-mode autograd through ``torch.linalg.eigh`` (one dual pass per
    Cartesian direction). Requires every eigenvalue coupling to the group to
    be simple at ``k_cart`` (see module docstring; use :func:`qgt_sos` at
    degenerate points). Returns (3, 3) complex: g = Re Q, Ω = −2 Im Q [Å²].
    """
    k = k_cart.detach().to(RDTYPE)
    dp = []
    p0: Tensor | None = None
    for mu in range(3):
        tangent = torch.zeros(3, dtype=RDTYPE, device=k.device)
        tangent[mu] = 1.0
        with fwAD.dual_level():
            pd = band_projector(h_fn, fwAD.make_dual(k, tangent), bands)
            prim, tang = fwAD.unpack_dual(pd)
        if tang is None:  # h_fn ignored k — projector is constant along μ
            tang = torch.zeros_like(prim)
        p0 = prim
        dp.append(tang)
    assert p0 is not None
    comp = torch.eye(p0.shape[0], dtype=p0.dtype, device=p0.device) - p0
    return torch.stack(
        [
            torch.stack([torch.einsum("ab,bc,ca->", dp[mu], comp, dp[nu]) for nu in range(3)])
            for mu in range(3)
        ]
    )


def _eigh_and_dh(h_fn: HFun, k_cart: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
    """(ε, U, [∂H/∂k_x, ∂H/∂k_y, ∂H/∂k_z]) at k — eigh under no_grad, the
    velocity matrices by one forward-mode dual pass per direction (never
    differentiates the eigendecomposition)."""
    k = k_cart.detach().to(RDTYPE)
    with torch.no_grad():
        w, u = torch.linalg.eigh(h_fn(k))
    dh = []
    for mu in range(3):
        tangent = torch.zeros(3, dtype=RDTYPE, device=k.device)
        tangent[mu] = 1.0
        with fwAD.dual_level():
            hd = h_fn(fwAD.make_dual(k, tangent))
            _, tang = fwAD.unpack_dual(hd)
        dh.append(tang if tang is not None else torch.zeros_like(u))
    return w, u, dh


def _crossgap_a(
    w: Tensor, u: Tensor, dh: list[Tensor], bands: Sequence[int] | Tensor
) -> tuple[list[Tensor], Tensor, Tensor]:
    """Cross-gap tangents A_μ[m,n] = ⟨u_m|∂_μH|u_n⟩/(ε_n−ε_m) = ⟨u_m|∂̃_μu_n⟩
    (m outside the group, n inside), plus (ε_out, ε_in)."""
    idx = torch.as_tensor(bands, dtype=torch.int64, device=u.device)
    out_mask = torch.ones(w.shape[0], dtype=torch.bool, device=u.device)
    out_mask[idx] = False
    ug, uo = u[:, idx], u[:, out_mask]
    denom = w[idx][None, :] - w[out_mask][:, None]  # (nout, ng): ε_n − ε_m
    a = [(uo.mH @ dhm @ ug) / denom.to(u.dtype) for dhm in dh]
    return a, w[out_mask], w[idx]


def qgt_sos(
    h_fn: HFun, k_cart: Tensor, bands: Sequence[int] | Tensor
) -> Tensor:
    """QGT by first-order perturbation theory: autograd only through ∂H/∂k.

        Q_μν = Σ_{n∈g} Σ_{m∉g} ⟨u_n|∂_μH|u_m⟩⟨u_m|∂_νH|u_n⟩ / (ε_n−ε_m)²

    Exact on the dense basis (complete sum over states). Never differentiates
    the eigendecomposition, so internal degeneracies of the band group are
    fine; only a nonzero gap between the group and its complement is needed.
    """
    w, u, dh = _eigh_and_dh(h_fn, k_cart)
    a, _, _ = _crossgap_a(w, u, dh, bands)
    q = torch.empty((3, 3), dtype=u.dtype, device=u.device)
    for mu in range(3):
        for nu in range(3):
            q[mu, nu] = (a[mu].conj() * a[nu]).sum()
    return q


def metric_curvature(q: Tensor) -> tuple[Tensor, Tensor]:
    """(g_μν, Ω_μν) [Å²] from a QGT: Fubini–Study metric Re Q and Berry
    curvature −2 Im Q."""
    return q.real, -2.0 * q.imag


def _ctvr_tensor_dense(
    h_fn: HFun, k_cart: Tensor, occ: Sequence[int] | Tensor, mu_chem: float
) -> Tensor:
    """The full (3, 3) complex CTVR tensor t_μν(k) = Σ_{n∈occ,m∉occ}
    A_μ[m,n]*(ε_m + ε_n − 2μ)A_ν[m,n] — the dense-eigh twin of the
    Sternheimer route's per-k tensor (route-equivalence tests compare these;
    :func:`orbmag_tensor` antisymmetrizes the imaginary part)."""
    w, u, dh = _eigh_and_dh(h_fn, k_cart)
    a, e_out, e_in = _crossgap_a(w, u, dh, occ)
    weight = (e_out[:, None] + e_in[None, :] - 2.0 * mu_chem).to(u.dtype)
    t = torch.empty((3, 3), dtype=u.dtype, device=u.device)
    for mu in range(3):
        for nu in range(3):
            t[mu, nu] = (a[mu].conj() * weight * a[nu]).sum()
    return t


def orbmag_tensor(
    h_fn: HFun, k_cart: Tensor, occ: Sequence[int] | Tensor, mu_chem: float
) -> Tensor:
    """Local orbital-magnetization integrand I_γ(k) of the modern theory
    (Ceresoli–Thonhauser–Vanderbilt–Resta), gauge-invariant covariant form:

        I_γ(k) = ε_γμν Im Σ_{n∈occ} ⟨∂̃_μu_n| (H_k + ε_n − 2 μ_chem) |∂̃_νu_n⟩
               = ε_γμν Im Σ_{n∈occ, m∉occ} A_μ[m,n]* (ε_m + ε_n − 2 μ_chem) A_ν[m,n]

    with ∂̃ = (1−P_occ)∂ the covariant derivative and A the cross-gap
    tangents of :func:`qgt_sos` — degeneracy-safe (only cross-gap
    denominators), gauge invariant under any mixing of the occupied states,
    and exactly linear in μ_chem with slope dI_γ/dμ_chem = 2 Ω_γ(k) (our
    Berry-curvature convention Ω = −2 Im Q; μ_chem drops entirely for an
    insulator once the BZ integral of Ω vanishes, i.e. Chern-trivial planes).

    Units: [H] · [k]⁻² — eV·Å² for a Bloch h_fn (k in Å⁻¹); model units for
    toys. Convert to a magnetic moment with :func:`orbital_magnetization`.
    """
    t = _ctvr_tensor_dense(h_fn, k_cart, occ, mu_chem)
    return torch.stack(
        [
            (t[1, 2] - t[2, 1]).imag,
            (t[2, 0] - t[0, 2]).imag,
            (t[0, 1] - t[1, 0]).imag,
        ]
    )


def orbital_magnetization(
    res: SCFResult,
    occ: Sequence[int],
    mu_chem: float,
    *,
    mesh: tuple[int, int, int] = (2, 2, 2),
    shift: tuple[float, float, float] = (0.11, 0.23, 0.37),
    spin: int = 0,
    ecut: float | None = None,
) -> Tensor:
    """Modern-theory orbital magnetization of a frozen-potential NC system:
    the magnetic moment per cell (3,) in Bohr magnetons μ_B.

    Unit chain (the classic silent-bug site, so spelled out): the k-space
    formula gives the moment density M_γ = (e/2ħ) Im ∫ d³k/(2π)³ Σ_n^occ
    ⟨∂̃u|×(H+ε−2μ)|∂̃u⟩. Per cell, with ∫d³k → ((2π)³/V_cell)·⟨mesh avg⟩,
    the V_cell cancels: m_γ = (e/2ħ)·avg_k I_γ with I_γ in eV·Å²
    (:func:`orbmag_tensor`; H in eV, ∂/∂k in Å). In Bohr magnetons
    μ_B = eħ/2mₑ:  m/μ_B = (e·I/2ħ)/(eħ/2mₑ) = I·mₑ/ħ² = I / (2·HBAR2_2M),
    since constants.HBAR2_2M = ħ²/2mₑ in eV·Å² — no new unit constants.

    Sign convention — CALIBRATED: m = +avg(I)/(2·HBAR2_2M) is the physical
    ELECTRON moment. Anchored thermodynamically (tests/unit/
    test_orbmag_sign.py) against Peierls-flux magnetic supercells of a
    Chern-trivial TR-broken lattice model with honest electron phases
    (q = −e): the measured dE/dφ equals −avg(I)/2 in radian-k units to
    ~1e-5 relative after Richardson extrapolation, so M = −∂E/∂B = +(e/2ħ)·
    avg(I). Consistent with this module's Ω = −2 Im Q throughout
    (dm/dμ_chem ∝ Chern is exact).

    A shifted (generic) mesh avoids band degeneracies at symmetry points;
    each mesh point gets its own G-sphere at its own k. ``occ`` is the
    occupied band-index group; ``mu_chem`` [eV] any energy in the gap.
    """
    n1, n2, n3 = mesh
    acc = torch.zeros(3, dtype=RDTYPE)
    for i in range(n1):
        for j in range(n2):
            for l3 in range(n3):
                kf = (np.asarray([i, j, l3], dtype=float) + np.asarray(shift)) / (
                    n1,
                    n2,
                    n3,
                ) - 0.5
                hk = BlochHK.from_scf(res, kf, spin=spin, ecut=ecut)
                acc = acc + orbmag_tensor(hk.h, hk.k_cart(kf), occ, mu_chem)
    return acc / (n1 * n2 * n3) / (2.0 * HBAR2_2M)


# --------------------------------------------------------------------------- #
# Sternheimer route: orbital magnetization without dense eigh                 #
# --------------------------------------------------------------------------- #


class VelocityApply:
    """Matrix-free velocity v_μ|ψ⟩ = (∂H/∂k_μ)|ψ⟩ on the SCF's batched mesh.

    The local potential is k-independent, so v_μ = analytic kinetic diagonal
    2·HBAR2_2M·(k+G)_μ plus the KB nonlocal k-derivative
    Σ_ij D_ij (|∂_μp_i⟩⟨p_j| + |p_i⟩⟨∂_μp_j|), with (p, ∂p) built ONCE per k
    by :meth:`KBProjectors.p_and_dp` (analytic forward-mode derivative of the
    traceable jl_t/Ylm/phase assembly — no finite differences) and padded to
    the (nk, nproj, npw_max) batch layout. The p/∂p tensors keep any
    reverse-mode graph of the frozen inputs (positions), so a later
    ∂M/∂params can differentiate through the apply.
    """

    def __init__(self, system: System) -> None:
        bk = system.batch
        assert bk is not None, "System.batch required (setup_system builds it)"
        self.bk = bk
        nk, m = bk.nk, bk.npw_max
        nproj = int(bk.dij_full.shape[0])
        self.p = torch.zeros(nk, nproj, m, dtype=CDTYPE)
        self.dp = [torch.zeros(nk, nproj, m, dtype=CDTYPE) for _ in range(3)]
        kb: KBProjectors | None = None
        for ik, sph in enumerate(system.spheres):
            kb = KBProjectors.for_sphere(system, sph, template=kb)
            p_k, dp_k = kb.p_and_dp(sph.k_cart.to(RDTYPE))
            self.p[ik, :, : kb.npw] = p_k
            for mu in range(3):
                self.dp[mu][ik, :, : kb.npw] = dp_k[mu]
        self.dij = bk.dij_full.to(CDTYPE)

    def apply(self, c: Tensor, mu: int, k_index: Tensor | None = None) -> Tensor:
        """v_μ c for a batched block c (nk, nb, npw_max).

        ``k_index`` selects/reorders the mesh k of each batch row (the finite-q
        machinery applies v at the k+q partner points); None means identity."""
        kpg, mask = self.bk.kpg, self.bk.mask
        p, dp = self.p, self.dp[mu]
        if k_index is not None:
            kpg, mask, p, dp = kpg[k_index], mask[k_index], p[k_index], dp[k_index]
        out = (2.0 * HBAR2_2M) * kpg[:, None, :, mu] * c
        if p.shape[1]:
            b_p = torch.einsum("kpg,kbg->kbp", p.conj(), c)
            b_dp = torch.einsum("kpg,kbg->kbp", dp.conj(), c)
            out = out + torch.einsum("kbp,pq,kqg->kbg", b_p, self.dij, dp)
            out = out + torch.einsum("kbp,pq,kqg->kbg", b_dp, self.dij, p)
        return out * mask[:, None, :]


def _sternheimer_ctvr(
    res: SCFResult,
    mu_chem: float,
    *,
    cg_tol: float = 1e-9,
    cg_max_iter: int = 400,
) -> tuple[Tensor, Tensor]:
    """Per-k CTVR tensors t_μν(k) (nk, 3, 3) and the k-weights, by
    conduction-projected Sternheimer solves at the SCF's own mesh:

        (H − ε_n + s·P_occ)|∂̃_μu_n⟩ = (1−P_occ) v_μ|u_n⟩
        t_μν(k) = Σ_n ⟨∂̃_μu_n|H|∂̃_νu_n⟩ + (ε_n − 2μ_chem)⟨∂̃_μu_n|∂̃_νu_n⟩

    identical to the dense route's Σ_m A*_μ(ε_m + ε_n − 2μ)A_ν on the same
    basis, but with no dense eigh — cost is 3 CG solves over the occupied
    block. Insulators, nspin = 1, full (symmetry-unreduced) k-mesh.
    """
    from gradwave.core.batch import BatchedHamiltonian, projectors_b
    from gradwave.postscf._response import (
        cg_sternheimer,
        insulator_window,
        pad_coeffs,
        sternheimer_shift,
    )

    system = res.system
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("orbital magnetization: nspin=1 only")
    if system.sym is not None:
        raise NotImplementedError(
            "orbital magnetization needs the full k-mesh: M_orb is a "
            "pseudovector, so IBZ star-unfolding is not implemented — run the "
            "SCF with use_symmetry=False"
        )
    bk = system.batch
    assert bk is not None

    nocc = insulator_window(
        res.occupations, 2.0, "orbital magnetization requires insulating occupations"
    )
    coeffs = res.coeffs
    assert coeffs is not None
    # SCFResult.coeffs is per-k ragged (list[Tensor]) for the NC formalism —
    # the same cast precedent as postscf/dielectric.py's dielectric_born
    c_occ = pad_coeffs(cast("list[Tensor]", coeffs), bk.npw_max)[:, :nocc]
    eps_occ = res.eigenvalues[:, :nocc].to(RDTYPE)
    shift = sternheimer_shift(eps_occ)
    h = BatchedHamiltonian(
        bk, system.grid.shape, res.v_eff, projectors_b(bk, system.positions)
    )

    def p_c(x: Tensor) -> Tensor:
        ov = torch.einsum("kng,kbg->kbn", c_occ.conj(), x)
        return x - torch.einsum("kbn,kng->kbg", ov, c_occ)

    v = VelocityApply(system)
    dpsi = []
    for mu in range(3):
        rhs = p_c(v.apply(c_occ, mu))
        dpsi.append(
            cg_sternheimer(h, bk, c_occ, eps_occ, rhs, torch.zeros_like(rhs),
                           shift, tol=cg_tol, max_iter=cg_max_iter)
        )
    hd = [h.apply(d) for d in dpsi]
    t = torch.zeros(bk.nk, 3, 3, dtype=CDTYPE)
    e_w = (eps_occ - 2.0 * mu_chem).to(CDTYPE)  # (nk, nocc)
    for mu in range(3):
        for nu in range(3):
            band = torch.einsum("kng,kng->kn", dpsi[mu].conj(), hd[nu])
            band = band + e_w * torch.einsum("kng,kng->kn", dpsi[mu].conj(), dpsi[nu])
            t[:, mu, nu] = band.sum(dim=1)
    return t, system.kweights.to(RDTYPE)


def orbital_magnetization_sternheimer(
    res: SCFResult,
    mu_chem: float,
    *,
    cg_tol: float = 1e-9,
    cg_max_iter: int = 400,
) -> Tensor:
    """Modern-theory orbital magnetization, moment per cell (3,) in μ_B —
    the Sternheimer route: no dense eigh, evaluated at the SCF's own k-mesh
    with the matrix-free :class:`VelocityApply` and
    ``postscf._response.cg_sternheimer``. Same integrand, units, and sign
    convention as :func:`orbital_magnetization` (see the unit chain there);
    agreement between the two routes is tested at low ecut. Scales as the
    SCF itself (CG in the sphere basis) instead of O(npw³) eigh.
    """
    t, w = _sternheimer_ctvr(res, mu_chem, cg_tol=cg_tol, cg_max_iter=cg_max_iter)
    t_avg = torch.einsum("k,kab->ab", w.to(CDTYPE), t)
    i_vec = torch.stack(
        [
            (t_avg[1, 2] - t_avg[2, 1]).imag,
            (t_avg[2, 0] - t_avg[0, 2]).imag,
            (t_avg[0, 1] - t_avg[1, 0]).imag,
        ]
    )
    return i_vec / (2.0 * HBAR2_2M)
