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
  F_i is evaluated by a DIRECT spherical Bessel contraction with
  ``pseudo.radial_torch.jl_t`` — plain tensor math, so it is traceable by
  both reverse- and forward-mode autograd (``pseudo.radial_torch.sbt_t`` is a
  custom Function without a jvp, so forward-mode cannot pass through it; at
  dense-H scale, npw·nmesh is tiny, so tracing the kernel is free). This
  matches the SCF's splined tables to ~1e-9 (the spline is fitted to the
  same transform).

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

Caveat: H(k) is non-smooth where k+G = 0 for a sphere G (i.e. k on a
reciprocal-lattice point, e.g. Γ for a Γ-centred sphere): |k+G| has a cusp
there. Values are exact (l ≥ 1 channels vanish at q = 0) but the ∂/∂k
contribution of that single row's l ≥ 1 projectors is masked to zero, like
``core.ylm.ylm_all``'s zero-row guard. Evaluate derivatives at generic k.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor

from gradwave.constants import HBAR2_2M, MINUS_I_POW
from gradwave.core.fftbox import r_to_g
from gradwave.core.ylm import ylm_all
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_gsphere, reciprocal_cell
from gradwave.pseudo.radial_torch import jl_t, simpson_weights
from gradwave.pseudo.upf import UPFData

if TYPE_CHECKING:  # leaf module: no runtime scf import (import-linter layering)
    from gradwave.scf.loop import SCFResult

HFun = Callable[[Tensor], Tensor]
"""A dense Bloch Hamiltonian: Cartesian k (3,) real → (n, n) Hermitian complex."""

# |k+G|² below this [Å⁻²] counts as the k+G = 0 cusp row (see module docstring).
_Q2_ZERO = 1e-24


@dataclass(frozen=True)
class _BetaChannel:
    """One radial KB channel: l and the frozen quadrature for F(q) = Σ_r gw·j_l(qr)."""

    l: int
    r: Tensor  # (nmesh,) radial mesh [bohr-scaled as parsed]
    gw: Tensor  # (nmesh,) (r·β)(r)·r · simpson weights — contract against j_l(qr)


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
            channels.append(_BetaChannel(l=beta.l, r=r, gw=gw))
        chan_of.append(idxs)
    return tuple(channels), chan_of


def _toeplitz_index(miller: Tensor, shape: tuple[int, int, int]) -> Tensor:
    """(npw, npw) flat FFT-box index of G_i − G_j (the dense local-term gather)."""
    n1, n2, n3 = shape
    nbox = torch.tensor((n1, n2, n3), device=miller.device)
    diff = (miller[:, None, :] - miller[None, :, :]) % nbox
    return diff[..., 0] * (n2 * n3) + diff[..., 1] * n3 + diff[..., 2]


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
        dij_full = (
            torch.block_diag(*blocks)
            if blocks
            else torch.zeros((0, 0), dtype=RDTYPE)
        )

        return cls(
            b=torch.as_tensor(reciprocal_cell(grid.cell), dtype=RDTYPE),
            k_ref_cart=sphere.k_cart.to(RDTYPE),
            miller=sphere.miller,
            g_cart=sphere.kpg.to(RDTYPE) - sphere.k_cart.to(RDTYPE),
            vmat=vmat,
            tau=system.positions.detach().cpu().to(RDTYPE),
            dij_full=dij_full,
            col_atom=torch.tensor(col_atom, dtype=torch.int64),
            col_chan=torch.tensor(col_chan, dtype=torch.int64),
            col_lm=torch.tensor(col_lm, dtype=torch.int64),
            channels=channels,
            lmax=max((c.l for c in channels), default=0),
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

    def projectors(self, k_cart: Tensor) -> Tensor:
        """KB projectors p (nproj_tot, npw) at continuous k, differentiable in k."""
        kpg = self.g_cart + k_cart  # (npw, 3)
        q2 = (kpg * kpg).sum(-1)
        cusp = q2 < _Q2_ZERO
        q = torch.sqrt(torch.where(cusp, torch.ones_like(q2), q2))
        q = torch.where(cusp, torch.zeros_like(q), q)

        fvals = [ch.gw @ jl_t(ch.l, q[None, :] * ch.r[:, None]) for ch in self.channels]
        y = ylm_all(self.lmax, kpg)  # (npw, (lmax+1)²)
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


def qgt_sos(
    h_fn: HFun, k_cart: Tensor, bands: Sequence[int] | Tensor
) -> Tensor:
    """QGT by first-order perturbation theory: autograd only through ∂H/∂k.

        Q_μν = Σ_{n∈g} Σ_{m∉g} ⟨u_n|∂_μH|u_m⟩⟨u_m|∂_νH|u_n⟩ / (ε_n−ε_m)²

    Exact on the dense basis (complete sum over states). Never differentiates
    the eigendecomposition, so internal degeneracies of the band group are
    fine; only a nonzero gap between the group and its complement is needed.
    """
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
    idx = torch.as_tensor(bands, dtype=torch.int64, device=u.device)
    out_mask = torch.ones(w.shape[0], dtype=torch.bool, device=u.device)
    out_mask[idx] = False
    ug, uo = u[:, idx], u[:, out_mask]
    denom = (w[idx][None, :] - w[out_mask][:, None]) ** 2  # (nout, ng)
    a = [uo.mH @ dhm @ ug for dhm in dh]  # ⟨u_m|∂H|u_n⟩, (nout, ng)
    q = torch.empty((3, 3), dtype=u.dtype, device=u.device)
    for mu in range(3):
        for nu in range(3):
            q[mu, nu] = (a[mu].conj() * a[nu] / denom).sum()
    return q


def metric_curvature(q: Tensor) -> tuple[Tensor, Tensor]:
    """(g_μν, Ω_μν) [Å²] from a QGT: Fubini–Study metric Re Q and Berry
    curvature −2 Im Q."""
    return q.real, -2.0 * q.imag
