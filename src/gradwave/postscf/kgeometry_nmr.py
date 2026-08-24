"""Finite-q velocity perturbations for the magnetic (NMR/GIPAW) response
(milestone 7 of the QGT track).

A uniform magnetic field enters as the q → 0 limit of monochromatic vector
potentials A ∝ ê_μ e^{±iq·r} (Mauri–Pfrommer–Louie). The elementary
perturbation is the symmetrized velocity ½{v_μ, e^{iq·r}} with v the FULL
gauge-covariant velocity ∂H/∂k (kinetic + KB nonlocal commutator — the
matrix-free ``kgeometry.VelocityApply``). In cell-periodic matrix elements
at fixed Miller index the anticommutator is EXACT:

    ⟨k+q, G|½{v_μ, e^{iqr}}|k, G⟩ = ½ (v_μ^{(k)} + v_μ^{(k+q)})[G, G],

and on a Monkhorst-Pack mesh every k+q folds to another mesh point
k_j = k+q − G0 whose sphere IS the k+q sphere relabeled (|k_j + G_j| =
|k+q+G| with G_j = G + G0), so v^{(k+q)} is exactly the mesh velocity at
k_j. The perturbation apply is therefore

    O_{q,μ}|u_nk⟩ = ½ [ T(v_μ^{(k)} u_nk) + v_μ^{(k_j)} (T u_nk) ],

with T the k → k_j sphere transfer through the dense box carrying the
umklapp phase e^{iG0·r} — exactly ``postscf.dfpt_q``'s embedding
(kpq_map / _g0_phase). The first-order responses come from the same
conduction-projected Sternheimer solve as ``dfpt_q._chi0_q_batched``:

    (H_{k+q} − ε_{nk} + s·P_occ^{k+q}) |δu^{(μ,q)}_{nk}⟩ = −P_c^{k+q} O_{q,μ}|u_nk⟩.

``paramagnetic_tensor`` contracts them into the bare (unscreened)
paramagnetic current-current response block

    P_μν(k; q) = Σ_n ⟨O_{q,μ} u_nk| G_{k+q}(ε_nk) P_c |O_{q,ν} u_nk⟩,

the object whose antisymmetrized q-derivative builds the induced orbital
current (milestone 8). Validation is route equivalence: a dense twin
(``paramagnetic_tensor_dense``) evaluates the same P_μν(k; q) from explicit
``BlochHK`` matrices at k and the UNFOLDED k+q (continuous q, absolute-
Miller-matched transfer, sum over dense eigenstates) — agreement at
commensurate q pins the umklapp embedding, the operator symmetrization,
and the solver in one number. At q = 0 the machinery reduces to the M5
velocity Sternheimer solves.

Milestone 9 (the shielding finish line) adds the two remaining physics
pieces on top of the M8 induced current:

- the KB NONLOCAL CURRENT (:class:`_NLPairVelocity`): the [r, V_NL]
  commutator contribution to the current operator, in the exact
  line-averaged pair-velocity form q·Ā = V_NL^{(k+q)} − V_NL^{(k)}, which
  closes the continuity equation q·j(q) = s(q) that the kinetic current
  alone violates. The closure is exact to CG tolerance once the explicit
  (ecut-vanishing) basis-truncation term is accounted
  (:func:`continuity_truncation_term`) — all three tested.
- the q → 0 ANTISYMMETRIC ASSEMBLY (:func:`sigma_shielding`): uniform B as
  the limit of two transverse A-waves (Pickard–Mauri), the ±q branch
  currents antisymmetrized (the (1/2q)[S(q) − S(−q)] linear-in-q
  extraction), Biot–Savart'ed to B_ind at the nuclei, prefactor
  4π·ALPHA_FS²/E2 validated against the analytic Lamb term.

Milestone 10 replaces the finite ±q antisymmetric difference by the ANALYTIC
∂S/∂q at q = 0 (:func:`sigma_shielding_dq` / :class:`ShieldingDq`): per mesh
k a dense fixed-sphere context (eigh, ∂H/∂k by forward-mode AD, the mixed
second derivatives q̂·∇(∂H/∂k) by nested ``torch.func.jvp``), the response
derivative δu' = dδu/ds taken implicitly through the Sternheimer
characterization (never differentiating the eigendecomposition — the
``qgt_sos`` degeneracy-safe pattern), and the Biot–Savart assembled by the
exact product rule at s = 0, whose kernel-derivative terms carry the
diamagnetic (Lamb) content. This removes the O(q²) finite-q error and the
mesh-commensurability constraint entirely; the finite-q route above stays
as the cross-validation reference (Richardson q → 0 extrapolation).

Insulators, nspin = 1, full (symmetry-unreduced) k-mesh, q commensurate
with the mesh for the Sternheimer route (the dense twin and the analytic
∂/∂q route take any q / any full mesh).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor

from gradwave.constants import ALPHA_FS, E2, HBAR2_2M
from gradwave.core.batch import (
    BatchedHamiltonian,
    BatchedK,
    box_to_sphere_b,
    build_batched,
    g_to_r_b,
    projectors_b,
)
from gradwave.core.fftbox import g_to_r, g_to_r_box, r_to_g
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_gsphere, reciprocal_cell
from gradwave.postscf._response import (
    DenseSternheimerSolver,
    cg_sternheimer,
    dense_sternheimer_for,
    insulator_window,
    pad_coeffs,
    sternheimer_shift,
)
from gradwave.postscf.dfpt_q import _g0_phase, _reindex_bk, kpq_map
from gradwave.postscf.gipaw import R_E_ANG, PAWOnSite, ground_state_shielding
from gradwave.postscf.kgeometry import (
    BlochHK,
    KBProjectors,
    OverlapVelocity,
    VelocityApply,
    _eigh_and_dh,
    _overlap_kbprojectors,
    _toeplitz_index,
)
from gradwave.postscf.kgeometry_topo import _pack_miller

if TYPE_CHECKING:
    from gradwave.scf.loop import SCFResult, System
    from gradwave.scf.uspp_setup import USPPSystem
    from gradwave.symmetry import RhoSymmetrizer


@dataclass(frozen=True)
class VelocityQSolves:
    """First-order responses to ½{v_μ, e^{iqr}} for all three μ, on the
    folded k+q spheres, plus the frozen context of the solve."""

    q_frac: np.ndarray  # (3,)
    o_c: Tensor  # (3, nk, nocc, npw_max): O_{q,μ}|u_nk⟩ on the k_j spheres
    dpsi: Tensor  # (3, nk, nocc, npw_max): −G_{k+q} P_c O_{q,μ}|u_nk⟩
    c_occ_k: Tensor  # (nk, nocc, npw_max) occupied u at k
    c_occ_kq: Tensor  # (nk, nocc, npw_max) occupied u at the folded k+q
    eps_k: Tensor  # (nk, nocc)
    jidx: np.ndarray  # (nk,) mesh index of fold(k+q)
    g0: np.ndarray  # (nk, 3) umklapp
    weights: Tensor  # (nk,) k-weights (sum 1)
    bk_kq: BatchedK  # the reindexed k+q batch
    # frozen solve context reused by the downstream current/screening
    # assembly (each was rebuilt per induced_current_q call before):
    ph: Tensor | None = None  # (nk, 1, *shape) e^{iG0·r} stack
    h_kq: BatchedHamiltonian | None = None  # the k+q Hamiltonian of the solve
    dense: DenseSternheimerSolver | None = None  # eigh-prepass drafter (or None)


def _guard(res: SCFResult) -> None:
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("finite-q velocity response: nspin=1 only")
    if res.system.sym is not None:
        raise NotImplementedError(
            "finite-q velocity response needs the full k-mesh: run the SCF "
            "with use_symmetry=False"
        )


def _g0_phase_stack(shape: tuple[int, int, int], g0: np.ndarray,
                    device: torch.device) -> Tensor:
    """(nk, 1, n1, n2, n3) stack of e^{iG0·r} phases. The distinct G0 rows are
    computed once and broadcast (on any commensurate q most rows are zero —
    the previous per-k rebuild was a measured setup hotspot)."""
    uniq, inv = np.unique(g0, axis=0, return_inverse=True)
    phases = torch.stack([_g0_phase(shape, row, device) for row in uniq])
    return phases[torch.as_tensor(inv.reshape(-1), dtype=torch.long)][:, None]


# --------------------------------------------------------------------------- #
# USPP/PAW S-metric response backend (Phase 2): batched H/S on the smooth       #
# valence spheres from the frozen ground state, feeding velocity_perturbation_q #
# --------------------------------------------------------------------------- #


class _USPPBatchedHS:
    """Batched USPP/PAW Hamiltonian H and overlap S on one set of (folded)
    smooth-valence spheres, the batched twin of ``scf.uspp_loop._HkS``.

        H c = t·c + V̂_loc[c] + Σ_ij Dscr_ij |β_i⟩⟨β_j| c
        S c = c   +           Σ_ij q_ij   |β_i⟩⟨β_j| c

    ``p`` are the KB β-projectors on these spheres (the same
    :class:`~gradwave.postscf.kgeometry.OverlapVelocity`/``_overlap_kbprojectors``
    build the ∂S/∂k from, so H, S and their k-derivatives share one β source),
    ``dscr`` the per-spin SCREENED descreened D (``uspp_frozen.screened_dscr``),
    ``q_full`` the m-expanded ∫Q overlap weights. ``.apply`` is H (the
    ``cg_sternheimer`` operator); ``.s_apply`` is S (its ``s_apply`` hook)."""

    def __init__(self, bk: BatchedK, shape: tuple[int, int, int],
                 v_eff_r: Tensor, p: Tensor, dscr: Tensor, q_full: Tensor) -> None:
        self.bk = bk
        self.shape = shape
        self.v_eff_r = v_eff_r.to(CDTYPE)
        self.p = p  # (nk, nproj, npw_max)
        self.dscr = dscr.to(CDTYPE)
        self.q = q_full.to(CDTYPE)
        self.mask = bk.mask

    def apply(self, c: Tensor) -> Tensor:
        out = self.bk.t[:, None, :].to(c.dtype) * c
        psi = g_to_r_b(c, self.bk, self.shape)
        out = out + box_to_sphere_b(psi * self.v_eff_r, self.bk)
        if self.p.shape[1]:
            b = torch.einsum("kpg,kbg->kbp", self.p.conj(), c)
            out = out + torch.einsum("kbp,pq,kqg->kbg", b, self.dscr, self.p)
        return out * self.mask[:, None, :]

    def s_apply(self, c: Tensor) -> Tensor:
        out = c
        if self.p.shape[1]:
            b = torch.einsum("kpg,kbg->kbp", self.p.conj(), c)
            out = out + torch.einsum("kbp,pq,kqg->kbg", b, self.q, self.p)
        return out * self.mask[:, None, :]


class _USPPHVelocity:
    """∂H/∂k_μ for the USPP smooth Hamiltonian: the analytic kinetic diagonal
    2·HBAR2_2M·(k+G)_μ plus the screened KB nonlocal derivative
    Σ_ij Dscr_ij(|∂_μβ_i⟩⟨β_j| + |β_i⟩⟨∂_μβ_j|) — the ``VelocityApply`` structure
    with the SCREENED D and the PAW β. It shares the β/∂β mesh tensors of an
    :class:`~gradwave.postscf.kgeometry.OverlapVelocity` (same
    ``_overlap_kbprojectors`` build), so ∂H/∂k and ∂S/∂k use one projector
    source. ``k_index`` selects/reorders the mesh k of each batch row exactly
    like ``VelocityApply``/``OverlapVelocity``."""

    def __init__(self, system: USPPSystem, dscr: Tensor,
                 ov: OverlapVelocity | None = None) -> None:
        ov = OverlapVelocity(system) if ov is None else ov
        self.p, self.dp = ov.p, ov.dp
        self.dij = dscr.to(CDTYPE)
        nk, m = ov.nk, ov.npw_max
        self.kpg = torch.zeros(nk, m, 3, dtype=RDTYPE)
        self.mask = torch.zeros(nk, m, dtype=torch.bool)
        for ik, sph in enumerate(system.spheres):
            n = int(sph.kpg.shape[0])
            self.kpg[ik, :n] = sph.kpg.to(RDTYPE)
            self.mask[ik, :n] = True

    def apply(self, c: Tensor, mu: int, k_index: Tensor | None = None) -> Tensor:
        p, dp, kpg, mask = self.p, self.dp[mu], self.kpg, self.mask
        if k_index is not None:
            p, dp, kpg, mask = p[k_index], dp[k_index], kpg[k_index], mask[k_index]
        out = (2.0 * HBAR2_2M) * kpg[:, None, :, mu].to(c.dtype) * c
        if p.shape[1]:
            b_p = torch.einsum("kpg,kbg->kbp", p.conj(), c)
            b_dp = torch.einsum("kpg,kbg->kbp", dp.conj(), c)
            out = out + torch.einsum("kbp,pq,kqg->kbg", b_p, self.dij, dp)
            out = out + torch.einsum("kbp,pq,kqg->kbg", b_dp, self.dij, p)
        return out * mask[:, None, :]


@dataclass
class USPPResponseCtx:
    """Frozen-ground-state backend that lets :func:`velocity_perturbation_q`
    run the S-metric magnetic Sternheimer on a USPP/PAW SCF result (which
    carries no ``System.batch``/``v_eff``). Build with
    :func:`build_uspp_response_ctx`; pass as ``velocity_perturbation_q(...,
    uspp=ctx)``.

    Holds the batched smooth-sphere kinematics (``bk``), the frozen local
    potential (``v_eff_r``) and screened D (``dscr``), the ∫Q weights
    (``q_full``), and the shared β source for H/S and their velocities:
    ``ov`` (an :class:`OverlapVelocity`, ≡ ∂S/∂k = ``dsvel``) and ``v_h`` (∂H/∂k).
    """

    system: USPPSystem
    bk: BatchedK
    shape: tuple[int, int, int]
    v_eff_r: Tensor
    dscr: Tensor
    q_full: Tensor
    ov: OverlapVelocity
    v_h: _USPPHVelocity

    def build_hkq(self, jsel: Tensor) -> tuple[_USPPBatchedHS, BatchedK]:
        """(H/S operator, BatchedK) on the folded k+q spheres ``jsel``."""
        bk_kq = self.bk.reindex(jsel)
        p_kq = self.ov.p[jsel]
        h = _USPPBatchedHS(bk_kq, self.shape, self.v_eff_r, p_kq,
                           self.dscr, self.q_full)
        return h, bk_kq


def build_uspp_response_ctx(res: Any, xc: Any) -> USPPResponseCtx:
    """Assemble the :class:`USPPResponseCtx` for a converged USPP/PAW ``res``
    (a ``USPPResult`` or the equivalent dict) at the SCF's own ``xc``.

    Rebuilds the frozen ground-state operator the way ``uspp_implicit`` /
    ``uspp_frozen`` do: ``v_eff = v_H + v_loc + v_xc`` on the dense grid and the
    screened D (bare D + ∫v_eff Q + PAW one-center ddd). nspin = 1 only."""
    from gradwave.postscf.uspp_frozen import frozen_veff, screened_dscr

    system = res["system"]
    if getattr(res, "get", None) is not None and res.get("nspin", 1) != 1:
        raise NotImplementedError("USPP magnetic response: nspin=1 only")
    veff = frozen_veff(res, xc)
    dscr = screened_dscr(res, xc, veff)
    bk = build_batched(system.spheres, system.proj_data)
    ov = OverlapVelocity(system)
    v_h = _USPPHVelocity(system, dscr[0], ov=ov)
    return USPPResponseCtx(
        system=system, bk=bk, shape=tuple(system.grid.shape),
        v_eff_r=veff[0], dscr=dscr[0], q_full=system.q_full, ov=ov, v_h=v_h)


def velocity_perturbation_q(
    res: SCFResult,
    q_frac: np.ndarray | list[float] | tuple[float, float, float],
    *,
    cg_tol: float = 1e-9,
    max_iter: int = 400,
    dense: DenseSternheimerSolver | str | None = "auto",
    v: VelocityApply | None = None,
    s_apply: Callable[[Tensor], Tensor] | None = None,
    dsvel: OverlapVelocity | None = None,
    uspp: USPPResponseCtx | None = None,
    chunk_k: int | None = None,
) -> VelocityQSolves:
    """Solve the k+q Sternheimer equations for the three symmetrized velocity
    perturbations ½{v_μ, e^{iqr}} at every mesh k (see module docstring).

    q must be commensurate with the SCF mesh (kpq_map raises otherwise);
    q = 0 reduces exactly to the M5 velocity solves.

    S-metric (ultrasoft/PAW) hooks — both default off, so norm-conserving
    callers are byte-unchanged:

    - ``s_apply``: the batched overlap operator S (on the folded k+q spheres).
      When given, the RHS conduction projector, the occupied projector, and
      the Sternheimer resolvent switch to the S-metric via
      ``_response.cg_sternheimer(..., s_apply=, s_occ=)`` (which forces the
      plain-CG path; the NC-only dense eigh draft is skipped), and the
      returned δu is S-orthogonal to the occupied block (⟨ψ_o|S|δu⟩ = 0).
    - ``dsvel``: a :class:`kgeometry.OverlapVelocity` supplying ∂S/∂k, so the
      elementary velocity becomes the generalized v = ∂H/∂k − ε ∂S/∂k. Absent
      (norm-conserving) the perturbation is the plain kinetic+KB velocity.

    With an explicit identity S (``s_apply`` = the identity, ``dsvel`` = None)
    the projectors and solve reduce to the norm-conserving arithmetic — the
    NC-limit reduction gate for the S-metric machinery.

    ``dense`` selects the eigh-prepass draft path: "auto" (default) builds or
    reuses a size-gated :class:`~gradwave.postscf._response.
    DenseSternheimerSolver` for the mesh (see ``dense_sternheimer_for``),
    ``None`` forces plain CG, or pass a prebuilt solver (``sigma_shielding``
    shares one across all its axis/sign/polarization solves). Drafts are
    always certified: the CG polish re-measures every (k, band) column
    against the true batched operator and iterates any column above
    ``cg_tol``. ``v`` reuses a prebuilt :class:`~gradwave.postscf.kgeometry.
    VelocityApply` (its constructor is a serial per-k projector build worth
    hoisting across the 4-6 calls of a tensor evaluation).

    ``chunk_k`` streams the Sternheimer solve over blocks of at most that many
    mesh-k at a time (``None`` = the whole mesh at once, the historical path).
    Each mesh-k is an independent Sternheimer system, so the chunked result is
    bit-identical to the full solve up to ``cg_tol``; the win is that the CG /
    Hamiltonian-apply real-space FFT boxes (the peak transient for many-atom
    cells) are held for one block rather than the whole mesh. The returned
    ``o_c``/``dpsi`` are still the full-mesh stacks (reassembled from the
    blocks) — chunking bounds only the solve's peak, not the stored result."""
    _guard(res)
    if uspp is not None:
        system = cast("System", uspp.system)
        bk = uspp.bk
    else:
        system = res.system
        bk = system.batch
        assert bk is not None
    grid = system.grid
    shape = grid.shape
    q_frac = np.asarray(q_frac, dtype=float)

    k_frac = np.stack([sph.k_frac for sph in system.spheres])
    jidx, g0 = kpq_map(k_frac, q_frac)
    jsel = torch.as_tensor(jidx, dtype=torch.long)
    nk = len(k_frac)

    nocc = insulator_window(res.occupations, 2.0, "insulating occupations required")
    c_all = pad_coeffs(cast("list[Tensor]", res.coeffs), bk.npw_max)
    c_occ_k = c_all[:, :nocc]
    c_occ_kq = c_all[jsel, :nocc]
    eps_k = res.eigenvalues[:, :nocc].to(RDTYPE)
    shift = sternheimer_shift(eps_k)

    if uspp is not None:
        # USPP/PAW S-metric backend: batched H/S on the folded spheres from the
        # frozen ground state; the generalized velocity v = ∂H/∂k − ε ∂S/∂k.
        solver = None
        h_kq, bk_kq = uspp.build_hkq(jsel)
        s_apply = h_kq.s_apply
        if dsvel is None:
            dsvel = uspp.ov
        if v is None:
            v = cast("VelocityApply", uspp.v_h)
    else:
        # The S-metric solve uses plain CG (the dense eigh draft factorizes the
        # NORM-CONSERVING operator only); force it off when an overlap is given.
        solver = None if s_apply is not None else (
            dense_sternheimer_for(res) if dense == "auto" else dense)
        assert not isinstance(solver, str)
        if solver is not None:
            h_kq, bk_kq = solver.h_for(jsel)
        else:
            bk_kq = _reindex_bk(bk, jsel)
            h_kq = BatchedHamiltonian(
                bk_kq, shape, res.v_eff, projectors_b(bk_kq, system.positions))
    ph = _g0_phase_stack(shape, g0, c_all.device)  # (nk, 1, *shape): e^{iG0·r}

    # S-metric occupied block (≡ c_occ_kq when S = I, so the conduction
    # projector and the solver collapse to the norm-conserving path).
    s_occ_kq = c_occ_kq if s_apply is None else s_apply(c_occ_kq)

    if v is None:
        v = VelocityApply(system)

    def _solve_block(idx_k: Tensor | None, jsel_x: Tensor, c_k: Tensor,
                     c_kq: Tensor, eps_x: Tensor, ph_x: Tensor, bk_k: BatchedK,
                     bk_kq_x: BatchedK,
                     h_kq_x: "BatchedHamiltonian | _USPPBatchedHS",
                     s_occ_x: Tensor,
                     s_apply_x: Callable[[Tensor], Tensor] | None,
                     ) -> tuple[Tensor, Tensor]:
        """The three velocity Sternheimer axes for one k-subset, returning the
        (3, nsub, nocc, npw) ``o``/``δu`` stacks. ``idx_k`` is the mesh-k gather
        for :meth:`VelocityApply.apply` (``None`` = the whole mesh, identity, so
        the ``chunk_k=None`` call is byte-for-byte the historical loop)."""

        def transfer_x(w: Tensor) -> Tensor:
            return box_to_sphere_b(g_to_r_b(w, bk_k, shape) * ph_x, bk_kq_x)

        def p_c_x(x: Tensor) -> Tensor:
            ov = torch.einsum("kng,kbg->kbn", c_kq.conj(), x)
            return x - torch.einsum("kbn,kng->kbg", ov, s_occ_x)

        c_t = transfer_x(c_k)
        o_l, d_l = [], []
        for mu in range(3):
            vg_k = v.apply(c_k, mu, k_index=idx_k)
            vg_t = v.apply(c_t, mu, k_index=jsel_x)
            if dsvel is not None:  # generalized velocity v = ∂H/∂k − ε ∂S/∂k
                e = eps_x[..., None].to(CDTYPE)
                vg_k = vg_k - e * dsvel.apply(c_k, mu, k_index=idx_k)
                vg_t = vg_t - e * dsvel.apply(c_t, mu, k_index=jsel_x)
            o_mu = 0.5 * (transfer_x(vg_k) + vg_t)
            rhs = -p_c_x(o_mu)
            x0 = (solver.solve(jsel_x, eps_x, rhs, shift, nocc)
                  if solver is not None else torch.zeros_like(rhs))
            d_l.append(
                cg_sternheimer(h_kq_x, bk_kq_x, c_kq, eps_x, rhs,
                               x0, shift, tol=cg_tol, max_iter=max_iter,
                               s_apply=s_apply_x, s_occ=s_occ_x)
            )
            o_l.append(o_mu)
        return torch.stack(o_l), torch.stack(d_l)

    if chunk_k is None:
        o_c, dpsi = _solve_block(None, jsel, c_occ_k, c_occ_kq, eps_k, ph, bk,
                                 bk_kq, h_kq, s_occ_kq, s_apply)
    else:
        o_parts: list[Tensor] = []
        d_parts: list[Tensor] = []
        for lo in range(0, nk, int(chunk_k)):
            hi = min(lo + int(chunk_k), nk)
            idx = torch.arange(lo, hi)
            jsel_c = jsel[idx]
            bk_c = _reindex_bk(bk, idx)  # this block's k spheres
            if uspp is not None:
                h_c, bk_kq_c = uspp.build_hkq(jsel_c)
                s_apply_c = h_c.s_apply
            elif solver is not None:
                h_c, bk_kq_c = solver.h_for(jsel_c)
                s_apply_c = s_apply
            else:
                bk_kq_c = _reindex_bk(bk, jsel_c)
                h_c = BatchedHamiltonian(
                    bk_kq_c, shape, res.v_eff,
                    projectors_b(bk_kq_c, system.positions))
                s_apply_c = s_apply
            c_kq_c = c_occ_kq[idx]
            s_occ_c = c_kq_c if s_apply_c is None else s_apply_c(c_kq_c)
            o_b, d_b = _solve_block(idx, jsel_c, c_occ_k[idx], c_kq_c,
                                    eps_k[idx], ph[idx], bk_c, bk_kq_c, h_c,
                                    s_occ_c, s_apply_c)
            o_parts.append(o_b)
            d_parts.append(d_b)
        o_c = torch.cat(o_parts, dim=1)
        dpsi = torch.cat(d_parts, dim=1)

    return VelocityQSolves(
        q_frac=q_frac,
        o_c=o_c,
        dpsi=dpsi,
        c_occ_k=c_occ_k,
        c_occ_kq=c_occ_kq,
        eps_k=eps_k,
        jidx=jidx,
        g0=g0,
        weights=system.kweights.to(RDTYPE),
        bk_kq=bk_kq,
        ph=ph,
        h_kq=cast("BatchedHamiltonian | None", h_kq),
        dense=solver,
    )


def paramagnetic_tensor(sol: VelocityQSolves) -> Tensor:
    """Per-k bare paramagnetic response P_μν(k; q) (nk, 3, 3) complex:

        P_μν(k) = Σ_n ⟨O_{q,μ}u_nk| G_{k+q}(ε_nk) P_c |O_{q,ν}u_nk⟩
                = Σ_n ⟨O_{q,μ}u_nk | δu^{(μ→ν,q)}_{nk}⟩

    (rhs = −P_c O u makes the solver return δu = +G_{k+q}(ε_nk) P_c O u with
    G = Σ_m |m⟩⟨m|/(ε_nk − ε_m)). Gauge invariant under occupied phases and
    degenerate rotations; BZ-average with ``sol.weights``."""
    p = torch.empty(sol.o_c.shape[1], 3, 3, dtype=CDTYPE)
    for mu in range(3):
        for nu in range(3):
            band = torch.einsum("kng,kng->kn", sol.o_c[mu].conj(), sol.dpsi[nu])
            p[:, mu, nu] = band.sum(dim=1)
    return p


# --------------------------------------------------------------------------- #
# dense twin (continuous q, no mesh constraint)                               #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# KB nonlocal current: the [r, V_NL] closure of the continuity equation       #
# --------------------------------------------------------------------------- #


class _NLPairVelocity:
    """Line-averaged KB velocity Ā_μ for every (k, k+q) Sternheimer pair.

    The kinetic current alone does not satisfy the continuity equation in the
    presence of the nonlocal KB potential: for the transition pair (u_nk,
    δu_{n,k+q}) the exact identity (from (H − ε_nk)|δψ⟩ = |χ⟩ and
    [H, e^{-iqr}] algebra) is

        q·j_kin(q) + q·j_NL(q) = s(q),   s(q) = Σ f w ⟨ψ|e^{-iqr}|χ⟩/Ω,

    where the nonlocal piece must satisfy q·j_NL(q) =
    Σ f w ⟨u_k|V_NL^{(k+q)} − V_NL^{(k)}|δu⟩/Ω (matrix elements in the fixed
    k-referenced Miller basis). The velocity form of the nonlocal current
    (∂V_NL/∂k at a single k) closes this only to O(q³) because V_NL(k) is not
    quadratic in k. This class instead uses the exact mean-value form

        Ā_μ = ∫₀¹ ds ∂V_NL/∂k_μ |_{k+sq},   q·Ā = V_NL^{(k+q)} − V_NL^{(k)}

    (fundamental theorem of calculus, entrywise in the fixed Miller basis),
    evaluated by Gauss–Legendre quadrature over the analytic KB projector
    build (``KBProjectors.p_and_dp`` at the shifted k's) — machine-precision
    exact for the small q of the magnetic response. At q = 0, Ā_μ reduces to
    the ordinary KB velocity ∂V_NL/∂k of ``VelocityApply``. The current
    density field is the same symmetrized cross density as the kinetic term:

        j_NL,μ(r) = Σ f w · ½[ũ*·(Ā_μ δu)~ + (Ā_μ u)~*·δũ](r) / Ω,

    whose cell average is exactly ⟨u|Ā_μ|δu⟩ (Ā Hermitian), so the closure
    holds at the G = 0 Fourier component to CG/quadrature precision. The
    G ≠ 0 components inherit the intrinsic transverse ambiguity of any
    nonlocal-potential current — the standard GIPAW caveat; the linear-in-q
    (shielding) content is unaffected.

    Everything is assembled per k on the union of the k-sphere Millers and
    the (k+q)-sphere Millers re-referenced to k (M − G0), so the umklapp
    phase is absorbed and no coefficients are truncated.
    """

    def __init__(self, system: System, sol: VelocityQSolves, nquad: int = 8) -> None:
        self.shape = system.grid.shape
        self.volume = system.grid.volume
        b = reciprocal_cell(system.grid.cell)
        q_cart = torch.as_tensor(sol.q_frac @ b, dtype=RDTYPE)
        # Gauss–Legendre on [0, 1]
        x, w = np.polynomial.legendre.leggauss(nquad)
        s_nodes = 0.5 * (x + 1.0)
        s_w = 0.5 * w

        self.u_idx: list[Tensor] = []  # union slot of each k-sphere coeff
        self.d_idx: list[Tensor] = []  # union slot of each (k+q)-sphere coeff
        self.flat: list[Tensor] = []  # FFT-box flat index of each union Miller
        self.p: list[Tensor] = []  # (nq, nproj, npwU) quadrature projectors
        self.dp: list[Tensor] = []  # (nq, 3, nproj, npwU)
        self.w_s = torch.as_tensor(s_w, dtype=RDTYPE)
        self.dij = system.batch.dij_full.to(CDTYPE) if system.batch is not None \
            else torch.zeros((0, 0), dtype=CDTYPE)
        self.nproj = int(self.dij.shape[0])

        n1, n2, n3 = self.shape
        # the k-independent projector context (radial quadratures, D blocks,
        # column maps) is built once and reused for every k's union sphere
        kb0 = (KBProjectors.for_sphere(system, system.spheres[0])
               if self.nproj else None)
        for ik, sph in enumerate(system.spheres):
            m_k = sph.miller.numpy()
            m_d = system.spheres[int(sol.jidx[ik])].miller.numpy() - sol.g0[ik][None, :]
            union = np.unique(np.concatenate([m_k, m_d], axis=0), axis=0)
            keys_u = _pack_miller(union)
            order = np.argsort(keys_u)
            keys_sorted = keys_u[order]
            u_pos = order[np.searchsorted(keys_sorted, _pack_miller(m_k))]
            d_pos = order[np.searchsorted(keys_sorted, _pack_miller(m_d))]
            self.u_idx.append(torch.as_tensor(u_pos, dtype=torch.int64))
            self.d_idx.append(torch.as_tensor(d_pos, dtype=torch.int64))
            mod = union % np.array([n1, n2, n3])
            self.flat.append(torch.as_tensor(
                (mod[:, 0] * n2 + mod[:, 1]) * n3 + mod[:, 2], dtype=torch.int64))

            if kb0 is not None:
                kb = KBProjectors(
                    g_cart=torch.as_tensor(union.astype(float) @ b, dtype=RDTYPE),
                    tau=kb0.tau, dij_full=kb0.dij_full, col_atom=kb0.col_atom,
                    col_chan=kb0.col_chan, col_lm=kb0.col_lm,
                    channels=kb0.channels, lmax=kb0.lmax, volume=kb0.volume,
                )
                k_cart = sph.k_cart.to(RDTYPE)
                ps, dps = [], []
                for s in s_nodes:
                    p_s, dp_s = kb.p_and_dp(k_cart + float(s) * q_cart)
                    ps.append(p_s)
                    dps.append(torch.stack(dp_s))
                self.p.append(torch.stack(ps))
                self.dp.append(torch.stack(dps))

    def _apply(self, ik: int, mu: int, c: Tensor) -> Tensor:
        """Ā_μ c for coefficients c (nb, npwU) on the union sphere of k ``ik``."""
        p, dp = self.p[ik], self.dp[ik][:, mu]
        w = self.w_s.to(CDTYPE)
        b_p = torch.einsum("spg,bg->sbp", p.conj(), c)
        b_dp = torch.einsum("spg,bg->sbp", dp.conj(), c)
        out = torch.einsum("s,sbp,pq,sqg->bg", w, b_p, self.dij, dp)
        out = out + torch.einsum("s,sbp,pq,sqg->bg", w, b_dp, self.dij, p)
        return out

    def field(self, sol: VelocityQSolves, dpsi: Tensor) -> Tensor:
        """BZ-weighted nonlocal current density (3, n1, n2, n3), the periodic
        part at wavevector q, normalized exactly like ``induced_current_q``'s
        ``j_para`` (f = 2, 1/Ω, k-weights folded). ``dpsi`` is the first-order
        response block (nk, nocc, npw_max) on the folded k+q spheres."""
        j = torch.zeros(3, *self.shape, dtype=CDTYPE)
        if not self.nproj:
            return j
        f_occ = 2.0
        for ik in range(len(self.flat)):
            npw_u = int(self.flat[ik].shape[0])
            n_k = int(self.u_idx[ik].shape[0])
            n_d = int(self.d_idx[ik].shape[0])
            nocc = dpsi.shape[1]
            u_u = torch.zeros(nocc, npw_u, dtype=CDTYPE)
            d_u = torch.zeros(nocc, npw_u, dtype=CDTYPE)
            u_u[:, self.u_idx[ik]] = sol.c_occ_k[ik, :, :n_k]
            d_u[:, self.d_idx[ik]] = dpsi[ik, :, :n_d]
            u_box = g_to_r(u_u, self.flat[ik], self.shape)
            d_box = g_to_r(d_u, self.flat[ik], self.shape)
            w_k = float(sol.weights[ik])
            for mu in range(3):
                a_du = g_to_r(self._apply(ik, mu, d_u), self.flat[ik], self.shape)
                a_u = g_to_r(self._apply(ik, mu, u_u), self.flat[ik], self.shape)
                j[mu] += (w_k * f_occ / self.volume) * 0.5 * (
                    u_box.conj() * a_du + a_u.conj() * d_box
                ).sum(dim=0)
        return j


# --------------------------------------------------------------------------- #
# induced current density and K_Hxc screening (milestone 8)                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CurrentResponseQ:
    """First-order induced current density for the perturbation λ·O_{q,ν}
    (per unit λ; the physical field adds the −q branch)."""

    q_frac: np.ndarray
    e_pol: np.ndarray  # (3,) Cartesian polarization vector of the perturbation
    j_para: Tensor  # (3, n1, n2, n3) complex: BZ-weighted canonical current
    j_para_k: Tensor  # (nk, 3, n1, n2, n3): per-k, (f/Ω)-normalized, no w_k
    j_dia: Tensor  # (3, n1, n2, n3): diamagnetic term 2·HBAR2_2M·e_pol_μ·ρ(r)
    j_nl: Tensor  # (3, n1, n2, n3): KB nonlocal current (continuity closure)
    drho_bare: Tensor  # (n1, n2, n3) complex: bare induced density (periodic part)
    drho: Tensor  # screened induced density (== bare when screen=False)
    n_dyson: int
    dpsi: Tensor  # (nk, nocc, npw_max): total (screened) first-order response

    @property
    def nu(self) -> int:
        """Cartesian index of a one-hot polarization (back-compat accessor)."""
        return int(np.argmax(np.abs(self.e_pol)))

    @property
    def j_total(self) -> Tensor:
        """j_para + j_nl + j_dia — the conserved induced current field."""
        return self.j_para + self.j_nl + self.j_dia


def _drho_q(sol: VelocityQSolves, dpsi: Tensor, bk: BatchedK, shape, ph: Tensor,
            volume: float) -> Tensor:
    """Periodic part of the induced density: Σ_kn f w u*_nk δu_nk e^{−iG0r}/Ω."""
    u_box = g_to_r_b(sol.c_occ_k, bk, shape)
    du_box = g_to_r_b(dpsi, sol.bk_kq, shape)
    contrib = (u_box.conj() * du_box * ph.conj()).sum(dim=1)  # (nk, *shape)
    w = (2.0 * sol.weights).to(CDTYPE)
    return torch.einsum("k,k...->...", w, contrib) / volume


def induced_current_q(
    res: SCFResult,
    q_frac: np.ndarray | list[float] | tuple[float, float, float],
    nu: int | np.ndarray | list[float] | tuple[float, float, float],
    *,
    xc: object | None = None,
    screen: bool = False,
    cg_tol: float = 1e-9,
    cg_max_iter: int = 400,
    dyson_beta: float = 0.4,
    dyson_tol: float = 1e-7,
    dyson_iter: int = 40,
    history: int = 8,
    sol: VelocityQSolves | None = None,
    nl_quad: int = 8,
    _nl: _NLPairVelocity | None = None,
) -> CurrentResponseQ:
    """Induced (canonical) current density of the perturbation λ·O_{q,pol},
    O_{q,pol} = Σ_ν e_pol[ν]·½{v_ν, e^{iqr}}. ``nu`` is either a Cartesian
    index (one-hot polarization, the historical interface) or a full
    Cartesian polarization vector e_pol.

    Paramagnetic part from the (u_nk, δu_nk) Sternheimer pairs with the
    KINETIC current operator — the cell-periodic cross density
    ½[ũ*(v_kin δu)~ + (v_kin u)~* δũ] referenced to (k, k+q) via the umklapp
    phase; per-k means obey the exact f-sum d⟨v_μ⟩/dk_ν identity (tested).
    Diamagnetic part 2·HBAR2_2M·e_pol_μ·ρ(r) (the A·A term of the same
    coupling; at q = 0 its cell average cancels the paramagnetic one).
    The KB nonlocal current (``j_nl``, :class:`_NLPairVelocity`) closes the
    continuity equation q·j(q) = s(q) that the kinetic current alone
    violates (tested); ``j_total`` is the conserved sum. The remaining
    systematic is the GIPAW core reconstruction (out of scope here).

    ``sol`` reuses an existing :func:`velocity_perturbation_q` solve (the
    shielding assembly evaluates several polarizations per q); it must match
    ``q_frac``.

    ``screen=True`` adds the self-consistent K_Hxc response: the induced
    density is converged through the Dyson fixed point δρ = χ₀[O] +
    χ₀[K_Hxc^q δρ] (reusing dfpt_q.chi0_q / _k_hxc_q with Anderson mixing)
    and the final δu gains the local-field solve for the converged
    δV_Hxc — the current is then assembled from the SCREENED δu. For a
    time-reversal-symmetric ground state the q = 0 velocity perturbation
    induces no density (δρ is TR-odd), so screening is inert there (tested).
    """
    _guard(res)
    system = res.system
    grid = system.grid
    shape = grid.shape
    bk = system.batch
    assert bk is not None
    if isinstance(nu, (int, np.integer)):
        e_pol = np.zeros(3)
        e_pol[int(nu)] = 1.0
    else:
        e_pol = np.asarray(nu, dtype=float)
    if sol is None:
        sol = velocity_perturbation_q(res, q_frac, cg_tol=cg_tol, max_iter=cg_max_iter)
    else:
        assert np.allclose(sol.q_frac, np.asarray(q_frac, dtype=float)), \
            "sol was solved at a different q"
    nk = sol.c_occ_k.shape[0]
    ph = sol.ph if sol.ph is not None else _g0_phase_stack(
        shape, sol.g0, sol.c_occ_k.device)

    e_pol_t = torch.as_tensor(e_pol, dtype=RDTYPE).to(CDTYPE)
    dpsi = torch.einsum("m,mkng->kng", e_pol_t, sol.dpsi)
    drho_bare = _drho_q(sol, dpsi, bk, shape, ph, grid.volume)
    drho = drho_bare
    n_dyson = 0

    if screen:
        assert xc is not None, "screen=True needs the xc functional"
        from gradwave.core._anderson import AndersonMixer
        from gradwave.postscf.dfpt_q import _k_hxc_q, chi0_q

        b = 2.0 * np.pi * np.linalg.inv(np.asarray(grid.cell, float)).T
        q_cart = torch.as_tensor(np.asarray(q_frac, float) @ b, dtype=RDTYPE)
        # At q = 0 the ±q branches coincide and the perturbation λ·v_ν is
        # itself Hermitian: the physical density response is 2·Re[Σ u*δu]
        # (TR-odd ⇒ ≈ 0 for a TR ground state), while the imaginary part of
        # the single-branch sum is a branch-splitting artifact that must NOT
        # drive K_Hxc. At q ≠ 0 the branch is a genuine wavevector-q field.
        at_gamma = float(np.abs(np.asarray(q_frac, float)).max()) < 1e-12

        def phys(u: Tensor) -> Tensor:
            return u.real.to(CDTYPE) if at_gamma else u

        def g(u: Tensor) -> Tensor:
            return drho_bare + chi0_q(
                res, q_frac, _k_hxc_q(res, xc, phys(u), q_cart), tol=cg_tol,
                dense=sol.dense,
            )

        u = drho_bare.clone()
        mixer = AndersonMixer(history, dyson_beta)
        for it in range(dyson_iter):
            r = g(u) - u
            step = float(torch.linalg.norm(r)) / max(1.0, float(torch.linalg.norm(u)))
            n_dyson = it + 1
            if step < dyson_tol:
                break
            u = mixer.step(u.reshape(-1), r.reshape(-1)).reshape(u.shape)
        else:
            raise RuntimeError(f"current-response Dyson not converged after {dyson_iter}")
        drho = u

        # final local-field solve: δu gains the converged δV_Hxc response
        dv = _k_hxc_q(res, xc, phys(drho), q_cart)
        u_box = g_to_r_b(sol.c_occ_k, bk, shape)
        rhs_s = box_to_sphere_b(u_box * dv[None, None] * ph, sol.bk_kq)
        ov = torch.einsum("kng,kbg->kbn", sol.c_occ_kq.conj(), rhs_s)
        rhs = -(rhs_s - torch.einsum("kbn,kng->kbg", ov, sol.c_occ_kq))
        h_kq = sol.h_kq if sol.h_kq is not None else BatchedHamiltonian(
            sol.bk_kq, shape, res.v_eff, projectors_b(sol.bk_kq, system.positions)
        )
        shift = sternheimer_shift(sol.eps_k)
        nocc = sol.c_occ_kq.shape[1]
        x0 = (sol.dense.solve(torch.as_tensor(sol.jidx, dtype=torch.long),
                              sol.eps_k, rhs, shift, nocc)
              if sol.dense is not None else torch.zeros_like(rhs))
        dpsi = dpsi + cg_sternheimer(h_kq, sol.bk_kq, sol.c_occ_kq, sol.eps_k,
                                     rhs, x0, shift,
                                     tol=cg_tol, max_iter=cg_max_iter)

    # canonical paramagnetic current density, cell-periodic part at q
    f_occ = 2.0
    u_box = g_to_r_b(sol.c_occ_k, bk, shape)
    du_box = g_to_r_b(dpsi, sol.bk_kq, shape) * ph.conj()
    j_k = torch.empty(nk, 3, *shape, dtype=CDTYPE)
    for mu in range(3):
        wu = (2.0 * HBAR2_2M) * bk.kpg[:, None, :, mu] * sol.c_occ_k
        wdu = (2.0 * HBAR2_2M) * sol.bk_kq.kpg[:, None, :, mu] * dpsi
        wu_box = g_to_r_b(wu, bk, shape)
        wdu_box = g_to_r_b(wdu, sol.bk_kq, shape) * ph.conj()
        j_k[:, mu] = (f_occ / grid.volume) * 0.5 * (
            u_box.conj() * wdu_box + wu_box.conj() * du_box
        ).sum(dim=1)
    j_para = torch.einsum("k,km...->m...", sol.weights.to(CDTYPE), j_k)
    j_dia = torch.einsum(
        "m,...->m...", e_pol_t, (2.0 * HBAR2_2M) * res.rho.to(CDTYPE))
    nl = _NLPairVelocity(system, sol, nquad=nl_quad) if _nl is None else _nl
    j_nl = nl.field(sol, dpsi)

    return CurrentResponseQ(
        q_frac=np.asarray(q_frac, dtype=float),
        e_pol=e_pol,
        j_para=j_para,
        j_para_k=j_k,
        j_dia=j_dia,
        j_nl=j_nl,
        drho_bare=drho_bare,
        drho=drho,
        n_dyson=n_dyson,
        dpsi=dpsi,
    )


def _dense_velocity_matrices(hh: BlochHK, k_cart: Tensor) -> list[Tensor]:
    """[v_x, v_y, v_z] (npw, npw) at k on hh's sphere — the dense form of
    ``VelocityApply``: kinetic diagonal + dpᵀD p* + pᵀD dp*."""
    kbp = KBProjectors(
        g_cart=hh.g_cart, tau=hh.tau, dij_full=hh.dij_full,
        col_atom=hh.col_atom, col_chan=hh.col_chan, col_lm=hh.col_lm,
        channels=hh.channels, lmax=hh.lmax, volume=hh.volume,
    )
    p, dp = kbp.p_and_dp(k_cart)
    dij = hh.dij_full.to(CDTYPE)
    kpg = hh.g_cart + k_cart
    out = []
    for mu in range(3):
        v = torch.diag_embed((2.0 * HBAR2_2M * kpg[:, mu]).to(CDTYPE))
        if p.shape[0]:
            v = v + dp[mu].mT @ (dij @ p.conj()) + p.mT @ (dij @ dp[mu].conj())
        out.append(v)
    return out


def paramagnetic_tensor_dense(
    res: SCFResult,
    k_frac: np.ndarray | list[float],
    q_frac: np.ndarray | list[float],
    nocc: int,
) -> Tensor:
    """P_μν(k; q) (3, 3) from explicit dense H at k and the UNFOLDED k+q:
    eigh at both points (direct beta tables), the operator built from the
    same two-term symmetrization as the mesh route — O = ½(T v_k + v_{k+q} T)
    with T the absolute-Miller-matched sphere transfer — and the resolvent
    as an explicit sum over dense conduction states. Continuous q (no mesh
    commensurability): the reference the Sternheimer route is validated
    against, and the probe for genuine q → 0 limits."""
    k = np.asarray(k_frac, dtype=float)
    q = np.asarray(q_frac, dtype=float)
    hk = BlochHK.from_scf(res, k)
    hkq = BlochHK.from_scf(res, k + q)
    kc, kqc = hk.k_cart(k), hkq.k_cart(k + q)
    w_k, u_k = torch.linalg.eigh(hk.h(kc))
    w_q, u_q = torch.linalg.eigh(hkq.h(kqc))

    # sphere transfer T: plane wave (k+m) → the (k+q)-sphere slot of the SAME
    # Miller m (the operator ½{v, e^{iqr}} is Miller-diagonal)
    keys_k = _pack_miller(hk.miller.numpy())
    keys_q = _pack_miller(hkq.miller.numpy())
    _, i_q, i_k = np.intersect1d(keys_q, keys_k, return_indices=True)
    t_mat = torch.zeros(hkq.npw, hk.npw, dtype=CDTYPE)
    t_mat[torch.as_tensor(i_q), torch.as_tensor(i_k)] = 1.0

    v_k = _dense_velocity_matrices(hk, kc)
    v_q = _dense_velocity_matrices(hkq, kqc)

    u_occ = u_k[:, :nocc]
    u_con = u_q[:, nocc:]
    eps_occ = w_k[:nocc]
    denom = (eps_occ[None, :] - w_q[nocc:][:, None]).to(CDTYPE)  # (ncon, nocc)

    phi = [0.5 * (t_mat @ (v_k[mu] @ u_occ) + v_q[mu] @ (t_mat @ u_occ)) for mu in range(3)]
    a = [(u_con.mH @ phi[mu]) for mu in range(3)]  # ⟨u_m|O_μ u_n⟩, (ncon, nocc)
    p = torch.empty(3, 3, dtype=CDTYPE)
    for mu in range(3):
        for nu in range(3):
            p[mu, nu] = (a[mu].conj() * a[nu] / denom).sum()
    return p


# --------------------------------------------------------------------------- #
# bare shielding tensor: q→0 antisymmetric assembly (Pickard–Mauri)           #
# --------------------------------------------------------------------------- #


def continuity_source_q(res: SCFResult, sol: VelocityQSolves,
                        e_pol: np.ndarray | list[float]) -> Tensor:
    """Periodic source field s(r) of the exact continuity identity

        q_cart · mean[j_para + j_nl] = mean s,
        s(r) = Σ_nk f w_k ũ*_nk χ̃_nk e^{−iG0r} / Ω,   χ = −P_c O_{q,pol} u,

    i.e. the cross density of the occupied states with the Sternheimer RHS.
    This is the "∂ρ/∂t" side of ∇·j + ∂ρ/∂t = 0 for the single-branch
    response pair — the KB nonlocal current test's reference (the kinetic
    current alone fails it by the [V_NL, e^{−iqr}] commutator)."""
    system = res.system
    shape = system.grid.shape
    bk = system.batch
    assert bk is not None
    ph = sol.ph if sol.ph is not None else _g0_phase_stack(
        shape, sol.g0, sol.c_occ_k.device)
    e_pol_t = torch.as_tensor(np.asarray(e_pol, dtype=float), dtype=RDTYPE).to(CDTYPE)
    o_pol = torch.einsum("m,mkng->kng", e_pol_t, sol.o_c)
    ov = torch.einsum("kng,kbg->kbn", sol.c_occ_kq.conj(), o_pol)
    rhs = -(o_pol - torch.einsum("kbn,kng->kbg", ov, sol.c_occ_kq))
    return _drho_q(sol, rhs, bk, shape, ph, system.grid.volume)


def _antisym_field_g(sp: Tensor, sm: Tensor) -> Tensor:
    """Fourier coefficients F̂(q+G) (3, n1, n2, n3) of the antisymmetric
    combination F(r) = Im{e^{iq·r}S₊(r) − e^{−iq·r}S₋(r)} — the physical
    (real) induced-current field of a uniform B, per the ±q branch fields
    S₊/S₋ (periodic parts): F̂(q+G) = [Ŝ₊(G) + conj(Ŝ₋(−G))]/2i."""
    sp_g = r_to_g(sp)
    rev = r_to_g(sm).conj()
    for d in (-3, -2, -1):
        rev = torch.roll(torch.flip(rev, dims=(d,)), 1, dims=d)
    return (sp_g + rev) / 2j


def _biot_savart_sigma_cols(sp: Tensor, sm: Tensor, q_cart: Tensor,
                            g_cart: Tensor, sites: Tensor) -> Tensor:
    """Shielding column σ·(q̂×e_pol) at each site: (nsite, 3) real.

    Biot–Savart of the physical induced current in G-space, B̂_ind(K) =
    (4π/c)·iK×ĵ_c(K)/K² at K = ±q+G, evaluated at the nuclei; the −q
    components are the conjugates of the +q ones, so the K-sum is
    2·Re Σ_G. Unit chain (all from gradwave.constants, mirroring the
    calibrated orbital-magnetization collapse): the wave coupling and the
    charge current carry e²/(c²ħ²) = ALPHA_FS²/E2 [1/(eV·Å)], leaving

        σ·(q̂×e_pol) = (4π·ALPHA_FS²/(E2·q)) · 2 Re Σ_{G≠0}
                        e^{i(q+G)·r_s} [i(q+G)×F̂(q+G)] / |q+G|².

    The G = 0 term (the macroscopic, sample-shape/susceptibility
    contribution) is omitted — the standard bare-crystal convention; it
    cancels in chemical-shift differences between same-shape samples."""
    f_g = _antisym_field_g(sp, sm).permute(1, 2, 3, 0)  # (n1, n2, n3, 3)
    kk = g_cart.to(RDTYPE) + q_cart.to(RDTYPE)
    k2 = (kk * kk).sum(dim=-1)
    term = 1j * torch.linalg.cross(kk.to(CDTYPE), f_g, dim=-1) / k2[..., None].to(CDTYPE)
    term[0, 0, 0] = 0.0  # macroscopic (shape) term
    qmag = float(torch.linalg.norm(q_cart))
    pref = 4.0 * math.pi * ALPHA_FS**2 / (E2 * qmag)
    phases = torch.exp(1j * torch.einsum("sa,ijka->sijk", sites.to(RDTYPE),
                                         kk).to(CDTYPE))
    return pref * 2.0 * torch.einsum("sijk,ijkm->sm", phases, term).real


def _transverse_frame(q_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right-handed orthonormal (t1, t2) with q̂×t1 = t2, q̂×t2 = −t1."""
    a = np.zeros(3)
    a[int(np.argmin(np.abs(q_hat)))] = 1.0
    t1 = a - (a @ q_hat) * q_hat
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(q_hat, t1)


def sigma_shielding(
    res: SCFResult,
    *,
    q_index: int = 1,
    xc: object | None = None,
    screen: bool = False,
    cg_tol: float = 1e-9,
    cg_max_iter: int = 400,
    nl_quad: int = 8,
    sites: Tensor | None = None,
    core_ppm: Tensor | np.ndarray | list[float] | None = None,
) -> Tensor:
    """Bare (pseudo) NMR chemical-shielding tensor σ_ij per site, in ppm:
    σ_ij = −∂B_ind,i(r_site)/∂B_ext,j (positive = shielded), shape
    (nsite, 3, 3).

    Pickard–Mauri construction: a uniform B_ext is the q → 0 limit of two
    transverse vector-potential waves A = ê_pol sin(q·r)/q. For each usable
    mesh axis the ±q Sternheimer responses (:func:`velocity_perturbation_q`)
    are assembled — per transverse polarization — into the conserved induced
    current S = j_para + j_nl + j_dia (kinetic + KB-nonlocal continuity
    closure + diamagnetic), antisymmetrized over ±q (which cancels the
    even-in-q response and realizes the (1/2q)[S(q) − S(−q)] linear-in-q
    extraction), Biot–Savart'ed to B_ind at the nuclei
    (:func:`_biot_savart_sigma_cols`), and the tensor is least-squares
    reconstructed from the measured columns σ·(q̂×e_pol).

    Known systematics, by design of this bare layer:

    - NO GIPAW core augmentation/reconstruction: this is the pseudo (bare)
      shielding of the smooth valence states only. Absolute σ therefore
      misses the (large, element-specific, mostly isotropic and
      site-transferable) core/reconstruction contribution.
    - The G = 0 macroscopic shape/susceptibility term is omitted (standard
      bare-crystal convention).
    - Finite-q truncation: q = q_index/n_i per mesh axis; the residual error
      is O(q²) (the antisymmetric combination cancels even orders) PLUS a
      slowly-decaying sphere-boundary term: δu lives on the k+q sphere,
      which differs from the k sphere by an O(q) shell of tail-amplitude
      plane waves, leaving an ecut-dependent non-q² component in the 1/q
      assembly (measured — see :func:`sigma_shielding_dq`, which has
      neither error, and the ecut scan in the milestone-10 PR).

    ``screen=True`` (with ``xc``) uses the K_Hxc-screened first-order
    responses. Requires an insulating, symmetry-unreduced SCF with at least
    two mesh axes of more than one point (else the 3×3 tensor is
    underdetermined and a ValueError is raised).

    This is the finite-q *validation* route (route-equivalence against the
    dense twin). For production shielding use the analytic :func:`sigma_shielding_dq`,
    which is ~3× faster at every mesh and carries neither the O(q²) nor the
    sphere-boundary finite-q error.

    ``core_ppm`` (opt-in): a per-site frozen-core Lamb shielding [ppm], added as
    an isotropic constant σ_ij += core_ppm[s]·δ_ij to each site's tensor — the
    GIPAW σ_core term (:func:`postscf.gipaw.core_lamb_shielding`). Turns the bare
    layer's *relative* shielding into a physically meaningful *absolute* one; it
    cancels in chemical-shift differences. The diamagnetic/paramagnetic
    augmentation are added separately (see ``postscf.gipaw``).
    """
    _guard(res)
    system = res.system
    grid = system.grid
    b = reciprocal_cell(grid.cell)
    k_frac = np.stack([sph.k_frac for sph in system.spheres])
    mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
    axes = [i for i in range(3) if mesh_n[i] > 1]
    if len(axes) < 2:
        raise ValueError(
            f"sigma_shielding needs >=2 k-mesh axes with n>1 (mesh {mesh_n}): "
            "the shielding tensor is underdetermined from a single q direction")
    if sites is None:
        sites = system.positions.detach().cpu().to(RDTYPE)

    axis_jobs: list[tuple[np.ndarray, Tensor, np.ndarray, list[np.ndarray]]] = []
    for i in axes:
        q_frac = np.zeros(3)
        q_frac[i] = q_index / mesh_n[i]
        q_cart = torch.as_tensor(q_frac @ b, dtype=RDTYPE)
        q_hat = (q_frac @ b) / np.linalg.norm(q_frac @ b)
        axis_jobs.append((q_frac, q_cart, q_hat, list(_transverse_frame(q_hat))))

    # per-branch conserved-current fields, in fixed (axis, sign) order with
    # (polarization) order inside — the deterministic accumulation key
    v = VelocityApply(system)
    dense = dense_sternheimer_for(res)
    j_branches = []
    for q_frac, _q_cart, _q_hat, pols in axis_jobs:
        for sign in (1.0, -1.0):
            sol = velocity_perturbation_q(
                res, sign * q_frac, cg_tol=cg_tol, max_iter=cg_max_iter,
                dense=dense, v=v)
            nl = _NLPairVelocity(system, sol, nquad=nl_quad)
            j_branches.append([
                induced_current_q(res, sign * q_frac, pol, sol=sol,
                                  _nl=nl, xc=xc, screen=screen,
                                  cg_tol=cg_tol,
                                  cg_max_iter=cg_max_iter).j_total
                for pol in pols
            ])

    b_rows: list[np.ndarray] = []
    m_rows: list[Tensor] = []
    for a, (_q_frac, q_cart, q_hat, pols) in enumerate(axis_jobs):
        j_p, j_m = j_branches[2 * a], j_branches[2 * a + 1]
        for ip, pol in enumerate(pols):
            m_rows.append(_biot_savart_sigma_cols(
                j_p[ip], j_m[ip], q_cart, grid.g_cart, sites))
            b_rows.append(np.cross(q_hat, pol))

    bmat = torch.as_tensor(np.stack(b_rows), dtype=RDTYPE)  # (nr, 3)
    mmat = torch.stack(m_rows)  # (nr, nsite, 3)
    ns = mmat.shape[1]
    out = torch.empty(ns, 3, 3, dtype=RDTYPE)
    for s in range(ns):
        out[s] = torch.linalg.lstsq(bmat, mmat[:, s, :]).solution.mT
    out = out * 1e6
    if core_ppm is not None:
        core = torch.as_tensor(core_ppm, dtype=RDTYPE)
        if core.shape != (ns,):
            raise ValueError(
                f"core_ppm must have one value per site ({ns}); got {tuple(core.shape)}")
        out = out + core[:, None, None] * torch.eye(3, dtype=RDTYPE)
    return out


def continuity_truncation_term(res: SCFResult, sol: VelocityQSolves,
                               dpsi: Tensor) -> Tensor:
    """The exact basis-truncation remainder T of the continuity closure

        q · mean[j_para + j_nl] = mean s + T,

    (s from :func:`continuity_source_q`). In a complete basis T = 0; on
    finite G-spheres the pair (u_nk on the k-sphere, δu on the k+q-sphere)
    leaks through the sphere boundaries:

        T = Σ f w/Ω [ ⟨Π̄_q u|(Ĥ_q − ε)|δu⟩ − ⟨(Ĥ_k − ε)u|Π̄_k δu⟩ ],

    with Π̄ the complement projectors and Ĥ the untruncated (union-Miller)
    application of the k / k+q Hamiltonians — only V_loc and V_NL survive
    (the kinetic term is diagonal and both states vanish on the respective
    complements). Computed independently of j/s (dense union projectors +
    the Toeplitz local product), so asserting the identity above to the CG
    tolerance validates the KB nonlocal current construction exactly while
    quantifying the (ecut-vanishing) truncation systematic."""
    system = res.system
    grid = system.grid
    shape = grid.shape
    n1, n2, n3 = shape
    b = reciprocal_cell(grid.cell)
    q_cart = torch.as_tensor(sol.q_frac @ b, dtype=RDTYPE)
    v_eff = res.v_eff if res.v_eff.dim() == 3 else res.v_eff[0]
    v_eff = v_eff.to(CDTYPE)
    f_occ = 2.0
    total = torch.zeros((), dtype=CDTYPE)
    kb0 = KBProjectors.for_sphere(system, system.spheres[0])
    for ik, sph in enumerate(system.spheres):
        m_k = sph.miller.numpy()
        m_d = system.spheres[int(sol.jidx[ik])].miller.numpy() - sol.g0[ik][None, :]
        union = np.unique(np.concatenate([m_k, m_d], axis=0), axis=0)
        keys_u = _pack_miller(union)
        order = np.argsort(keys_u)
        keys_sorted = keys_u[order]
        u_pos = order[np.searchsorted(keys_sorted, _pack_miller(m_k))]
        d_pos = order[np.searchsorted(keys_sorted, _pack_miller(m_d))]
        npw_u = union.shape[0]
        in_k = np.zeros(npw_u, dtype=bool)
        in_k[u_pos] = True
        in_q = np.zeros(npw_u, dtype=bool)
        in_q[d_pos] = True
        mod = union % np.array([n1, n2, n3])
        flat = torch.as_tensor((mod[:, 0] * n2 + mod[:, 1]) * n3 + mod[:, 2],
                               dtype=torch.int64)

        nocc = dpsi.shape[1]
        u_u = torch.zeros(nocc, npw_u, dtype=CDTYPE)
        d_u = torch.zeros(nocc, npw_u, dtype=CDTYPE)
        u_u[:, torch.as_tensor(u_pos)] = sol.c_occ_k[ik, :, : len(u_pos)]
        d_u[:, torch.as_tensor(d_pos)] = dpsi[ik, :, : len(d_pos)]

        def vloc_apply(c: Tensor) -> Tensor:
            box = g_to_r(c, flat, shape) * v_eff  # noqa: B023
            return r_to_g(box).reshape(*c.shape[:-1], -1).index_select(-1, flat)  # noqa: B023

        kb = KBProjectors(
            g_cart=torch.as_tensor(union.astype(float) @ b, dtype=RDTYPE),
            tau=kb0.tau, dij_full=kb0.dij_full, col_atom=kb0.col_atom,
            col_chan=kb0.col_chan, col_lm=kb0.col_lm,
            channels=kb0.channels, lmax=kb0.lmax, volume=kb0.volume,
        )
        dij = kb.dij_full.to(CDTYPE)

        def vnl_apply(p: Tensor, c: Tensor) -> Tensor:
            if not p.shape[0]:
                return torch.zeros_like(c)
            beta = torch.einsum("jg,bg->bj", p.conj(), c)  # ⟨p_j|c⟩
            return torch.einsum("ij,bj,ig->bg", dij, beta, p)  # noqa: B023

        k_cart = sph.k_cart.to(RDTYPE)
        p_k = kb.p(k_cart)
        p_q = kb.p(k_cart + q_cart)

        h_q_du = vloc_apply(d_u) + vnl_apply(p_q, d_u)
        h_k_u = vloc_apply(u_u) + vnl_apply(p_k, u_u)

        only_k = torch.as_tensor(in_k & ~in_q)
        only_q = torch.as_tensor(in_q & ~in_k)
        piece1 = (u_u[:, only_k].conj() * h_q_du[:, only_k]).sum()
        piece2 = (h_k_u[:, only_q].conj() * d_u[:, only_q]).sum()
        total = total + float(sol.weights[ik]) * (f_occ / grid.volume) * (
            piece1 - piece2)
    return total


def uspp_smooth_continuity(
    res: Any, ctx: USPPResponseCtx, sol: VelocityQSolves,
    e_pol: np.ndarray | list[float],
) -> dict[str, complex]:
    """Continuity closure of the USPP/PAW SMOOTH+nonlocal induced current at
    the cell-average (G = 0) Fourier component — the physical validation of the
    S-metric PAW δu (the Phase-2 twin of the norm-conserving KB closure, #330).

    For the single-branch response pair (u_nk, δu_{n,k+q}) of the polarization
    ``e_pol`` the exact identity is

        q · mean[j_kin + j_nl] = mean s + (augmentation current + truncation),

    with ``s`` the smooth cross density of the occupied states and the
    Sternheimer RHS χ = −P_c^S O u (:func:`_drho_q` of χ), and the KB nonlocal
    current carrying the SCREENED D (``ctx.dscr``). Returns
    ``{"source", "q_j_kin", "q_j_nl"}`` (all complex, the G = 0 values), so
    ``q_j_kin`` alone is broken by the [V_NL, e^{-iqr}] commutator while
    ``q_j_kin + q_j_nl`` closes to ``mean s`` up to the (ecut-vanishing) basis
    truncation and the on-site augmentation-charge current — the small remainder
    that quantifies the PAW augmentation piece. The nonlocal divergence uses the
    exact mean-value form q·mean[j_nl] = Σ f w/Ω ⟨u|(V_NL^{k+q} − V_NL^{k})|δu⟩
    on the k-referenced union Millers (no line integral needed for the
    divergence)."""
    system = ctx.system
    grid = system.grid
    shape = grid.shape
    vol = grid.volume
    b = reciprocal_cell(grid.cell)
    q_cart = torch.as_tensor(sol.q_frac @ b, dtype=RDTYPE)
    e_t = torch.as_tensor(np.asarray(e_pol, dtype=float), dtype=RDTYPE).to(CDTYPE)
    dpsi = torch.einsum("m,mkng->kng", e_t, sol.dpsi)
    o_pol = torch.einsum("m,mkng->kng", e_t, sol.o_c)
    ph = sol.ph
    assert ph is not None

    # source: smooth cross density of u_nk and χ = −P_c^S O u
    hkq, _ = ctx.build_hkq(torch.as_tensor(sol.jidx, dtype=torch.long))
    s_occ_kq = hkq.s_apply(sol.c_occ_kq)
    ov = torch.einsum("kng,kbg->kbn", sol.c_occ_kq.conj(), o_pol)
    chi = -(o_pol - torch.einsum("kbn,kng->kbg", ov, s_occ_kq))
    source = complex(_drho_q(sol, chi, ctx.bk, shape, ph, vol).mean())

    # kinetic current field mean (the induced_current_q j_para assembly)
    npts = shape[0] * shape[1] * shape[2]
    u_box = g_to_r_b(sol.c_occ_k, ctx.bk, shape)
    du_box = g_to_r_b(dpsi, sol.bk_kq, shape) * ph.conj()
    q_j_kin = 0.0 + 0.0j
    for mu in range(3):
        wu = (2.0 * HBAR2_2M) * ctx.bk.kpg[:, None, :, mu] * sol.c_occ_k
        wdu = (2.0 * HBAR2_2M) * sol.bk_kq.kpg[:, None, :, mu] * dpsi
        wu_box = g_to_r_b(wu, ctx.bk, shape)
        wdu_box = g_to_r_b(wdu, sol.bk_kq, shape) * ph.conj()
        jk = (2.0 / vol) * 0.5 * (
            u_box.conj() * wdu_box + wu_box.conj() * du_box).sum(dim=1)
        jm = torch.einsum("k,k...->", sol.weights.to(CDTYPE), jk) / npts
        q_j_kin = q_j_kin + complex(q_cart[mu].to(CDTYPE) * jm)

    # nonlocal divergence q·mean[j_nl] via union-sphere matrix elements (dscr)
    dscr = ctx.dscr.to(CDTYPE)
    n1, n2, n3 = shape
    q_j_nl = 0.0 + 0.0j
    for ik, sph in enumerate(system.spheres):
        kb = _overlap_kbprojectors(system, sph)  # KBProjectors carrying paws β
        m_k = sph.miller.numpy()
        m_d = system.spheres[int(sol.jidx[ik])].miller.numpy() - sol.g0[ik][None, :]
        union = np.unique(np.concatenate([m_k, m_d], axis=0), axis=0)
        keys_u = _pack_miller(union)
        order = np.argsort(keys_u)
        ks = keys_u[order]
        u_pos = order[np.searchsorted(ks, _pack_miller(m_k))]
        d_pos = order[np.searchsorted(ks, _pack_miller(m_d))]
        nu, nocc = union.shape[0], dpsi.shape[1]
        u_u = torch.zeros(nocc, nu, dtype=CDTYPE)
        d_u = torch.zeros(nocc, nu, dtype=CDTYPE)
        u_u[:, torch.as_tensor(u_pos)] = sol.c_occ_k[ik, :, : len(u_pos)]
        d_u[:, torch.as_tensor(d_pos)] = dpsi[ik, :, : len(d_pos)]
        kbu = KBProjectors(
            g_cart=torch.as_tensor(union.astype(float) @ b, dtype=RDTYPE),
            tau=kb.tau, dij_full=kb.dij_full, col_atom=kb.col_atom,
            col_chan=kb.col_chan, col_lm=kb.col_lm, channels=kb.channels,
            lmax=kb.lmax, volume=kb.volume)
        k_cart = sph.k_cart.to(RDTYPE)
        p_k, p_q = kbu.p(k_cart), kbu.p(k_cart + q_cart)

        def vnl(p: Tensor, c: Tensor) -> Tensor:
            if not p.shape[0]:
                return torch.zeros_like(c)
            beta = torch.einsum("jg,bg->bj", p.conj(), c)
            return torch.einsum("ij,bj,ig->bg", dscr, beta, p)  # noqa: B023

        diff = vnl(p_q, d_u) - vnl(p_k, d_u)
        me = (u_u.conj() * diff).sum()
        q_j_nl = q_j_nl + float(sol.weights[ik]) * (2.0 / vol) * complex(me)

    return {"source": source, "q_j_kin": q_j_kin, "q_j_nl": q_j_nl}


# --------------------------------------------------------------------------- #
# analytic ∂/∂q shielding: the O(q²) finite-q error removed (milestone 10)     #
# --------------------------------------------------------------------------- #


def _resolvent_apply(uc: Tensor, wc: Tensor, eps: Tensor, x: Tensor) -> Tensor:
    """y_j = Σ_{m∈cond} |u_m⟩⟨u_m|x_j⟩/(ε_j − ε_m), column-wise.

    ``x`` (npw, ncol) with per-column reference energies ``eps`` (ncol,);
    (uc, wc) the conduction eigenpairs. The conduction-projected Green's
    function G(ε)P_c of the dense eigensystem — gauge invariant under
    degenerate conduction rotations (only the subspace enters)."""
    a = uc.mH @ x
    return uc @ (a / (eps[None, :] - wc[:, None]).to(a.dtype))


def _d2_mats(
    mat_fn: Callable[[Tensor], Tensor], kc: Tensor, q_hat: Tensor
) -> list[Tensor]:
    """Mixed second derivatives M_μ = q̂·∇_k(∂mat/∂k_μ) at kc (3 × (npw, npw))
    of any traceable dense matrix ``mat_fn(k)`` (H or S).

    By nested forward-mode ``torch.func.jvp`` (equality of mixed partials:
    M_μ = ∂_μ(q̂·∇mat), so the inner pass is along q̂ and the outer along ê_μ).
    Complex outputs compose cleanly; no eigendecomposition is differentiated.
    The overlap-second-derivative ∂²S/∂k² is this same machinery on the
    traceable ``S(k)`` (scoping-measured FD-matched at ~1e-9)."""

    def w_fn(k: Tensor) -> Tensor:
        return torch.func.jvp(mat_fn, (k,), (q_hat,))[1]

    out = []
    for mu in range(3):
        e_mu = torch.zeros(3, dtype=RDTYPE, device=kc.device)
        e_mu[mu] = 1.0
        out.append(torch.func.jvp(w_fn, (kc,), (e_mu,))[1])
    return out


def _d2h_mats(hk: BlochHK, kc: Tensor, q_hat: Tensor) -> list[Tensor]:
    """Mixed second derivatives M_μ = q̂·∇_k(∂H/∂k_μ) at kc (3 × (npw, npw))."""
    return _d2_mats(hk.h, kc, q_hat)


def _gen_velocity(
    v: list[Tensor], dsv: list[Tensor], x: Tensor, e: Tensor, coef: Tensor
) -> Tensor:
    """Generalized velocity Σ_μ coef_μ (∂H/∂k_μ x − ε ∂S/∂k_μ x), with ``e`` the
    per-column reference energy of ``x`` (broadcast over its columns). ``v`` is
    the screened-D ∂H/∂k, ``dsv`` the ∂S/∂k; ``coef`` picks the direction
    (q̂ for the covariant k-derivative, ê_pol for the perturbation O)."""
    acc = coef[0] * (v[0] @ x - (dsv[0] @ x) * e)
    acc = acc + coef[1] * (v[1] @ x - (dsv[1] @ x) * e)
    return acc + coef[2] * (v[2] @ x - (dsv[2] @ x) * e)


def _resolvent_apply_s(
    uc: Tensor, wc: Tensor, eps: Tensor, x: Tensor, s_mat: Tensor
) -> Tensor:
    """S-metric conduction resolvent y_j = Σ_{m∈cond} |u_m⟩⟨u_m|S|x_j⟩/(ε_j−ε_m).

    The generalized-eigenproblem twin of :func:`_resolvent_apply`: the
    conduction columns ``uc`` are S-orthonormal (uc.mH @ S @ uc = I) and the
    projection onto them uses the S inner product ⟨u_m|S|x⟩ = (uc.mH S x), so
    G(ε)P_c is the S-orthogonal conduction Green's function. Reduces to
    :func:`_resolvent_apply` exactly when S = I."""
    a = uc.mH @ (s_mat @ x)
    return uc @ (a / (eps[None, :] - wc[:, None]).to(a.dtype))


# --------------------------------------------------------------------------- #
# USPP/PAW S-metric dense context for the analytic ∂/∂q shielding (PR-A)        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BlochHKS:
    """Dense differentiable H(k) AND overlap S(k) at a frozen USPP/PAW ground
    state — the S-metric generalization of :class:`~gradwave.postscf.kgeometry.
    BlochHK` used by the analytic shielding route.

        H(k) = V̂_loc(G−G') + kinetic + Σ_ij Dscr_ij |β_i(k)⟩⟨β_j(k)|
        S(k) = I                       + Σ_ij q_ij   |β_i(k)⟩⟨β_j(k)|

    with ``Dscr`` the SCREENED descreened D (``uspp_frozen.screened_dscr``) and
    ``q`` the ∫Q augmentation overlap weights. Both are Hermitian and traceable
    in Cartesian k through the shared β source ``kb`` (a
    :class:`~gradwave.postscf.kgeometry.KBProjectors`), so ∂H/∂k, ∂S/∂k, ∂²H/∂k²
    and ∂²S/∂k² all come from the same forward-mode AD the NC route uses for
    ∂²H (no new augmentation second-derivative — scoping §3).

    Two builders: :meth:`from_uspp_ctx` (a real USPP/PAW result, via the shipped
    :class:`USPPResponseCtx`) and :meth:`from_nc` (a norm-conserving result with
    q = 0 ⇒ S = I, Dscr = D — the NC-limit reduction gate, exercising every
    S-metric code path while reducing to the plain-NC arithmetic)."""

    b: Tensor  # (3,3) reciprocal-cell rows [Å⁻¹]
    k_ref_cart: Tensor  # (3,) reference (sphere-defining) k [Å⁻¹]
    miller: Tensor  # (npw, 3) int64 Miller indices of the fixed sphere
    g_cart: Tensor  # (npw, 3) Cartesian G of the fixed sphere [Å⁻¹]
    vmat: Tensor  # (npw, npw) complex V̂_loc(G−G') [eV]
    kb: KBProjectors  # β source (NC or USPP overlap projectors)
    dscr: Tensor  # (nproj, nproj) screened D (= D for NC) [eV]
    qint: Tensor  # (nproj, nproj) ∫Q overlap weights (= 0 for NC)
    volume: float

    @property
    def npw(self) -> int:
        return int(self.g_cart.shape[0])

    @classmethod
    def from_nc(cls, res: SCFResult, k_frac: Sequence[float] | np.ndarray) -> BlochHKS:
        """NC-limit gate context: q = 0 ⇒ S = I, Dscr = D. Reuses
        :meth:`BlochHK.from_scf` so H(k) is byte-identical to the NC route's."""
        hk = BlochHK.from_scf(res, k_frac)
        kb = KBProjectors(
            g_cart=hk.g_cart, tau=hk.tau, dij_full=hk.dij_full,
            col_atom=hk.col_atom, col_chan=hk.col_chan, col_lm=hk.col_lm,
            channels=hk.channels, lmax=hk.lmax, volume=hk.volume,
        )
        return cls(
            b=hk.b, k_ref_cart=hk.k_ref_cart, miller=hk.miller, g_cart=hk.g_cart,
            vmat=hk.vmat, kb=kb, dscr=hk.dij_full,
            qint=torch.zeros_like(hk.dij_full), volume=hk.volume,
        )

    @classmethod
    def from_uspp_ctx(
        cls, ctx: USPPResponseCtx, k_frac: Sequence[float] | np.ndarray
    ) -> BlochHKS:
        """Real USPP/PAW context from the shipped :class:`USPPResponseCtx`
        (``build_uspp_response_ctx``): β + ∫Q on the sphere at ``k_frac`` from
        ``_overlap_kbprojectors``, the screened D and frozen local potential
        from the frozen ground state. The screened-D column order matches
        ``_overlap_kbprojectors`` (both are the (atom, channel, m) order the
        ctx built ``dscr`` in), so the two share one projector layout."""
        system = ctx.system
        grid = system.grid
        sphere = build_gsphere(grid, system.ecut, np.asarray(k_frac, dtype=float))
        kb = _overlap_kbprojectors(system, sphere)
        vhat = r_to_g(ctx.v_eff_r.detach().cpu().to(CDTYPE)).reshape(-1)
        vmat = vhat[_toeplitz_index(sphere.miller, grid.shape)]
        return cls(
            b=torch.as_tensor(reciprocal_cell(grid.cell), dtype=RDTYPE),
            k_ref_cart=sphere.k_cart.to(RDTYPE),
            miller=sphere.miller,
            g_cart=kb.g_cart,
            vmat=vmat,
            kb=kb,
            dscr=ctx.dscr.to(RDTYPE),
            qint=kb.dij_full,
            volume=float(grid.volume),
        )

    def k_cart(self, k_frac: Sequence[float] | np.ndarray | Tensor) -> Tensor:
        kf = (
            k_frac.to(RDTYPE)
            if isinstance(k_frac, Tensor)
            else torch.as_tensor(np.asarray(k_frac, dtype=float), dtype=RDTYPE)
        )
        return kf @ self.b

    def h(self, k_cart: Tensor) -> Tensor:
        """Dense Hermitian H(k) (npw, npw) [eV], differentiable in Cartesian k."""
        kpg = self.g_cart + k_cart
        kin = HBAR2_2M * (kpg * kpg).sum(-1)
        hmat = self.vmat + torch.diag_embed(kin.to(CDTYPE))
        if self.kb.dij_full.shape[0]:
            p = self.kb.p(k_cart)
            hmat = hmat + p.mT @ (self.dscr.to(CDTYPE) @ p.conj())
        return hmat

    def s(self, k_cart: Tensor) -> Tensor:
        """Dense Hermitian overlap S(k) (npw, npw), differentiable in k.
        S = I + Σ_ij q_ij |β_i⟩⟨β_j| (= I exactly when q = 0)."""
        npw = self.npw
        eye = torch.eye(npw, dtype=CDTYPE, device=self.g_cart.device)
        if self.kb.dij_full.shape[0]:
            p = self.kb.p(k_cart)
            eye = eye + p.mT @ (self.qint.to(CDTYPE) @ p.conj())
        return eye


def _geigh_and_dhds(
    hks: BlochHKS, k_cart: Tensor
) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
    """(ε, U, [∂H/∂k_μ], [∂S/∂k_μ]) of the generalized eigenproblem H U = S U ε.

    The eigensolve is Cholesky-whitened (S = L Lᴴ; eigh of L⁻¹ H L⁻ᴴ) so the
    returned U is S-ORTHONORMAL (Uᴴ S U = I) — under ``no_grad`` exactly like
    :func:`_eigh_and_dh` (the eigendecomposition is never differentiated; the
    S-orthonormal back-transform does not touch the forward ∂/∂k passes). The
    velocity matrices ∂H/∂k (screened D) and ∂S/∂k come from one forward-mode
    dual pass per direction through the traceable ``h``/``s``. With S = I,
    Cholesky(I) = I and this reduces to :func:`_eigh_and_dh` with ∂S/∂k = 0."""
    k = k_cart.detach().to(RDTYPE)
    with torch.no_grad():
        hmat = hks.h(k)
        smat = hks.s(k)
        ell = torch.linalg.cholesky(smat)
        ell_inv = torch.linalg.solve_triangular(
            ell, torch.eye(hks.npw, dtype=CDTYPE, device=k.device), upper=False
        )
        a = ell_inv @ hmat @ ell_inv.mH
        a = 0.5 * (a + a.mH)
        w, y = torch.linalg.eigh(a)
        u = ell_inv.mH @ y  # S-orthonormal eigenvectors
    dh, ds = [], []
    for mu in range(3):
        tangent = torch.zeros(3, dtype=RDTYPE, device=k.device)
        tangent[mu] = 1.0
        dh.append(torch.func.jvp(hks.h, (k,), (tangent,))[1])
        ds.append(torch.func.jvp(hks.s, (k,), (tangent,))[1])
    return w, u, dh, ds


@dataclass(frozen=True)
class _DenseKCtx:
    """Frozen per-k dense context of the analytic ∂/∂q route.

    NC route: ``hk`` a :class:`~gradwave.postscf.kgeometry.BlochHK`, ``ds``/``s0``
    ``None`` (S = I implicit). S-metric (USPP/PAW) route: ``hk`` a
    :class:`BlochHKS`, ``v`` the screened-D ∂H/∂k, ``ds`` the ∂S/∂k, ``s0`` the
    dense S(kc), ``u`` S-orthonormal."""

    hk: BlochHK | BlochHKS
    kc: Tensor  # (3,) Cartesian k [Å⁻¹]
    w: Tensor  # (npw,) eigenvalues [eV]
    u: Tensor  # (npw, npw) eigenvector columns
    v: list[Tensor]  # 3 × (npw, npw): ∂H/∂k_μ at kc
    kpg: Tensor  # (npw, 3) Cartesian k+G
    flat: Tensor  # (npw,) FFT-box flat index of the sphere Millers
    weight: float  # k-weight (mesh weights sum to 1)
    ds: list[Tensor] | None = None  # 3 × (npw, npw): ∂S/∂k_μ (None ⇒ S = I)
    s0: Tensor | None = None  # (npw, npw) S(kc) (None ⇒ S = I)


class ShieldingDq:
    """Analytic q → 0 machinery for the bare shielding: the exact s-derivative
    of the single-branch induced current S(s) of O_{sq̂,pol} at s = 0.

    Per mesh k a dense fixed-sphere context: eigh of H(k) (never
    differentiated), velocity matrices ∂H/∂k by forward-mode AD, and — per q̂
    direction — the mixed second derivatives M_μ = q̂·∇(∂H/∂k_μ) by nested
    jvp. The response derivative comes from differentiating the Sternheimer
    characterization of δu(s) = G_{k+sq̂}(ε_nk) P_c(s) O(s) u_nk implicitly:

        δu'_n = R(ε_n)[W δu⁰_n − Σ_j d̃u_j⟨u_j|O₀u_n⟩ + ½M_pol u_n]
                − Σ_j u_j⟨d̃u_j|δu⁰_n⟩,

    with R(ε) = G(ε)P_c the conduction resolvent, W = q̂·v, d̃u_j = R(ε_j)Wu_j
    the covariant k-derivative (cross-gap denominators only — degenerate
    occupied or conduction manifolds are safe, the ``qgt_sos`` pattern), and
    ½M_pol = O'(0) from O(s) = ½Σ_μ pol_μ(v_μ(0) + v_μ(s)). The field
    derivative adds the explicit operator derivatives: the δu-side kinetic
    weight 2·HBAR2_2M·(k+sq̂+G)_μ contributes 2·HBAR2_2M·q̂_μ, and the
    line-averaged KB pair velocity Ā_μ(s) = ∫₀¹dt ∂V_NL/∂k_μ|_{k+tsq̂}
    contributes Ā'_μ(0) = ½ q̂·∇(∂V_NL/∂k_μ) on BOTH sides of the cross
    density; the diamagnetic term is s-independent. Everything on the fixed
    k-sphere — the s → 0 limit of the finite-q machinery, where the k+q
    sphere and the umklapp embedding reduce to the identity.
    """

    def __init__(
        self,
        res: SCFResult,
        *,
        k_frac: np.ndarray | Sequence[np.ndarray] | None = None,
        uspp: USPPResponseCtx | str | None = None,
        chunk_k: int | None = None,
    ) -> None:
        """Build the per-k dense contexts (eigh + ∂H/∂k) at each mesh k.

        ``k_frac`` overrides ``system.spheres`` with an explicit set of k-points
        — the symmetry-reduced wedge (or the union of per-axis wedges) of the
        opt-in reduced path (:func:`sigma_shielding_dq`). Contexts built with an
        explicit ``k_frac`` carry no meaningful k-weight (``branch_fields_axis``
        is then always passed the per-axis star weights); default weights are the
        full-mesh ``system.kweights`` for the unreduced route.

        ``uspp`` selects the S-metric (ultrasoft/PAW) build for the SMOOTH
        (valence) shielding: a :class:`USPPResponseCtx` from
        :func:`build_uspp_response_ctx` (real USPP/PAW), or the string
        ``"nc-gate"`` (build the S-metric context from ``res``'s NC pseudo with
        q = 0 ⇒ S = I, Dscr = D — the machine-precision reduction gate). Each
        per-k context then carries the screened-D ∂H/∂k, ∂S/∂k and the dense
        S(kc); the generalized eigenproblem H U = S U ε gives S-orthonormal U.
        With ``uspp=None`` the plain norm-conserving (S = I) route is unchanged.
        """
        _guard(res)
        self.res = res
        self.uspp = uspp
        ctx = uspp if isinstance(uspp, USPPResponseCtx) else None
        system = ctx.system if ctx is not None else res.system
        self.shape: tuple[int, int, int] = system.grid.shape
        self.volume = float(system.grid.volume)
        self.nocc = insulator_window(
            res.occupations, 2.0, "insulating occupations required"
        )
        n1, n2, n3 = self.shape
        nbox = np.array([n1, n2, n3])
        if k_frac is None:
            kfs = [sph.k_frac for sph in system.spheres]
            weights = [float(system.kweights[ik]) for ik in range(len(kfs))]
        else:
            kfs = [np.asarray(k, dtype=float) for k in np.asarray(k_frac, dtype=float)]
            weights = [float("nan")] * len(kfs)
        self._kfs = kfs
        self._weights = weights
        self._ctx = ctx
        self._nbox = nbox
        self._chunk_k = None if chunk_k is None else int(chunk_k)
        # Eager (chunk_k=None) holds every mesh k's dense context at once — the
        # historical path, byte-for-byte. Streaming (chunk_k set) defers the
        # per-k build to branch_fields_axis, which builds and frees one context
        # at a time, so the peak is O(1) dense (npw×npw) velocity matrices
        # rather than nk of them — the memory route for many-atom cells. Each
        # context is a pure function of its k, so the streamed and eager fields
        # are bit-identical (the per-axis rebuild is the only cost).
        self.ks: list[_DenseKCtx] | None = (
            None if self._chunk_k is not None
            else [self._build_ctx(kf, wk)
                  for kf, wk in zip(kfs, weights, strict=True)])

    def _build_ctx(self, kf: np.ndarray, wk: float) -> _DenseKCtx:
        """One frozen dense per-k context: eigh of H(k) (never differentiated)
        and ∂H/∂k, plus the S-metric ∂S/∂k and S(kc) on the USPP/PAW route.
        A pure function of ``kf`` — building it lazily (streaming) yields the
        same w/u/v as the eager build."""
        res, ctx, uspp = self.res, self._ctx, self.uspp
        _n1, n2, n3 = self.shape
        if uspp is None:
            hk: BlochHK | BlochHKS = BlochHK.from_scf(res, kf)
            kc = hk.k_ref_cart
            w, u, dh = _eigh_and_dh(hk.h, kc)
            ds: list[Tensor] | None = None
            s0: Tensor | None = None
        else:
            hk = (
                BlochHKS.from_nc(res, kf)
                if ctx is None
                else BlochHKS.from_uspp_ctx(ctx, kf)
            )
            kc = hk.k_ref_cart
            w, u, dh, ds = _geigh_and_dhds(hk, kc)
            s0 = hk.s(kc)
        mod = hk.miller.numpy() % self._nbox
        flat = torch.as_tensor(
            (mod[:, 0] * n2 + mod[:, 1]) * n3 + mod[:, 2], dtype=torch.int64
        )
        return _DenseKCtx(hk=hk, kc=kc, w=w, u=u, v=dh, kpg=hk.g_cart + kc,
                          flat=flat, weight=wk, ds=ds, s0=s0)

    def branch_fields_axis(
        self,
        q_hat: np.ndarray | list[float],
        pols: Sequence[np.ndarray],
        *,
        k_indices: Sequence[int] | None = None,
        weights: Sequence[float] | None = None,
        fold: _LittleGroupFold | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """[(S⁰, ∂S/∂s), …] per polarization for the direction q̂: the
        BZ-weighted single-branch conserved current field (kinetic + KB
        nonlocal + diamagnetic, each (3, n1, n2, n3)) of λ·O_{sq̂,pol} and
        its exact s-derivative at s = 0. The per-axis second derivatives and
        covariant k-derivatives are computed once and shared across ``pols``.

        ``k_indices`` / ``weights`` restrict the BZ sum to a subset of the
        stored contexts with explicit (star-multiplicity) weights — the
        little-group-of-q wedge of the opt-in reduced path. ``fold`` then
        reconstructs the full-BZ field from that wedge by the little-group
        orbit action on the current⊗polarization tensor field (see
        :class:`_LittleGroupFold`); the two must be paired (the wedge k-sum is
        NOT the full field on its own). With all three ``None`` this is the
        exact full-mesh route (every context, its own k-weight, no fold)."""
        qh = torch.as_tensor(np.asarray(q_hat, dtype=float), dtype=RDTYPE)
        out = [
            (torch.zeros(3, *self.shape, dtype=CDTYPE),
             torch.zeros(3, *self.shape, dtype=CDTYPE))
            for _ in pols
        ]
        n_all = len(self.ks) if self.ks is not None else len(self._kfs)
        idxs = list(range(n_all)) if k_indices is None else list(k_indices)
        if weights is not None:
            ws = list(weights)
        elif self.ks is not None:
            ws = [self.ks[i].weight for i in idxs]
        else:
            ws = [self._weights[i] for i in idxs]
        # Eager: index the prebuilt contexts. Streaming (self.ks is None): build
        # each dense context here and let it fall out of scope after its k, so
        # only one is alive at a time.
        for i, w_k in zip(idxs, ws, strict=True):
            dk = (self.ks[i] if self.ks is not None
                  else self._build_ctx(self._kfs[i], self._weights[i]))
            self._accumulate_k(dk, w_k, qh, pols, out)
        return self._finish_axis(out, pols, fold)

    def _accumulate_k(
        self,
        dk: _DenseKCtx,
        w_k: float,
        qh: Tensor,
        pols: Sequence[np.ndarray],
        out: list[tuple[Tensor, Tensor]],
    ) -> None:
        """Add one k-context's single-branch current contribution to the per-pol
        (S⁰, ∂S/∂s) accumulators ``out``. ``qh`` is the RDTYPE q̂ tensor; ``pols``
        the polarization list; ``w_k`` the k-weight. Pure accumulation (mutates
        ``out``) — the per-k body factored out of :meth:`branch_fields_axis` so a
        single context can be contributed to many axes (see
        :meth:`branch_fields_all_axes`)."""
        nocc = self.nocc
        f_occ = 2.0
        uo, wo = dk.u[:, :nocc], dk.w[:nocc]
        uc, wc = dk.u[:, nocc:], dk.w[nocc:]
        pref = w_k * f_occ / self.volume
        if dk.ds is None:
            # ---- norm-conserving (S = I) route (unchanged) ----
            m_mats = _d2h_mats(cast("BlochHK", dk.hk), dk.kc, qh)
            wmat = torch.einsum("m,mij->ij", qh.to(CDTYPE), torch.stack(dk.v))
            dtu = _resolvent_apply(uc, wc, wo, wmat @ uo)
            u_box = g_to_r(uo.mT, dk.flat, self.shape)
            vu = [dk.v[mu] @ uo for mu in range(3)]
            vu_box = [g_to_r(x.mT, dk.flat, self.shape) for x in vu]
            for ip, pol in enumerate(pols):
                ep = np.asarray(pol, dtype=float)
                o0u = ep[0] * vu[0] + ep[1] * vu[1] + ep[2] * vu[2]
                du0 = _resolvent_apply(uc, wc, wo, o0u)
                me_u = 0.5 * (
                    ep[0] * (m_mats[0] @ uo)
                    + ep[1] * (m_mats[1] @ uo)
                    + ep[2] * (m_mats[2] @ uo)
                )
                inner = wmat @ du0 - dtu @ (uo.mH @ o0u) + me_u
                du1 = _resolvent_apply(uc, wc, wo, inner) - uo @ (dtu.mH @ du0)
                du0_box = g_to_r(du0.mT, dk.flat, self.shape)
                du1_box = g_to_r(du1.mT, dk.flat, self.shape)
                s0_acc, ds_acc = out[ip]
                for mu in range(3):
                    kq = HBAR2_2M * float(qh[mu])
                    vdu0_box = g_to_r((dk.v[mu] @ du0).mT, dk.flat, self.shape)
                    vdu1_box = g_to_r((dk.v[mu] @ du1).mT, dk.flat, self.shape)
                    # δu-side operator derivative ½M_μ + κq̂_μ (kinetic + ½Ā'),
                    # u-side ½M_μ − κq̂_μ (½Ā' only), κ = HBAR2_2M
                    dop_du0 = 0.5 * (m_mats[mu] @ du0) + kq * du0
                    dop_u = 0.5 * (m_mats[mu] @ uo) - kq * uo
                    dop_du0_box = g_to_r(dop_du0.mT, dk.flat, self.shape)
                    dop_u_box = g_to_r(dop_u.mT, dk.flat, self.shape)
                    s0_acc[mu] += pref * 0.5 * (
                        u_box.conj() * vdu0_box + vu_box[mu].conj() * du0_box
                    ).sum(dim=0)
                    ds_acc[mu] += pref * 0.5 * (
                        u_box.conj() * (dop_du0_box + vdu1_box)
                        + dop_u_box.conj() * du0_box
                        + vu_box[mu].conj() * du1_box
                    ).sum(dim=0)
            return
        # ---- S-metric (USPP/PAW smooth) route ----
        # The generalized eigenproblem's covariant velocity and perturbation
        # use v_gen = ∂H/∂k − ε ∂S/∂k (per-column reference energy ε), the
        # S-orthogonal resolvent projects with ⟨u_m|S|·⟩, and O'(0) carries
        # the −ε ∂²S piece; the PHYSICAL current operator stays the plain
        # screened-D ∂H/∂k (scoping L2/L3/L4/L6/L7). All reduce to the NC
        # arithmetic above when S = I, ∂S = 0 (the NC-limit gate).
        hks = cast("BlochHKS", dk.hk)
        dsv = dk.ds
        s0m = cast("Tensor", dk.s0)
        wo_c = wo.to(CDTYPE)
        qh_c = qh.to(CDTYPE)
        m_h = _d2_mats(hks.h, dk.kc, qh)  # ∂²H/∂k²
        m_s = _d2_mats(hks.s, dk.kc, qh)  # ∂²S/∂k²

        dtu = _resolvent_apply_s(
            uc, wc, wo, _gen_velocity(dk.v, dsv, uo, wo_c, qh_c), s0m
        )
        u_box = g_to_r(uo.mT, dk.flat, self.shape)
        vcur = [dk.v[mu] @ uo for mu in range(3)]  # current op (screened-D ∂H/∂k)
        vcur_box = [g_to_r(x.mT, dk.flat, self.shape) for x in vcur]
        for ip, pol in enumerate(pols):
            ep = torch.as_tensor(np.asarray(pol, dtype=float), dtype=CDTYPE)
            o0u = _gen_velocity(dk.v, dsv, uo, wo_c, ep)  # perturbation O u
            du0 = _resolvent_apply_s(uc, wc, wo, o0u, s0m)
            # O'(0) u = ½ Σ ep_μ (M^H_μ uo − ε M^S_μ uo)
            me_u = 0.5 * (
                ep[0] * (m_h[0] @ uo - (m_s[0] @ uo) * wo_c)
                + ep[1] * (m_h[1] @ uo - (m_s[1] @ uo) * wo_c)
                + ep[2] * (m_h[2] @ uo - (m_s[2] @ uo) * wo_c)
            )
            inner = (
                _gen_velocity(dk.v, dsv, du0, wo_c, qh_c)
                - dtu @ (uo.mH @ (s0m @ o0u))
                + me_u
            )
            du1 = _resolvent_apply_s(uc, wc, wo, inner, s0m) - uo @ (
                dtu.mH @ (s0m @ du0)
            )
            du0_box = g_to_r(du0.mT, dk.flat, self.shape)
            du1_box = g_to_r(du1.mT, dk.flat, self.shape)
            s0_acc, ds_acc = out[ip]
            for mu in range(3):
                kq = HBAR2_2M * float(qh[mu])
                vdu0_box = g_to_r((dk.v[mu] @ du0).mT, dk.flat, self.shape)
                vdu1_box = g_to_r((dk.v[mu] @ du1).mT, dk.flat, self.shape)
                # current-operator s-derivative: ½M^H_μ (screened-D ∂H/∂k,
                # NOT generalized) + κq̂_μ (kinetic) on δu, −κq̂_μ on u.
                dop_du0 = 0.5 * (m_h[mu] @ du0) + kq * du0
                dop_u = 0.5 * (m_h[mu] @ uo) - kq * uo
                dop_du0_box = g_to_r(dop_du0.mT, dk.flat, self.shape)
                dop_u_box = g_to_r(dop_u.mT, dk.flat, self.shape)
                s0_acc[mu] += pref * 0.5 * (
                    u_box.conj() * vdu0_box + vcur_box[mu].conj() * du0_box
                ).sum(dim=0)
                ds_acc[mu] += pref * 0.5 * (
                    u_box.conj() * (dop_du0_box + vdu1_box)
                    + dop_u_box.conj() * du0_box
                    + vcur_box[mu].conj() * du1_box
                ).sum(dim=0)

    def _finish_axis(
        self,
        out: list[tuple[Tensor, Tensor]],
        pols: Sequence[np.ndarray],
        fold: _LittleGroupFold | None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Add the s-independent diamagnetic (Lamb) term 2κ ê ⊗ ρ to each pol's
        S⁰ accumulator and apply the little-group ``fold`` (if any), returning the
        finished [(S⁰, ∂S/∂s), …]. Shared by :meth:`branch_fields_axis` and
        :meth:`branch_fields_all_axes` so the post-processing stays identical."""
        rho = self.res.rho if self.res.rho.dim() == 3 else self.res.rho[0]
        for ip, pol in enumerate(pols):
            ep_t = torch.as_tensor(np.asarray(pol, dtype=float), dtype=RDTYPE)
            out[ip][0].add_(torch.einsum(
                "m,ijk->mijk", ep_t.to(CDTYPE),
                (2.0 * HBAR2_2M) * rho.to(CDTYPE)))
        if fold is not None:
            out = fold.apply(out)
        return out

    def branch_fields_all_axes(
        self,
        specs: Sequence[
            tuple[
                np.ndarray | list[float],
                Sequence[np.ndarray],
                Sequence[int] | None,
                Sequence[float] | None,
                _LittleGroupFold | None,
            ]
        ],
    ) -> list[list[tuple[Tensor, Tensor]]]:
        """specs: list of (q_hat, pols, k_indices, weights, fold). Build each
        union-k context ONCE and contribute it to every axis that includes it,
        then free it — the single-pass twin of calling :meth:`branch_fields_axis`
        per axis, so streaming (chunk_k) rebuilds nothing. Returns a list (one per
        spec) of the :meth:`branch_fields_axis` result."""
        n_all = len(self.ks) if self.ks is not None else len(self._kfs)
        pols_a = [pols for (_q, pols, _ki, _w, _f) in specs]
        qh_a = [
            torch.as_tensor(np.asarray(q_hat, dtype=float), dtype=RDTYPE)
            for (q_hat, _p, _ki, _w, _f) in specs
        ]
        out_a: list[list[tuple[Tensor, Tensor]]] = [
            [
                (torch.zeros(3, *self.shape, dtype=CDTYPE),
                 torch.zeros(3, *self.shape, dtype=CDTYPE))
                for _ in pols
            ]
            for (_q, pols, _ki, _w, _f) in specs
        ]
        # Per-axis membership: union-context index -> k-weight. Same weight rule
        # as branch_fields_axis (explicit star weights on the reduced wedge, else
        # the stored per-k weight — self._weights[i] == self.ks[i].weight).
        mem_a: list[dict[int, float]] = []
        for (_q, _p, k_indices, weights, _f) in specs:
            if k_indices is None:
                mem = {i: self._weights[i] for i in range(n_all)}
            else:
                k_idx = list(k_indices)
                mem = {
                    k_idx[j]: (
                        float(weights[j]) if weights is not None
                        else self._weights[k_idx[j]]
                    )
                    for j in range(len(k_idx))
                }
            mem_a.append(mem)
        # ONE outer pass over the union of contexts: build (or index) each once,
        # contribute to every axis that includes it, then let it fall out of
        # scope. Streaming therefore builds each context exactly once total, not
        # once per axis.
        for i in range(n_all):
            dk = (self.ks[i] if self.ks is not None
                  else self._build_ctx(self._kfs[i], self._weights[i]))
            for a in range(len(specs)):
                if i in mem_a[a]:
                    self._accumulate_k(dk, mem_a[a][i], qh_a[a], pols_a[a], out_a[a])
        return [
            self._finish_axis(out_a[a], pols_a[a], specs[a][4])
            for a in range(len(specs))
        ]


def _branch_field_fixed_sphere(
    eng: ShieldingDq,
    ik: int,
    q_hat: np.ndarray | list[float],
    e_pol: np.ndarray | list[float],
    s: float,
    nquad: int = 8,
) -> Tensor:
    """Validation twin: the single-k single-branch conserved current field
    S(s) (3, n1, n2, n3) of O_{sq̂,pol} at FINITE s on the FIXED k-sphere —
    dense eigh at k+sq̂ for the resolvent, the two-point symmetrized velocity
    operator, k / k+sq̂ kinetic weights, Gauss–Legendre line-averaged KB Ā(s);
    no diamagnetic term (s-independent). Central differences of this twin pin
    every term of :meth:`ShieldingDq.branch_fields_axis`'s analytic ∂S/∂s;
    at s = 0 it reproduces the S⁰ assembly (minus dia) exactly."""
    assert eng.ks is not None, "the finite-s twin needs the eager (chunk_k=None) build"
    dk = eng.ks[ik]
    nocc = eng.nocc
    qh = torch.as_tensor(np.asarray(q_hat, dtype=float), dtype=RDTYPE)
    ep = np.asarray(e_pol, dtype=float)
    # The finite-s twin is the NC (S = I) validation reference only.
    hk_nc = cast("BlochHK", dk.hk)
    ks = dk.kc + s * qh
    with torch.no_grad():
        ws, us = torch.linalg.eigh(hk_nc.h(ks))
    vs = _dense_velocity_matrices(hk_nc, ks)
    uo, wo = dk.u[:, :nocc], dk.w[:nocc]
    o_u = 0.5 * (
        ep[0] * ((dk.v[0] + vs[0]) @ uo)
        + ep[1] * ((dk.v[1] + vs[1]) @ uo)
        + ep[2] * ((dk.v[2] + vs[2]) @ uo)
    )
    du = _resolvent_apply(us[:, nocc:], ws[nocc:], wo, o_u)

    x, wq = np.polynomial.legendre.leggauss(nquad)
    abar = [torch.zeros_like(dk.v[0]) for _ in range(3)]
    for xi, wi in zip(x, wq, strict=True):
        t = 0.5 * (float(xi) + 1.0)
        kt = dk.kc + (t * s) * qh
        vt = _dense_velocity_matrices(hk_nc, kt)
        kin_t = (2.0 * HBAR2_2M) * (hk_nc.g_cart + kt)
        for mu in range(3):
            abar[mu] += (0.5 * float(wi)) * (
                vt[mu] - torch.diag_embed(kin_t[:, mu].to(CDTYPE))
            )

    kpg_d = dk.kpg + s * qh
    u_box = g_to_r(uo.mT, dk.flat, eng.shape)
    du_box = g_to_r(du.mT, dk.flat, eng.shape)
    out = torch.zeros(3, *eng.shape, dtype=CDTYPE)
    pref = dk.weight * 2.0 / eng.volume
    for mu in range(3):
        wu = (2.0 * HBAR2_2M) * dk.kpg[:, mu].to(CDTYPE)[:, None] * uo \
            + abar[mu] @ uo
        wdu = (2.0 * HBAR2_2M) * kpg_d[:, mu].to(CDTYPE)[:, None] * du \
            + abar[mu] @ du
        out[mu] = pref * 0.5 * (
            u_box.conj() * g_to_r(wdu.mT, dk.flat, eng.shape)
            + g_to_r(wu.mT, dk.flat, eng.shape).conj() * du_box
        ).sum(dim=0)
    return out


def _biot_savart_sigma_cols_dq(
    s0: Tensor, ds: Tensor, q_hat_cart: Tensor, g_cart: Tensor, sites: Tensor
) -> tuple[Tensor, Tensor]:
    """Analytic q → 0 shielding column σ·(q̂×e_pol) at each site: (nsite, 3),
    plus the s = 0 null (the coefficient of the spurious 1/q divergence).

    The finite-q column is C·Φ̃(q)/q with C = 4π·ALPHA_FS²/E2 and Φ̃ the ±q
    antisymmetrized Biot–Savart sum; Φ̃(0) = 0 exactly (the q = 0 A-wave is
    pure gauge / TR-null), so the limit is C·Φ̃'(0) by the product rule:

        col = C·2 Re Σ_{G≠0} e^{iG·r_s}/G² { iG×F̂'(G) + iq̂×F̂⁰(G)
                + [i(q̂·r_s) − 2(G·q̂)/G²]·iG×F̂⁰(G) },

    with F̂⁰(G) = [Ŝ⁰(G) + conj(Ŝ⁰(−G))]/2i the s = 0 antisymmetric field
    (its own Biot–Savart sum is the returned null — measured ~0) and
    F̂'(G) = [Ŝ'(G) − conj(Ŝ'(−G))]/2i its s-derivative (S₋(s) = S₊(−s)).
    The kernel-derivative terms carry the diamagnetic (Lamb) content that
    the finite-q route picks up from the O(q) kernel expansion. G = 0
    (macroscopic shape term) omitted as in the finite-q assembly."""
    f0 = _antisym_field_g(s0, s0).permute(1, 2, 3, 0)  # (n1, n2, n3, 3)
    f1 = _antisym_field_g(ds, -ds).permute(1, 2, 3, 0)
    g = g_cart.to(RDTYPE)
    g2 = (g * g).sum(dim=-1)
    g2[0, 0, 0] = 1.0  # dummy; the G = 0 slot is zeroed below
    qh = q_hat_cart.to(RDTYPE)
    cross0 = 1j * torch.linalg.cross(g.to(CDTYPE), f0, dim=-1)
    cross1 = 1j * torch.linalg.cross(g.to(CDTYPE), f1, dim=-1)
    crossq = 1j * torch.linalg.cross(qh.to(CDTYPE).expand_as(f0), f0, dim=-1)
    g2c = g2[..., None].to(CDTYPE)
    a_field = (cross1 + crossq
               - (2.0 * (g @ qh) / g2)[..., None].to(CDTYPE) * cross0) / g2c
    b_field = cross0 / g2c
    a_field[0, 0, 0] = 0.0
    b_field[0, 0, 0] = 0.0
    pref = 4.0 * math.pi * ALPHA_FS**2 / E2
    phases = torch.exp(
        1j * torch.einsum("sa,ijka->sijk", sites.to(RDTYPE), g).to(CDTYPE)
    )
    sum_a = torch.einsum("sijk,ijkm->sm", phases, a_field)
    sum_b = torch.einsum("sijk,ijkm->sm", phases, b_field)
    qdotr = (sites.to(RDTYPE) @ qh).to(CDTYPE)
    col = pref * 2.0 * (sum_a + 1j * qdotr[:, None] * sum_b).real
    null = pref * 2.0 * sum_b.real
    return col, null


# --------------------------------------------------------------------------- #
# little-group-of-q IBZ wedge reduction (opt-in, exact, symmetry-gated)        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _LittleGroupFold:
    """Little-group orbit fold of the analytic route's current⊗polarization
    tensor field, the vector-field analogue of ``chi0_q``'s scalar star-unfold.

    ``branch_fields_axis`` on the G_q wedge produces, per transverse
    polarization ê_β, the BZ-partial single-branch current field
    Θ[μ, β](r) — a rank-2 field: index μ is the (POLAR) induced current, index
    β the perturbation polarization (both transverse to q̂). Under g ∈ G_q
    (which fixes q̂, hence maps the transverse plane to itself) it transforms as
    Θ_{gk} = S_g Θ_k Q_gᵀ with S_g = Aᵀ W_g A⁻ᵀ the plain Cartesian rotation (no
    det — the current is polar, NOT axial: risk-#1 bookkeeping) acting on μ, and
    Q_g = Tᵀ S_g T its restriction to the transverse frame T = [t1 t2] acting on
    β. Averaging g·Θ over G_q is the projector onto the G_q-covariant subspace,
    so folding the star-multiplicity-weighted wedge sum reconstructs the
    full-BZ field exactly:

        Θ_sym[μ,β](G) = (1/|G_q|) Σ_g S_g[μ,μ'] Q_g[β,β'] e^{−2πi m·w_g}
                                        Θ[μ',β'](W_gᵀ G).

    The scalar G-fold (Miller map, non-symmorphic phase, density-sphere mask)
    is reused verbatim from a :class:`~gradwave.symmetry.VectorFieldSymmetrizer`
    on G_q; only the extra Q_g mixing on β is new. The diamagnetic term
    2κ ê_β ⊗ ρ is a fixed point of this fold (ρ is G_q-invariant and the Q_g/S_g
    average returns ê_β), so it may be added before folding."""

    rho_sym: RhoSymmetrizer  # idx/phase/mask scalar G-fold on G_q
    rot: Tensor  # (nop, 3, 3) polar Cartesian S_g = Aᵀ W A⁻ᵀ
    qmat: Tensor  # (nop, 2, 2) transverse-frame restriction Q_g = Tᵀ S_g T
    shape: tuple[int, int, int]

    def _fold(self, theta: Tensor) -> Tensor:
        """Fold a rank-2 field theta (3, 2, n1, n2, n3) complex → same."""
        rs = self.rho_sym
        tg = r_to_g(theta)
        flat = tg.reshape(3, 2, -1) * rs.mask
        gathered = flat[:, :, rs.idx]  # (3, 2, nop, N)
        mixed = torch.einsum(
            "oai,ocb,ibon,on->acn",
            self.rot.to(tg.dtype), self.qmat.to(tg.dtype), gathered, rs.phase,
        ) / rs.idx.shape[0]
        box = (mixed * rs.mask).reshape(3, 2, *self.shape)
        return g_to_r_box(box)

    def apply(
        self, fields: list[tuple[Tensor, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        """Fold the (S⁰, ∂S/∂s) pair of each of the two transverse pols."""
        if len(fields) != 2:
            raise ValueError("little-group fold expects the 2 transverse pols")
        s0 = torch.stack([fields[0][0], fields[1][0]], dim=1)  # (3, 2, *shape)
        ds = torch.stack([fields[0][1], fields[1][1]], dim=1)
        s0f, dsf = self._fold(s0), self._fold(ds)
        return [(s0f[:, 0], dsf[:, 0]), (s0f[:, 1], dsf[:, 1])]


def _build_axis_fold(shape, q_hat, tframe, lg, cell, dens_mask):
    """Assemble the :class:`_LittleGroupFold` for one q̂ axis, or None if the
    FFT box is not closed under the little group (a mesh/box mismatch — the
    caller falls back to the exact full mesh for that axis)."""
    from gradwave.symmetry import VectorFieldSymmetrizer

    try:
        vfs = VectorFieldSymmetrizer(shape, lg, cell, dens_mask=dens_mask)
    except ValueError:
        return None
    t_mat = torch.as_tensor(np.asarray(tframe, dtype=float), dtype=torch.float64)
    qmat = torch.einsum("ia,oij,jb->oab", t_mat, vfs.rot, t_mat)  # (nop, 2, 2)
    return _LittleGroupFold(rho_sym=vfs.rho_sym, rot=vfs.rot, qmat=qmat,
                            shape=tuple(shape))


def _plan_wedge_reduction(res, axes, use_symmetry):
    """Plan the per-axis little-group-of-q k-reduction of the analytic route.

    Returns ``None`` to run the exact full-mesh path (symmetry off, or no axis
    reduces by ≥1.1× on top of the time-reversal fold the route already banks).
    Otherwise returns ``(union_kfrac, per_axis)``: the union of every axis's
    wedge k-points (one shared ``ShieldingDq`` setup pass) and, per axis, the
    context indices into that union, the star-multiplicity weights, and the
    :class:`_LittleGroupFold` (``None`` for a non-reduced axis, which then sums
    the full time-reversal-folded mesh unchanged)."""
    if use_symmetry is False:
        return None
    from gradwave.symmetry import (
        SpaceGroup,
        _k_ops,
        _orbit_reduce,
        find_spacegroup,
        little_cogroup,
    )

    system = res.system
    cell = np.asarray(system.grid.cell, dtype=float)
    frac = system.positions.detach().cpu().numpy() @ np.linalg.inv(cell)
    sg = system.sym or find_spacegroup(cell, frac, system.species_of_atom)
    # True Monkhorst–Pack dimensions per axis. ``mesh_n`` (unique k per axis of
    # the STORED spheres) undercounts when the route runs on the time-reversal
    # folded mesh — e.g. a 4³ mesh folds to 3 unique kₓ (0, ¼, ½). Recover the
    # full-BZ dims by TR-unfolding (∪ of k and −k is closed = the MP grid), so
    # ``_orbit_reduce`` tiles the correct Γ-centered mesh.
    k_all = np.stack([sph.k_frac for sph in system.spheres])
    k_pm = np.concatenate([k_all, -k_all], axis=0) % 1.0
    true_n = [len(np.unique(np.round(k_pm[:, i], 6))) for i in range(3)]
    mesh: tuple[int, int, int] = (true_n[0], true_n[1], true_n[2])
    b = reciprocal_cell(cell)
    ident = np.eye(3, dtype=np.int64)
    tr_reps, tr_w = _orbit_reduce(mesh, [ident, -ident])  # TR-folded full mesh
    baseline_nk = len(tr_reps)
    dens_mask = system.grid.dens_mask
    shape = system.grid.shape

    plans, any_reducible = [], False
    for i in axes:
        q_hat = np.asarray(b[i], dtype=float) / np.linalg.norm(b[i])
        q_frac = np.zeros(3)
        q_frac[i] = 1.0 / mesh[i]
        lg_full, g0 = little_cogroup(q_frac, sg)
        keep = [j for j in range(len(lg_full.rotations)) if np.all(g0[j] == 0)]
        lg = SpaceGroup(
            rotations=lg_full.rotations[keep],
            translations=lg_full.translations[keep],
            atom_map=lg_full.atom_map[keep],
            international=sg.international, origin_shift=sg.origin_shift,
        )
        ops_t = _k_ops(lg.rotations)
        reps_kf, reps_w = _orbit_reduce(mesh, ops_t + [-w for w in ops_t])
        tframe = np.stack(list(_transverse_frame(q_hat)), axis=1)  # (3, 2)
        fold = None
        if len(lg.rotations) > 1 and baseline_nk >= 1.1 * len(reps_kf):
            fold = _build_axis_fold(shape, q_hat, tframe, lg, cell, dens_mask)
        if fold is not None:
            any_reducible = True
            plans.append((i, q_hat, reps_kf, reps_w, fold))
        else:
            plans.append((i, q_hat, tr_reps, tr_w, None))  # exact full path
    if not any_reducible:
        return None

    union: list[np.ndarray] = []
    keyidx: dict[tuple[float, ...], int] = {}
    per_axis = []
    for i, q_hat, kf, w, fold in plans:
        idx = []
        for row in kf:
            key = tuple(np.round(row, 6))
            if key not in keyidx:
                keyidx[key] = len(union)
                union.append(row)
            idx.append(keyidx[key])
        per_axis.append((i, q_hat, idx, list(map(float, w)), fold))
    return np.stack(union), per_axis


def _symmetrize_site_tensors(out: Tensor, system: object) -> Tensor:
    """Project the per-atom shielding tensors onto the crystal space group:

        σ_sym[a] = (1/N) Σ_g Rᵀ_g σ_raw[g·a] R_g,   R_g = Cᵀ W_g C⁻ᵀ,

    with W_g the fractional rotation, C the cell (rows = lattice vectors), and
    g·a = ``atom_map[g, a]``. R_g is the Cartesian image of W_g; the shielding
    is an ordinary rank-2 Cartesian tensor under it (the two axial-vector signs
    of an improper op cancel), so σ' = R σ Rᵀ for every operation. The average
    enforces that symmetry-equivalent atoms carry equal (rotated) tensors and
    symmetrizes each atom under its own site group — removing the residual the
    2-axis analytic reconstruction leaves on rotation-related sites. Identity
    for a P1 cell."""
    from gradwave.symmetry import find_spacegroup

    sysa: Any = cast("Any", system)  # System | USPPSystem; attrs read below

    def _np(x: Any) -> np.ndarray:  # grid.cell is a numpy array on the api path,
        # a tensor from setup_system; positions is a tensor — accept both.
        if hasattr(x, "detach"):
            x = x.detach().cpu()
        return np.asarray(x, dtype=float)

    cell = _np(sysa.grid.cell)
    pos = _np(sysa.positions)
    if pos.shape[0] != out.shape[0]:  # custom site subset — cannot map to atoms
        return out
    frac = pos @ np.linalg.inv(cell)
    species = [int(s) for s in sysa.species_of_atom]
    sg = find_spacegroup(cell, frac, species)
    ct = cell.T
    ct_inv = np.linalg.inv(ct)
    sig = out.detach().cpu().numpy()
    acc = np.zeros_like(sig)
    for op in range(sg.n_ops):
        r = ct @ sg.rotations[op].astype(float) @ ct_inv
        acc += np.einsum("ji,ajk,kl->ail", r, sig[sg.atom_map[op]], r)
    acc /= sg.n_ops
    return torch.as_tensor(acc, dtype=out.dtype, device=out.device)


def sigma_shielding_dq(
    res: SCFResult,
    *,
    sites: Tensor | None = None,
    use_symmetry: bool | str = "auto",
    uspp: USPPResponseCtx | str | None = None,
    chunk_k: int | None = None,
    symmetrize: bool = True,
) -> Tensor:
    """Bare (pseudo) NMR shielding tensor by the ANALYTIC ∂/∂q at q = 0:
    σ_ij = −∂B_ind,i(r_site)/∂B_ext,j in ppm, shape (nsite, 3, 3).

    Same physical content and conventions as :func:`sigma_shielding` (bare
    valence only — no GIPAW core reconstruction; the G = 0 macroscopic shape
    term omitted), but the (1/2q)[S(q) − S(−q)] finite difference is replaced
    by the exact q-derivative of the induced-current assembly
    (:class:`ShieldingDq`), removing the O(q²) finite-q error and the
    q-commensurability constraint. The joint (mesh n, q = b/n) refinement of
    the finite-q route converges to this result.

    q̂ DIRECTIONS ARE THE SAMPLED MESH AXES (n_i > 1), as in the finite-q
    driver — not free: the analytic column is a BZ sum of a k-DERIVATIVE,
    and a uniform-mesh Riemann sum of ∂f/∂k along q̂ retains the spurious
    terms i(q̂·R)f_R over the mesh SUPERLATTICE vectors R (f_R the
    Wannier-decaying Fourier coefficients of the k-periodic integrand —
    ∫∂f = 0 exactly, but the mesh sum only kills R off the superlattice).
    Along a mesh axis q̂ = b̂_i every unsampled-direction R has q̂·R = 0
    exactly and the residual is the n_i-cell-neighbor coefficient
    (exponentially small in n_i for an insulator — measured directly as the
    longitudinal gauge null); along an arbitrary q̂ the n = 1 axes leak at
    O(1). Needs ≥ 2 sampled axes to determine the tensor.

    Bare (unscreened) responses only: for a TR-symmetric insulator the K_Hxc
    screening correction to the antisymmetric q-derivative vanishes at q = 0
    (the density response linear in B is TR-odd). Measured on Si columns of
    ~30–80 ppm: |screened − bare| decays monotonically 0.92 → 0.50 → 0.30 →
    0.22 ppm over q = b/2 → b/16, and the screened vs bare q → 0
    extrapolations agree to 0.2 ppm — any residual screening correction is
    bounded below the finite-q route's own extrapolation uncertainty.

    ``use_symmetry`` (default ``"auto"``) enables the little-group-of-q IBZ
    wedge reduction: the per-axis k-sum of ``branch_fields_axis`` is solved only
    on the wedge of the little group G_q of that axis's q̂ direction (on top of
    the time-reversal fold the route already banks) and the full-BZ current
    field is reconstructed by the little-group orbit action
    (:class:`_LittleGroupFold`). This is exact — bit-equivalent to the full-mesh
    result to solver tolerance — and opt-in with auto-detection: an axis is
    reduced only when G_q is non-trivial and buys ≥1.1× on top of TR, otherwise
    it runs the exact full mesh, so the reduced path is never slower than
    ``use_symmetry=False``. Measured ~2.7×/3.4× on Si (4³/6³) and ~1.5×/2.0× on
    rutile TiO₂; a cell with trivial symmetry falls back with no slowdown.

    ``chunk_k`` streams the per-k dense contexts (eigh of H(k) + the ∂H/∂k
    velocity matrices, the (npw×npw) blocks that dominate the footprint):
    ``None`` (default) holds all mesh k at once — the historical path — while a
    positive value builds and frees one k's context at a time inside the per-
    axis BZ sum, dropping the peak from nk dense contexts to one at the cost of
    rebuilding each context once per sampled axis. Bit-identical to the full-
    mesh build (each context is a pure function of its k); the memory route for
    many-atom cells whose full-mesh contexts do not fit.

    ``symmetrize`` (default True) projects the per-site tensors onto the crystal
    space group — σ_sym[a] = (1/N) Σ_g Rᵀ_g σ_raw[g·a] R_g with R_g = Cᵀ W_g C⁻ᵀ
    the Cartesian image of the fractional rotation. The analytic reconstruction
    uses only the ≥2 sampled mesh axes, so sites related by a rotation whose
    image direction is not itself sampled (a hexagonal 3-fold, an orthorhombic
    screw) otherwise pick up a site-dependent residual and come out UNEQUAL even
    though they are crystallographically equivalent (measured: α-quartz Si-site
    spread ~8 ppm, flat in k, present with the wedge off — cubic Si is immune).
    Symmetrization enforces the exact equivalence and is a no-op for a P1 cell.
    Only applied to the default all-atom site list; an explicit ``sites`` subset
    (which cannot be mapped to the atom permutation) skips it."""
    _guard(res)
    system = uspp.system if isinstance(uspp, USPPResponseCtx) else res.system
    sites_is_default = sites is None
    if sites is None:
        sites = system.positions.detach().cpu().to(RDTYPE)
    b = reciprocal_cell(system.grid.cell)
    k_frac = np.stack([sph.k_frac for sph in system.spheres])
    mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
    axes = [i for i in range(3) if mesh_n[i] > 1]
    if len(axes) < 2:
        raise ValueError(
            f"sigma_shielding_dq needs >=2 k-mesh axes with n>1 (mesh {mesh_n}): "
            "the tensor is underdetermined from a single q̂ direction, and the "
            "BZ sum of the analytic q-derivative only converges along sampled "
            "mesh axes (see docstring)")

    # The little-group wedge reduction folds the current field only (it is
    # orthogonal to the H/S metric); keep the S-metric route on the exact full
    # mesh (the smooth-USPP build is validated by the NC-limit gate there).
    plan = None if uspp is not None else _plan_wedge_reduction(res, axes, use_symmetry)
    if plan is None:  # exact full-mesh path (symmetry off or nothing reduces)
        eng = ShieldingDq(res, uspp=uspp, chunk_k=chunk_k)
        axis_specs = [
            (np.asarray(b[i], dtype=float) / np.linalg.norm(b[i]), None, None, None)
            for i in axes
        ]
    else:  # opt-in little-group-of-q wedge reduction
        union_kfrac, per_axis = plan
        eng = ShieldingDq(res, k_frac=union_kfrac, chunk_k=chunk_k)
        axis_specs = [
            (q_hat, k_idx, w, fold) for _i, q_hat, k_idx, w, fold in per_axis
        ]

    # Single all-axes pass: build each union-k context ONCE and contribute it to
    # every axis (branch_fields_all_axes), so the streamed (chunk_k) path rebuilds
    # each context once TOTAL rather than once per sampled axis. _transverse_frame
    # is computed once per axis and reused for both the spec and the b_rows below
    # so the pol ordering matches the returned fields.
    specs = [
        (q_hat, list(_transverse_frame(q_hat)), k_idx, w, fold)
        for q_hat, k_idx, w, fold in axis_specs
    ]
    all_fields = eng.branch_fields_all_axes(specs)

    b_rows: list[np.ndarray] = []
    m_rows: list[Tensor] = []
    for (q_hat, pols, _k_idx, _w, _fold), fields in zip(specs, all_fields, strict=True):
        for pol, (s0f, dsf) in zip(pols, fields, strict=True):
            col, _ = _biot_savart_sigma_cols_dq(
                s0f, dsf, torch.as_tensor(q_hat, dtype=RDTYPE),
                system.grid.g_cart, sites)
            m_rows.append(col)
            b_rows.append(np.cross(q_hat, pol))
    bmat = torch.as_tensor(np.stack(b_rows), dtype=RDTYPE)
    mmat = torch.stack(m_rows)
    ns = mmat.shape[1]
    out = torch.empty(ns, 3, 3, dtype=RDTYPE)
    for s in range(ns):
        out[s] = torch.linalg.lstsq(bmat, mmat[:, s, :]).solution.mT
    if symmetrize and sites_is_default:
        out = _symmetrize_site_tensors(out, system)
    return out * 1e6


# --------------------------------------------------------------------------- #
# On-site paramagnetic augmentation (GIPAW σ_para_aug): X_β cross density + σ   #
# --------------------------------------------------------------------------- #

_ParaBranch = tuple[np.ndarray, np.ndarray, np.ndarray, Tensor, Tensor, float]


def _para_branch_cross(
    res: SCFResult,
    ctx: USPPResponseCtx,
    *,
    cg_tol: float = 1e-9,
    max_iter: int = 800,
) -> tuple[list[_ParaBranch], np.ndarray, "USPPSystem"]:
    """Per-(axis, pol) raw ±q on-site cross densities for the magnetic response.

    For every sampled mesh axis the ±q generalized-velocity Sternheimer solves
    (:func:`velocity_perturbation_q` with the USPP ctx, v = ∂H/∂k − ε ∂S/∂k)
    give the first-order response δu; projected onto the on-site overlap
    projectors ⟨p̃(k±q)|δu⟩ and contracted (BZ- and occupation-weighted) with
    the ground projection ⟨p̃(k)|ψ̃⁰⟩ (≡ ``becps``) this yields the raw ±q
    cross densities per transverse polarization

        x_±[a, b] = Σ_{k,o} f w_k ⟨ψ̃⁰_o|p̃_a(k)⟩ ⟨p̃_b(k±q)|δu_o^{pol,±q}⟩

    in the full m-expanded projector layout (all atoms). The per-nucleus gauge
    referencing and the (1/2iq) linear-in-q extraction are applied downstream in
    :func:`para_augmentation_shielding`. Returns the branch list
    ``[(q_cart, q_hat, pol, x_p, x_m, q_mag)]``, the projector→atom column map,
    and the (USPP) system.
    """
    system = ctx.system
    nocc = insulator_window(res.occupations, 2.0, "insulating occupations required")
    ov = ctx.ov
    nk, npw_ov = ov.nk, ov.npw_max
    c_occ = torch.zeros(nk, nocc, npw_ov, dtype=CDTYPE)
    for ik, c in enumerate(cast("list[Tensor]", res.coeffs)):
        n = min(c.shape[1], npw_ov)
        c_occ[ik, :, :n] = c[:nocc, :n]
    g_gnd = torch.einsum("kag,kog->koa", ov.p.conj(), c_occ)  # ⟨p̃_a(k)|ψ̃⁰_o⟩

    kb = _overlap_kbprojectors(system, system.spheres[0])
    col_atom = kb.col_atom.numpy()
    b = reciprocal_cell(system.grid.cell)
    k_frac = np.stack([sph.k_frac for sph in system.spheres])
    mesh_n = [len(np.unique(np.round(k_frac[:, i], 6))) for i in range(3)]
    axes = [i for i in range(3) if mesh_n[i] > 1]
    if len(axes) < 2:
        raise ValueError(
            "para augmentation needs >=2 k-mesh axes with n>1 to determine the "
            "shielding tensor (run the SCF on a >=2-axis unreduced mesh)")
    wf = (2.0 * system.kweights.to(RDTYPE))[:, None, None].to(CDTYPE)  # f_occ·w_k
    gw = (wf * g_gnd).conj()

    branches: list[_ParaBranch] = []
    for i in axes:
        q_frac = np.zeros(3)
        q_frac[i] = 1.0 / mesh_n[i]
        q_cart = q_frac @ b
        q_mag = float(np.linalg.norm(q_cart))
        q_hat = q_cart / q_mag
        pols = list(_transverse_frame(q_hat))
        contrib: dict[float, Tensor] = {}
        for sgn in (+1.0, -1.0):
            sol = velocity_perturbation_q(
                res, sgn * q_frac, cg_tol=cg_tol, max_iter=max_iter, uspp=ctx)
            jsel = torch.as_tensor(sol.jidx, dtype=torch.long)
            p_kq = ov.p[jsel]
            npw = min(p_kq.shape[-1], sol.dpsi.shape[-1])
            contrib[sgn] = torch.einsum(
                "kbg,mkog->mkob", p_kq[:, :, :npw].conj(), sol.dpsi[..., :npw])
        for pol in pols:
            pol_t = torch.as_tensor(pol, dtype=CDTYPE)
            r_p = torch.einsum("m,mkob->kob", pol_t, contrib[+1.0])
            r_m = torch.einsum("m,mkob->kob", pol_t, contrib[-1.0])
            x_p = torch.einsum("koa,kob->ab", gw, r_p)
            x_m = torch.einsum("koa,kob->ab", gw, r_m)
            branches.append((q_cart, q_hat, np.asarray(pol, dtype=float),
                             x_p, x_m, q_mag))
    return branches, col_atom, system


def para_augmentation_shielding(
    res: SCFResult,
    ctx: USPPResponseCtx,
    onsites: dict[int, PAWOnSite],
    *,
    pref: float = R_E_ANG,
    cg_tol: float = 1e-9,
    max_iter: int = 800,
) -> Tensor:
    """On-site paramagnetic-augmentation shielding tensor σ_para_aug
    (nsite, 3, 3) in ppm — the hard GIPAW magnetic-response term
    [Yates–Pickard–Mauri, PRB 76, 024401 (2007), Eq. (85)-(86)].

    Assembles the per-field ground×response on-site cross density X_β (the
    ``becps`` ⊗ ⟨p̃|δu⟩ product, :func:`_para_branch_cross`), references the
    vector potential A(r) = pol·sin(q·(r−R))/q to each nucleus R by the gauge
    phase e^{∓iq·R} on the ±q branches (so off-origin sites are treated on the
    same footing as the origin — the covariant-position term), extracts the
    uniform-field (1/2iq) linear-in-q part, least-squares reconstructs the
    Cartesian field response X_β from the (q̂×pol) branches, and contracts it
    with the on-site current operator via :meth:`PAWOnSite.para_aug_tensor`.

    ``onsites`` maps species index → :class:`~gradwave.postscf.gipaw.PAWOnSite`;
    a species absent from the map (a norm-conserving atom with no partial waves)
    contributes a zero tensor. ``pref`` is the paramagnetic prefactor: the
    classical electron radius R_E (the same constant carrying the
    diamagnetic/Lamb terms) reproduces the diamond ²⁹Si absolute σ with no free
    scale — the two symmetry-equivalent Si sites agree to <1 ppm as the internal
    consistency gate (see the PR-analytic-B notes)."""
    branches, col_atom, system = _para_branch_cross(
        res, ctx, cg_tol=cg_tol, max_iter=max_iter)
    tau = system.positions.detach().cpu().numpy()
    species_of_atom = [int(s) for s in system.species_of_atom]
    nsite = len(species_of_atom)
    out = torch.zeros(nsite, 3, 3, dtype=RDTYPE)
    for a in range(nsite):
        onsite = onsites.get(species_of_atom[a])
        if onsite is None:
            continue
        idx = np.where(col_atom == a)[0]
        rows: list[Tensor] = []
        bmat_rows: list[np.ndarray] = []
        for (q_cart, q_hat, pol, x_p, x_m, q_mag) in branches:
            # gauge-reference to nucleus R (A ∝ sin(q·(r−R))/q ⇒ e^{∓iq·R} on
            # the ±q responses) and the (1/2iq) uniform-field extraction.
            ph = np.exp(-1j * float(q_cart @ tau[a]))
            block = 1j * (ph * x_p[np.ix_(idx, idx)]
                          - np.conj(ph) * x_m[np.ix_(idx, idx)]) / (2.0 * q_mag)
            rows.append(block)
            bmat_rows.append(np.cross(q_hat, pol))
        bmat = torch.as_tensor(np.stack(bmat_rows), dtype=CDTYPE)
        xs = torch.stack(rows)
        nbr, n, _ = xs.shape
        xcart = torch.linalg.lstsq(
            bmat, xs.reshape(nbr, -1)).solution.reshape(3, n, n)
        out[a] = onsite.para_aug_tensor(xcart, pref=pref)
    return out


def sigma_shielding_gipaw(
    res: SCFResult,
    ctx: USPPResponseCtx,
    paws: Sequence[Any],
    *,
    pref: float = R_E_ANG,
    use_symmetry: bool | str = False,
    cg_tol: float = 1e-9,
    max_iter: int = 800,
) -> dict[str, Tensor]:
    """Full absolute GIPAW chemical-shielding tensor per site (nsite, 3, 3) ppm,

        σ = σ_bare + σ_core + σ_dia_aug + σ_para_aug,

    for an all-PAW USPP ground state. Sums the smooth (bare valence) analytic-
    USPP shielding (:func:`sigma_shielding_dq` through the S-metric ctx), the
    frozen-core Lamb + on-site diamagnetic augmentation
    (:func:`gradwave.postscf.gipaw.ground_state_shielding`), and the on-site
    paramagnetic augmentation (:func:`para_augmentation_shielding`). ``paws`` is
    one :class:`~gradwave.pseudo.upf_paw.PAWData` per species. Returns a dict
    ``{"total", "bare", "core", "dia_aug", "para_aug"}`` (all nsite, 3, 3)."""
    system = ctx.system
    species_of_atom = [int(s) for s in system.species_of_atom]
    sig_bare = sigma_shielding_dq(res, uspp=ctx, use_symmetry=use_symmetry)
    gs = ground_state_shielding(
        list(paws), species_of_atom, cast("Any", res).rho_ij_atoms)
    onsites = {sp: PAWOnSite.from_paw(p) for sp, p in enumerate(paws)}
    sig_para = para_augmentation_shielding(
        res, ctx, onsites, pref=pref, cg_tol=cg_tol, max_iter=max_iter)
    eye = torch.eye(3, dtype=RDTYPE)
    core = gs["core"][:, None, None] * eye
    total = sig_bare + core + gs["dia_aug"] + sig_para
    return {"total": total, "bare": sig_bare, "core": core,
            "dia_aug": gs["dia_aug"], "para_aug": sig_para}
