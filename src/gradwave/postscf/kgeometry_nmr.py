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

Insulators, nspin = 1, full (symmetry-unreduced) k-mesh, q commensurate
with the mesh for the Sternheimer route (the dense twin takes any q).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import Tensor

from gradwave.constants import ALPHA_FS, E2, HBAR2_2M
from gradwave.core.batch import (
    BatchedHamiltonian,
    BatchedK,
    box_to_sphere_b,
    g_to_r_b,
    projectors_b,
)
from gradwave.core.fftbox import g_to_r, r_to_g
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import reciprocal_cell
from gradwave.postscf._response import (
    cg_sternheimer,
    insulator_window,
    pad_coeffs,
    sternheimer_shift,
)
from gradwave.postscf.dfpt_q import _g0_phase, _reindex_bk, kpq_map
from gradwave.postscf.kgeometry import BlochHK, KBProjectors, VelocityApply
from gradwave.postscf.kgeometry_topo import _pack_miller

if TYPE_CHECKING:
    from gradwave.scf.loop import SCFResult, System


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


def _guard(res: SCFResult) -> None:
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("finite-q velocity response: nspin=1 only")
    if res.system.sym is not None:
        raise NotImplementedError(
            "finite-q velocity response needs the full k-mesh: run the SCF "
            "with use_symmetry=False"
        )


def velocity_perturbation_q(
    res: SCFResult,
    q_frac: np.ndarray | list[float] | tuple[float, float, float],
    *,
    cg_tol: float = 1e-9,
    max_iter: int = 400,
) -> VelocityQSolves:
    """Solve the k+q Sternheimer equations for the three symmetrized velocity
    perturbations ½{v_μ, e^{iqr}} at every mesh k (see module docstring).

    q must be commensurate with the SCF mesh (kpq_map raises otherwise);
    q = 0 reduces exactly to the M5 velocity solves."""
    _guard(res)
    system = res.system
    grid = system.grid
    shape = grid.shape
    bk = system.batch
    assert bk is not None
    q_frac = np.asarray(q_frac, dtype=float)

    k_frac = np.stack([sph.k_frac for sph in system.spheres])
    jidx, g0 = kpq_map(k_frac, q_frac)
    nk = len(k_frac)
    jsel = torch.as_tensor(jidx, dtype=torch.long)
    bk_kq = _reindex_bk(bk, jsel)

    nocc = insulator_window(res.occupations, 2.0, "insulating occupations required")
    c_all = pad_coeffs(cast("list[Tensor]", res.coeffs), bk.npw_max)
    c_occ_k = c_all[:, :nocc]
    c_occ_kq = c_all[jsel, :nocc]
    eps_k = res.eigenvalues[:, :nocc].to(RDTYPE)
    shift = sternheimer_shift(eps_k)

    h_kq = BatchedHamiltonian(bk_kq, shape, res.v_eff, projectors_b(bk_kq, system.positions))
    ph = torch.stack(
        [_g0_phase(shape, g0[ik], c_all.device) for ik in range(nk)]
    )[:, None]  # (nk, 1, *shape): e^{iG0·r}

    def transfer(w: Tensor) -> Tensor:
        """k-sphere coefficients → the same plane waves on the k_j spheres."""
        return box_to_sphere_b(g_to_r_b(w, bk, shape) * ph, bk_kq)

    def p_c(x: Tensor) -> Tensor:
        ov = torch.einsum("kng,kbg->kbn", c_occ_kq.conj(), x)
        return x - torch.einsum("kbn,kng->kbg", ov, c_occ_kq)

    v = VelocityApply(system)
    c_t = transfer(c_occ_k)
    o_list, d_list = [], []
    for mu in range(3):
        o_mu = 0.5 * (transfer(v.apply(c_occ_k, mu)) + v.apply(c_t, mu, k_index=jsel))
        rhs = -p_c(o_mu)
        d_list.append(
            cg_sternheimer(h_kq, bk_kq, c_occ_kq, eps_k, rhs,
                           torch.zeros_like(rhs), shift, tol=cg_tol, max_iter=max_iter)
        )
        o_list.append(o_mu)
    return VelocityQSolves(
        q_frac=q_frac,
        o_c=torch.stack(o_list),
        dpsi=torch.stack(d_list),
        c_occ_k=c_occ_k,
        c_occ_kq=c_occ_kq,
        eps_k=eps_k,
        jidx=jidx,
        g0=g0,
        weights=system.kweights.to(RDTYPE),
        bk_kq=bk_kq,
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

            if self.nproj:
                kb0 = KBProjectors.for_sphere(system, sph)
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
    ph = torch.stack(
        [_g0_phase(shape, sol.g0[ik], sol.c_occ_k.device) for ik in range(nk)]
    )[:, None]

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
                res, q_frac, _k_hxc_q(res, xc, phys(u), q_cart), tol=cg_tol
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
        h_kq = BatchedHamiltonian(
            sol.bk_kq, shape, res.v_eff, projectors_b(sol.bk_kq, system.positions)
        )
        shift = sternheimer_shift(sol.eps_k)
        dpsi = dpsi + cg_sternheimer(h_kq, sol.bk_kq, sol.c_occ_kq, sol.eps_k,
                                     rhs, torch.zeros_like(rhs), shift,
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
    nk = sol.c_occ_k.shape[0]
    ph = torch.stack(
        [_g0_phase(shape, sol.g0[ik], sol.c_occ_k.device) for ik in range(nk)]
    )[:, None]
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
      is O(q²) (the antisymmetric combination cancels even orders) — check
      stability by comparing ``q_index`` values on a finer mesh.

    ``screen=True`` (with ``xc``) uses the K_Hxc-screened first-order
    responses. Requires an insulating, symmetry-unreduced SCF with at least
    two mesh axes of more than one point (else the 3×3 tensor is
    underdetermined and a ValueError is raised).
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

    b_rows: list[np.ndarray] = []
    m_rows: list[Tensor] = []
    for i in axes:
        q_frac = np.zeros(3)
        q_frac[i] = q_index / mesh_n[i]
        q_cart = torch.as_tensor(q_frac @ b, dtype=RDTYPE)
        q_hat = (q_frac @ b) / np.linalg.norm(q_frac @ b)
        sol_p = velocity_perturbation_q(res, q_frac, cg_tol=cg_tol,
                                        max_iter=cg_max_iter)
        sol_m = velocity_perturbation_q(res, -q_frac, cg_tol=cg_tol,
                                        max_iter=cg_max_iter)
        nl_p = _NLPairVelocity(system, sol_p, nquad=nl_quad)
        nl_m = _NLPairVelocity(system, sol_m, nquad=nl_quad)
        for pol in _transverse_frame(q_hat):
            cur_p = induced_current_q(res, q_frac, pol, sol=sol_p, _nl=nl_p,
                                      xc=xc, screen=screen, cg_tol=cg_tol,
                                      cg_max_iter=cg_max_iter)
            cur_m = induced_current_q(res, -q_frac, pol, sol=sol_m, _nl=nl_m,
                                      xc=xc, screen=screen, cg_tol=cg_tol,
                                      cg_max_iter=cg_max_iter)
            m_rows.append(_biot_savart_sigma_cols(
                cur_p.j_total, cur_m.j_total, q_cart, grid.g_cart, sites))
            b_rows.append(np.cross(q_hat, pol))

    bmat = torch.as_tensor(np.stack(b_rows), dtype=RDTYPE)  # (nr, 3)
    mmat = torch.stack(m_rows)  # (nr, nsite, 3)
    ns = mmat.shape[1]
    out = torch.empty(ns, 3, 3, dtype=RDTYPE)
    for s in range(ns):
        out[s] = torch.linalg.lstsq(bmat, mmat[:, s, :]).solution.mT
    return out * 1e6


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

        kb0 = KBProjectors.for_sphere(system, sph)
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
