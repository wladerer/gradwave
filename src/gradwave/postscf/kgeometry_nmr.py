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

Insulators, nspin = 1, full (symmetry-unreduced) k-mesh, q commensurate
with the mesh for the Sternheimer route (the dense twin takes any q).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import Tensor

from gradwave.core.batch import (
    BatchedHamiltonian,
    BatchedK,
    box_to_sphere_b,
    g_to_r_b,
    projectors_b,
)
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf._response import (
    cg_sternheimer,
    insulator_window,
    pad_coeffs,
    sternheimer_shift,
)
from gradwave.postscf.dfpt_q import _g0_phase, _reindex_bk, kpq_map
from gradwave.postscf.kgeometry import BlochHK, VelocityApply
from gradwave.postscf.kgeometry_topo import _pack_miller

if TYPE_CHECKING:
    from gradwave.scf.loop import SCFResult


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
# induced current density and K_Hxc screening (milestone 8)                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CurrentResponseQ:
    """First-order induced current density for the perturbation λ·O_{q,ν}
    (per unit λ; the physical field adds the −q branch)."""

    q_frac: np.ndarray
    nu: int  # perturbation direction
    j_para: Tensor  # (3, n1, n2, n3) complex: BZ-weighted canonical current
    j_para_k: Tensor  # (nk, 3, n1, n2, n3): per-k, (f/Ω)-normalized, no w_k
    j_dia: Tensor  # (3, n1, n2, n3): diamagnetic term 2·HBAR2_2M·δ_μν·ρ(r)
    drho_bare: Tensor  # (n1, n2, n3) complex: bare induced density (periodic part)
    drho: Tensor  # screened induced density (== bare when screen=False)
    n_dyson: int
    dpsi: Tensor  # (nk, nocc, npw_max): total (screened) first-order response


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
    nu: int,
    *,
    xc: object | None = None,
    screen: bool = False,
    cg_tol: float = 1e-9,
    cg_max_iter: int = 400,
    dyson_beta: float = 0.4,
    dyson_tol: float = 1e-7,
    dyson_iter: int = 40,
    history: int = 8,
) -> CurrentResponseQ:
    """Induced (canonical) current density of the perturbation λ·O_{q,ν}.

    Paramagnetic part from the (u_nk, δu_nk) Sternheimer pairs with the
    KINETIC current operator — the cell-periodic cross density
    ½[ũ*(v_kin δu)~ + (v_kin u)~* δũ] referenced to (k, k+q) via the umklapp
    phase; per-k means obey the exact f-sum d⟨v_μ⟩/dk_ν identity (tested).
    Diamagnetic part 2·HBAR2_2M·δ_μν·ρ(r) (the A·A term of the same
    coupling; at q = 0 its cell average cancels the paramagnetic one).
    The KB nonlocal current contribution is NOT included — that is the
    GIPAW reconstruction gap, tracked explicitly.

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
    sol = velocity_perturbation_q(res, q_frac, cg_tol=cg_tol, max_iter=cg_max_iter)
    nk = sol.c_occ_k.shape[0]
    ph = torch.stack(
        [_g0_phase(shape, sol.g0[ik], sol.c_occ_k.device) for ik in range(nk)]
    )[:, None]

    dpsi = sol.dpsi[nu]
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
    from gradwave.constants import HBAR2_2M

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
    j_dia = torch.zeros(3, *shape, dtype=CDTYPE)
    j_dia[nu] = (2.0 * HBAR2_2M) * res.rho.to(CDTYPE)

    return CurrentResponseQ(
        q_frac=np.asarray(q_frac, dtype=float),
        nu=nu,
        j_para=j_para,
        j_para_k=j_k,
        j_dia=j_dia,
        drho_bare=drho_bare,
        drho=drho,
        n_dyson=n_dyson,
        dpsi=dpsi,
    )


def _dense_velocity_matrices(hh: BlochHK, k_cart: Tensor) -> list[Tensor]:
    """[v_x, v_y, v_z] (npw, npw) at k on hh's sphere — the dense form of
    ``VelocityApply``: kinetic diagonal + dpᵀD p* + pᵀD dp*."""
    from gradwave.constants import HBAR2_2M
    from gradwave.postscf.kgeometry import KBProjectors

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
