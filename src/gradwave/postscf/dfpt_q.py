"""q≠0 DFPT: the bare density response χ₀ to a perturbation of crystal
wavevector q (Phase 2 of the little-group star-unfold; see
docs/design/little-group-star-unfold.md).

A perturbation δV_q(r) = e^{iq·r} v(r) (v cell-periodic) couples a Bloch state at
k to a first-order response at k+q. The Sternheimer equation becomes

    (H_{k+q} − ε_{n,k} + α P_{k+q}) |δψ_{n,k}⟩ = −(1 − P_{k+q}) δV_q |ψ_{n,k}⟩,

and the response density is δρ_q(r) = e^{iq·r} Σ_{nk} f_{nk} w_k ψ*_{n,k}(r) δψ_{n,k+q}(r).
This is the same operator as the q=0 Sternheimer (``scf.implicit._sternheimer``)
with three changes: the Hamiltonian, occupied manifold, and projector move to
k+q, while the subtracted eigenvalue stays ε_{n,k}. On a Monkhorst-Pack mesh with
``time_reversal=False`` every k+q folds to another mesh point k_j = k+q − G0
(integer umklapp G0); the difference is carried by the box phase e^{iG0·r}.

Insulator, nspin=1 (matching the q=0 insulator χ₀ path). Requires a
``use_symmetry=False`` result whose k-mesh is the full unfolded MP grid.
"""

from __future__ import annotations

import numpy as np
import torch

from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf._response import sternheimer_shift
from gradwave.scf.implicit import _hamiltonians, _occupied, projected_cg
from gradwave.scf.loop import SCFResult
from gradwave.solvers.precond import teter


def _fold(x: np.ndarray) -> np.ndarray:
    return -((-x + 0.5) % 1.0 - 0.5)


def kpq_map(k_frac: np.ndarray, q_frac: np.ndarray, tol: float = 1e-6):
    """Index of k+q on the mesh and the umklapp G0 = round(k+q − fold(k+q)).

    Returns ``(jidx (nk,), G0 (nk,3) int)`` with ``fold(k+q) == k_frac[jidx]``.
    Raises if k+q is not a mesh point (q not commensurate, or a TR-reduced mesh).
    """
    k_frac = np.asarray(k_frac, dtype=float)
    q = np.asarray(q_frac, dtype=float)
    kq = k_frac + q
    kqf = _fold(kq)
    g0 = np.round(kq - kqf).astype(np.int64)
    # match each folded k+q against the mesh (round to 1e-6 like monkhorst_pack)
    key = {tuple(np.round(k, 6)): i for i, k in enumerate(k_frac)}
    jidx = np.empty(len(k_frac), dtype=np.int64)
    for i, kf in enumerate(kqf):
        j = key.get(tuple(np.round(kf, 6)))
        if j is None:
            raise ValueError(
                f"k+q={kq[i]} folds to {kf} which is not a mesh point — q must be "
                "commensurate with the k-mesh and the mesh must be un-reduced "
                "(time_reversal=False, use_symmetry=False)")
        jidx[i] = j
    return jidx, g0


def _g0_phase(shape, g0: np.ndarray, device) -> torch.Tensor:
    """e^{iG0·r} on the real-space box (n1,n2,n3), G0 in integer Miller units."""
    axes = [torch.arange(n, device=device, dtype=RDTYPE) / n for n in shape]
    grids = torch.meshgrid(*axes, indexing="ij")
    ph = 2 * np.pi * (int(g0[0]) * grids[0] + int(g0[1]) * grids[1]
                      + int(g0[2]) * grids[2])
    return torch.exp(1j * ph.to(CDTYPE))


def _sternheimer_kq(h_kq, c_occ_kq, eps_k, rhs_r, sphere_kq, alpha, tol, max_iter):
    """Solve (H_{k+q} − ε_{n,k} + α P_{k+q}) δψ = −P_c^{k+q}(rhs_r) at one k.

    ``rhs_r`` is the real-space perturbation field δV_q·ψ_{n,k} already carried
    onto the k+q sphere's periodic convention (n_occ, *box). ``eps_k`` are the k
    eigenvalues (not k+q). Returns δψ (n_occ, npw_{k+q}) in the k+q conduction
    space."""
    def p_c(x):
        return x - (x @ c_occ_kq.conj().T) @ c_occ_kq

    w_psi = box_to_sphere(r_to_g(rhs_r), sphere_kq.flat_idx)
    rhs = -p_c(w_psi)

    def a_apply(x):
        hx = h_kq.apply(x) - eps_k[:, None] * x
        return p_c(hx) + alpha * ((x @ c_occ_kq.conj().T) @ c_occ_kq)

    x = torch.zeros_like(rhs)
    r = rhs - a_apply(x)
    t_g = h_kq.t
    t_band = torch.clamp(
        torch.einsum("bg,g,bg->b", c_occ_kq.conj(), t_g.to(c_occ_kq.dtype),
                     c_occ_kq).real, min=1e-6)
    x = projected_cg(a_apply, lambda rr: teter(rr, t_g, t_band), x, r, tol, max_iter)
    return p_c(x)


def chi0_q(res: SCFResult, q_frac, v_box: torch.Tensor, tol: float = 1e-8,
           max_iter: int = 200, *, k_indices=None, k_weights=None,
           symmetrizer=None) -> torch.Tensor:
    """Bare density response δρ_q = χ₀[δV_q] for wavevector q (nspin=1 insulator).

    ``v_box`` is the cell-periodic part v(r) of δV_q(r) = e^{iq·r} v(r), a complex
    field on the FFT box. Returns the cell-periodic part of the **+q Fourier
    component** δρ_{+q}(r) = Σ_{nk} f w_k ψ*_{n,k} δψ_{n,k+q} (so the physical
    δρ_q(r) = e^{iq·r} · result). The −q term δψ* ψ belongs to δρ_{−q}. At q=0 the
    two coincide, so the full real q=0 response is δρ_0 = 2·Re δρ_{+q}|_{q=0} =
    ``scf.implicit.apply_chi0`` (a use_symmetry=False result)."""
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("chi0_q: nspin=1 only")
    if getattr(res.system, "sym", None) is not None:
        raise ValueError("chi0_q needs a use_symmetry=False result (full k-mesh)")
    system = res.system
    grid = system.grid
    shape = grid.shape
    hs = _hamiltonians(res)  # nspin=1 → flat per-k list
    k_frac = np.stack([sph.k_frac for sph in system.spheres])
    jidx, g0 = kpq_map(k_frac, q_frac)
    v_box = v_box.to(CDTYPE)
    ks = range(len(system.spheres)) if k_indices is None else list(k_indices)
    kw = system.kweights if k_weights is None else k_weights
    dr_r = torch.zeros(shape, dtype=CDTYPE, device=grid.g2.device)
    for ik in ks:
        c_occ_k, eps_k = _occupied(res, 0, ik)
        j = int(jidx[ik])
        c_occ_kq, _eps_kq = _occupied(res, 0, j)
        h_kq = hs[j]
        sphere_kq = system.spheres[j]
        ph = _g0_phase(shape, g0[ik], grid.g2.device)     # e^{iG0·r}
        psi_box = g_to_r(c_occ_k, system.spheres[ik].flat_idx, shape)  # u_{n,k}
        # δV_q·ψ_k periodic part relative to k_j = (v·u_k)·e^{iG0·r}
        rhs_r = psi_box * v_box.unsqueeze(0) * ph.unsqueeze(0)
        alpha = sternheimer_shift(eps_k)
        dpsi = _sternheimer_kq(h_kq, c_occ_kq, eps_k, rhs_r, sphere_kq,
                               alpha, tol, max_iter)
        dpsi_box = g_to_r(dpsi, sphere_kq.flat_idx, shape)
        # δρ_q periodic part: ψ*_k δψ_{k+q} with the umklapp phase undone
        dr_r += 2.0 * float(kw[ik]) * (
            psi_box.conj() * dpsi_box * ph.conj().unsqueeze(0)).sum(dim=0)
    dr_r = dr_r / grid.volume
    if symmetrizer is not None:
        dr_r = symmetrizer.apply(r_to_g(dr_r))
        from gradwave.core.fftbox import g_to_r_box
        dr_r = g_to_r_box(dr_r)
    return dr_r


def _k_hxc_q(res, xc, u_r, q_cart):
    """(K_Hxc^q u)(r): the q-shifted Hartree kernel 4πe²/|q+G|² plus the local
    f_xc, applied to a complex wavevector-q field. Unlike q=0 there is no G=0
    exclusion (|q+G| ≠ 0 for q ≠ Γ); at Γ the G=0 term is dropped. f_xc is a real
    local operator, so it acts on the real and imaginary parts separately."""
    import math

    from gradwave.constants import E2
    from gradwave.core.fftbox import g_to_r_box
    from gradwave.postscf._response import fxc_hvp

    grid = res.system.grid
    core = res.system.rho_core
    rho_xc = res.rho if core is None else res.rho + core
    qpg2 = ((grid.g_cart + q_cart) ** 2).sum(dim=-1)             # |q+G|²
    inv = torch.where(qpg2 > 1e-12, 1.0 / torch.clamp(qpg2, min=1e-12),
                      torch.zeros_like(qpg2))
    kh = g_to_r_box(4.0 * math.pi * E2 * r_to_g(u_r.to(CDTYPE)) * inv)  # complex
    fx = (fxc_hvp(xc, rho_xc, grid, u_r.real)
          + 1j * fxc_hvp(xc, rho_xc, grid, u_r.imag)).to(CDTYPE)
    return kh + fx


def screened_response_q(res: SCFResult, xc, q_frac, dv_ext: torch.Tensor,
                        tol: float = 1e-8, max_iter: int = 200,
                        dyson_tol: float = 1e-8, dyson_iter: int = 60,
                        beta: float = 0.4, history: int = 8) -> torch.Tensor:
    """Screened (self-consistent) +q density response to a bare perturbation
    ``dv_ext`` (periodic part), by the Dyson fixed point
    δρ = χ₀[δV_ext] + χ₀[K_Hxc^q δρ] (nspin=1 insulator).

    ``dv_ext`` is the cell-periodic part of the bare perturbation δV_ext,q(r) =
    e^{iq·r} dv_ext(r); returns the periodic part of the screened δρ_{+q}. K_Hxc
    is a local field, so the fixed point stays at wavevector q. This is the
    interacting density response at finite q (the object phonon DFPT screens
    with); at q=0 for a real perturbation it is real."""
    from gradwave.core._anderson import AndersonMixer

    b = 2.0 * np.pi * np.linalg.inv(np.asarray(res.system.grid.cell, float)).T
    q_cart = torch.as_tensor(np.asarray(q_frac, float) @ b,
                             dtype=RDTYPE, device=res.system.grid.g2.device)

    def bare(field):
        return chi0_q(res, q_frac, field, tol=tol, max_iter=max_iter)

    drho0 = bare(dv_ext)

    def g(u):
        return drho0 + bare(_k_hxc_q(res, xc, u, q_cart))

    u = drho0.clone()
    mixer = AndersonMixer(history, beta)
    step = float("inf")
    for _ in range(dyson_iter):
        r = g(u) - u
        step = float(torch.linalg.norm(r)) / max(1.0, float(torch.linalg.norm(u)))
        if step < dyson_tol:
            return u
        u = mixer.step(u.reshape(-1), r.reshape(-1)).reshape(u.shape)
    raise RuntimeError(f"screened q-response not converged ({step:.2e} after {dyson_iter})")


def _little_group_orbits(k_frac: np.ndarray, sg, q_frac):
    """Partition the mesh k-points into orbits under the little co-group of q
    (reciprocal action W⁻ᵀk). Returns (rep_indices, multiplicities) — one
    representative per orbit and the orbit size."""
    from gradwave.symmetry import _k_ops, little_cogroup

    lg, _ = little_cogroup(q_frac, sg)
    ops_t = _k_ops(lg.rotations)
    n = len(k_frac)
    key = {tuple(np.round(_fold(k), 6)): i for i, k in enumerate(k_frac)}
    owner = -np.ones(n, dtype=np.int64)
    reps, mult = [], []
    for i in range(n):
        if owner[i] >= 0:
            continue
        orbit = set()
        for w_t in ops_t:
            j = key.get(tuple(np.round(_fold(w_t @ k_frac[i]), 6)))
            if j is not None:
                orbit.add(j)
        for j in orbit:
            owner[j] = len(reps)
        reps.append(i)
        mult.append(len(orbit))
    return np.asarray(reps), np.asarray(mult)


def chi0_q_reduced(res: SCFResult, q_frac, v_box: torch.Tensor, sg,
                   tol: float = 1e-8, max_iter: int = 200) -> torch.Tensor:
    """The q≠0 star-unfold: chi0_q summed over the little-group IBZ of q and
    folded by ``QFieldSymmetrizer``, reproducing the full-mesh chi0_q at
    ~|little co-group|-fold lower k-cost.

    Valid when the perturbation v is invariant under the little co-group of q
    (an ordinary-rep q — construction raises for a projective small rep). The
    k-points reduce to the little-group orbits (rep × multiplicity); the folded
    response reconstructs the star. ``sg`` is the crystal ``SpaceGroup``."""
    from gradwave.symmetry import QFieldSymmetrizer

    system = res.system
    grid = system.grid
    qsym = QFieldSymmetrizer(grid.shape, q_frac, sg, grid.cell, grid.g2,
                             grid.dens_mask)
    k_frac = np.stack([sph.k_frac for sph in system.spheres])
    reps, mult = _little_group_orbits(k_frac, sg, q_frac)
    kw = system.kweights
    k_weights = {int(r): float(kw[r]) * int(m) for r, m in zip(reps, mult, strict=True)}
    # per-rep weight = its own kweight × orbit size (the star multiplicity)
    weights = torch.zeros(len(system.spheres), dtype=kw.dtype, device=kw.device)
    for r, w in k_weights.items():
        weights[r] = w
    return chi0_q(res, q_frac, v_box, tol, max_iter,
                  k_indices=[int(r) for r in reps], k_weights=weights,
                  symmetrizer=qsym)
