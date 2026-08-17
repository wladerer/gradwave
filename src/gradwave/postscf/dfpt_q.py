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
           max_iter: int = 200) -> torch.Tensor:
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
    dr_r = torch.zeros(shape, dtype=CDTYPE, device=grid.g2.device)
    for ik in range(len(system.spheres)):
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
        dr_r += 2.0 * float(system.kweights[ik]) * (
            psi_box.conj() * dpsi_box * ph.conj().unsqueeze(0)).sum(dim=0)
    return dr_r / grid.volume
