"""Macroscopic dielectric tensor ε∞ and Born effective charges (E-field DFPT).

Continues the autodiff-DFPT theme of hubbard_u.py — the three derivative
objects are obtained without hand-coding any of the classic DFPT calculus:

- P_c r|ψ⟩ (the position operator on Bloch states) from a Sternheimer solve
  with (∂H/∂k)|ψ⟩ on the RHS:  (H − ε_v)|ξ^α_v⟩ = −i P_c (∂H/∂k_α)|ψ_v⟩.
  ∂H/∂k_α = analytic kinetic part + central finite difference IN K of the
  KB nonlocal tables (radial SBT + Ylm rebuilt at k ± δk e_α; the form
  factors are smooth, so δk = 1e-3 Å⁻¹ is exact to O(δk²)).
- The self-consistent field response screens through K_Hxc from the autograd
  HVP of E_Hxc (scf/implicit.apply_k_hxc) — no hand-coded f_xc.
- Born charges are the mixed derivative ∂²E/∂E_α∂τ_sβ: with Δψ^α in hand,
  ONE autograd backward through the position-differentiable pseudopotential
  (structure-factor local part + projector phases — the same graph the
  Hellmann–Feynman forces use) yields all (atom, β) components at once:
      Z*_{s,αβ} = Z_s δ_αβ − ∂/∂τ_{s,β} T^α(τ),
      T^α(τ) = ∫ Δρ^α v_loc(τ) + Σ_kv 4 w Re⟨Δψ^α|V_NL(τ)|ψ⟩.

ε∞_αβ = δ_αβ − (16π e²/Ω) Σ_kv w_k Re⟨ξ^α_v|Δψ^β_v⟩       (f = 2 folded in)

Insulators, nspin=1, scalar-relativistic pseudos, no symmetry reduction
(time-reversal reduction is fine — the response quantities fold evenly).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from gradwave.constants import E2, HBAR2_2M
from gradwave.core.batch import g_to_r_b, projectors_b
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.fftbox import r_to_g
from gradwave.core.hamiltonian import build_projector_data, projectors
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf._anderson import AndersonMixer
from gradwave.postscf.hubbard_u import _cg_sternheimer_b, _pad
from gradwave.pseudo.kb import beta_form_factors
from gradwave.scf.implicit import apply_k_hxc


def _shifted_projectors(system, dkvec: torch.Tensor) -> torch.Tensor:
    """Full KB projectors (nk, nproj, npw_max) rebuilt at k+G shifted by dkvec
    — radial form factors re-evaluated by SBT at the shifted |k+G|, Ylm and
    the e^{−i(k+G)τ} phases at the shifted vectors."""
    bk = system.batch
    beta_ls = [[b.l for b in upf.betas] for upf in system.upfs]
    dij_species = [torch.as_tensor(upf.dij, dtype=RDTYPE) for upf in system.upfs]
    nproj = bk.proj_phase_free.shape[1]
    out = torch.zeros(len(system.spheres), nproj, bk.npw_max, dtype=CDTYPE,
                      device=bk.kpg.device)
    for ik, sph in enumerate(system.spheres):
        kpg_s = sph.kpg + dkvec
        q = torch.linalg.norm(kpg_s, dim=1).cpu().numpy()
        beta_tables = [torch.as_tensor(beta_form_factors(upf, q), dtype=RDTYPE,
                                       device=kpg_s.device)
                       for upf in system.upfs]
        shim = SimpleNamespace(kpg=kpg_s, npw=sph.npw)
        pd = build_projector_data(shim, system.species_of_atom, beta_tables,
                                  beta_ls, dij_species, system.grid.volume)
        out[ik, :, : sph.npw] = projectors(pd, system.positions)
    return out


def _vnl_apply(p: torch.Tensor, dij: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    b = torch.einsum("kpg,kbg->kbp", p.conj(), c)
    return torch.einsum("kbp,pq,kqg->kbg", b, dij, p)


@torch.no_grad()
def _dhdk_psi(system, c_occ: torch.Tensor, alpha: int, dk: float) -> torch.Tensor:
    """(∂H/∂k_α)|ψ⟩: analytic kinetic derivative + FD-in-k of the nonlocal."""
    bk = system.batch
    kin = (2.0 * HBAR2_2M) * bk.kpg[:, None, :, alpha] * c_occ
    if bk.proj_phase_free.shape[1] == 0:
        return kin
    ek = torch.zeros(3, dtype=RDTYPE, device=bk.kpg.device)
    ek[alpha] = dk
    p_p = _shifted_projectors(system, ek)
    p_m = _shifted_projectors(system, -ek)
    dij = bk.dij_full.to(CDTYPE)
    dnl = (_vnl_apply(p_p, dij, c_occ) - _vnl_apply(p_m, dij, c_occ)) / (2.0 * dk)
    return kin + dnl


@torch.no_grad()
def dielectric_born(res, xc, *, dk: float = 1e-3, cg_tol: float = 1e-9,
                    beta: float = 0.5, outer_tol: float = 1e-7,
                    max_outer: int = 80, history: int = 8,
                    verbose: bool = False) -> dict:
    """ε∞ (3,3) and Born effective charges Z* (na,3,3) from E-field DFPT."""
    system = res.system
    if getattr(res, "nspin", 1) != 1:
        raise NotImplementedError("dielectric response: nspin=1 insulators only")
    if system.is_fr:
        raise NotImplementedError("dielectric response: scalar-relativistic only")
    if system.sym is not None:
        raise NotImplementedError("dielectric response requires use_symmetry=False")
    bk, grid = system.batch, system.grid
    kw = system.kweights
    vol = grid.volume

    occ = res.occupations
    nocc = int((occ[0] > 1.0).sum())
    if not ((occ[:, :nocc] - 2.0).abs().max() < 1e-6
            and (occ[:, nocc:].abs().max() < 1e-6 if occ.shape[1] > nocc else True)):
        raise NotImplementedError("insulating occupations (f=2) required")
    c_occ = _pad(res.coeffs, bk.npw_max)[:, :nocc]
    eps_occ = res.eigenvalues[:, :nocc].to(RDTYPE)
    shift = 2.0 * float(eps_occ.max() - eps_occ.min()) + 10.0

    from gradwave.core.batch import BatchedHamiltonian, box_to_sphere_b

    h = BatchedHamiltonian(bk, grid.shape, res.v_eff, projectors_b(bk, system.positions))

    def p_c(x):
        ov = torch.einsum("kng,kbg->kbn", c_occ.conj(), x)
        return x - torch.einsum("kbn,kng->kbg", ov, c_occ)

    # ξ^α = P_c r_α ψ via Sternheimer with the ∂H/∂k commutator RHS
    xi = []
    for a in range(3):
        rhs = -1j * p_c(_dhdk_psi(system, c_occ, a, dk))
        xi.append(_cg_sternheimer_b(h, bk, c_occ, eps_occ, rhs,
                                    torch.zeros_like(rhs), shift, tol=cg_tol))

    psi_r = g_to_r_b(c_occ, bk, grid.shape)
    n_pts = grid.n_points
    eps_mat = torch.zeros(3, 3, dtype=RDTYPE)
    dpsi_all, drho_all = [], []
    for b_dir in range(3):
        # Anderson-accelerated fixed point on u = K_Hxc[Δρ(E-probe + u)]
        u_flat = torch.zeros(n_pts, dtype=RDTYPE, device=c_occ.device)
        mixer = AndersonMixer(history, beta)
        dpsi = torch.zeros_like(c_occ)
        col_prev = None
        for it in range(1, max_outer + 1):
            rhs = -xi[b_dir]
            if it > 1:
                u_r = u_flat.reshape(grid.shape)
                rhs = rhs - p_c(box_to_sphere_b(psi_r * u_r.to(psi_r.dtype), bk))
            dpsi = _cg_sternheimer_b(h, bk, c_occ, eps_occ, rhs, dpsi, shift,
                                     tol=cg_tol)
            dpsi_r = g_to_r_b(dpsi, bk, grid.shape)
            drho = 4.0 * (kw[:, None, None, None, None]
                          * (psi_r.conj() * dpsi_r).real).sum(dim=(0, 1)) / vol
            col = torch.tensor([
                float((kw[:, None] * torch.einsum(
                    "kbg,kbg->kb", xi[a].conj(), dpsi).real).sum())
                for a in range(3)])
            if verbose:
                print(f"  E{b_dir} it {it:3d}: eps col = "
                      f"{[round(1 - 16 * math.pi * E2 / vol * c, 6) for c in col.tolist()]}")
            if col_prev is not None and float((col - col_prev).abs().max()) < outer_tol:
                break
            col_prev = col
            r_vec = apply_k_hxc(res, xc, drho).reshape(-1).to(u_flat.device) - u_flat
            u_flat = mixer.step(u_flat, r_vec)
        else:
            raise RuntimeError(f"E-field response ({b_dir}) not converged")
        dpsi_all.append(dpsi)
        drho_all.append(drho)
        eps_mat[:, b_dir] = 1.0 * torch.eye(3)[:, b_dir] \
            - (16.0 * math.pi * E2 / vol) * col

    # Born charges: mixed derivative via autograd over the τ-differentiable
    # pseudopotential, one backward per field direction.
    na = len(system.species_of_atom)
    born = torch.zeros(na, 3, 3, dtype=RDTYPE)
    dij_c = bk.dij_full.to(CDTYPE)
    for a in range(3):
        pos = system.positions.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            vloc_g = local_potential_g(pos, system.species_index,
                                       system.vloc_tables, grid.g_cart, vol)
            t_loc = local_energy(r_to_g(drho_all[a].to(CDTYPE)), vloc_g, vol)
            p = projectors_b(bk, pos)
            b_c = torch.einsum("kpg,kbg->kbp", p.conj(), c_occ)
            b_d = torch.einsum("kpg,kbg->kbp", p.conj(), dpsi_all[a])
            t_nl = 4.0 * torch.einsum("k,kbp,pq,kbq->", kw.to(CDTYPE),
                                      b_d.conj(), dij_c, b_c).real
            (grad,) = torch.autograd.grad(t_loc + t_nl, pos)
        for s in range(na):
            born[s, a] = -grad[s]
            born[s, a, a] += float(system.charges[s])
    asr = born.sum(dim=0)
    return {"eps": eps_mat, "born": born, "asr": asr,
            "eps_iso": float(torch.diagonal(eps_mat).mean())}
