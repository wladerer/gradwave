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
from gradwave.core.hamiltonian import projectors
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf._anderson import AndersonMixer
from gradwave.postscf._kb import projector_data_at_k, species_projector_tables
from gradwave.postscf._response import (
    cg_sternheimer,
    fxc_hvp_spin,
    hartree_kernel,
    insulator_window,
    pad_coeffs,
    sternheimer_shift,
)
from gradwave.scf.implicit import apply_k_hxc


def _shifted_projectors(system, dkvec: torch.Tensor) -> torch.Tensor:
    """Full KB projectors (nk, nproj, npw_max) rebuilt at k+G shifted by dkvec
    — radial form factors re-evaluated by SBT at the shifted |k+G|, Ylm and
    the e^{−i(k+G)τ} phases at the shifted vectors."""
    bk = system.batch
    beta_ls, dij_species = species_projector_tables(system.upfs)
    nproj = bk.proj_phase_free.shape[1]
    out = torch.zeros(len(system.spheres), nproj, bk.npw_max, dtype=CDTYPE,
                      device=bk.kpg.device)
    for ik, sph in enumerate(system.spheres):
        shim = SimpleNamespace(kpg=sph.kpg + dkvec, npw=sph.npw)
        pd = projector_data_at_k(shim, system.species_of_atom, system.upfs,
                                 beta_ls, dij_species, system.grid.volume,
                                 device=shim.kpg.device)
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
    """ε∞ (3,3) and Born effective charges Z* (na,3,3) from E-field DFPT.

    Collinear spin (nspin=2) is threaded per channel: the Sternheimer solve runs
    independently for each spin (H is block-diagonal in spin), and the screening
    field u^σ couples the two channels through the spin Hxc kernel K_Hxc^{σσ'}
    (Hartree on the total Δρ + f_xc^{σσ'}), exactly as the linear-response Hubbard
    U does. In the nonmagnetic limit the two channels are identical and the result
    reduces to the nspin=1 value.
    """
    system = res.system
    nspin = int(getattr(res, "nspin", 1))
    if nspin not in (1, 2):
        raise NotImplementedError("dielectric response: nspin must be 1 or 2")
    if system.is_fr:
        raise NotImplementedError("dielectric response: scalar-relativistic only")
    if system.sym is not None:
        raise NotImplementedError("dielectric response requires use_symmetry=False")
    if nspin == 2:
        return _dielectric_born_spin(
            res, xc, dk=dk, cg_tol=cg_tol, beta=beta, outer_tol=outer_tol,
            max_outer=max_outer, history=history, verbose=verbose)
    bk, grid = system.batch, system.grid
    kw = system.kweights
    vol = grid.volume

    nocc = insulator_window(res.occupations, 2.0,
                            "insulating occupations (f=2) required")
    c_occ = pad_coeffs(res.coeffs, bk.npw_max)[:, :nocc]
    eps_occ = res.eigenvalues[:, :nocc].to(RDTYPE)
    shift = sternheimer_shift(eps_occ)

    from gradwave.core.batch import BatchedHamiltonian, box_to_sphere_b

    h = BatchedHamiltonian(bk, grid.shape, res.v_eff, projectors_b(bk, system.positions))

    def p_c(x):
        ov = torch.einsum("kng,kbg->kbn", c_occ.conj(), x)
        return x - torch.einsum("kbn,kng->kbg", ov, c_occ)

    # ξ^α = P_c r_α ψ via Sternheimer with the ∂H/∂k commutator RHS
    xi = []
    for a in range(3):
        rhs = -1j * p_c(_dhdk_psi(system, c_occ, a, dk))
        xi.append(cg_sternheimer(h, bk, c_occ, eps_occ, rhs,
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
            dpsi = cg_sternheimer(h, bk, c_occ, eps_occ, rhs, dpsi, shift,
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


def _k_hxc_spin(res, xc, dru, drd):
    """(Δv↑, Δv↓) = K_Hxc^{σσ'} Δρ^{σ'}: Hartree kernel on the total Δρ (G=0
    excluded) plus the spin f_xc Hessian-vector product at the SCF spin densities
    (NLCC core split half/half per channel, exactly as the SCF potential built
    it). The shared postscf._response primitives, matching hubbard_u._k_hxc_spin."""
    core = res.system.rho_core
    cu2 = 0.0 if core is None else 0.5 * core
    kh = hartree_kernel(res.system.grid, dru + drd)
    fu, fd = fxc_hvp_spin(xc, res.rho_spin[0] + cu2, res.rho_spin[1] + cu2,
                          res.system.grid, dru, drd)
    return kh + fu, kh + fd


@torch.no_grad()
def _dielectric_born_spin(res, xc, *, dk, cg_tol, beta, outer_tol, max_outer,
                          history, verbose) -> dict:
    """ε∞ and Born charges for a collinear spin-polarized insulator (nspin=2).

    Mirrors ``dielectric_born`` per spin channel: the ∂H/∂k RHS and the
    conduction-projected Sternheimer solve run independently for each spin (H is
    block-diagonal in spin), while the self-consistent screening field u^σ folds
    the two channels together through K_Hxc^{σσ'}. Each channel carries one
    electron (f=1), so the prefactors halve per channel and sum over spin — 8π
    instead of 16π on ε, 2 instead of 4 on the density/Born terms — reducing to
    the nspin=1 value in the nonmagnetic limit."""
    from gradwave.core.batch import BatchedHamiltonian, box_to_sphere_b

    system = res.system
    bk, grid = system.batch, system.grid
    kw = system.kweights
    vol = grid.volume

    # per-spin occupied window, Hamiltonian, Sternheimer shift, conduction proj
    projs_b = projectors_b(bk, system.positions)
    c_occ, eps_occ, hs, shift = [], [], [], []
    for sp in range(2):
        nocc = insulator_window(res.occupations[sp], 1.0,
                                "insulating occupations (f=1 per spin) required")
        c_occ.append(pad_coeffs(res.coeffs[sp], bk.npw_max)[:, :nocc])
        eps_occ.append(res.eigenvalues[sp][:, :nocc].to(RDTYPE))
        hs.append(BatchedHamiltonian(bk, grid.shape, res.v_eff[sp], projs_b))
        shift.append(sternheimer_shift(eps_occ[sp]))

    def p_c(x, sp):
        ov = torch.einsum("kng,kbg->kbn", c_occ[sp].conj(), x)
        return x - torch.einsum("kbn,kng->kbg", ov, c_occ[sp])

    # ξ^α_σ = P_c r_α ψ_σ per spin channel (∂H/∂k is spin-independent)
    xi = [[None, None, None] for _ in range(2)]
    for sp in range(2):
        for a in range(3):
            rhs = -1j * p_c(_dhdk_psi(system, c_occ[sp], a, dk), sp)
            xi[sp][a] = cg_sternheimer(hs[sp], bk, c_occ[sp], eps_occ[sp], rhs,
                                       torch.zeros_like(rhs), shift[sp], tol=cg_tol)

    psi_r = [g_to_r_b(c_occ[sp], bk, grid.shape) for sp in range(2)]
    n_pts = grid.n_points
    eps_mat = torch.zeros(3, 3, dtype=RDTYPE)
    dpsi_all = [[], []]          # dpsi_all[sp] -> per field direction
    drho_tot_all = []            # total Δρ per field direction (Born local part)
    for b_dir in range(3):
        # Anderson fixed point on the two-channel screening field u = (u↑, u↓)
        u_flat = torch.zeros(2 * n_pts, dtype=RDTYPE, device=c_occ[0].device)
        mixer = AndersonMixer(history, beta)
        dpsi = [torch.zeros_like(c_occ[sp]) for sp in range(2)]
        col_prev = None
        for it in range(1, max_outer + 1):
            drho = []
            for sp in range(2):
                rhs = -xi[sp][b_dir]
                if it > 1:
                    u_r = u_flat[sp * n_pts:(sp + 1) * n_pts].reshape(grid.shape)
                    rhs = rhs - p_c(
                        box_to_sphere_b(psi_r[sp] * u_r.to(psi_r[sp].dtype), bk), sp)
                dpsi[sp] = cg_sternheimer(hs[sp], bk, c_occ[sp], eps_occ[sp], rhs,
                                          dpsi[sp], shift[sp], tol=cg_tol)
                dpsi_r = g_to_r_b(dpsi[sp], bk, grid.shape)
                drho.append(2.0 * (kw[:, None, None, None, None]
                                   * (psi_r[sp].conj() * dpsi_r).real).sum(dim=(0, 1))
                            / vol)
            # ε column (summed over spin): f=1 per channel, so 8π not 16π below
            col = torch.tensor([
                float(sum(
                    (kw[:, None] * torch.einsum(
                        "kbg,kbg->kb", xi[sp][a].conj(), dpsi[sp]).real).sum()
                    for sp in range(2)))
                for a in range(3)])
            if verbose:
                print(f"  E{b_dir} it {it:3d}: eps col = "
                      f"{[round(1 - 8 * math.pi * E2 / vol * c, 6) for c in col.tolist()]}")
            if col_prev is not None and float((col - col_prev).abs().max()) < outer_tol:
                break
            col_prev = col
            du, dd = _k_hxc_spin(res, xc, drho[0], drho[1])
            r_vec = torch.cat([du.reshape(-1), dd.reshape(-1)]) - u_flat
            u_flat = mixer.step(u_flat, r_vec)
        else:
            raise RuntimeError(f"E-field response ({b_dir}) not converged")
        for sp in range(2):
            dpsi_all[sp].append(dpsi[sp])
        drho_tot_all.append(drho[0] + drho[1])
        eps_mat[:, b_dir] = 1.0 * torch.eye(3)[:, b_dir] \
            - (8.0 * math.pi * E2 / vol) * col

    # Born charges: mixed derivative ∂²E/∂E_α∂τ via one autograd backward per
    # field direction. Local part sees the total Δρ; the nonlocal part sums the
    # two spin channels (factor 2 per channel = f·c.c.).
    na = len(system.species_of_atom)
    born = torch.zeros(na, 3, 3, dtype=RDTYPE)
    dij_c = bk.dij_full.to(CDTYPE)
    for a in range(3):
        pos = system.positions.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            vloc_g = local_potential_g(pos, system.species_index,
                                       system.vloc_tables, grid.g_cart, vol)
            t_loc = local_energy(r_to_g(drho_tot_all[a].to(CDTYPE)), vloc_g, vol)
            p = projectors_b(bk, pos)
            t_nl = 0.0
            for sp in range(2):
                b_c = torch.einsum("kpg,kbg->kbp", p.conj(), c_occ[sp])
                b_d = torch.einsum("kpg,kbg->kbp", p.conj(), dpsi_all[sp][a])
                t_nl = t_nl + 2.0 * torch.einsum(
                    "k,kbp,pq,kbq->", kw.to(CDTYPE), b_d.conj(), dij_c, b_c).real
            (grad,) = torch.autograd.grad(t_loc + t_nl, pos)
        for s in range(na):
            born[s, a] = -grad[s]
            born[s, a, a] += float(system.charges[s])
    asr = born.sum(dim=0)
    return {"eps": eps_mat, "born": born, "asr": asr,
            "eps_iso": float(torch.diagonal(eps_mat).mean())}
