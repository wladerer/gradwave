"""Stress for ultrasoft/PAW — strain autograd (extends postscf/stress.py).

On top of the norm-conserving strain terms (see stress.py), USPP/PAW adds:

- strained augmentation form factors Q̃_ij(G(ε)) (differentiable SBT per L +
  Y_LM(Ĝ(ε)), L up to 4) entering ρ_aug(ε) in every density term,
- strained projectors in becp(ε) → E_NL, becsum, and the S-constraint term
  −Σ w f ε_n ⟨ψ|S(ε)|ψ⟩ (also carries the 1/√Ω normalization of β),
- the one-center chain Σ_a ddd_a·ρ^a_ij(ε) (ddd is strain-independent — the
  radial one-center integrals never see the cell),
- the smooth-coefficient split: ρ̃_s fixed per spin, ρ_aug(ε) rebuilt.

σ = (1/Ω)∂E/∂ε as in stress.py; validated against QE tstress on Si kjpaw
(nspin=1) and displaced ferromagnetic Ni (nspin=2).
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import E2, HBAR2_2M
from gradwave.core.fftbox import r_to_g
from gradwave.core.ylm import ylm_all
from gradwave.postscf.paw_forces import _aug_at_fixed, _normalize_spin
from gradwave.postscf.stress import _box_millers, _ewald_strained
from gradwave.pseudo.radial_torch import RadialTables, sbt_t, simpson_weights

_MINUS_I_POW = [1.0 + 0.0j, -1.0j, -1.0 + 0.0j, 1.0j, 1.0 + 0.0j]  # (−i)^L, L ≤ 4


def stress_uspp(res: dict, xc) -> torch.Tensor:
    """σ (3,3) [eV/Å³] for a converged scf_uspp result (nspin 1 or 2)."""
    system = res["system"]
    eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    e = _energy_strained_uspp(res, xc, eps)
    (grad,) = torch.autograd.grad(e, eps)
    return 0.5 * (grad + grad.T) / system.grid.volume


def _strained_aug(system, rho_ij, tabs, gaunt, y_aug, q_sph, phases, omega):
    """ρ_aug(G(ε)) on the sphere from one spin channel's strained becsum."""
    cdt = torch.complex128
    aug_sph = torch.zeros(q_sph.shape[0], dtype=cdt)
    for a, sp in enumerate(system.species_of_atom):
        paw = system.paws[sp]
        idx = []
        for i, bb in enumerate(paw.betas):
            for m in range(2 * bb.l + 1):
                idx.append((i, bb.l * bb.l + m))
        n_aug = paw.aug_cutoff_idx
        w_aug = torch.as_tensor(simpson_weights(paw.rab[:n_aug]))
        r_aug = torch.as_tensor(paw.r[:n_aug])
        acc_a = torch.zeros(q_sph.shape[0], dtype=cdt)
        for (i, j, ll), qfun in paw.qijl.items():
            rows_i = [k for k, (ci, _) in enumerate(idx) if ci == i]
            rows_j = [k for k, (cj, _) in enumerate(idx) if cj == j]
            lm_i = [idx[k][1] for k in rows_i]
            lm_j = [idx[k][1] for k in rows_j]
            cblk = gaunt[ll * ll:(ll + 1) ** 2][:, lm_i][:, :, lm_j].to(cdt)
            b_lm = torch.einsum("Mij,ij->M", cblk, rho_ij[a][rows_i][:, rows_j])
            if i != j:  # (j,i) partner uses the transposed becsum block
                b_lm = b_lm + torch.einsum(
                    "Mij,ji->M", cblk, rho_ij[a][rows_j][:, rows_i])
            if float(b_lm.detach().abs().max()) < 1e-14:
                continue
            fq = sbt_t(ll, torch.as_tensor(qfun), r_aug, w_aug, q_sph)
            ang = y_aug[:, ll * ll:(ll + 1) ** 2].to(cdt) @ b_lm
            acc_a = acc_a + _MINUS_I_POW[ll] * fq.to(cdt) * ang
        aug_sph = aug_sph + phases[:, a] * 4.0 * math.pi * acc_a
        _ = tabs
    return aug_sph / omega.to(cdt)


def _energy_strained_uspp(res: dict, xc, eps: torch.Tensor) -> torch.Tensor:
    system = res["system"]
    grid = system.grid
    shape = grid.shape
    rdt = torch.float64
    cdt = torch.complex128
    nspin, coeffs_s, occ_s, eigs_s, becsum_s, rho_sp_mixed = _normalize_spin(res)

    f_map = torch.eye(3, dtype=rdt) + eps
    a0 = torch.as_tensor(grid.cell, dtype=rdt)
    a_e = a0 @ f_map.T
    b_e = 2.0 * math.pi * torch.linalg.inv(a_e).T
    omega0 = grid.volume
    omega = torch.linalg.det(a_e)
    omega = omega * torch.sign(omega.detach())
    pos_e = system.positions.detach() @ f_map.T

    mask = grid.dens_mask.reshape(-1)
    m_box = _box_millers(shape, None)
    m_sph = m_box[mask]
    g_sph = m_sph @ b_e
    g2_sph = (g_sph**2).sum(-1)
    is_g0 = g2_sph.detach() < 1e-12
    q_sph = torch.sqrt(torch.where(is_g0, torch.ones_like(g2_sph), g2_sph))
    q_sph = torch.where(is_g0, torch.zeros_like(q_sph), q_sph)
    sphere_idx = system.sphere_idx

    kw = system.kweights
    tabs = [RadialTables(p) for p in system.paws]
    lmax_b = max(b.l for p in system.paws for b in p.betas)

    from gradwave.core.gaunt import real_gaunt_table

    gaunt = torch.as_tensor(real_gaunt_table(lmax_b))
    y_aug = ylm_all(2 * lmax_b, g_sph)
    phase_arg = g_sph @ pos_e.T
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))

    is_paw = any(p.is_paw for p in system.paws)
    onec = None
    if is_paw:
        from gradwave.scf.paw_onsite import OneCenter

        onec = {sp: OneCenter(system.paws[sp], xc)
                for sp in set(system.species_of_atom)}

    e_total = _ewald_strained(pos_e, system.charges, a_e, b_e, omega, grid.cell)
    q_full = system.q_full.to(cdt)
    rho_sph_chans, rho_r_chans = [], []
    for isp in range(nspin):
        coeffs = [c.detach() for c in coeffs_s[isp]]
        occ = occ_s[isp].detach()
        eigs = eigs_s[isp].detach()
        rho_ij = [torch.zeros(s1 - s0, s1 - s0, dtype=cdt)
                  for (s0, s1) in system.atom_slices]
        for ik, sph in enumerate(system.spheres):
            kfrac = torch.as_tensor(sph.k_frac, dtype=rdt)
            kpg = (sph.miller.to(rdt) + kfrac) @ b_e
            kpg2 = (kpg**2).sum(-1)
            c = coeffs[ik]
            band = torch.einsum("bg,g->b", (c.real**2 + c.imag**2), HBAR2_2M * kpg2)
            e_total = e_total + (kw[ik] * occ[ik] * band).sum()

            q_k = torch.sqrt(kpg2.clamp_min(1e-30))
            q_k = torch.where(kpg2.detach() < 1e-24, torch.zeros_like(q_k), q_k)
            y = ylm_all(lmax_b, kpg)
            pref = 4.0 * math.pi / torch.sqrt(omega)
            cols = []
            for sp in system.species_of_atom:
                tab = tabs[sp]
                for i, ell in enumerate(tab.beta_l):
                    f = tab.beta_of_g(i, q_k)
                    for m_col in range(2 * ell + 1):
                        cols.append(
                            (pref * f * y[:, ell * ell + m_col]).to(cdt)
                            * _MINUS_I_POW[ell]
                        )
            p = torch.stack(cols, dim=0)
            parg = kpg @ pos_e.T
            ph = torch.exp(torch.complex(torch.zeros_like(parg), -parg))
            pd = system.proj_data[ik]
            p = p * ph[:, pd.atom_index].T
            b_ovl = c @ p.conj().T
            quad_d = torch.einsum(
                "bi,ij,bj->b", b_ovl.conj(), pd.dij_full.to(cdt), b_ovl).real
            e_total = e_total + (kw[ik] * occ[ik] * quad_d).sum()
            quad_q = torch.einsum("bi,ij,bj->b", b_ovl.conj(), q_full, b_ovl).real
            e_total = e_total - (kw[ik] * occ[ik] * eigs[ik] * quad_q).sum()
            w = (kw[ik] * occ[ik]).to(cdt)
            for a, (s0, s1) in enumerate(system.atom_slices):
                ba = b_ovl[:, s0:s1]
                rho_ij[a] = rho_ij[a] + torch.einsum("b,bi,bj->ij", w, ba.conj(), ba)
        rho_ij = [0.5 * (m + m.conj().T) for m in rho_ij]

        if is_paw:  # one-center chain with per-spin ddd at the converged becsum
            for a, sp in enumerate(system.species_of_atom):
                bec = (becsum_s[0][a] if nspin == 1
                       else [becsum_s[0][a], becsum_s[1][a]])
                _, ddd = onec[sp].energy_and_ddd(bec)
                ddd_isp = ddd if nspin == 1 else ddd[isp]
                e_total = e_total + (ddd_isp.to(cdt) * rho_ij[a]).sum().real

        aug_sph = _strained_aug(system, rho_ij, tabs, gaunt, y_aug, q_sph,
                                phases, omega)
        rho_s_fix = (rho_sp_mixed[isp].detach()
                     - _aug_at_fixed(res, system, isp)).detach()
        rho_st = (r_to_g(rho_s_fix.to(cdt)) * omega0).reshape(-1)[mask]
        rho_sph_sp = rho_st / omega.to(cdt) + aug_sph
        rho_sph_chans.append(rho_sph_sp)
        n_pts = grid.n_points
        rho_box = torch.zeros(n_pts, dtype=cdt)
        rho_box[sphere_idx] = rho_sph_sp
        rho_r_chans.append(torch.fft.ifftn(
            rho_box.reshape(shape) * n_pts, dim=(-3, -2, -1)).real)

    rho_sph_tot = sum(rho_sph_chans)
    g2_safe = torch.where(is_g0, torch.ones_like(g2_sph), g2_sph)
    inv_g2 = torch.where(is_g0, torch.zeros_like(g2_sph), 1.0 / g2_safe)
    e_total = e_total + 0.5 * 4.0 * math.pi * E2 * omega * (
        (rho_sph_tot.abs() ** 2) * inv_g2).sum()

    # local pseudopotential
    for sp, tab in enumerate(tabs):
        atoms = [a for a, sa in enumerate(system.species_of_atom) if sa == sp]
        if not atoms:
            continue
        s_sp = phases[:, atoms].sum(dim=1)
        v = torch.zeros_like(q_sph)
        v[~is_g0] = tab.vloc_of_g(q_sph[~is_g0])
        v[is_g0] = tab.alpha
        e_total = e_total + (rho_sph_tot.conj() * s_sp * v.to(cdt)).sum().real

    # XC on the strained real-space densities (+ strained NLCC core)
    rho_core_e = None
    if system.rho_core is not None:
        core = torch.zeros(q_sph.shape[0], dtype=cdt)
        for sp, tab in enumerate(tabs):
            if tab.core_g is None:
                continue
            atoms = [a for a, sa in enumerate(system.species_of_atom) if sa == sp]
            if not atoms:
                continue
            f_core = tab.core_of_g(q_sph)
            core = core + phases[:, atoms].sum(dim=1) * f_core.to(cdt) / omega.to(cdt)
        core_box = torch.zeros(grid.n_points, dtype=cdt)
        core_box[sphere_idx] = core
        rho_core_e = torch.fft.ifftn(
            core_box.reshape(shape) * grid.n_points, dim=(-3, -2, -1)).real
    from gradwave.core.density import sigma_from_rho

    g_box = None
    if xc.needs_gradient:
        g_box = (m_box @ b_e).reshape(*shape, 3)
    if nspin == 1:
        rho_xc = rho_r_chans[0] if rho_core_e is None else rho_r_chans[0] + rho_core_e
        sigma = sigma_from_rho(rho_xc, g_box) if xc.needs_gradient else None
        e_total = e_total + xc.energy(rho_xc, omega, sigma)
    else:
        c2 = 0.0 if rho_core_e is None else 0.5 * rho_core_e
        r_u, r_d = rho_r_chans[0] + c2, rho_r_chans[1] + c2
        if xc.needs_gradient:
            s_uu = sigma_from_rho(r_u, g_box)
            s_dd = sigma_from_rho(r_d, g_box)
            s_tt = sigma_from_rho(r_u + r_d, g_box)
        else:
            s_uu = s_dd = s_tt = None
        e_total = e_total + xc.energy(r_u, r_d, omega, s_uu, s_dd, s_tt)

    return e_total
