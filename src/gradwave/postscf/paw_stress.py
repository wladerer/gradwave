"""Stress for ultrasoft/PAW — strain autograd (extends postscf/stress.py).

On top of the norm-conserving strain terms (see stress.py), USPP/PAW adds:

- strained augmentation form factors Q̃_ij(G(ε)) (differentiable SBT per L +
  Y_LM(Ĝ(ε)), L up to 4) entering ρ_aug(ε) in every density term,
- strained projectors in becp(ε) → E_NL, becsum, and the S-constraint term
  −Σ w f ε_n ⟨ψ|S(ε)|ψ⟩ (also carries the 1/√Ω normalization of β),
- the one-center chain Σ_a ddd_a·ρ^a_ij(ε) (ddd is strain-independent — the
  radial one-center integrals never see the cell),
- the smooth-coefficient split: ρ̃_s fixed, ρ_aug(ε) rebuilt.

σ = (1/Ω)∂E/∂ε as in stress.py; validated against QE tstress on Si kjpaw.
"""

from __future__ import annotations

import math

import torch

from gradwave.constants import E2, HBAR2_2M
from gradwave.core.fftbox import r_to_g
from gradwave.core.ylm import ylm_all
from gradwave.postscf.paw_forces import _aug_at_fixed
from gradwave.postscf.stress import _box_millers, _ewald_strained
from gradwave.pseudo.radial_torch import RadialTables, sbt_t, simpson_weights

_MINUS_I_POW = [1.0 + 0.0j, -1.0j, -1.0 + 0.0j, 1.0j, 1.0 + 0.0j]  # (−i)^L, L ≤ 4


def stress_uspp(res: dict, xc) -> torch.Tensor:
    """σ (3,3) [eV/Å³] for a converged scf_uspp result (nspin=1)."""
    if res.get("nspin", 1) != 1:
        raise NotImplementedError("USPP/PAW stress for nspin=2 not implemented yet")
    system = res["system"]
    eps = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    e = _energy_strained_uspp(res, xc, eps)
    (grad,) = torch.autograd.grad(e, eps)
    return 0.5 * (grad + grad.T) / system.grid.volume


def _energy_strained_uspp(res: dict, xc, eps: torch.Tensor) -> torch.Tensor:
    system = res["system"]
    grid = system.grid
    shape = grid.shape
    rdt = torch.float64
    cdt = torch.complex128

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

    coeffs = [c.detach() for c in res["coeffs"]]
    occ = res["occupations"].detach()
    eigs = res["eigenvalues"].detach()
    kw = system.kweights
    tabs = [RadialTables(p) for p in system.paws]
    lmax_b = max(b.l for p in system.paws for b in p.betas)

    # ---- kinetic + nonlocal + S-term + strained becsum, per k
    e_kin = torch.zeros((), dtype=rdt)
    e_nl = torch.zeros((), dtype=rdt)
    e_sconstr = torch.zeros((), dtype=rdt)
    q_full = system.q_full.to(cdt)
    rho_ij = [torch.zeros(s1 - s0, s1 - s0, dtype=cdt)
              for (s0, s1) in system.atom_slices]
    for ik, sph in enumerate(system.spheres):
        kfrac = torch.as_tensor(sph.k_frac, dtype=rdt)
        kpg = (sph.miller.to(rdt) + kfrac) @ b_e
        kpg2 = (kpg**2).sum(-1)
        c = coeffs[ik]
        band = torch.einsum("bg,g->b", (c.real**2 + c.imag**2), HBAR2_2M * kpg2)
        e_kin = e_kin + (kw[ik] * occ[ik] * band).sum()

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
        quad_d = torch.einsum("bi,ij,bj->b", b_ovl.conj(), pd.dij_full.to(cdt), b_ovl).real
        e_nl = e_nl + (kw[ik] * occ[ik] * quad_d).sum()
        quad_q = torch.einsum("bi,ij,bj->b", b_ovl.conj(), q_full, b_ovl).real
        e_sconstr = e_sconstr + (kw[ik] * occ[ik] * eigs[ik] * quad_q).sum()
        # ALSO the Σ|c|² part of ⟨ψ|S|ψ⟩ is strain-independent → omitted
        w = (kw[ik] * occ[ik]).to(cdt)
        for a, (s0, s1) in enumerate(system.atom_slices):
            ba = b_ovl[:, s0:s1]
            rho_ij[a] = rho_ij[a] + torch.einsum("b,bi,bj->ij", w, ba.conj(), ba)
    rho_ij = [0.5 * (m + m.conj().T) for m in rho_ij]

    # ---- strained augmentation density on the sphere
    from gradwave.core.gaunt import real_gaunt_table

    gaunt = torch.as_tensor(real_gaunt_table(lmax_b))  # (L2, nlmb, nlmb)
    y_aug = ylm_all(2 * lmax_b, g_sph)  # (nGm, (2lb+1)²)
    phase_arg = g_sph @ pos_e.T
    phases = torch.exp(torch.complex(torch.zeros_like(phase_arg), -phase_arg))
    aug_sph = torch.zeros(g_sph.shape[0], dtype=cdt)
    for a, sp in enumerate(system.species_of_atom):
        paw = system.paws[sp]
        tab = tabs[sp]
        # per-atom angular coefficients B_{LM} for each (channel pair, L)
        idx = []
        for i, bb in enumerate(paw.betas):
            for m in range(2 * bb.l + 1):
                idx.append((i, bb.l * bb.l + m))
        n_aug = paw.aug_cutoff_idx
        w_aug = torch.as_tensor(simpson_weights(paw.rab[:n_aug]))
        r_aug = torch.as_tensor(paw.r[:n_aug])
        acc_a = torch.zeros(g_sph.shape[0], dtype=cdt)
        for (i, j, ll), qfun in paw.qijl.items():
            # B_LM = Σ_{ma,mb} c[LM, lm_a, lm_b] ρ_ab over the (i,j) channels
            rows_i = [k for k, (ci, _) in enumerate(idx) if ci == i]
            rows_j = [k for k, (cj, _) in enumerate(idx) if cj == j]
            lm_i = [idx[k][1] for k in rows_i]
            lm_j = [idx[k][1] for k in rows_j]
            cblk = gaunt[ll * ll:(ll + 1) ** 2][:, lm_i][:, :, lm_j].to(cdt)
            rblk = rho_ij[a][rows_i][:, rows_j]
            b_lm = torch.einsum("Mij,ij->M", cblk, rblk)
            if i != j:  # (j,i) partner uses the transposed becsum block
                b_lm = b_lm + torch.einsum("Mij,ji->M", cblk, rho_ij[a][rows_j][:, rows_i])
            if float(b_lm.detach().abs().max()) < 1e-14:
                continue
            fq = sbt_t(ll, torch.as_tensor(qfun), r_aug, w_aug, q_sph)
            ang = (y_aug[:, ll * ll:(ll + 1) ** 2].to(cdt) @ b_lm)
            acc_a = acc_a + _MINUS_I_POW[ll] * fq.to(cdt) * ang
        aug_sph = aug_sph + phases[:, a] * 4.0 * math.pi * acc_a
        _ = tab
    aug_sph = aug_sph / omega.to(cdt)

    # ---- density coefficients: fixed smooth part + strained augmentation
    rho_s = (res["rho"].detach() - _aug_at_fixed(res, system)).detach()
    rho_st = (r_to_g(rho_s.to(cdt)) * omega0).reshape(-1)[mask]  # [e], fixed
    rho_sph = rho_st / omega.to(cdt) + aug_sph

    g2_safe = torch.where(is_g0, torch.ones_like(g2_sph), g2_sph)
    inv_g2 = torch.where(is_g0, torch.zeros_like(g2_sph), 1.0 / g2_safe)
    e_h = 0.5 * 4.0 * math.pi * E2 * omega * ((rho_sph.abs() ** 2) * inv_g2).sum()

    # ---- local pseudopotential
    e_loc = torch.zeros((), dtype=rdt)
    for sp, tab in enumerate(tabs):
        atoms = [a for a, sa in enumerate(system.species_of_atom) if sa == sp]
        if not atoms:
            continue
        s_sp = phases[:, atoms].sum(dim=1)
        v = torch.zeros_like(q_sph)
        v[~is_g0] = tab.vloc_of_g(q_sph[~is_g0])
        v[is_g0] = tab.alpha
        e_loc = e_loc + (rho_sph.conj() * s_sp * v.to(cdt)).sum().real

    # ---- XC (assemble real-space ρ(ε) from the strained coefficients)
    n_pts = grid.n_points
    rho_box = torch.zeros(n_pts, dtype=cdt)
    rho_box[sphere_idx] = rho_sph
    rho_r = torch.fft.ifftn(rho_box.reshape(shape) * n_pts, dim=(-3, -2, -1)).real
    rho_xc = rho_r
    if system.rho_core is not None:
        core = torch.zeros(g_sph.shape[0], dtype=cdt)
        for sp, tab in enumerate(tabs):
            if tab.core_g is None:
                continue
            atoms = [a for a, sa in enumerate(system.species_of_atom) if sa == sp]
            if not atoms:
                continue
            f_core = tab.core_of_g(q_sph)
            core = core + phases[:, atoms].sum(dim=1) * f_core.to(cdt) / omega.to(cdt)
        core_box = torch.zeros(n_pts, dtype=cdt)
        core_box[sphere_idx] = core
        rho_xc = rho_r + torch.fft.ifftn(
            core_box.reshape(shape) * n_pts, dim=(-3, -2, -1)
        ).real
    sigma = None
    if xc.needs_gradient:
        from gradwave.core.density import sigma_from_rho

        g_box = (m_box @ b_e).reshape(*shape, 3)
        sigma = sigma_from_rho(rho_xc, g_box)
    e_xc = xc.energy(rho_xc, omega, sigma)

    # ---- one-center chain (ddd strain-independent) + Ewald
    e_1c = torch.zeros((), dtype=rdt)
    if any(p.is_paw for p in system.paws):
        from gradwave.scf.paw_onsite import OneCenter

        onec = {sp: OneCenter(system.paws[sp], xc)
                for sp in set(system.species_of_atom)}
        for a, sp in enumerate(system.species_of_atom):
            _, ddd = onec[sp].energy_and_ddd(res["rho_ij_atoms"][a])
            e_1c = e_1c + (ddd.to(cdt) * rho_ij[a]).sum().real
    e_ew = _ewald_strained(pos_e, system.charges, a_e, b_e, omega, grid.cell)

    return e_kin + e_h + e_xc + e_loc + e_nl + e_ew + e_1c - e_sconstr
