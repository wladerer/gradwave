"""Ultrasoft/PAW plane-wave SCF — stage 1: augmentation + S-operator (Layer B).

The ultrasoft generalized eigenproblem H|ψ⟩ = ε S|ψ⟩ with

    S = 1 + Σ_a Σ_ij q_ij |β^a_i⟩⟨β^a_j|,   q_ij = ∫ Q_ij(r⃗) d³r
    ρ(r) = ρ_smooth(r) + ρ_aug(r),
    ρ_aug(G) = (1/Ω) Σ_a e^{−iG·τ_a} Σ_ij ρ^a_ij Q̃_ij(G)
    ρ^a_ij = Σ_nk w_k f_nk ⟨ψ|β^a_i⟩⟨β^a_j|ψ⟩
    D^scr_ij = D_ij + ∫ v_eff(r) Q_ij(r⃗−τ_a) d³r        (rebuilt each iteration)

with the augmentation form factors from the UPF's per-L radial functions:

    Q̃_(i,mi),(j,mj)(G) = 4π Σ_{LM} (−i)^L c^{LM}_{limi,ljmj} Y_LM(Ĝ) ∫ q^L_ij(r) j_L(Gr) dr

(c = real Gaunt coefficients, core/gaunt.py). The plane-wave energy terms are
assembled exactly as in the NC loop but with ρ = ρ_s + ρ_aug and the BARE
D_ij in E_NL; the one-center PAW corrections (stage 2, postscf/paw_onsite.py)
add per-atom radial-grid terms on top and are required to match QE's total.

Deliberately per-k and unbatched — correctness first; the batched fast path
can absorb it later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.hartree import hartree_energy, hartree_potential_g
from gradwave.core.energies.kinetic import kinetic_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.energies.total import EnergyBreakdown
from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g
from gradwave.core.hamiltonian import ProjectorData, becp, projectors
from gradwave.core.occupations import (
    SCHEMES,
)
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.scf.common import shared_fermi_occupations
from gradwave.scf.guess import sad_density
from gradwave.scf.layout import MixLayout
from gradwave.scf.loop import vxc_potential
from gradwave.scf.mixing import BroydenMixer, JohnsonMixer, PulayMixer
from gradwave.scf.uspp_setup import USPPSystem
from gradwave.solvers.precond import teter


class _HkS:
    """H and S applies at one k for fixed v_eff and screened D."""

    def __init__(self, sphere, shape, v_eff_r, pd: ProjectorData, p, dscr, q_full,
                 hub_sphi=None, hub_d=None):
        self.sphere, self.shape = sphere, shape
        self.v_eff_r = v_eff_r
        self.p = p
        self.dscr = dscr.to(CDTYPE)
        self.q = q_full.to(CDTYPE)
        self.t = HBAR2_2M * sphere.kpg2
        # DFT+U: S-dressed atomic-orbital projectors + Dudarev D (apply
        # convention wants D^T = conj(D) for Hermitian D, like the NC path)
        self.hub_sphi = hub_sphi
        self.hub_d = hub_d

    def h(self, c):
        out = self.t * c
        psi = g_to_r(c, self.sphere.flat_idx, self.shape)
        out = out + box_to_sphere(r_to_g(psi * self.v_eff_r), self.sphere.flat_idx)
        b = becp(self.p, c)
        out = out + (b @ self.dscr) @ self.p
        if self.hub_sphi is not None:
            bh = becp(self.hub_sphi, c)
            out = out + (bh @ self.hub_d) @ self.hub_sphi
        return out

    def s(self, c):
        b = becp(self.p, c)
        return c + (b @ self.q) @ self.p


def davidson_gen(hs: _HkS, x0: torch.Tensor, nbands: int, tol: float,
                 max_iter: int = 60, max_dim: int | None = None):
    """Block Davidson for H x = ε S x. x0 (nb0 ≥ nbands, npw).

    HV/SV are cached — H and S act only on the block of new directions each
    iteration; restarts contract them through the Ritz rotation.
    """
    npw = x0.shape[1]
    max_dim = max_dim or min(npw, max(4 * nbands, nbands + 24))

    def ortho_block(d, v_prev):
        """Orthonormalize d against v_prev and internally (two GS passes)."""
        for _ in range(2):
            if v_prev is not None:
                d = d - (d @ v_prev.conj().T) @ v_prev
            q, _ = torch.linalg.qr(d.T, mode="reduced")
            d = q.T.contiguous()
        return d

    v_sub = ortho_block(x0, None)
    hv, sv = hs.h(v_sub), hs.s(v_sub)
    eps = x = hx = sx = None
    for _ in range(max_iter):
        h_sub = v_sub.conj() @ hv.T
        s_sub = v_sub.conj() @ sv.T
        h_sub = 0.5 * (h_sub + h_sub.conj().T)
        s_sub = 0.5 * (s_sub + s_sub.conj().T)
        ell = torch.linalg.cholesky(s_sub)
        a = torch.linalg.solve_triangular(ell, h_sub, upper=False)  # L⁻¹H
        # (L⁻¹H)L⁻† = (L⁻¹ (L⁻¹H)†)† — solve with L again, then dagger
        a = torch.linalg.solve_triangular(ell, a.conj().T, upper=False).conj().T
        w, u = torch.linalg.eigh(0.5 * (a + a.conj().T))
        u = torch.linalg.solve_triangular(ell.conj().T, u, upper=True)
        eps = w[:nbands].real
        u_r = u[:, :nbands].T.to(CDTYPE)
        x, hx, sx = u_r @ v_sub, u_r @ hv, u_r @ sv
        r = hx - eps[:, None].to(CDTYPE) * sx
        rnorm = torch.linalg.norm(r, dim=1)
        if float(rnorm.max()) < tol:
            return eps, x
        active = rnorm > tol
        d = teter(r[active], hs.t, eps[active])
        if v_sub.shape[0] + int(active.sum()) > max_dim:
            # restart from the Ritz vectors, RE-ORTHONORMALIZED in the
            # standard metric (they are S-orthonormal; using them as-is lets
            # the basis drift toward linear dependence and the ill-conditioned
            # overlap Cholesky then produces spurious below-minimum states).
            # HV/SV rotate with the same triangular transform: x = RᵀQᵀ ⇒
            # basis Qᵀ = R⁻ᵀ x.
            qq, rr = torch.linalg.qr(x.T, mode="reduced")
            v_sub = qq.T.contiguous()
            rt = rr.T  # x = RᵀQᵀ (plain transpose), so Qᵀ = (Rᵀ)⁻¹x
            hv = torch.linalg.solve_triangular(rt, hx, upper=False)
            sv = torch.linalg.solve_triangular(rt, sx, upper=False)
        d = ortho_block(d, v_sub)
        v_sub = torch.cat([v_sub, d], dim=0)
        hv = torch.cat([hv, hs.h(d)], dim=0)
        sv = torch.cat([sv, hs.s(d)], dim=0)
    return eps, x



@dataclass
class _IterOps:
    """Frozen per-run operators and tables for `_scf_iteration` — everything
    the one-iteration SCF map needs beyond the state it maps. Built once by
    `_build_iter_ops`; newton.py and the mixer rig construct it directly to
    evaluate the raw map without the driver."""

    system: USPPSystem
    xc: object
    nspin: int
    smearing: str
    width: float
    batched: bool
    projs: list
    bk: object
    p_b: object
    hub: object
    vloc_g: torch.Tensor
    vloc_r: torch.Tensor
    phase_pos: torch.Tensor
    is_paw: bool
    onec: list | None
    grid: object
    vol: float
    dev: object
    shape: tuple
    mask_flat: torch.Tensor
    g_spin: int
    nk: int
    nb: int


def _build_iter_ops(system: USPPSystem, xc, *, nspin=1, smearing="none",
                    width=0.1, batched=True, hubbard=None) -> _IterOps:
    grid = system.grid
    vol = grid.volume
    dev = system.positions.device
    projs = [projectors(pd, system.positions) for pd in system.proj_data]
    bk = p_b = None
    if batched or hubbard:
        from gradwave.core.batch import build_batched, projectors_b

        bk = build_batched(system.spheres, system.proj_data, device=dev)
        p_b = projectors_b(bk, system.positions)
    hub = None
    if hubbard:
        from gradwave.scf.uspp_hubbard import build_uspp_hubbard

        hub = build_uspp_hubbard(system, hubbard, bk, p_b)
    vloc_g = local_potential_g(system.positions,
                               torch.tensor(system.species_of_atom, device=dev),
                               system.vloc_tables, grid.g_cart, vol)
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real
    phase_arg = system.g_sphere @ system.positions.T  # (nGm, na)
    phase_pos = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))
    is_paw = any(p.is_paw for p in system.paws)
    onec = None
    if is_paw:
        from gradwave.scf.paw_onsite import OneCenter

        onec = [OneCenter(p, xc) for p in system.paws]
    return _IterOps(system=system, xc=xc, nspin=nspin, smearing=smearing,
                    width=width, batched=batched, projs=projs, bk=bk, p_b=p_b,
                    hub=hub, vloc_g=vloc_g, vloc_r=vloc_r, phase_pos=phase_pos,
                    is_paw=is_paw, onec=onec, grid=grid, vol=vol, dev=dev,
                    shape=grid.shape, mask_flat=grid.dens_mask.reshape(-1),
                    g_spin=2 if nspin == 1 else 1, nk=len(system.spheres),
                    nb=system.nbands)


@torch.no_grad()
def _scf_iteration(ops: _IterOps, rho_s, rho_ij_mix, coeffs, coeffs_b,
                   n_hub_s, tol_eff, seed_salt):
    """ONE evaluation of the SCF map at (rho_s, rho_ij_mix): potentials →
    screened D (+ one-center ddd from the MIXER-side becsum) → generalized
    Davidson (warm-started via coeffs/coeffs_b, mutated in place) →
    shared-Fermi occupations (+U matrices) → fresh densities/becsum →
    energy assembly. No mixing, no convergence judgment, no rescue —
    those belong to the driver (or to newton/the rig, which call this
    directly)."""
    system, xc, nspin = ops.system, ops.xc, ops.nspin
    smearing, width, batched = ops.smearing, ops.width, ops.batched
    projs, bk, p_b, hub = ops.projs, ops.bk, ops.p_b, ops.hub
    vloc_g, vloc_r, phase_pos = ops.vloc_g, ops.vloc_r, ops.phase_pos
    is_paw, onec = ops.is_paw, ops.onec
    grid, vol, dev, shape = ops.grid, ops.vol, ops.dev, ops.shape
    mask_flat, nk, nb = ops.mask_flat, ops.nk, ops.nb
    if batched:
        from gradwave.core.batch import becp_b
        from gradwave.scf.uspp_batch import BatchedHS, davidson_gen_batched
    if hub is not None:
        from gradwave.core.hubbard import (
            hubbard_dmatrix,
            hubbard_energy,
            occupation_matrices,
        )

    def _becsum_for_onec(a):
        if nspin == 1:
            return rho_ij_mix[0][a]
        return [rho_ij_mix[0][a], rho_ij_mix[1][a]]

    rho_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    rho_g_box = r_to_g(rho_tot.to(CDTYPE))
    v_h = (torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2),
                           dim=(-3, -2, -1)) * grid.n_points).real
    core = system.rho_core
    if nspin == 1:
        v_xc, _ = vxc_potential(xc, rho_tot if core is None else rho_tot + core,
                                grid)
        veff_s = [v_h + v_xc + vloc_r]
    else:
        from gradwave.scf.loop import vxc_spin_potential

        c2 = None if core is None else 0.5 * core
        v_up, v_dn, _ = vxc_spin_potential(
            xc,
            rho_s[0] if core is None else rho_s[0] + c2,
            rho_s[1] if core is None else rho_s[1] + c2,
            grid,
        )
        veff_s = [v_h + v_up + vloc_r, v_h + v_dn + vloc_r]

    # screened D per spin/atom: D_ij + Σ_G ṽ_σ(G) e^{iGτ} Q̃_ij(G)*
    dscr_s = []
    for isp in range(nspin):
        v_eff_g = r_to_g(veff_s[isp].to(CDTYPE)).reshape(-1)[mask_flat]
        dscr = torch.zeros_like(system.q_full)
        for a, sp in enumerate(system.species_of_atom):
            s0, s1 = system.atom_slices[a]
            contr = torch.einsum(
                "ijg,g->ij", system.aug[sp].q_g.conj(), v_eff_g * phase_pos[:, a]
            )
            herm = 0.5 * (contr + contr.conj().T)
            dscr[s0:s1, s0:s1] = herm.real
        dscr_s.append(dscr + system.proj_data[0].dij_full)
    e_onec = torch.zeros((), dtype=RDTYPE, device=dev)
    if is_paw:
        dscr_s = [d.clone() for d in dscr_s]
        for a, sp in enumerate(system.species_of_atom):
            s0, s1 = system.atom_slices[a]
            # one-center runs on CPU (per-atom radial work); ddd crosses back
            e1c, ddd = onec[sp].energy_and_ddd(_becsum_for_onec(a))
            e_onec = e_onec + e1c
            if nspin == 1:
                dscr_s[0][s0:s1, s0:s1] += ddd.to(dev)
            else:
                for isp in range(nspin):
                    dscr_s[isp][s0:s1, s0:s1] += ddd[isp].to(dev)

    eigs_s = []
    for isp in range(nspin):
        hub_d = None
        if hub is not None:
            hub_d = hubbard_dmatrix(n_hub_s[isp], hub.sites, hub.nproj,
                                    dev).conj()  # apply wants D^T
        if batched:
            hs_b = BatchedHS(bk, shape, veff_s[isp], p_b, dscr_s[isp],
                             system.q_full, hub_sphi=hub.sphi if hub else None,
                             hub_d=hub_d)
            if coeffs_b[isp] is None:
                # per-k CPU seeds (identical to the per-k path), padded
                x0 = torch.zeros(nk, nb + 4, bk.npw_max, dtype=CDTYPE,
                                 device=dev)
                for ik, sph in enumerate(system.spheres):
                    gen = torch.Generator().manual_seed(
                        1234 + ik + 7777 * isp + seed_salt)
                    xk = torch.randn(nb + 4, sph.npw, generator=gen,
                                     dtype=torch.float64) \
                        + 1j * torch.randn(nb + 4, sph.npw, generator=gen,
                                           dtype=torch.float64)
                    xk = xk.to(dev) * torch.exp(
                        -0.5 * HBAR2_2M * sph.kpg2 / system.ecut * 4.0)
                    x0[ik, :, :sph.npw] = xk.to(CDTYPE)
            else:
                x0 = coeffs_b[isp]
            eig_b, x_b = davidson_gen_batched(hs_b, x0, nb, tol=tol_eff)
            b_all = becp_b(p_b, x_b)
            snorm = (x_b.abs() ** 2).sum(dim=-1) + torch.einsum(
                "kbi,ij,kbj->kb", b_all.conj(),
                system.q_full.to(CDTYPE), b_all).real
            x_b = x_b / torch.sqrt(snorm)[..., None]
            coeffs_b[isp] = x_b
            for ik, sph in enumerate(system.spheres):
                coeffs[isp][ik] = x_b[ik, :, :sph.npw]
            eigs_s.append(eig_b)
            continue
        eigs_l = []
        for ik, sph in enumerate(system.spheres):
            hs = _HkS(sph, shape, veff_s[isp], system.proj_data[ik], projs[ik],
                      dscr_s[isp], system.q_full,
                      hub_sphi=(hub.sphi[ik, :, :sph.npw] if hub else None),
                      hub_d=hub_d)
            if coeffs[isp][ik] is None:
                # seed on CPU (device-independent determinism), then move
                gen = torch.Generator().manual_seed(
                    1234 + ik + 7777 * isp + seed_salt)
                x0 = torch.randn(nb + 4, sph.npw, generator=gen, dtype=torch.float64) \
                    + 1j * torch.randn(nb + 4, sph.npw, generator=gen,
                                       dtype=torch.float64)
                x0 = x0.to(dev) * torch.exp(
                    -0.5 * HBAR2_2M * sph.kpg2 / system.ecut * 4.0)
                x0 = x0.to(CDTYPE)
            else:
                x0 = coeffs[isp][ik]
            e_k, c_k = davidson_gen(hs, x0, nb, tol=tol_eff)
            b = becp(projs[ik], c_k)
            snorm = (c_k.abs() ** 2).sum(dim=1).real + torch.einsum(
                "bi,ij,bj->b", b.conj(), system.q_full.to(CDTYPE), b
            ).real
            c_k = c_k / torch.sqrt(snorm)[:, None]
            coeffs[isp][ik] = c_k
            eigs_l.append(e_k)
        eigs_s.append(torch.stack(eigs_l))

    occ_s, mu, entropy_term = shared_fermi_occupations(
        eigs_s, system.kweights, smearing, width, system.n_electrons,
        nspin, dev)

    # DFT+U: fresh S-metric occupation matrices + Dudarev E_U (lags one
    # step into V_U like the NC path; nspin=1 splits [0,2] occupations
    # into two equal channels)
    e_hub = torch.zeros((), dtype=RDTYPE, device=dev)
    if hub is not None:
        def _padded_coeffs(isp, _cb=coeffs_b):
            if batched:
                return _cb[isp]
            cp = torch.zeros(nk, nb, bk.npw_max, dtype=CDTYPE, device=dev)
            for ik, sph in enumerate(system.spheres):
                cp[ik, :, :sph.npw] = coeffs[isp][ik]
            return cp

        if nspin == 2:
            for isp in range(nspin):
                n_hub_s[isp] = occupation_matrices(
                    hub.sphi, _padded_coeffs(isp), occ_s[isp],
                    system.kweights, hub.sites)
            e_hub = sum(hubbard_energy(n_hub_s[isp], hub.sites)
                        for isp in range(nspin))
        else:
            n_half = occupation_matrices(
                hub.sphi, _padded_coeffs(0), 0.5 * occ_s[0],
                system.kweights, hub.sites)
            n_hub_s = [n_half]
            e_hub = 2.0 * hubbard_energy(n_half, hub.sites)

    # smooth densities + per-spin becsum + augmentation
    rho_out_s, becps_s = [], []
    rho_ij_s = [[torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=dev)
                 for (s0, s1) in system.atom_slices] for _ in range(nspin)]
    for isp in range(nspin):
        rho_sp = torch.zeros(shape, dtype=RDTYPE, device=dev)
        becps = []
        for ik, sph in enumerate(system.spheres):
            c = coeffs[isp][ik]
            psi_r = g_to_r(c, sph.flat_idx, shape)
            w = system.kweights[ik] * occ_s[isp][ik]
            rho_sp = rho_sp + torch.einsum("b,bxyz->xyz", w, (psi_r.abs() ** 2)) / vol
            b = becp(projs[ik], c)
            becps.append(b)
            for a, (s0, s1) in enumerate(system.atom_slices):
                ba = b[:, s0:s1]
                rho_ij_s[isp][a] = rho_ij_s[isp][a] + torch.einsum(
                    "b,bi,bj->ij", w.to(CDTYPE), ba.conj(), ba
                )
        rho_ij_s[isp] = [0.5 * (m + m.conj().T) for m in rho_ij_s[isp]]
        if system.becsum_sym is not None:
            rho_ij_s[isp] = system.becsum_sym.apply(rho_ij_s[isp])
        becps_s.append(becps)

        aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE, device=dev)
        for a, sp in enumerate(system.species_of_atom):
            aug_sph = aug_sph + phase_pos[:, a].conj() * torch.einsum(
                "ij,ijg->g", rho_ij_s[isp][a], system.aug[sp].q_g
            )
        aug_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=dev)
        aug_box[system.sphere_idx] = aug_sph / vol
        rho_aug = (torch.fft.ifftn(aug_box.reshape(shape) * grid.n_points,
                                   dim=(-3, -2, -1))).real
        rho_out_sp = rho_sp + rho_aug
        if system.rho_symmetrizer is not None:
            sym_g = system.rho_symmetrizer.apply(r_to_g(rho_out_sp.to(CDTYPE)))
            rho_out_sp = torch.fft.ifftn(
                sym_g * grid.n_points, dim=(-3, -2, -1)).real
        rho_out_s.append(rho_out_sp)
    rho_tot_out = rho_out_s[0] if nspin == 1 else rho_out_s[0] + rho_out_s[1]

    n_tot = float(rho_tot_out.sum()) * vol / grid.n_points
    assert abs(n_tot - system.n_electrons) < 1e-5, (
        f"charge not conserved: {n_tot:.8f} vs {system.n_electrons}"
    )

    rho_g_out = r_to_g(rho_tot_out.to(CDTYPE))
    from gradwave.core.density import sigma_from_rho

    if nspin == 1:
        rho_xc_out = rho_tot_out if core is None else rho_tot_out + core
        sigma = sigma_from_rho(rho_xc_out, grid.g_cart) if xc.needs_gradient else None
        e_xc = xc.energy(rho_xc_out, vol, sigma)
    else:
        c2 = 0.0 if core is None else 0.5 * core
        r_u, r_d = rho_out_s[0] + c2, rho_out_s[1] + c2
        if xc.needs_gradient:
            s_uu = sigma_from_rho(r_u, grid.g_cart)
            s_dd = sigma_from_rho(r_d, grid.g_cart)
            s_tt = sigma_from_rho(r_u + r_d, grid.g_cart)
        else:
            s_uu = s_dd = s_tt = None
        e_xc = xc.energy(r_u, r_d, vol, s_uu, s_dd, s_tt)
    energies = EnergyBreakdown(
        kinetic=sum(kinetic_energy(coeffs[isp], occ_s[isp], system.kweights,
                                   system.spheres) for isp in range(nspin)),
        hartree=hartree_energy(rho_g_out, grid.g2, vol),
        xc=e_xc,
        local=local_energy(rho_g_out, vloc_g, vol),
        nonlocal_=sum(nonlocal_energy(becps_s[isp], system.proj_data[0].dij_full,
                                      occ_s[isp], system.kweights)
                      for isp in range(nspin)),
        ewald=ewald_energy(system.positions, system.charges, grid.cell),
        smearing=entropy_term,
        hubbard=e_hub if hub is not None else 0.0,
        onecenter=e_onec,
    )
    return dict(eigs_s=eigs_s, occ_s=occ_s, mu=mu, n_hub_s=n_hub_s,
                rho_out_s=rho_out_s, rho_ij_s=rho_ij_s, becps_s=becps_s,
                energies=energies)


@torch.no_grad()
def scf_uspp(system: USPPSystem, xc, *, nspin: int = 1, start_mag=None,
             smearing="none", width=0.1, max_iter=60, etol=1e-8, rhotol=1e-7,
             diago_tol=1e-9, mixing_alpha=0.7, mixing_history=None,
             trust_factor=20.0, batched=True, hubbard=None, start_from=None,
             criterion="drho", rho_safety=1e-2, adapt_step=False,
             mixing_scheme="pulay", mixing_kerker=None, mixing_metric="plain",
             spin_precond=False, opts=None, verbose=True):
    """USPP/PAW SCF. nspin=2 takes a SpinXC functional and per-species
    start_mag (list, in [-1, 1]); mixing then runs in the (total,
    magnetization) basis with Kerker on the total for smeared systems.
    batched=True solves all k in one padded generalized-Davidson block
    (identical eigenpairs; batched=False is the reference per-k path).
    hubbard: list[HubbardManifold] — Dudarev DFT+U with S-metric occupation
    matrices (QE U_projection_type='atomic' convention for USPP).
    start_from: a previous scf_uspp result on the SAME FFT grid and spin
    count — seeds (ρ, becsum) from its converged state instead of SAD +
    atomic occupations. The density is rescaled by the volume ratio so the
    electron count is conserved. This is the right start for scans (EOS
    volumes, displacements): adjacent points are small perturbations, so
    the warm start both cuts iterations and keeps trajectory-dependent
    branches (FM vs NM) from flipping between points.
    criterion: "drho" (default) demands both etol and rhotol; "energy"
    converges on a settled 3-iteration free-energy tail (< etol) with only
    the loose rho_safety residual bound — the honest criterion for smeared
    metals, whose residual floors at occupation noise (O(res²) energy
    error) while F is long converged.
    adapt_step: OPT-IN per-block adaptive damping. Blocks whose residual
    grows across iterations get their damped step cut by the observed
    gain, with a plateau-triggered global halving on top. On FM Ni at the
    default mixing_alpha this prevents the silent collapse to the NM
    branch (m and F land at the validated values), but it does NOT reach
    tight convergence — the monotone multipliers over-react to transient
    startup growth (measured head-to-head: static alpha=0.3 converges to
    |dρ| 2e-3 where adaptive stalls at 2e-2 with the ρ-block floored).
    Use it as a stabilizer for exploratory runs at unknown damping; for
    production FM metals keep hand-set mixing_alpha (0.3 for Ni).
    mixing_scheme: "pulay" (default) or "broyden" — limited-memory
    Broyden-II, whose sequential secant updates keep directional gain
    estimates that Pulay's residual-span extrapolation loses (the QE
    default scheme; candidate replacement for hand-set damping on FM
    metals).
    spin_precond: Stoner preconditioner on the magnetization channel
    (smeared nspin=2 only; scf/spin_precond.py) — the physics-informed
    treatment of the Stoner-expansive mode, applied to residuals before
    damping/mixing.
    opts: an SCFOptions object (scf/options.py) — the readable form of
    all of the above; when given it overrides the flat kwargs."""
    mixing_w0, bec_step_scale = 0.01, None  # MixerOptions defaults
    if opts is not None:
        smearing, width = opts.smearing, opts.width
        max_iter, etol, rhotol = opts.max_iter, opts.etol, opts.rhotol
        diago_tol, criterion = opts.diago_tol, opts.criterion
        rho_safety, batched, verbose = opts.rho_safety, opts.batched, \
            opts.verbose
        mx = opts.mixer
        mixing_alpha, mixing_history = mx.alpha, mx.history
        mixing_scheme, mixing_kerker = mx.scheme, mx.kerker
        mixing_metric, trust_factor = mx.metric, mx.trust_factor
        adapt_step, spin_precond = mx.adapt_step, mx.spin_precond
        mixing_w0, bec_step_scale = mx.w0, mx.bec_step_scale
    grid = system.grid
    vol = grid.volume
    dev = system.positions.device
    nk = len(system.spheres)
    shape = grid.shape

    if start_from is not None:
        prev_grid = start_from["system"].grid
        if tuple(prev_grid.shape) != tuple(shape):
            raise ValueError("start_from requires the same FFT grid "
                             f"({tuple(prev_grid.shape)} vs {tuple(shape)})")
        if start_from.get("nspin", 1) != nspin:
            raise ValueError("start_from nspin mismatch")
        # ρ carries a 1/Ω normalization: rescale by the volume ratio so the
        # electron count is exactly conserved on the new cell
        chg = float(prev_grid.volume) / float(vol)
        if nspin == 1:
            rho_s = [start_from["rho"].detach().to(dev) * chg]
            spin_frac = [None]
        else:
            rho_s = [r.detach().to(dev) * chg for r in start_from["rho_spin"]]
            mags = list(start_mag or [0.0] * len(system.paws))
            spin_frac = [[(1.0 + m) / 2.0 for m in mags],
                         [(1.0 - m) / 2.0 for m in mags]]
    elif nspin == 1:
        rho_s = [sad_density(grid, system.positions, system.species_of_atom,
                             system.paws, system.n_electrons)]
        spin_frac = [None]
    else:
        mags = list(start_mag or [0.0] * len(system.paws))
        up = [(1.0 + m) / 2.0 for m in mags]
        dn = [(1.0 - m) / 2.0 for m in mags]
        n_up = sum(float(system.charges[a]) * up[sp]
                   for a, sp in enumerate(system.species_of_atom))
        rho_s = [
            sad_density(grid, system.positions, system.species_of_atom,
                        system.paws, n_up, species_scale=up),
            sad_density(grid, system.positions, system.species_of_atom,
                        system.paws, system.n_electrons - n_up, species_scale=dn),
        ]
        spin_frac = [up, dn]

    ops = _build_iter_ops(system, xc, nspin=nspin, smearing=smearing,
                          width=width, batched=batched, hubbard=hubbard)
    hub = ops.hub
    n_hub_s = None
    if hub is not None:
        n_hub_s = [[torch.zeros(s["dim"], s["dim"], dtype=CDTYPE, device=dev)
                    for s in hub.sites] for _ in range(nspin)]

    # Mixing vector = [ρ channels on the density sphere, flattened becsum per
    # spin]. Mixing becsum TOGETHER with ρ (QE keeps it inside rho%mix the
    # same way) is essential for metals: the D-feedback loops (∫v_eff Q and
    # the one-center ddd) must see a becsum coherent with the mixed density —
    # a fresh or independently-damped becsum gives a gain>1 charge
    # oscillation for semicore-metal PAW (fcc Ni diverges with ×9/iteration).
    # Kerker damps the ρ-TOTAL block only (becsum is localized; the
    # magnetization channel must keep its G=0 free for ↑↓ transfer).
    # MixLayout owns the composite-vector structure: packing, Kerker mask
    # (ρ-total block only — becsum is localized and the magnetization
    # channel keeps its G=0 free for ↑↓ transfer), the becsum step scale
    # (the on-site becsum↔ddd feedback is the stiffest direction), and the
    # adaptive-damping block ids
    if bec_step_scale is None:
        # Johnson handles the on-site becsum↔ddd mode without extra
        # damping (FM Ni 27→16 it); the 0.4 stays for pulay/broyden
        bec_step_scale = 1.0 if mixing_scheme == "johnson" else 0.4
    layout = MixLayout(grid, nspin, system.atom_slices, device=dev,
                       bec_step_scale=bec_step_scale)
    g2_mix, ng, nbec = layout.g2_sphere, layout.ng, layout.nbec
    g2_full, kerker_mask = layout.g2_full, layout.kerker_mask
    step_scale = layout.step_scale
    adapt_ids = layout.block_ids if adapt_step else None
    use_kerker = (smearing != "none") if mixing_kerker is None \
        else bool(mixing_kerker)
    if mixing_history is None:
        # per-scheme defaults, measured on FM Ni (see JohnsonMixer)
        mixing_history = 12 if mixing_scheme == "johnson" else 8
    metric_w = None
    if mixing_metric == "coulomb":
        # QE rho_ddot: Coulomb-metric inner products (long-range emphasis).
        # G=0 is EXCLUDED (zero weight — 1/G² there poisons the Gram
        # matrix); becsum components get unit weight
        wg = torch.where(g2_mix > 1e-12, 1.0 / g2_mix.clamp_min(1e-12),
                         torch.zeros_like(g2_mix))
        metric_w = torch.cat([wg] * nspin
                             + [torch.ones(nbec, device=dev)] * nspin)
    if mixing_scheme == "broyden":
        mixer = BroydenMixer(g2_full, alpha=mixing_alpha,
                             history=mixing_history, kerker=use_kerker,
                             kerker_mask=kerker_mask, check_g0=False,
                             step_scale=step_scale)
    elif mixing_scheme == "johnson":
        mixer = JohnsonMixer(g2_full, alpha=mixing_alpha,
                             history=mixing_history, kerker=use_kerker,
                             kerker_mask=kerker_mask, check_g0=False,
                             step_scale=step_scale, metric_w=metric_w,
                             w0=mixing_w0)
    else:
        mixer = PulayMixer(g2_full, alpha=mixing_alpha,
                           history=mixing_history,
                           kerker=use_kerker,
                           kerker_mask=kerker_mask, check_g0=False,
                           step_scale=step_scale, adapt_blocks=adapt_ids)
    coeffs = [[None] * nk for _ in range(nspin)]
    coeffs_b = [None] * nspin
    e_free_prev, history, converged = None, [], False
    rescue_count, seed_salt = 0, 0  # solver-blowup rescue state (task #55)
    last_reset_it = -10  # trust-region reset cooldown (task #55)
    occ_s = eigs_s = mu = None
    energies = None

    # PAW one-center machinery; becsum seeded from the reference atomic
    # occupations (spin-split by start_mag; zeros for bare USPP where the UPF
    # carries no PP_OCCUPATIONS). rho_ij_mix is the MIXER-side becsum used
    # for the one-center ddd; rho_ij_s holds each iteration's fresh becsum.
    is_paw, onec = ops.is_paw, ops.onec
    is_paw = any(p.is_paw for p in system.paws)
    onec = None
    if is_paw:
        from gradwave.scf.paw_onsite import OneCenter

        onec = [OneCenter(p, xc) for p in system.paws]
    rho_ij_s = [[] for _ in range(nspin)]
    if start_from is not None:
        prev_bec = start_from["rho_ij_atoms"]
        for isp in range(nspin):
            src = prev_bec if nspin == 1 else prev_bec[isp]
            rho_ij_s[isp] = [m.detach().to(device=dev, dtype=CDTYPE).clone()
                             for m in src]
    else:
        for sp in system.species_of_atom:
            paw = system.paws[sp]
            nm = sum(2 * b.l + 1 for b in paw.betas)
            for isp in range(nspin):
                m0 = torch.zeros(nm, nm, dtype=CDTYPE, device=dev)
                if paw.paw_occ is not None:
                    frac = 0.5 if nspin == 1 else spin_frac[isp][sp]
                    col = 0
                    for i, b in enumerate(paw.betas):
                        for _m in range(2 * b.l + 1):
                            m0[col, col] = paw.paw_occ[i] / (2 * b.l + 1) * (
                                2.0 * frac if nspin == 1 else frac)
                            col += 1
                rho_ij_s[isp].append(m0)
    rho_ij_mix = [[m.clone() for m in ch] for ch in rho_ij_s]

    for it in range(1, max_iter + 1):
        if it == 1:
            # SAD starts don't deserve a tight first solve; warm starts do
            # (their density is already near a fixed point — scan/rig
            # callers control precision through diago_tol)
            tol_eff = max(diago_tol, 1e-3) if start_from is None else diago_tol
        else:
            r_prev = history[-1]["res"]
            tol_eff = max(diago_tol, min(1e-3, 0.1 * r_prev * r_prev / system.n_electrons))

        step = _scf_iteration(ops, rho_s, rho_ij_mix, coeffs, coeffs_b,
                              n_hub_s, tol_eff, seed_salt)
        eigs_s, occ_s, mu = step["eigs_s"], step["occ_s"], step["mu"]
        n_hub_s = step["n_hub_s"]
        rho_out_s, rho_ij_s = step["rho_out_s"], step["rho_ij_s"]
        becps_s, energies = step["becps_s"], step["energies"]
        e_free = float(energies.free_energy)

        to_mix = layout.pack

        # solver-blowup rescue: a warm-started Davidson can deterministically
        # return states ~10² eV up from a CONVERGED-quality density (F falls
        # off a cliff, |Δρ| freezes, mixer resets change nothing — task #55
        # fingerprint). Detect the jump, throw away the poisoned warm starts,
        # and re-solve from salted fresh seeds WITHOUT feeding the garbage
        # into the mixer.
        if (e_free_prev is not None and history
                and abs(e_free - e_free_prev) > 5.0
                and history[-1]["res"] < 1e-2 and rescue_count < 2):
            rescue_count += 1
            seed_salt = 104729 * rescue_count
            coeffs_b = [None] * nspin
            for isp in range(nspin):
                coeffs[isp] = [None] * nk
            if verbose:
                print(f"  USPP {it:3d}  [solver blowup: dE = "
                      f"{abs(e_free - e_free_prev):.1f} eV from residual "
                      f"{history[-1]['res']:.1e} — reseeding eigensolver]")
            continue

        rho_in_vec = to_mix(rho_s, rho_ij_mix)
        rho_out_vec = to_mix(rho_out_s, rho_ij_s)
        res_norm = float(
            torch.linalg.norm(rho_out_vec[: ng * nspin] - rho_in_vec[: ng * nspin])
        ) * vol
        de = abs(e_free - e_free_prev) if e_free_prev is not None else float("inf")
        history.append({"iter": it, "free_energy": e_free, "dE": de, "res": res_norm})
        if verbose:
            mag = ""
            if nspin == 2:
                m = float((rho_out_s[0] - rho_out_s[1]).sum()) * vol / grid.n_points
                mag = f"  m = {m:+.4f} muB"
            print(f"  USPP {it:3d}  F = {e_free:+.10f} eV  dE = {de:.3e}  "
                  f"|drho| = {res_norm:.3e}{mag}")
        if criterion == "energy":
            # QE-style energy criterion: the free energy is variational, its
            # error is O(residual²), and for smeared metals the residual
            # floors at occupation noise long after F has settled. Require a
            # settled 3-iteration tail (a single small dE can be a sloshing
            # coincidence) plus a loose residual safety that excludes frozen
            # limit cycles without demanding the unreachable plateau floor.
            tail = [h["free_energy"] for h in history[-3:]]
            done = (len(tail) == 3 and max(tail) - min(tail) < etol
                    and res_norm < rho_safety
                    and tol_eff <= diago_tol * 1.01)
        else:
            done = (de < etol and res_norm < rhotol
                    and tol_eff <= diago_tol * 1.01)
        if done:
            converged = True
            rho_s = rho_out_s
            if is_paw:  # report the one-center energy at the FINAL fresh becsum
                e_onec = torch.zeros((), dtype=RDTYPE, device=dev)
                for a, sp in enumerate(system.species_of_atom):
                    fresh = (rho_ij_s[0][a] if nspin == 1
                             else [rho_ij_s[0][a], rho_ij_s[1][a]])
                    e1c, _ = onec[sp].energy_and_ddd(fresh)
                    e_onec = e_onec + e1c
                energies.onecenter = e_onec
            break
        e_free_prev = e_free
        # trust region: a residual jump means the DIIS history is lying about
        # the curvature (typical of the wild first USPP/PAW iterations from
        # the SAD + atomic-becsum start); discard it and restart from a plain
        # damped step rather than extrapolating into nonsense.
        # WINDOWED baseline + cooldown (task #55): with an all-time best, one
        # touch of the eigensolver noise floor poisons the criterion forever —
        # every subsequent iteration resets the mixer, DIIS never re-learns,
        # and the SCF degenerates into pure damped iteration, which DIVERGES
        # geometrically (×2.5/iter observed) for gain>1 sloshing modes and
        # ends 148 eV up in a frozen limit cycle. Recent-best forgets old
        # floors; the cooldown stops a reset that didn't help from re-firing
        # before DIIS has curvature again.
        best_res = min(h["res"] for h in history[-10:])
        if (it > 1 and res_norm > trust_factor * best_res
                and it - last_reset_it >= 5):
            mixer.reset()
            last_reset_it = it
            if verbose:
                print(f"  USPP {it:3d}  [mixer reset: residual jumped "
                      f"{res_norm:.2e} > {trust_factor:g}x best {best_res:.2e}]")
        if spin_precond and nspin == 2 and smearing != "none":
            # Stoner preconditioner on the m-channel (arXiv:2606.26693):
            # rebuilt each iteration from the current orbitals; neutralizes
            # the Stoner-expansive magnetization mode that plain damping
            # amplifies and history mixing cannot hold
            from gradwave.scf.spin_precond import build_stoner_precond

            sp = build_stoner_precond(
                system, coeffs, eigs_s, mu, SCHEMES[smearing], width,
                rho_out_s[0] + rho_out_s[1], rho_out_s[0] - rho_out_s[1], xc)
            if sp is None:
                mixer.extra_precond = None
            else:
                def _spin_pc(rvec, _sp=sp):
                    out = rvec.clone()
                    out[ng:2 * ng] = _sp.apply(rvec[ng:2 * ng])
                    return out
                mixer.extra_precond = _spin_pc
        mixed = mixer.step(rho_in_vec, rho_out_vec)
        rho_s, bec_mixed = layout.unpack(mixed)
        # Hermitize the mixed becsum (linear combinations preserve it up
        # to roundoff) for the next iteration's one-center ddd
        for isp in range(nspin):
            for a in range(len(system.atom_slices)):
                m = bec_mixed[isp][a]
                rho_ij_mix[isp][a] = 0.5 * (m + m.conj().T)

    rho_final = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    out = dict(
        converged=converged, n_iter=len(history), energies=energies,
        eigenvalues=eigs_s[0] if nspin == 1 else torch.stack(eigs_s),
        occupations=occ_s[0] if nspin == 1 else torch.stack(occ_s),
        coeffs=coeffs[0] if nspin == 1 else coeffs, rho=rho_final,
        rho_ij_atoms=rho_ij_s[0] if nspin == 1 else rho_ij_s,
        becps=becps_s[0] if nspin == 1 else becps_s, history=history,
        fermi=mu, system=system, nspin=nspin, smearing=smearing, width=width,
        mixer_mult=(dict(mixer._block_mult)
                    if getattr(mixer, "_block_mult", None) else None),
        rho_out_spin=rho_out_s,  # RAW map output (pre-mixing) — rig/diagnostics
    )
    if hub is not None:
        out["hub_occ"] = n_hub_s
        out["hub_sites"] = hub.sites
    if nspin == 2:
        out["rho_spin"] = rho_s
        out["mag_total"] = float((rho_s[0] - rho_s[1]).sum()) * vol / grid.n_points
        out["mag_abs"] = float((rho_s[0] - rho_s[1]).abs().sum()) * vol / grid.n_points
    return out
