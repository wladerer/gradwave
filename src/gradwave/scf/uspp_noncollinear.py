"""Non-collinear (2-component spinor) ultrasoft/PAW SCF.

Fuses the two existing loops: the spinor structure of ``scf/noncollinear.py``
(doubled coefficient axis, Pauli-decomposed density ρ + m⃗, the 2×2 grid potential
v·1 + B⃗·σ) with the ultrasoft machinery of ``scf/uspp_loop.py`` (the generalized
eigenproblem H|ψ⟩ = εS|ψ⟩, the augmentation charge, and the PAW one-center
corrections). The PAW-specific non-collinear pieces come from
``scf/paw_noncollinear.py``, each validated in isolation against the collinear
code: the 2×2-in-spin on-site becsum, and the one-center energy + 2×2 ddd.

Structure per iteration (mirroring the collinear ``_scf_iteration``):
  potentials  v_h + (v_xc, B⃗_xc) + v_loc on the dense grid
  screened D  four channels — D_n = ∫v Q + D_bare + ddd_n, D_i = ∫B_i Q + ddd_i —
              assembled into the 2×2 blocks D↑↑ = D_n+D_z, D↓↓ = D_n−D_z,
              D↑↓ = D_x − i·D_y (the projector-space image of the grid potential)
  solve       generalized Davidson (``davidson_gen_batched``) on doubled vectors,
              S = 1+Σq|β⟩⟨β| acting per spin block (S ⊗ 1₂ — no SOC in S)
  densities   Pauli grid channels + the 2×2 becsum + a 4-channel augmentation
              (n_aug → ρ, m⃗_aug → m⃗ from the corresponding becsum channels)
  mixing      one Pulay vector: 4 G-space grid channels (Kerker on ρ only,
              separate magnetization step) + 4 flattened becsum channels

Validation: the collinear limit (all moments ∥ ẑ reproduces the collinear
nspin=2 ``scf_uspp`` free energy; nonmagnetic reproduces nspin=1) and global
rotation invariance (no SOC → the energy is independent of the moment axis, which
exercises the off-diagonal D↑↓ blocks that vanish in the collinear limit). LDA
only (the on-site non-collinear XC is LDA-only); magnetic runs need
use_symmetry=False, exactly like the norm-conserving spinor loop.
"""

from __future__ import annotations

import torch

from gradwave.core.batch import becp_b, box_to_sphere_b, g_to_r_b
from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.hartree import hartree_energy, hartree_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.energies.total import EnergyBreakdown
from gradwave.core.fftbox import r_to_g
from gradwave.core.occupations import SCHEMES, find_fermi, occupations_and_entropy
from gradwave.core.xc.noncollinear import NoncollinearXC, energy_with_grid, vxc_and_bxc
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.scf.common import symmetrize_rho
from gradwave.scf.guess import sad_density
from gradwave.scf.mixing import PulayMixer
from gradwave.scf.paw_noncollinear import (
    onsite_nc_energy_and_ddd,
    spinor_onsite_becsum,
)
from gradwave.scf.uspp_loop import _build_iter_ops
from gradwave.scf.uspp_setup import USPPSystem


class _SpinorBK:
    """The doubled-axis view of BatchedK that davidson_gen_batched reads."""

    def __init__(self, bk):
        self.mask = torch.cat([bk.mask, bk.mask], dim=-1)
        self.npw = 2 * bk.npw
        self.npw_max = 2 * bk.npw_max


class SpinorBatchedHS:
    """H and S applies on doubled vectors (nk, nb, 2·npw_max) for USPP/PAW.

    The 2×2 potential acts on the grid (v ± B_z diagonal, B_x − iB_y off-diagonal)
    and in projector space through the four screened-D channels; S = 1 + Σq|β⟩⟨β|
    is spin-diagonal (no SOC), applied per component with the same q_full."""

    def __init__(self, bk, shape, v_r, b_vec_r, p, d_chan, q_full):
        self.inner = bk
        self.bk = _SpinorBK(bk)
        self.shape = shape
        self.p = p
        self.q_full = q_full.to(CDTYPE)
        self.m = bk.npw_max
        self.t = torch.cat([bk.t, bk.t], dim=-1)
        d_n, d_x, d_y, d_z = (d.to(CDTYPE) for d in d_chan)
        self._d_uu = d_n + d_z
        self._d_dd = d_n - d_z
        self._d_ud = d_x - 1j * d_y            # image of v_ud = B_x − iB_y
        bx, by, bz = b_vec_r[0], b_vec_r[1], b_vec_r[2]
        self.b_zero = float(b_vec_r.abs().max()) == 0.0
        self._v_uu = v_r + bz
        self._v_dd = v_r - bz
        self._v_ud = torch.complex(bx, -by)

    def h(self, c):
        bk, m = self.inner, self.m
        cu, cd = c[..., :m], c[..., m:]
        t_r = bk.t
        out_u = t_r[:, None, :] * cu
        out_d = t_r[:, None, :] * cd
        # local 2×2 mix (dense grid; correctness-first — no smooth dual box)
        psi = g_to_r_b(torch.cat([cu, cd], dim=1), bk, self.shape)
        nb = cu.shape[1]
        psi_u, psi_d = psi[:, :nb], psi[:, nb:]
        if self.b_zero:
            h_u = psi_u * self._v_uu
            h_d = psi_d * self._v_dd
        else:
            h_u = psi_u * self._v_uu + psi_d * self._v_ud
            h_d = psi_u * self._v_ud.conj() + psi_d * self._v_dd
        hud = box_to_sphere_b(torch.cat([h_u, h_d], dim=1), bk)
        out_u = out_u + hud[:, :nb]
        out_d = out_d + hud[:, nb:]
        # nonlocal: 2×2 screened D in projector space
        p = self.p
        bu = becp_b(p, cu)
        bd = becp_b(p, cd)
        out_u = out_u + torch.einsum(
            "kbp,pq,kqg->kbg", bu, self._d_uu, p) + torch.einsum(
            "kbp,pq,kqg->kbg", bd, self._d_ud, p)
        out_d = out_d + torch.einsum(
            "kbp,pq,kqg->kbg", bu, self._d_ud.conj(), p) + torch.einsum(
            "kbp,pq,kqg->kbg", bd, self._d_dd, p)
        mask = bk.mask[:, None, :]
        return torch.cat([out_u * mask, out_d * mask], dim=-1)

    def s(self, c):
        bk, m = self.inner, self.m
        mask = bk.mask[:, None, :]
        outs = []
        for blk in (c[..., :m], c[..., m:]):
            b = becp_b(self.p, blk)
            outs.append((blk + torch.einsum(
                "kbp,pq,kqg->kbg", b, self.q_full, self.p)) * mask)
        return torch.cat(outs, dim=-1)


@torch.no_grad()
def scf_uspp_noncollinear(
    system: USPPSystem,
    xc,                       # collinear SpinXC (e.g. LSDA_PW92) — grid + on-site
    mag_vec_init,             # (na, 3) initial moment fraction·direction per atom
    smearing: str = "gaussian",
    width: float = 0.1,
    max_iter: int = 150,
    etol: float = 1e-8,
    rhotol: float = 1e-7,
    mixing_alpha: float = 0.4,
    mixing_history: int = 8,
    mag_mixing_alpha: float | None = None,
    bec_step_scale: float = 0.4,
    diago_tol: float = 1e-9,
    verbose: bool = True,
) -> dict:
    if getattr(xc, "needs_gradient", False):
        raise NotImplementedError(
            "non-collinear USPP/PAW is LDA-only (the on-site NC XC is LDA-only)")
    # Magnetic-symmetry systems (setup_uspp(..., magmoms=...)) carry a
    # MagneticSymmetrizer + MagneticBecsumSymmetrizer pair and fold k into the
    # magnetic IBZ; (ρ, m⃗, becsum) are re-symmetrized over the full Shubnikov
    # group every iteration. Plain paramagnetic symmetrizers remain invalid
    # here (the space group and time reversal act on m⃗).
    mag_sym_active = hasattr(system.rho_symmetrizer, "apply_m")
    if not mag_sym_active and (
            system.rho_symmetrizer is not None or system.becsum_sym is not None):
        raise ValueError(
            "non-collinear USPP/PAW requires use_symmetry=False or a magnetic-"
            "symmetry system (setup_uspp(..., magmoms=...))")
    ncxc = NoncollinearXC(xc)
    ops = _build_iter_ops(system, xc, nspin=1, smearing=smearing, width=width,
                          batched=True)
    grid, vol, dev, shape = ops.grid, ops.vol, ops.dev, ops.shape
    bk, p_b = ops.bk, ops.p_b
    mask_flat, nk = ops.mask_flat, ops.nk
    nbands = 2 * system.nbands             # spinor bands hold one electron each
    m_pw = bk.npw_max
    mag_vec_init = torch.as_tensor(mag_vec_init, dtype=RDTYPE)
    na = len(system.species_of_atom)
    from gradwave.scf.uspp_batch import davidson_gen_batched

    # ---- seeds: grid (SAD + directed m⃗), 2×2 becsum (atomic occ + directed m) ----
    rho = sad_density(grid, system.positions, system.species_of_atom, system.paws,
                      system.n_electrons).to(dev)
    m = torch.stack([
        sad_density(grid, system.positions, system.species_of_atom, system.paws,
                    None, atom_scale=[float(mag_vec_init[a, i]) for a in range(na)])
        for i in range(3)]).to(dev)
    bec_chan = [[] for _ in range(4)]      # [n, mx, my, mz] per atom, real (nm, nm)
    for a, sp in enumerate(system.species_of_atom):
        paw = system.paws[sp]
        nm = sum(2 * b.l + 1 for b in paw.betas)
        n0 = torch.zeros(nm, nm, dtype=RDTYPE, device=dev)
        if paw.paw_occ is not None:
            col = 0
            for i, b in enumerate(paw.betas):
                for _ in range(2 * b.l + 1):
                    n0[col, col] = paw.paw_occ[i] / (2 * b.l + 1)
                    col += 1
        d = mag_vec_init[a]
        scale = min(float(d.norm()), 0.9)
        dirv = d / d.norm() if float(d.norm()) > 1e-12 else torch.zeros(3)
        bec_chan[0].append(n0)
        for i in range(3):
            bec_chan[i + 1].append(scale * float(dirv[i]) * n0)

    # ---- mixer: 4 grid channels (Kerker/step on ρ vs m⃗) + 4 becsum channels ----
    g2_vec = grid.g2.reshape(-1)[mask_flat]
    ng = int(mask_flat.sum())
    nbec = sum((s1 - s0) ** 2 for (s0, s1) in system.atom_slices)
    if mag_mixing_alpha is None:
        mag_mixing_alpha = max(mixing_alpha, 0.6)
    ratio = mag_mixing_alpha / mixing_alpha if mixing_alpha > 0 else 1.0
    kerker_mask = torch.cat([
        torch.ones(ng, dtype=torch.bool, device=dev),
        torch.zeros(3 * ng + 4 * nbec, dtype=torch.bool, device=dev)])
    step_scale = torch.cat([
        torch.ones(ng, dtype=RDTYPE, device=dev),
        torch.full((3 * ng,), float(ratio), dtype=RDTYPE, device=dev),
        torch.full((4 * nbec,), float(bec_step_scale), dtype=RDTYPE, device=dev)])
    mixer = PulayMixer(torch.cat([g2_vec] * 4 + [torch.zeros(4 * nbec, device=dev)]),
                       alpha=mixing_alpha, history=mixing_history, kerker=True,
                       check_g0=False, kerker_mask=kerker_mask, step_scale=step_scale)

    def pack(rho_, m_, bec_):
        gvecs = [r_to_g(f.to(CDTYPE)).reshape(-1)[mask_flat]
                 for f in (rho_, m_[0], m_[1], m_[2])]
        bflat = [torch.cat([c.reshape(-1).to(CDTYPE) for c in bec_[i]])
                 for i in range(4)]
        return torch.cat(gvecs + bflat)

    def unpack(v):
        fields = []
        for c4 in range(4):
            box = torch.zeros(grid.n_points, dtype=CDTYPE, device=dev)
            box[mask_flat] = v[c4 * ng:(c4 + 1) * ng]
            fields.append(torch.fft.ifftn(box.reshape(shape) * grid.n_points,
                                          dim=(-3, -2, -1)).real)
        bec = [[] for _ in range(4)]
        off = 4 * ng
        for i in range(4):
            for (s0, s1) in system.atom_slices:
                n = s1 - s0
                bec[i].append(v[off:off + n * n].reshape(n, n).real.clone())
                off += n * n
        return fields[0], torch.stack(fields[1:]), bec

    # ---- spinor seeds: alternate up/down lowest plane waves ----
    coeffs = torch.zeros(nk, nbands, 2 * m_pw, dtype=CDTYPE, device=dev)
    for b in range(nbands):
        coeffs[:, b, (b // 2) + (b % 2) * m_pw] = 1.0

    scheme = SCHEMES[smearing]
    e_free_prev, converged, history = None, False, []
    mu, it = 0.0, 0
    energies = None
    q_c = system.q_full.to(CDTYPE)
    dij_bare = system.proj_data[0].dij_full

    for it in range(1, max_iter + 1):
        # ---- potentials ----
        v_h = (torch.fft.ifftn(hartree_potential_g(r_to_g(rho.to(CDTYPE)), grid.g2),
                               dim=(-3, -2, -1)) * grid.n_points).real
        v_xc, b_xc, _ = vxc_and_bxc(ncxc, rho, m, grid, rho_core=system.rho_core)
        v_r = v_h + v_xc + ops.vloc_r

        # ---- screened D: four channels, ∫(v, B⃗) Q + bare D + one-center ddd ----
        pots = (v_r, b_xc[0], b_xc[1], b_xc[2])
        d_chan = [torch.zeros_like(system.q_full) for _ in range(4)]
        for c4, pot in enumerate(pots):
            pot_g = r_to_g(pot.to(CDTYPE)).reshape(-1)[mask_flat]
            for a, sp in enumerate(system.species_of_atom):
                s0, s1 = system.atom_slices[a]
                contr = torch.einsum("ijg,g->ij", system.aug[sp].q_g.conj(),
                                     pot_g * ops.phase_pos[:, a])
                d_chan[c4][s0:s1, s0:s1] = (0.5 * (contr + contr.conj().T)).real
        d_chan[0] = d_chan[0] + dij_bare
        e_onec = torch.zeros((), dtype=RDTYPE, device=dev)
        if ops.is_paw:
            for a, sp in enumerate(system.species_of_atom):
                s0, s1 = system.atom_slices[a]
                e1c, ddd = onsite_nc_energy_and_ddd(
                    ops.onec[sp], [bec_chan[i][a] for i in range(4)])
                e_onec = e_onec + e1c
                for c4 in range(4):
                    d_chan[c4][s0:s1, s0:s1] += ddd[c4].to(dev)

        # ---- spinor generalized eigensolve + S-normalization ----
        tol_eff = max(diago_tol, 1e-3) if it == 1 else \
            max(diago_tol, min(1e-3, 0.03 * history[-1]["res"]))
        hs = SpinorBatchedHS(bk, shape, v_r, b_xc, p_b, d_chan, system.q_full)
        eigs, x = davidson_gen_batched(hs, coeffs, nbands, tol=tol_eff)
        eigs = eigs.to(RDTYPE)
        bu = becp_b(p_b, x[..., :m_pw])
        bd = becp_b(p_b, x[..., m_pw:])
        snorm = (x.abs() ** 2).sum(dim=-1) \
            + torch.einsum("kbi,ij,kbj->kb", bu.conj(), q_c, bu).real \
            + torch.einsum("kbi,ij,kbj->kb", bd.conj(), q_c, bd).real
        coeffs = x / torch.sqrt(snorm)[..., None]

        # ---- occupations (spinor: one electron per band) ----
        mu = float(find_fermi(eigs, system.kweights, scheme, width,
                              system.n_electrons, degeneracy=1.0))
        mu_t = torch.tensor(mu, dtype=RDTYPE, device=dev)
        occ, s_ent = occupations_and_entropy(eigs, mu_t, scheme, width,
                                             degeneracy=1.0)
        entropy_term = -width * (system.kweights[:, None] * s_ent).sum()

        # ---- densities: Pauli grid channels + 2×2 becsum + 4-channel aug ----
        bu = becp_b(p_b, coeffs[..., :m_pw])
        bd = becp_b(p_b, coeffs[..., m_pw:])
        rho_out = torch.zeros(shape, dtype=RDTYPE, device=dev)
        m_out = torch.zeros(3, *shape, dtype=RDTYPE, device=dev)
        bec_out = [[torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE, device=dev)
                    for (s0, s1) in system.atom_slices] for _ in range(4)]
        for ik in range(nk):
            w = (system.kweights[ik] * occ[ik]).to(RDTYPE)
            cu = coeffs[ik:ik + 1, :, :m_pw]
            cd = coeffs[ik:ik + 1, :, m_pw:]
            bk1 = _slice_bk(bk, ik)
            pu = g_to_r_b(cu, bk1, shape)[0]
            pd = g_to_r_b(cd, bk1, shape)[0]
            uu = torch.einsum("b,bxyz->xyz", w, pu.real ** 2 + pu.imag ** 2)
            dd = torch.einsum("b,bxyz->xyz", w, pd.real ** 2 + pd.imag ** 2)
            ud = torch.einsum("b,bxyz->xyz", w.to(CDTYPE), pu.conj() * pd)
            rho_out += uu + dd
            m_out[0] += 2.0 * ud.real
            m_out[1] += 2.0 * ud.imag
            m_out[2] += uu - dd
            for a, (s0, s1) in enumerate(system.atom_slices):
                chans = spinor_onsite_becsum(bu[ik, :, s0:s1], bd[ik, :, s0:s1],
                                             w.to(CDTYPE))
                for c4 in range(4):
                    bec_out[c4][a] = bec_out[c4][a] + chans[c4]
        rho_out, m_out = rho_out / vol, m_out / vol
        bec_out_r = [[c.real for c in bec_out[c4]] for c4 in range(4)]
        if mag_sym_active:
            # symmetrize the becsum BEFORE building the augmentation charge so
            # the smooth and one-center densities carry the same symmetry
            bec_out_r = system.becsum_sym.apply(bec_out_r)

        # augmentation: n_aug → ρ, m⃗_aug → m⃗, from the matching becsum channel
        targets = [rho_out, m_out[0], m_out[1], m_out[2]]
        aug_fields = []
        for c4 in range(4):
            aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE,
                                  device=dev)
            for a, sp in enumerate(system.species_of_atom):
                aug_sph = aug_sph + ops.phase_pos[:, a].conj() * torch.einsum(
                    "ij,ijg->g", bec_out_r[c4][a].to(CDTYPE), system.aug[sp].q_g)
            aug_box = torch.zeros(grid.n_points, dtype=CDTYPE, device=dev)
            aug_box[system.sphere_idx] = aug_sph / vol
            aug_fields.append(torch.fft.ifftn(
                aug_box.reshape(shape) * grid.n_points, dim=(-3, -2, -1)).real)
        rho_out = targets[0] + aug_fields[0]
        m_out = torch.stack([targets[1 + i] + aug_fields[1 + i] for i in range(3)])
        if mag_sym_active:
            rho_out = symmetrize_rho(system.rho_symmetrizer, rho_out, grid)
            m_g = torch.stack([r_to_g(m_out[i].to(CDTYPE)) for i in range(3)])
            m_out = torch.fft.ifftn(system.rho_symmetrizer.apply_m(m_g)
                                    * grid.n_points, dim=(-3, -2, -1)).real

        n_tot = float(rho_out.sum()) * vol / grid.n_points
        assert abs(n_tot - system.n_electrons) < 1e-5, (
            f"charge not conserved: {n_tot:.8f} vs {system.n_electrons}")

        # ---- energies ----
        rho_g_out = r_to_g(rho_out.to(CDTYPE))
        t_occ = (system.kweights[:, None] * occ).to(RDTYPE)
        e_kin = torch.einsum("kb,kbg,kg->", t_occ,
                             coeffs.real ** 2 + coeffs.imag ** 2, hs.t)
        e_nl = nonlocal_energy([bu[ik] for ik in range(nk)], dij_bare, occ,
                               system.kweights) \
            + nonlocal_energy([bd[ik] for ik in range(nk)], dij_bare, occ,
                              system.kweights)
        energies = EnergyBreakdown(
            kinetic=e_kin,
            hartree=hartree_energy(rho_g_out, grid.g2, vol),
            xc=energy_with_grid(ncxc, rho_out, m_out, grid,
                                rho_core=system.rho_core),
            local=_local_energy(rho_g_out, ops.vloc_g, vol),
            nonlocal_=e_nl,
            ewald=ewald_energy(system.positions, system.charges, grid.cell),
            smearing=entropy_term,
            onecenter=e_onec,
        )
        e_free = float(energies.free_energy)

        # ---- residual, convergence, mixing ----
        vin = pack(rho, m, bec_chan)
        vout = pack(rho_out, m_out, bec_out_r)
        res_norm = float(torch.linalg.norm(vout - vin)) * vol
        de = abs(e_free - e_free_prev) if e_free_prev is not None else float("inf")
        history.append({"iter": it, "free_energy": e_free, "dE": de, "res": res_norm})
        if verbose:
            mv = [float(m_out[i].mean()) * vol for i in range(3)]
            print(f"  NC-USPP {it:3d}  F = {e_free:+.8f}  dE = {de:.2e}  "
                  f"|dρ,m,bec| = {res_norm:.2e}  "
                  f"m⃗ = ({mv[0]:+.3f},{mv[1]:+.3f},{mv[2]:+.3f})", flush=True)
        if de < etol and res_norm < rhotol and tol_eff <= diago_tol * 1.01:
            converged = True
            rho, m, bec_chan = rho_out, m_out, bec_out_r
            break
        e_free_prev = e_free
        rho, m, bec_chan = unpack(mixer.step(vin, vout))
        bec_chan = [[0.5 * (c + c.T) for c in bec_chan[c4]] for c4 in range(4)]

    m_int = [float(m[i].mean()) * vol for i in range(3)]
    m_norm = torch.sqrt((m ** 2).sum(dim=0))
    return dict(
        converged=converged, n_iter=it, energies=energies, fermi=mu,
        mag_vec=tuple(m_int), mag_abs=float(m_norm.mean()) * vol,
        rho=rho, m=m, eigenvalues=eigs, history=history,
        rho_ij_chan=bec_chan, coeffs=coeffs,
    )


def _local_energy(rho_g_box, vloc_g, vol):
    from gradwave.core.energies.local_pp import local_energy

    return local_energy(rho_g_box, vloc_g, vol)


def _slice_bk(bk, ik: int):
    import dataclasses

    return dataclasses.replace(
        bk, npw=bk.npw[ik:ik + 1], mask=bk.mask[ik:ik + 1],
        flat_idx=bk.flat_idx[ik:ik + 1], kpg=bk.kpg[ik:ik + 1], t=bk.t[ik:ik + 1],
        proj_phase_free=bk.proj_phase_free[ik:ik + 1])
