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

import dataclasses
import time

import torch

from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.hartree import hartree_energy, hartree_potential_g
from gradwave.core.energies.local_pp import local_energy
from gradwave.core.energies.total import EnergyBreakdown
from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.core.occupations import SCHEMES, find_fermi, occupations_and_entropy
from gradwave.core.xc.noncollinear import NoncollinearXC, energy_with_grid, vxc_and_bxc
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.scf.common import (
    adaptive_diago_tol,
    convergence_gate,
    record_iteration,
    symmetrize_rho,
)
from gradwave.scf.guess import sad_density
from gradwave.scf.mixing import PulayMixer
from gradwave.scf.paw_noncollinear import (
    onsite_nc_energy_and_ddd,
    spinor_onsite_becsum,
)
from gradwave.scf.results import USPPNCResult
from gradwave.scf.spinor_common import (
    apply_local_spinor,
    pack_grid_channels,
    pauli_density_accumulate,
    spinor_band_chunk,
    spinor_kinetic_energy,
    spinor_potential_blocks,
    spinor_pw_seed,
    spinor_scalar_nonlocal_energy,
    unpack_grid_channels,
)
from gradwave.scf.uspp_loop import (
    _build_iter_ops,
    _seed_becsum,
    _species_atoms,
    aug_dmat_batched,
)
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

    def __init__(self, bk, shape, v_r, b_vec_r, p, d_chan, q_full, smooth=None):
        self.inner = bk
        self.bk = _SpinorBK(bk)
        self.shape = shape
        self.p = p
        # constant per SCF iteration but consumed every Davidson round: cache
        # the resolved conjugate once instead of materializing p.conj() per apply
        self.p_conj = p.conj().resolve_conj()
        self.q_full = q_full.to(CDTYPE)
        self.m = bk.npw_max
        self.t = torch.cat([bk.t, bk.t], dim=-1)
        d_n, d_x, d_y, d_z = (d.to(CDTYPE) for d in d_chan)
        self._d_uu = d_n + d_z
        self._d_dd = d_n - d_z
        self._d_ud = d_x - 1j * d_y            # image of v_ud = B_x − iB_y
        self.b_zero, self._v_uu, self._v_dd, self._v_ud = \
            spinor_potential_blocks(v_r, b_vec_r)
        # dual grid (USPP/PAW): the local 2×2 mix runs on the smaller smooth
        # box — exact for ⟨ψ|V|ψ⟩ since ψ†ψ products live within twice the
        # wavefunction cutoff (same argument as the collinear BatchedHamiltonian;
        # the 2×2 blocks are plain grid fields, so it holds per block). The
        # kinetic and nonlocal terms are sphere-based and untouched. The caller
        # passes smooth = (shape_s, flat_idx_s) with the potentials already
        # filtered onto the smooth box.
        self._fft_bk = bk
        self._fft_shape = shape
        if smooth is not None:
            shape_s, flat_idx_s = smooth
            self._fft_bk = dataclasses.replace(bk, flat_idx=flat_idx_s)
            self._fft_shape = shape_s

    def _band_chunk(self, nk: int, device, elem_bytes: int = 16) -> int:
        """Bands per chunk for the local mix + nonlocal einsums — the shared
        spinor heuristic (scf/spinor_common.py), on the smooth box when the
        dual grid is active."""
        return spinor_band_chunk(self._fft_shape, nk, device, elem_bytes)

    def h(self, c):
        bk, m = self.inner, self.m
        cu, cd = c[..., :m], c[..., m:]
        t_r = bk.t
        out_u = t_r[:, None, :] * cu
        out_d = t_r[:, None, :] * cd
        nk, nb = c.shape[0], c.shape[1]
        chunk = self._band_chunk(nk, c.device, c.element_size())
        # local 2×2 mix — the shared fused-FFT band-chunked apply, on the
        # smooth box when the dual grid is active
        apply_local_spinor(out_u, out_d, cu, cd, self._fft_bk,
                           self._fft_shape, chunk, self._v_uu, self._v_dd,
                           self._v_ud, self.b_zero)
        # nonlocal: 2×2 screened D in projector space; one fused becp for
        # both spin components
        p, pc = self.p, self.p_conj
        bud = torch.einsum("kpg,kbg->kbp", pc, torch.cat([cu, cd], dim=1))
        bu, bd = bud[:, :nb], bud[:, nb:]
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
        nb = c.shape[1]
        bud = torch.einsum("kpg,kbg->kbp", self.p_conj,
                           torch.cat([c[..., :m], c[..., m:]], dim=1))
        outs = []
        for blk, b in ((c[..., :m], bud[:, :nb]), (c[..., m:], bud[:, nb:])):
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
) -> USPPNCResult:
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
    # the n-channel is the same reference atomic-occupation diagonal the
    # collinear USPP path seeds (spin-summed, i.e. the nspin=1 becsum); reuse
    # _seed_becsum for it and direct the moment onto the m-channels
    n_seed = _seed_becsum(system, 1, None, [None], dev)[0]
    for a in range(na):
        n0 = n_seed[a].real
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
        gvecs = pack_grid_channels((rho_, m_[0], m_[1], m_[2]), mask_flat)
        bflat = [torch.cat([c.reshape(-1).to(CDTYPE) for c in bec_[i]])
                 for i in range(4)]
        return torch.cat([gvecs] + bflat)

    def unpack(v):
        fields = unpack_grid_channels(v, 4, ng, mask_flat, shape,
                                      grid.n_points, dev)
        bec = [[] for _ in range(4)]
        off = 4 * ng
        for i in range(4):
            for (s0, s1) in system.atom_slices:
                n = s1 - s0
                bec[i].append(v[off:off + n * n].reshape(n, n).real.clone())
                off += n * n
        return fields[0], torch.stack(fields[1:]), bec

    # ---- spinor seeds: alternate up/down lowest plane waves ----
    coeffs = spinor_pw_seed(nk, nbands, m_pw, dev)

    scheme = SCHEMES[smearing]
    e_free_prev, converged, history = None, False, []
    mu, it = 0.0, 0
    energies = None
    q_c = system.q_full.to(CDTYPE)
    dij_bare = system.proj_data[0].dij_full
    # atoms grouped by species: the per-iteration ∫(v,B⃗)Q and augmentation
    # contractions batch over the atom axis (one einsum per species instead
    # of a Python loop of tiny kernels per atom per channel)
    sp_atoms = _species_atoms(system)
    smooth_geom = None
    if system.smooth_shape is not None:
        smooth_geom = (system.smooth_shape, system.smooth_flat_idx)

    # E_ewald is constant across the loop (positions frozen) — build it once.
    e_ew = ewald_energy(system.positions, system.charges, grid.cell)

    for it in range(1, max_iter + 1):
        t_it = time.perf_counter()
        # ---- potentials ----
        v_h = g_to_r_box(
            hartree_potential_g(r_to_g(rho.to(CDTYPE)), grid.g2), real=True)
        v_xc, b_xc, _ = vxc_and_bxc(ncxc, rho, m, grid, rho_core=system.rho_core)
        v_r = v_h + v_xc + ops.vloc_r

        # ---- screened D: four channels, ∫(v, B⃗) Q + bare D + one-center ddd ----
        # one FFT for all four channels, one einsum per species over
        # (channel, atom) — not a per-atom Python loop with a q_g.conj()
        # re-materialized 4·na times per iteration
        pots = torch.stack([v_r, b_xc[0], b_xc[1], b_xc[2]])
        pots_g_box = r_to_g(pots.to(CDTYPE)).reshape(4, -1)
        d_chan = list(aug_dmat_batched(system, pots_g_box[:, mask_flat],
                                       ops.phase_pos))
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
        tol_eff = adaptive_diago_tol(it, history, diago_tol,
                                     system.n_electrons, schedule="linear")
        smooth = None
        if smooth_geom is not None:
            # filter the 2×2 potential fields onto the smooth box (the dense
            # G-coeffs above restricted to the smooth sphere by shared Miller)
            # for the dual-grid H-apply; filtering is linear, so filtering the
            # (v, B⃗) fields and combining into the 2×2 blocks commutes
            vb_s = g_to_r_box(
                pots_g_box[:, system.smooth2dense].reshape(
                    4, *system.smooth_shape), real=True)
            v_r_h, b_xc_h = vb_s[0], vb_s[1:]
            smooth = smooth_geom
        else:
            v_r_h, b_xc_h = v_r, b_xc
        hs = SpinorBatchedHS(bk, shape, v_r_h, b_xc_h, p_b, d_chan,
                             system.q_full, smooth=smooth)
        eigs, x = davidson_gen_batched(hs, coeffs, nbands, tol=tol_eff)
        eigs = eigs.to(RDTYPE)
        # one fused becp for both spin components; ⟨β|ψ⟩ is linear, so the
        # S-normalized projections come from the same contraction (reused by
        # the density/becsum build below instead of a second becp pass)
        bud = torch.einsum("kpg,kbg->kbp", hs.p_conj,
                           torch.cat([x[..., :m_pw], x[..., m_pw:]], dim=1))
        bu, bd = bud[:, :nbands], bud[:, nbands:]
        snorm = (x.abs() ** 2).sum(dim=-1) \
            + torch.einsum("kbi,ij,kbj->kb", bu.conj(), q_c, bu).real \
            + torch.einsum("kbi,ij,kbj->kb", bd.conj(), q_c, bd).real
        sn = torch.sqrt(snorm)[..., None]
        coeffs = x / sn
        bu, bd = bu / sn, bd / sn

        # ---- occupations (spinor: one electron per band) ----
        mu = float(find_fermi(eigs, system.kweights, scheme, width,
                              system.n_electrons, degeneracy=1.0))
        mu_t = torch.tensor(mu, dtype=RDTYPE, device=dev)
        occ, s_ent = occupations_and_entropy(eigs, mu_t, scheme, width,
                                             degeneracy=1.0)
        entropy_term = -width * (system.kweights[:, None] * s_ent).sum()

        # ---- densities: Pauli grid channels + 2×2 becsum + 4-channel aug ----
        # the shared band-chunked, fused-FFT Pauli accumulation
        # (scf/spinor_common.py) on the DENSE grid (the density needs the
        # full ecutrho resolution, unlike the H-apply's smooth-box mix)
        w_kb = (system.kweights[:, None] * occ).to(RDTYPE)
        nbc = spinor_band_chunk(shape, nk, dev, coeffs.element_size())
        rho_out, m_out = pauli_density_accumulate(
            coeffs, w_kb, bk, shape, m_pw, nbands, nbc, dev)
        # becsum from the already-computed S-normalized projections, with k
        # folded into the band axis (the einsums contract over bands anyway)
        bec_out = [[None] * len(system.atom_slices) for _ in range(4)]
        w_flat = w_kb.reshape(-1).to(CDTYPE)
        bu_f = bu.reshape(nk * nbands, -1)
        bd_f = bd.reshape(nk * nbands, -1)
        for a, (s0, s1) in enumerate(system.atom_slices):
            chans = spinor_onsite_becsum(bu_f[:, s0:s1], bd_f[:, s0:s1], w_flat)
            for c4 in range(4):
                bec_out[c4][a] = chans[c4]
        rho_out, m_out = rho_out / vol, m_out / vol
        bec_out_r = [[c.real for c in bec_out[c4]] for c4 in range(4)]
        if mag_sym_active:
            # symmetrize the becsum BEFORE building the augmentation charge so
            # the smooth and one-center densities carry the same symmetry
            bec_out_r = system.becsum_sym.apply(bec_out_r)

        # augmentation: n_aug → ρ, m⃗_aug → m⃗, from the matching becsum channel —
        # all four channels and all atoms of a species in one einsum, one
        # batched FFT for the four aug fields
        aug_sph4 = torch.zeros(4, system.sphere_idx.shape[0], dtype=CDTYPE,
                               device=dev)
        for sp, atoms in sp_atoms.items():
            bec_sp = torch.stack([
                torch.stack([bec_out_r[c4][a].to(CDTYPE) for a in atoms])
                for c4 in range(4)])                       # (4, na_sp, nm, nm)
            aug_sph4 += torch.einsum("caij,ijg,ga->cg", bec_sp,
                                     system.aug[sp].q_g,
                                     ops.phase_pos[:, atoms].conj())
        aug_box = torch.zeros(4, grid.n_points, dtype=CDTYPE, device=dev)
        aug_box[:, system.sphere_idx] = aug_sph4 / vol
        aug_fields = g_to_r_box(aug_box.reshape(4, *shape), real=True)
        rho_out = rho_out + aug_fields[0]
        m_out = m_out + aug_fields[1:]
        if mag_sym_active:
            rho_out = symmetrize_rho(system.rho_symmetrizer, rho_out, grid)
            m_g = torch.stack([r_to_g(m_out[i].to(CDTYPE)) for i in range(3)])
            m_out = g_to_r_box(system.rho_symmetrizer.apply_m(m_g), real=True)

        n_tot = float(rho_out.sum()) * vol / grid.n_points
        if abs(n_tot - system.n_electrons) >= 1e-5:
            raise ValueError(
                f"charge not conserved: {n_tot:.8f} vs {system.n_electrons}")

        # ---- energies ----
        rho_g_out = r_to_g(rho_out.to(CDTYPE))
        t_occ = (system.kweights[:, None] * occ).to(RDTYPE)
        e_kin = spinor_kinetic_energy(t_occ, coeffs, hs.t)
        e_nl = spinor_scalar_nonlocal_energy(bu, bd, dij_bare, occ,
                                             system.kweights, nk)
        energies = EnergyBreakdown(
            kinetic=e_kin,
            hartree=hartree_energy(rho_g_out, grid.g2, vol),
            xc=energy_with_grid(ncxc, rho_out, m_out, grid,
                                rho_core=system.rho_core),
            local=local_energy(rho_g_out, ops.vloc_g, vol),
            nonlocal_=e_nl,
            ewald=e_ew,
            smearing=entropy_term,
            onecenter=e_onec,
        )
        e_free = float(energies.free_energy)

        # ---- residual, convergence, mixing ----
        vin = pack(rho, m, bec_chan)
        vout = pack(rho_out, m_out, bec_out_r)
        # Convergence residual is the (ρ, m⃗) density change only — the first
        # 4·ng grid channels, excluding the 4·nbec becsum tail. This matches
        # the canonical convention in the collinear USPP loop (residual on
        # rho_*_vec[: ng*nspin]) and the norm-conserving loops. Including the
        # stiff on-site becsum mode over-tightens the gate: a PAW magnet whose
        # becsum residual floors above rhotol while ρ and m⃗ are settled would
        # otherwise never satisfy convergence_gate and burn to max_iter.
        res_norm = float(torch.linalg.norm((vout - vin)[: 4 * ng])) * vol
        de = record_iteration(history, it, e_free, e_free_prev, res_norm, t_it)
        if verbose:
            mv = [float(m_out[i].mean()) * vol for i in range(3)]
            print(f"  NC-USPP {it:3d}  F = {e_free:+.8f}  dE = {de:.2e}  "
                  f"|dρ,m| = {res_norm:.2e}  "
                  f"m⃗ = ({mv[0]:+.3f},{mv[1]:+.3f},{mv[2]:+.3f})", flush=True)
        if convergence_gate(de, res_norm, tol_eff, etol, rhotol, diago_tol):
            converged = True
            rho, m, bec_chan = rho_out, m_out, bec_out_r
            break
        e_free_prev = e_free
        rho, m, bec_chan = unpack(mixer.step(vin, vout))
        bec_chan = [[0.5 * (c + c.T) for c in bec_chan[c4]] for c4 in range(4)]

    m_int = [float(m[i].mean()) * vol for i in range(3)]
    m_norm = torch.sqrt((m ** 2).sum(dim=0))
    return USPPNCResult(
        converged=converged, n_iter=it, energies=energies, fermi=mu,
        mag_vec=tuple(m_int), mag_abs=float(m_norm.mean()) * vol,
        rho=rho, m=m, eigenvalues=eigs, history=history,
        rho_ij_chan=bec_chan, coeffs=coeffs,
    )


