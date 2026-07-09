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

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.hartree import hartree_energy, hartree_potential_g
from gradwave.core.energies.kinetic import kinetic_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.energies.total import EnergyBreakdown
from gradwave.core.fftbox import box_to_sphere, g_to_r, r_to_g
from gradwave.core.gaunt import real_gaunt_table, ylm_np
from gradwave.core.hamiltonian import ProjectorData, becp, build_projector_data, projectors
from gradwave.core.occupations import (
    SCHEMES,
    find_fermi,
    fixed_occupations,
    occupations_and_entropy,
)
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_fft_grid, build_gsphere
from gradwave.pseudo.kb import beta_form_factors
from gradwave.pseudo.radial import sbt
from gradwave.pseudo.upf_paw import PAWData
from gradwave.scf.guess import sad_density
from gradwave.scf.loop import vxc_potential
from gradwave.scf.mixing import PulayMixer
from gradwave.solvers.precond import teter

_MINUS_I_POW_L = [1.0, -1.0j, -1.0, 1.0j, 1.0]  # (−i)^L, L ≤ 4


@dataclass
class AugSpecies:
    """m-expanded augmentation form factors of one species on the density
    sphere: q_g[i, j, :] = Q̃_(i,mi),(j,mj)(G) (dimensionless charge)."""

    q_g: torch.Tensor  # (nproj_m, nproj_m, nGm) complex128
    q_int: torch.Tensor  # (nproj_m, nproj_m) real — ∫Q d³r, the S weights


@dataclass
class USPPSystem:
    grid: object
    spheres: list
    kweights: torch.Tensor
    positions: torch.Tensor
    species_of_atom: list
    paws: list
    charges: torch.Tensor
    n_electrons: float
    nbands: int
    ecut: float
    proj_data: list  # per-k ProjectorData (dij_full = BARE D, m-expanded)
    q_full: torch.Tensor  # (nproj_tot, nproj_tot) m-expanded S weights
    aug: list  # per-species AugSpecies
    sphere_idx: torch.Tensor  # (nGm,) flat indices of the density sphere
    g_sphere: torch.Tensor  # (nGm, 3)
    vloc_tables: torch.Tensor
    rho_core: torch.Tensor | None
    atom_slices: list = field(default_factory=list)  # per-atom projector column ranges
    sym: object = None  # SpaceGroup when use_symmetry
    rho_symmetrizer: object = None
    becsum_sym: object = None


def _mexp_index_map(paw: PAWData):
    """[(channel i, l, m_col)] in projector-column order for one atom."""
    out = []
    for i, b in enumerate(paw.betas):
        for m_col in range(2 * b.l + 1):
            out.append((i, b.l, m_col))
    return out


def _aug_tables(paw: PAWData, g_sphere: np.ndarray) -> AugSpecies:
    """FT of every m-expanded augmentation function on the density sphere."""
    lmax_b = max(b.l for b in paw.betas)
    gnorm = np.linalg.norm(g_sphere, axis=1)
    uniq, inv = np.unique(np.round(gnorm, 10), return_inverse=True)
    y = ylm_np(2 * lmax_b, g_sphere)  # (nGm, (2lb+1)²)
    c_gaunt = real_gaunt_table(lmax_b)  # (LM, lm_i, lm_j)

    # radial transforms per (channel pair, L) on unique shells
    n = paw.aug_cutoff_idx
    rad: dict[tuple, np.ndarray] = {}
    for (i, j, ll), qfun in paw.qijl.items():
        rad[(i, j, ll)] = sbt(ll, qfun, paw.r[:n], paw.rab[:n], uniq)[inv]

    idx = _mexp_index_map(paw)
    nm = len(idx)
    q_g = np.zeros((nm, nm, len(g_sphere)), dtype=np.complex128)
    for a, (i, li, mi) in enumerate(idx):
        for b, (j, lj, mj) in enumerate(idx):
            key = (i, j) if (i, j) in {(p, q) for (p, q, _) in paw.qijl} else (j, i)
            lm_i, lm_j = li * li + mi, lj * lj + mj
            acc = np.zeros(len(g_sphere), dtype=np.complex128)
            for ll in range(abs(li - lj), li + lj + 1):
                if (key[0], key[1], ll) not in rad:
                    continue
                f_l = rad[(key[0], key[1], ll)]
                cy = c_gaunt[ll * ll : (ll + 1) ** 2, lm_i, lm_j]  # (2L+1,)
                ang = y[:, ll * ll : (ll + 1) ** 2] @ cy
                acc += _MINUS_I_POW_L[ll] * ang * f_l
            q_g[a, b] = 4.0 * math.pi * acc

    q_int = np.zeros((nm, nm))
    for a, (i, li, mi) in enumerate(idx):
        for b, (j, lj, mj) in enumerate(idx):
            if li == lj and mi == mj:
                q_int[a, b] = paw.q[i, j]
    return AugSpecies(
        q_g=torch.as_tensor(q_g, dtype=CDTYPE),
        q_int=torch.as_tensor(q_int, dtype=RDTYPE),
    )


def setup_uspp(
    cell,
    positions,
    species_of_atom,
    paws: list[PAWData],
    ecut: float,
    kmesh=(1, 1, 1),
    nbands: int | None = None,
    ecutrho: float | None = None,
    fft_shape=None,
    use_symmetry: bool = False,
    symprec: float = 1e-6,
) -> USPPSystem:
    from gradwave.kpoints import monkhorst_pack
    from gradwave.pseudo.local import alpha_z, vloc_of_g
    from gradwave.scf.loop import _unique_shells

    cell = np.asarray(cell, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if ecutrho is None:
        ecutrho = 4.0 * ecut
    sym = rho_symmetrizer = None
    if use_symmetry:
        from gradwave.symmetry import find_spacegroup

        frac = positions @ np.linalg.inv(cell)
        sym = find_spacegroup(cell, frac, list(species_of_atom), symprec=symprec)
        if sym.n_ops <= 1:
            sym = None
    # build_fft_grid derives the density sphere as 2·G_max(ecut_arg)
    grid = build_fft_grid(cell, ecutrho / 4.0, shape_override=fft_shape,
                          equal_dims=sym is not None)
    if sym is not None:
        from gradwave.symmetry import RhoSymmetrizer, reduce_mesh

        rho_symmetrizer = RhoSymmetrizer(grid.shape, sym, dens_mask=grid.dens_mask)
        kfrac, kw = reduce_mesh(kmesh, (0, 0, 0), sym, time_reversal=True)
    else:
        kfrac, kw = monkhorst_pack(kmesh, (0, 0, 0), time_reversal=True)
    spheres = [build_gsphere(grid, ecut, k) for k in kfrac]

    charges = torch.tensor([paws[s].z_valence for s in species_of_atom], dtype=RDTYPE)
    n_electrons = float(charges.sum())
    if nbands is None:
        nocc = int(np.ceil(n_electrons / 2.0))
        nbands = max(int(np.ceil(nocc * 1.2)), nocc + 4)

    g_flat = np.sqrt(grid.g2.reshape(-1).numpy())
    uniq, inverse = _unique_shells(g_flat)
    vloc_tables = []
    for paw in paws:
        tab = np.empty_like(uniq)
        tab[0] = alpha_z(paw)
        tab[1:] = vloc_of_g(paw, uniq[1:])
        vloc_tables.append(tab[inverse].reshape(grid.shape))
    vloc_tables = torch.as_tensor(np.stack(vloc_tables), dtype=RDTYPE)

    dij_species = [torch.as_tensor(p.dij, dtype=RDTYPE) for p in paws]
    # S weights from the SAME radial integrals as the augmentation tables —
    # PP_Q agrees with ∫q⁰_ij dr only to the file's print precision (~5e-8 per
    # pair), and any mismatch breaks exact charge conservation (ρ_aug carries
    # Q̃(0)=∫q⁰ while S-normalization would enforce PP_Q). QE's init_us_1
    # recomputes qq from qfuncl the same way.
    from gradwave.pseudo.radial import simpson as _simpson

    q_species = []
    for p in paws:
        qm = np.array(p.q, dtype=np.float64)
        for (i, j, ll), qfun in p.qijl.items():
            if ll == 0:
                val = float(_simpson(qfun, p.rab[: p.aug_cutoff_idx]))
                qm[i, j] = qm[j, i] = val
        q_species.append(torch.as_tensor(qm, dtype=RDTYPE))
    beta_ls = [[b.l for b in p.betas] for p in paws]
    proj_data, q_full = [], None
    for sph in spheres:
        q_of_k = np.sqrt(sph.kpg2.numpy())
        beta_tables = [
            torch.as_tensor(beta_form_factors(p, q_of_k), dtype=RDTYPE) for p in paws
        ]
        proj_data.append(
            build_projector_data(sph, species_of_atom, beta_tables, beta_ls,
                                 dij_species, grid.volume)
        )
        if q_full is None:  # m-expansion identical at every k — reuse the builder
            q_full = build_projector_data(
                sph, species_of_atom, beta_tables, beta_ls, q_species, grid.volume
            ).dij_full

    # density sphere and per-species augmentation tables
    mask = grid.dens_mask.reshape(-1)
    sphere_idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    g_sphere = grid.g_cart.reshape(-1, 3)[sphere_idx]
    aug = [_aug_tables(p, g_sphere.numpy()) for p in paws]

    rho_core = None
    if any(p.core_rho is not None for p in paws):
        from gradwave.core.structure import structure_factors
        from gradwave.pseudo.atomic import core_density_of_q

        core_g = torch.zeros(grid.n_points, dtype=CDTYPE)
        pos_t = torch.as_tensor(positions, dtype=RDTYPE)
        for sp_i, paw in enumerate(paws):
            tab = torch.as_tensor(core_density_of_q(paw, uniq), dtype=RDTYPE)
            shell = tab[torch.as_tensor(inverse)]
            atoms = [a for a, sa in enumerate(species_of_atom) if sa == sp_i]
            if not atoms:
                continue
            sfac = structure_factors(pos_t[atoms], grid.g_cart).sum(dim=0).reshape(-1)
            core_g += sfac * shell.to(CDTYPE) / grid.volume
        core_g = torch.where(mask, core_g, torch.zeros_like(core_g))
        rho_core = torch.fft.ifftn(
            core_g.reshape(grid.shape) * grid.n_points, dim=(-3, -2, -1)
        ).real

    # per-atom projector column ranges (atoms in order, matching build order)
    slices, start = [], 0
    for sp in species_of_atom:
        nm = sum(2 * b.l + 1 for b in paws[sp].betas)
        slices.append((start, start + nm))
        start += nm

    return USPPSystem(
        grid=grid, spheres=spheres,
        kweights=torch.as_tensor(kw, dtype=RDTYPE),
        positions=torch.as_tensor(positions, dtype=RDTYPE),
        species_of_atom=list(species_of_atom), paws=list(paws), charges=charges,
        n_electrons=n_electrons, nbands=nbands, ecut=ecut,
        proj_data=proj_data, q_full=q_full, aug=aug,
        sphere_idx=sphere_idx, g_sphere=g_sphere,
        vloc_tables=vloc_tables, rho_core=rho_core, atom_slices=slices,
        sym=sym, rho_symmetrizer=rho_symmetrizer,
        becsum_sym=(None if sym is None else _make_becsum_sym(
            sym, cell, paws, species_of_atom, slices)),
    )


def _make_becsum_sym(sym, cell, paws, species_of_atom, slices):
    from gradwave.scf.paw_symmetry import BecsumSymmetrizer

    return BecsumSymmetrizer(sym, cell, paws, species_of_atom, slices)


class _HkS:
    """H and S applies at one k for fixed v_eff and screened D."""

    def __init__(self, sphere, shape, v_eff_r, pd: ProjectorData, p, dscr, q_full):
        self.sphere, self.shape = sphere, shape
        self.v_eff_r = v_eff_r
        self.p = p
        self.dscr = dscr.to(CDTYPE)
        self.q = q_full.to(CDTYPE)
        self.t = HBAR2_2M * sphere.kpg2

    def h(self, c):
        out = self.t * c
        psi = g_to_r(c, self.sphere.flat_idx, self.shape)
        out = out + box_to_sphere(r_to_g(psi * self.v_eff_r), self.sphere.flat_idx)
        b = becp(self.p, c)
        return out + (b @ self.dscr) @ self.p

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


def scf_uspp(system: USPPSystem, xc, *, nspin: int = 1, start_mag=None,
             smearing="none", width=0.1, max_iter=60, etol=1e-8, rhotol=1e-7,
             diago_tol=1e-9, mixing_alpha=0.7, mixing_history=8,
             trust_factor=20.0, verbose=True):
    """USPP/PAW SCF. nspin=2 takes a SpinXC functional and per-species
    start_mag (list, in [-1, 1]); mixing then runs in the (total,
    magnetization) basis with Kerker on the total for smeared systems."""
    grid = system.grid
    vol = grid.volume
    nk, nb = len(system.spheres), system.nbands
    shape = grid.shape
    mask_flat = grid.dens_mask.reshape(-1)
    g_spin = 2 if nspin == 1 else 1

    if nspin == 1:
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

    projs = [projectors(pd, system.positions) for pd in system.proj_data]
    vloc_g = local_potential_g(system.positions, torch.tensor(system.species_of_atom),
                               system.vloc_tables, grid.g_cart, vol)
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real

    phase_arg = system.g_sphere @ system.positions.T  # (nGm, na)
    phase_pos = torch.exp(torch.complex(torch.zeros_like(phase_arg), phase_arg))

    # Mixing vector = [ρ channels on the density sphere, flattened becsum per
    # spin]. Mixing becsum TOGETHER with ρ (QE keeps it inside rho%mix the
    # same way) is essential for metals: the D-feedback loops (∫v_eff Q and
    # the one-center ddd) must see a becsum coherent with the mixed density —
    # a fresh or independently-damped becsum gives a gain>1 charge
    # oscillation for semicore-metal PAW (fcc Ni diverges with ×9/iteration).
    # Kerker damps the ρ-TOTAL block only (becsum is localized; the
    # magnetization channel must keep its G=0 free for ↑↓ transfer).
    g2_mix = grid.g2.reshape(-1)[mask_flat]
    ng = g2_mix.shape[0]
    nbec = sum((s1 - s0) ** 2 for (s0, s1) in system.atom_slices)
    g2_full = torch.cat([g2_mix] * nspin + [torch.zeros(nbec)] * nspin)
    kerker_mask = torch.cat([
        torch.ones(ng, dtype=torch.bool),
        torch.zeros(ng * (nspin - 1) + nbec * nspin, dtype=torch.bool),
    ])
    # the on-site becsum↔ddd feedback is the stiffest direction (on-site
    # Hartree curvature ~tens of eV) — take smaller plain steps on the becsum
    # block while DIIS accumulates curvature
    step_scale = torch.cat([
        torch.ones(ng * nspin, dtype=torch.float64),
        torch.full((nbec * nspin,), 0.4, dtype=torch.float64),
    ])
    mixer = PulayMixer(g2_full, alpha=mixing_alpha, history=mixing_history,
                       kerker=(smearing != "none"), kerker_mask=kerker_mask,
                       check_g0=False, step_scale=step_scale)
    coeffs = [[None] * nk for _ in range(nspin)]
    e_free_prev, history, converged = None, [], False
    occ_s = entropy_term = eigs_s = mu = None
    energies = None

    # PAW one-center machinery; becsum seeded from the reference atomic
    # occupations (spin-split by start_mag; zeros for bare USPP where the UPF
    # carries no PP_OCCUPATIONS). rho_ij_mix is the MIXER-side becsum used
    # for the one-center ddd; rho_ij_s holds each iteration's fresh becsum.
    is_paw = any(p.is_paw for p in system.paws)
    onec = None
    if is_paw:
        from gradwave.scf.paw_onsite import OneCenter

        onec = [OneCenter(p, xc) for p in system.paws]
    rho_ij_s = [[] for _ in range(nspin)]
    for sp in system.species_of_atom:
        paw = system.paws[sp]
        nm = sum(2 * b.l + 1 for b in paw.betas)
        for isp in range(nspin):
            m0 = torch.zeros(nm, nm, dtype=CDTYPE)
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

    def _becsum_for_onec(a):
        if nspin == 1:
            return rho_ij_mix[0][a]
        return [rho_ij_mix[0][a], rho_ij_mix[1][a]]

    for it in range(1, max_iter + 1):
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
        e_onec = torch.zeros((), dtype=RDTYPE)
        if is_paw:
            dscr_s = [d.clone() for d in dscr_s]
            for a, sp in enumerate(system.species_of_atom):
                s0, s1 = system.atom_slices[a]
                e1c, ddd = onec[sp].energy_and_ddd(_becsum_for_onec(a))
                e_onec = e_onec + e1c
                if nspin == 1:
                    dscr_s[0][s0:s1, s0:s1] += ddd
                else:
                    for isp in range(nspin):
                        dscr_s[isp][s0:s1, s0:s1] += ddd[isp]

        if it == 1:
            tol_eff = max(diago_tol, 1e-3)
        else:
            r_prev = history[-1]["res"]
            tol_eff = max(diago_tol, min(1e-3, 0.1 * r_prev * r_prev / system.n_electrons))

        eigs_s = []
        for isp in range(nspin):
            eigs_l = []
            for ik, sph in enumerate(system.spheres):
                hs = _HkS(sph, shape, veff_s[isp], system.proj_data[ik], projs[ik],
                          dscr_s[isp], system.q_full)
                if coeffs[isp][ik] is None:
                    gen = torch.Generator().manual_seed(1234 + ik + 7777 * isp)
                    x0 = torch.randn(nb + 4, sph.npw, generator=gen, dtype=torch.float64) \
                        + 1j * torch.randn(nb + 4, sph.npw, generator=gen,
                                           dtype=torch.float64)
                    x0 = x0 * torch.exp(-0.5 * HBAR2_2M * sph.kpg2 / system.ecut * 4.0)
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

        if smearing == "none":
            if nspin != 1:
                raise ValueError("nspin=2 requires smearing (shared Fermi level)")
            occ_s = [fixed_occupations(eigs_s[0], system.n_electrons)]
            mu = float(eigs_s[0][:, int(system.n_electrons // 2) - 1].max())
            entropy_term = torch.zeros((), dtype=RDTYPE)
        else:
            scheme = SCHEMES[smearing]
            eigs_cat = torch.cat(eigs_s, dim=0)
            kw_cat = torch.cat([system.kweights] * nspin)
            mu = float(find_fermi(eigs_cat, kw_cat, scheme, width,
                                  system.n_electrons, degeneracy=g_spin))
            mu_t = torch.tensor(mu, dtype=RDTYPE)
            occ_s, ent = [], torch.zeros((), dtype=RDTYPE)
            for isp in range(nspin):
                o, s_ent = occupations_and_entropy(
                    eigs_s[isp], mu_t, scheme, width, degeneracy=g_spin)
                occ_s.append(o)
                ent = ent - width * (g_spin * system.kweights[:, None] * s_ent).sum()
            entropy_term = ent

        # smooth densities + per-spin becsum + augmentation
        rho_out_s, becps_s = [], []
        rho_ij_s = [[torch.zeros(s1 - s0, s1 - s0, dtype=CDTYPE)
                     for (s0, s1) in system.atom_slices] for _ in range(nspin)]
        for isp in range(nspin):
            rho_sp = torch.zeros(shape, dtype=RDTYPE)
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

            aug_sph = torch.zeros(system.sphere_idx.shape[0], dtype=CDTYPE)
            for a, sp in enumerate(system.species_of_atom):
                aug_sph = aug_sph + phase_pos[:, a].conj() * torch.einsum(
                    "ij,ijg->g", rho_ij_s[isp][a], system.aug[sp].q_g
                )
            aug_box = torch.zeros(grid.n_points, dtype=CDTYPE)
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
            onecenter=e_onec,
        )
        e_free = float(energies.free_energy)

        def to_mix(chans, becs):
            vecs = [r_to_g(c.to(CDTYPE)).reshape(-1)[mask_flat] for c in chans]
            if nspin == 2:
                vecs = [vecs[0] + vecs[1], vecs[0] - vecs[1]]
            bec_flat = [torch.cat([m.reshape(-1) for m in becs[isp]])
                        for isp in range(nspin)]
            return torch.cat(vecs + bec_flat)

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
        if de < etol and res_norm < rhotol and tol_eff <= diago_tol * 1.01:
            converged = True
            rho_s = rho_out_s
            if is_paw:  # report the one-center energy at the FINAL fresh becsum
                e_onec = torch.zeros((), dtype=RDTYPE)
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
        # damped step rather than extrapolating into nonsense
        best_res = min(h["res"] for h in history)
        if it > 1 and res_norm > trust_factor * best_res:
            mixer.reset()
            if verbose:
                print(f"  USPP {it:3d}  [mixer reset: residual jumped "
                      f"{res_norm:.2e} > {trust_factor:g}x best {best_res:.2e}]")
        mixed = mixer.step(rho_in_vec, rho_out_vec)
        rho_block, bec_block = mixed[: ng * nspin], mixed[ng * nspin:]
        if nspin == 1:
            chan_vecs = [rho_block]
        else:
            tot, mag_v = rho_block[:ng], rho_block[ng:]
            chan_vecs = [(tot + mag_v) / 2.0, (tot - mag_v) / 2.0]
        rho_s = []
        for vec in chan_vecs:
            rho_g_new = torch.zeros(grid.n_points, dtype=CDTYPE)
            rho_g_new[mask_flat] = vec
            rho_s.append((torch.fft.ifftn(rho_g_new.reshape(shape) * grid.n_points,
                                          dim=(-3, -2, -1))).real)
        # unpack the mixed becsum (Hermitize — linear combinations preserve it
        # up to roundoff) for the next iteration's one-center ddd
        off = 0
        for isp in range(nspin):
            for a, (s0, s1) in enumerate(system.atom_slices):
                nm_a = s1 - s0
                m = bec_block[off:off + nm_a * nm_a].reshape(nm_a, nm_a)
                rho_ij_mix[isp][a] = 0.5 * (m + m.conj().T)
                off += nm_a * nm_a

    rho_final = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    out = dict(
        converged=converged, n_iter=len(history), energies=energies,
        eigenvalues=eigs_s[0] if nspin == 1 else torch.stack(eigs_s),
        occupations=occ_s[0] if nspin == 1 else torch.stack(occ_s),
        coeffs=coeffs[0] if nspin == 1 else coeffs, rho=rho_final,
        rho_ij_atoms=rho_ij_s[0] if nspin == 1 else rho_ij_s,
        becps=becps_s[0] if nspin == 1 else becps_s, history=history,
        fermi=mu, system=system, nspin=nspin,
    )
    if nspin == 2:
        out["rho_spin"] = rho_s
        out["mag_total"] = float((rho_s[0] - rho_s[1]).sum()) * vol / grid.n_points
        out["mag_abs"] = float((rho_s[0] - rho_s[1]).abs().sum()) * vol / grid.n_points
    return out
