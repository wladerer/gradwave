"""USPP/PAW system construction (Layer B) — split from uspp.py (stage 3).

Builds USPPSystem: G-grids and spheres, projector data with the BARE D,
m-expanded augmentation form factors Q̃(G) on the density sphere, S weights
recomputed from the same radial integrals as the aug tables (never PP_Q —
file-precision mismatch breaks exact charge conservation), local-potential
tables, NLCC core density, symmetry machinery. The SCF driver lives in
uspp_loop.py; import both through the scf.uspp facade.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from gradwave.core.gaunt import real_gaunt_table, ylm_np
from gradwave.core.hamiltonian import build_projector_data
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_fft_grid, build_gsphere
from gradwave.pseudo.kb import beta_form_factors
from gradwave.pseudo.radial import sbt
from gradwave.pseudo.upf_paw import PAWData

_MINUS_I_POW_L = [1.0, -1.0j, -1.0, 1.0j, 1.0]  # (−i)^L, L ≤ 4


def _smooth_to_dense_map(shape_s, shape_d) -> torch.Tensor:
    """(n_smooth,) map from smooth-box flat index to dense-box flat index by
    shared Miller vector, for filtering v_eff onto the smooth grid."""
    def freqs(n):
        return np.fft.fftfreq(n, d=1.0 / n).astype(np.int64)
    m0, m1, m2 = (m.reshape(-1) for m in np.meshgrid(
        freqs(shape_s[0]), freqs(shape_s[1]), freqs(shape_s[2]), indexing="ij"))
    n0, n1, n2 = shape_d
    dense = (m0 % n0) * (n1 * n2) + (m1 % n1) * n2 + (m2 % n2)
    return torch.as_tensor(dense, dtype=torch.int64)


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
    # dual grid: the H-apply local term runs on the smaller wavefunction box
    # (2·G_max(ecut)) instead of the dense ecutrho box. None disables it.
    smooth_shape: tuple = None
    smooth_flat_idx: torch.Tensor = None  # (nk, npw_max) into the smooth box
    smooth2dense: torch.Tensor = None  # (n_smooth,) smooth→dense flat, by Miller

    def to(self, device) -> "USPPSystem":
        """Copy with every tensor moved to `device` (mirrors System.to; the
        paws' numpy radial tables and the one-center machinery stay CPU)."""

        def mv(obj, fields):
            return dataclasses.replace(
                obj, **{f: getattr(obj, f).to(device) for f in fields}
            )

        return dataclasses.replace(
            self,
            grid=mv(self.grid, ["g_cart", "g2", "dens_mask"]),
            spheres=[mv(s, ["k_cart", "miller", "kpg", "kpg2", "flat_idx"])
                     for s in self.spheres],
            proj_data=[mv(pd, ["atom_index", "f_ylm_phase_free", "kpg", "dij_full"])
                       for pd in self.proj_data],
            aug=[mv(a, ["q_g", "q_int"]) for a in self.aug],
            kweights=self.kweights.to(device),
            positions=self.positions.to(device),
            charges=self.charges.to(device),
            q_full=self.q_full.to(device),
            sphere_idx=self.sphere_idx.to(device),
            g_sphere=self.g_sphere.to(device),
            vloc_tables=self.vloc_tables.to(device),
            rho_core=self.rho_core.to(device) if self.rho_core is not None else None,
            rho_symmetrizer=(self.rho_symmetrizer.to(device)
                             if self.rho_symmetrizer is not None else None),
            becsum_sym=(self.becsum_sym.to(device)
                        if self.becsum_sym is not None else None),
            smooth_flat_idx=(self.smooth_flat_idx.to(device)
                             if self.smooth_flat_idx is not None else None),
            smooth2dense=(self.smooth2dense.to(device)
                          if self.smooth2dense is not None else None),
        )


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
    # equalize only symmetry-COUPLED axes (a slab's vacuum axis stays
    # independent of the in-plane pair); a blanket cubic box would blow the slab
    # dense grid up by the vacuum-to-in-plane ratio — matches the NC setup path.
    if sym is not None:
        from gradwave.symmetry import coupled_axis_groups
        axis_groups = coupled_axis_groups(sym)
    else:
        axis_groups = False
    # build_fft_grid derives the density sphere as 2·G_max(ecut_arg)
    grid = build_fft_grid(cell, ecutrho / 4.0, shape_override=fft_shape,
                          equal_dims=axis_groups)
    if sym is not None:
        from gradwave.symmetry import RhoSymmetrizer, reduce_mesh

        rho_symmetrizer = RhoSymmetrizer(grid.shape, sym, dens_mask=grid.dens_mask)
        kfrac, kw = reduce_mesh(kmesh, (0, 0, 0), sym, time_reversal=True)
    else:
        kfrac, kw = monkhorst_pack(kmesh, (0, 0, 0), time_reversal=True)
    spheres = [build_gsphere(grid, ecut, k) for k in kfrac]

    # dual grid: a smaller box holding only the wavefunction-product sphere
    # (2·G_max(ecut)). The Davidson H-apply local term runs there rather than
    # on the dense ecutrho box. Exact for ⟨ψ|V|ψ⟩ because two wavefunction
    # G-vectors differ by at most 2·G_max(ecut), so V truncated to this sphere
    # reproduces the matrix elements. Norm-conserving already has ecutrho =
    # 4·ecutwfc, so its box is this box and the smooth path is a no-op there.
    smooth_grid = build_fft_grid(cell, ecut, equal_dims=axis_groups)
    smooth_shape = tuple(int(n) for n in smooth_grid.shape)
    npw_max_s = max(s.miller.shape[0] for s in spheres)
    smooth_flat_idx = torch.zeros(len(spheres), npw_max_s, dtype=torch.int64)
    for ik, k in enumerate(kfrac):
        ss = build_gsphere(smooth_grid, ecut, k)  # same Miller order as dense
        smooth_flat_idx[ik, : ss.flat_idx.shape[0]] = ss.flat_idx
    smooth2dense = _smooth_to_dense_map(smooth_shape, tuple(grid.shape))

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
        smooth_shape=smooth_shape, smooth_flat_idx=smooth_flat_idx,
        smooth2dense=smooth2dense,
        becsum_sym=(None if sym is None else _make_becsum_sym(
            sym, cell, paws, species_of_atom, slices)),
    )


def _make_becsum_sym(sym, cell, paws, species_of_atom, slices):
    from gradwave.scf.paw_symmetry import BecsumSymmetrizer

    return BecsumSymmetrizer(sym, cell, paws, species_of_atom, slices)
