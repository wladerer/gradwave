"""SCF driver (Layer B) — the torch.no_grad boundary.

setup_system() freezes everything geometry- and pseudo-dependent (grids,
spheres, form-factor tables) once; scf() iterates diagonalize → occupy →
density → mix. The returned SCFResult carries DETACHED converged tensors;
postscf/forces.py rebuilds the differentiable energy from them, and M4's
implicit.py wraps this loop in a custom autograd.Function.

Convergence: |ΔF| < etol on two consecutive iterations AND the density
residual ‖ρ_out − ρ_in‖·Ω/N_G < rhotol (electrons-scale measure).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch

from gradwave.core.density import sigma_from_rho
from gradwave.core.energies.hartree import hartree_potential_g
from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.energies.total import EnergyBreakdown, total_energy
from gradwave.core.fftbox import r_to_g
from gradwave.core.hamiltonian import build_projector_data
from gradwave.core.occupations import (
    SCHEMES,
    find_fermi,
    fixed_occupations,
    occupations_and_entropy,
)
from gradwave.core.xc.base import XCFunctional
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.grids import build_fft_grid, build_gsphere
from gradwave.kpoints import monkhorst_pack
from gradwave.pseudo.kb import beta_form_factors
from gradwave.pseudo.local import alpha_z, vloc_of_g
from gradwave.pseudo.upf import UPFData
from gradwave.scf.guess import sad_density
from gradwave.scf.mixing import PulayMixer


@dataclass
class System:
    """Frozen per-geometry setup (Layer B product)."""

    grid: object
    spheres: list
    kweights: torch.Tensor
    positions: torch.Tensor  # (na,3) Å, detached
    species_of_atom: list[int]
    upfs: list[UPFData]
    charges: torch.Tensor  # (na,) Z_val
    species_index: torch.Tensor
    vloc_tables: torch.Tensor  # (nspecies, n1,n2,n3) [eV·Å³], G=0 = alpha-Z
    proj_data: list  # per-k ProjectorData
    n_electrons: float
    nbands: int
    ecut: float = 0.0  # eV — needed to build additional G-spheres (band paths)
    batch: object = None  # core.batch.BatchedK — the padded k-batched tensors
    sym: object = None  # symmetry.SpaceGroup when IBZ reduction is active, else None
    rho_symmetrizer: object = None  # symmetry.RhoSymmetrizer (paired with sym)

    def to(self, device) -> "System":
        """Copy with every tensor moved to `device` (setup stays CPU/numpy-built)."""

        def mv(obj, fields):
            return dataclasses.replace(
                obj, **{f: getattr(obj, f).to(device) for f in fields}
            )

        grid = mv(self.grid, ["g_cart", "g2", "dens_mask"])
        spheres = [mv(s, ["k_cart", "miller", "kpg", "kpg2", "flat_idx"]) for s in self.spheres]
        proj_data = [
            mv(pd, ["atom_index", "f_ylm_phase_free", "kpg", "dij_full"])
            for pd in self.proj_data
        ]
        batch = mv(
            self.batch,
            ["npw", "mask", "flat_idx", "kpg", "t", "proj_phase_free",
             "proj_atom_index", "dij_full"],
        )
        return dataclasses.replace(
            self,
            grid=grid,
            spheres=spheres,
            proj_data=proj_data,
            batch=batch,
            kweights=self.kweights.to(device),
            positions=self.positions.to(device),
            charges=self.charges.to(device),
            species_index=self.species_index.to(device),
            vloc_tables=self.vloc_tables.to(device),
            rho_symmetrizer=(
                self.rho_symmetrizer.to(device) if self.rho_symmetrizer is not None else None
            ),
        )


def _unique_shells(vals: np.ndarray):
    uniq, inverse = np.unique(np.round(vals, 9), return_inverse=True)
    return uniq, inverse


def setup_system(
    cell: np.ndarray,
    positions: np.ndarray,  # (na,3) Cartesian Å
    species_of_atom: list[int],
    upfs: list[UPFData],
    ecut: float,
    kmesh=(1, 1, 1),
    kshift=(0, 0, 0),
    nbands: int | None = None,
    use_symmetry: bool = False,
    symprec: float = 1e-6,
    fft_shape=None,
) -> System:
    """use_symmetry: reduce k to the IBZ and symmetrize ρ each SCF step.
    Requires an unshifted (Γ-centered) mesh — shifted meshes fall back to
    time-reversal-only reduction. M4's implicit backward requires
    use_symmetry=False (a perturbation breaks the crystal symmetry).
    """
    cell = np.asarray(cell, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    sym = rho_symmetrizer = None
    if use_symmetry and tuple(kshift) == (0, 0, 0):
        from gradwave.symmetry import RhoSymmetrizer, find_spacegroup, reduce_mesh

        frac = positions @ np.linalg.inv(cell)
        sym = find_spacegroup(cell, frac, species_of_atom, symprec=symprec)
        if sym.n_ops <= 1:
            sym = None  # P1 — nothing to gain, keep the plain path

    # equalize only symmetry-COUPLED axes (a slab's z axis stays independent
    # of the in-plane pair — blanket cubic boxes would triple slab grids)
    axis_groups = False
    if sym is not None:
        coupled = np.zeros((3, 3), dtype=bool)
        for w in sym.rotations:
            coupled |= w != 0
        coupled |= coupled.T
        groups, seen = [], set()
        for i in range(3):
            if i in seen:
                continue
            group = {i}
            frontier = {i}
            while frontier:
                j = frontier.pop()
                for k2 in range(3):
                    if coupled[j, k2] and k2 not in group:
                        group.add(k2)
                        frontier.add(k2)
            seen |= group
            groups.append(tuple(sorted(group)))
        axis_groups = groups
    grid = build_fft_grid(cell, ecut, equal_dims=axis_groups, shape_override=fft_shape)
    if sym is not None:
        rho_symmetrizer = RhoSymmetrizer(grid.shape, sym, dens_mask=grid.dens_mask)
        kfrac, kw = reduce_mesh(kmesh, kshift, sym)
    else:
        kfrac, kw = monkhorst_pack(kmesh, kshift)
    spheres = [build_gsphere(grid, ecut, k) for k in kfrac]

    charges = torch.tensor([upfs[s].z_valence for s in species_of_atom], dtype=RDTYPE)
    n_electrons = float(charges.sum())
    if nbands is None:
        nocc = int(np.ceil(n_electrons / 2.0))
        nbands = max(int(np.ceil(nocc * 1.2)), nocc + 4)

    # local potential tables on the dense box, per species
    g_flat = np.sqrt(grid.g2.reshape(-1).numpy())
    uniq, inverse = _unique_shells(g_flat)
    vloc_tables = []
    for upf in upfs:
        tab = np.empty_like(uniq)
        tab[0] = alpha_z(upf)
        if len(uniq) > 1:
            tab[1:] = vloc_of_g(upf, uniq[1:])
        vloc_tables.append(tab[inverse].reshape(grid.shape))
    vloc_tables = torch.as_tensor(np.stack(vloc_tables), dtype=RDTYPE)

    # per-k projector data
    proj_data = []
    for sph in spheres:
        q = np.sqrt(sph.kpg2.numpy())
        beta_tables = [
            torch.as_tensor(beta_form_factors(upf, q), dtype=RDTYPE) for upf in upfs
        ]
        beta_ls = [[b.l for b in upf.betas] for upf in upfs]
        dij_species = [torch.as_tensor(upf.dij, dtype=RDTYPE) for upf in upfs]
        proj_data.append(
            build_projector_data(
                sph, species_of_atom, beta_tables, beta_ls, dij_species, grid.volume
            )
        )

    from gradwave.core.batch import build_batched

    return System(
        grid=grid,
        spheres=spheres,
        batch=build_batched(spheres, proj_data),
        kweights=torch.as_tensor(kw, dtype=RDTYPE),
        positions=torch.as_tensor(np.asarray(positions), dtype=RDTYPE),
        species_of_atom=list(species_of_atom),
        upfs=list(upfs),
        charges=charges,
        species_index=torch.tensor(species_of_atom, dtype=torch.int64),
        vloc_tables=vloc_tables,
        proj_data=proj_data,
        n_electrons=n_electrons,
        nbands=nbands,
        ecut=ecut,
        sym=sym,
        rho_symmetrizer=rho_symmetrizer,
    )


def vxc_potential(xc: XCFunctional, rho: torch.Tensor, grid) -> tuple[torch.Tensor, torch.Tensor]:
    """(v_xc(r) [eV], E_xc [eV]) via autograd — GGA divergence term included."""
    rho_leaf = rho.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        sigma = sigma_from_rho(rho_leaf, grid.g_cart) if xc.needs_gradient else None
        e_xc = xc.energy(rho_leaf, grid.volume, sigma)
        (v,) = torch.autograd.grad(e_xc, rho_leaf)
    return v * (grid.n_points / grid.volume), e_xc.detach()


@dataclass
class SCFResult:
    converged: bool
    n_iter: int
    energies: EnergyBreakdown
    fermi: float
    eigenvalues: torch.Tensor  # (nk, nb) [eV]; (nspin, nk, nb) when nspin=2
    occupations: torch.Tensor  # (nk, nb) in [0,2]; (nspin, nk, nb) in [0,1] for spin
    coeffs: list  # [(nb, npw_k)] per k; list-of-lists [spin][k] when nspin=2
    rho: torch.Tensor  # TOTAL density (n1,n2,n3) [e/Å³]
    v_eff: torch.Tensor  # (n1,n2,n3) [eV]; (nspin,n1,n2,n3) when nspin=2
    system: System
    history: list = field(default_factory=list)
    nspin: int = 1
    rho_spin: list | None = None  # [ρ↑, ρ↓] when nspin=2
    mag_total: float = 0.0  # ∫(ρ↑−ρ↓) dr [μB]
    mag_abs: float = 0.0  # ∫|ρ↑−ρ↓| dr [μB]


def vxc_spin_potential(xc, rho_up, rho_dn, grid):
    """(v↑, v↓, E_xc) via autograd on a SpinXC — GGA terms included."""
    ru = rho_up.detach().clone().requires_grad_(True)
    rd = rho_dn.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        if xc.needs_gradient:
            s_uu = sigma_from_rho(ru, grid.g_cart)
            s_dd = sigma_from_rho(rd, grid.g_cart)
            s_tot = sigma_from_rho(ru + rd, grid.g_cart)
        else:
            s_uu = s_dd = s_tot = None
        e_xc = xc.energy(ru, rd, grid.volume, s_uu, s_dd, s_tot)
        vu, vd = torch.autograd.grad(e_xc, (ru, rd))
    scale = grid.n_points / grid.volume
    return vu * scale, vd * scale, e_xc.detach()


@torch.no_grad()
def scf(
    system: System,
    xc: XCFunctional,
    smearing: str = "none",
    width: float = 0.1,
    max_iter: int = 100,
    etol: float = 1e-8,
    rhotol: float = 1e-7,
    mixing_alpha: float = 0.7,
    mixing_history: int = 8,
    kerker: bool | None = None,
    diago_tol: float = 1e-9,
    verbose: bool = True,
    nspin: int = 1,
    start_mag=None,  # initial moment fractions: per-species OR per-atom (nspin=2)
) -> SCFResult:
    grid, spheres = system.grid, system.spheres
    vol = grid.volume
    nk, nb = len(spheres), system.nbands
    if nspin not in (1, 2):
        raise ValueError("nspin must be 1 or 2 (noncollinear is future work)")
    if nspin == 2 and smearing == "none":
        raise ValueError("nspin=2 requires smearing (fixed magnetic occupations ambiguous)")
    g_spin = 2.0 / nspin
    if kerker is None:
        # auto policy: metals always; insulators once the cell is large
        # enough that long-wavelength charge sloshing dominates mixing —
        # the sloshing amplification goes like 4πe²χ/G²_min, so switch on
        # when the smallest nonzero |G| drops below ~0.8 Å⁻¹ (L ≳ 8 Å).
        g2_nonzero = grid.g2.reshape(-1)
        g2_min = float(g2_nonzero[g2_nonzero > 1e-12].min())
        kerker = (smearing != "none") or (g2_min < 0.64)

    if nspin == 1:
        rho_s = [sad_density(grid, system.positions, system.species_of_atom,
                             system.upfs, system.n_electrons)]
    else:
        na = len(system.species_of_atom)
        nspecies = len(system.upfs)
        if start_mag is None:
            mags_at = [0.5] * na
        elif len(start_mag) == na:
            mags_at = [float(m) for m in start_mag]
        elif len(start_mag) == nspecies:
            mags_at = [float(start_mag[sp]) for sp in system.species_of_atom]
        else:
            raise ValueError("start_mag must have one entry per atom or per species")
        mags_by_sp = {}
        for a, sp in enumerate(system.species_of_atom):
            mags_by_sp.setdefault(sp, set()).add(round(mags_at[a], 12))
        uniform_per_species = all(len(v) == 1 for v in mags_by_sp.values())
        if system.rho_symmetrizer is not None and not uniform_per_species:
            raise ValueError(
                "non-uniform per-atom moments break the chemical space group "
                "(magnetic group is smaller) — build the system with "
                "use_symmetry=False for AFM/ferrimagnetic configurations"
            )
        n_up = sum(float(system.charges[a]) * (1 + mags_at[a]) / 2 for a in range(na))
        n_dn = system.n_electrons - n_up
        rho_s = [
            sad_density(grid, system.positions, system.species_of_atom, system.upfs,
                        n_up, atom_scale=[(1 + m) / 2 for m in mags_at]),
            sad_density(grid, system.positions, system.species_of_atom, system.upfs,
                        n_dn, atom_scale=[(1 - m) / 2 for m in mags_at]),
        ]

    mask_flat = grid.dens_mask.reshape(-1)
    g2_vec = grid.g2.reshape(-1)[mask_flat]
    # spin mixing runs in the (total, magnetization) basis: Kerker damps
    # long-wavelength charge sloshing on the TOTAL block only — applying it
    # to both channels would freeze the per-channel electron counts (the
    # G=0 Kerker zero forbids charge transfer between spin channels)
    g2_mix = torch.cat([g2_vec] * nspin)
    kerker_mask = None
    if nspin == 2:
        kerker_mask = torch.cat([torch.ones_like(g2_vec, dtype=torch.bool),
                                 torch.zeros_like(g2_vec, dtype=torch.bool)])
    mixer = PulayMixer(g2_mix, alpha=mixing_alpha, history=mixing_history,
                       kerker=kerker, check_g0=nspin == 1, kerker_mask=kerker_mask)

    from gradwave.core.batch import BatchedHamiltonian, becp_b, density_b, projectors_b
    from gradwave.solvers.davidson import davidson_batched

    device = system.positions.device
    bk = system.batch

    # frozen projector matrices (positions fixed during SCF)
    projs_b = projectors_b(bk, system.positions)
    vloc_g = local_potential_g(
        system.positions, system.species_index, system.vloc_tables, grid.g_cart, vol
    )
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real

    # initial orbitals: lowest-kinetic plane waves (sphere ordering is by |k+G|²)
    c0 = torch.zeros(nk, nb, bk.npw_max, dtype=CDTYPE, device=device)
    c0[:, torch.arange(nb), torch.arange(nb)] = 1.0
    coeffs_b_s = [c0.clone() for _ in range(nspin)]

    e_free_prev, converged, history = None, False, []
    eigs_s = [torch.zeros(nk, nb, dtype=RDTYPE, device=device) for _ in range(nspin)]
    occ_s = [torch.zeros(nk, nb, dtype=RDTYPE, device=device) for _ in range(nspin)]
    mu, entropy_term = 0.0, torch.zeros((), dtype=RDTYPE, device=device)
    veff_s = [torch.zeros(grid.shape, dtype=RDTYPE, device=device) for _ in range(nspin)]

    def symmetrize(r_out):
        if system.rho_symmetrizer is None:
            return r_out
        sym_g = system.rho_symmetrizer.apply(r_to_g(r_out.to(CDTYPE)))
        return (torch.fft.ifftn(sym_g * grid.n_points, dim=(-3, -2, -1))).real

    for it in range(1, max_iter + 1):
        rho_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
        rho_g_box = r_to_g(rho_tot.to(CDTYPE))
        v_h_r = (
            torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2), dim=(-3, -2, -1))
            * grid.n_points
        ).real
        if nspin == 1:
            v_xc_r, _ = vxc_potential(xc, rho_tot, grid)
            veff_s = [v_h_r + v_xc_r + vloc_r]
        else:
            v_up, v_dn, _ = vxc_spin_potential(xc, rho_s[0], rho_s[1], grid)
            veff_s = [v_h_r + v_up + vloc_r, v_h_r + v_dn + vloc_r]

        # adaptive diagonalization tolerance (QE-style): loose while the
        # density is far from self-consistent, tightening with the residual
        if it == 1:
            tol_eff = max(diago_tol, 1e-3)
        else:
            tol_eff = max(diago_tol, min(1e-3, 0.03 * history[-1]["res"]))
        for sp in range(nspin):
            h = BatchedHamiltonian(bk, grid.shape, veff_s[sp], projs_b)
            dav = davidson_batched(h.apply, coeffs_b_s[sp], bk.t, bk.mask, tol=tol_eff)
            eigs_s[sp], coeffs_b_s[sp] = dav.eigenvalues, dav.eigenvectors

        if smearing == "none":
            occ_s = [fixed_occupations(eigs_s[0], system.n_electrons)]
            mu = float(eigs_s[0][:, int(system.n_electrons // 2) - 1].max())
            entropy_term = torch.zeros((), dtype=RDTYPE, device=device)
        else:
            scheme = SCHEMES[smearing]
            eigs_cat = torch.cat(eigs_s, dim=0)  # (nspin·nk, nb)
            kw_cat = torch.cat([system.kweights] * nspin)
            mu = float(find_fermi(eigs_cat, kw_cat, scheme, width,
                                  system.n_electrons, degeneracy=g_spin))
            # NB: bare torch.tensor(mu) would be float32 and shift N_e by ~1e-7
            mu_t = torch.tensor(mu, dtype=RDTYPE, device=device)
            occ_s, ent = [], torch.zeros((), dtype=RDTYPE, device=device)
            for sp in range(nspin):
                o, s_ent = occupations_and_entropy(
                    eigs_s[sp], mu_t, scheme, width, degeneracy=g_spin)
                occ_s.append(o)
                ent = ent - width * (g_spin * system.kweights[:, None] * s_ent).sum()
            entropy_term = ent

        rho_out_s = [
            symmetrize(density_b(coeffs_b_s[sp], occ_s[sp], system.kweights,
                                 bk, grid.shape, vol))
            for sp in range(nspin)
        ]
        rho_tot_out = rho_out_s[0] if nspin == 1 else rho_out_s[0] + rho_out_s[1]

        # energy at (orbitals, rho_out); per-k trimmed views for the assembly
        coeffs_list_s = [
            [coeffs_b_s[sp][ik, :, : int(bk.npw[ik])] for ik in range(nk)]
            for sp in range(nspin)
        ]
        becps_s = [
            [becp_b(projs_b, coeffs_b_s[sp])[ik] for ik in range(nk)]
            for sp in range(nspin)
        ]
        if nspin == 1:
            energies = total_energy(
                coeffs_per_k=coeffs_list_s[0], occ=occ_s[0], kweights=system.kweights,
                spheres=spheres, grid=grid, rho=rho_tot_out, positions=system.positions,
                charges=system.charges, species_index=system.species_index,
                vloc_tables=system.vloc_tables, becp_per_k=becps_s[0],
                dij_full=_stack_dij(system), xc=xc, entropy_term=entropy_term,
            )
        else:
            from gradwave.core.energies.ewald import ewald_energy
            from gradwave.core.energies.hartree import hartree_energy
            from gradwave.core.energies.kinetic import kinetic_energy
            from gradwave.core.energies.local_pp import local_energy
            from gradwave.core.energies.nl_pp import nonlocal_energy

            rho_g_out = r_to_g(rho_tot_out.to(CDTYPE))
            e_kin = sum(kinetic_energy(coeffs_list_s[sp], occ_s[sp],
                                       system.kweights, spheres)
                        for sp in range(nspin))
            e_h = hartree_energy(rho_g_out, grid.g2, vol)
            if xc.needs_gradient:
                s_uu = sigma_from_rho(rho_out_s[0], grid.g_cart)
                s_dd = sigma_from_rho(rho_out_s[1], grid.g_cart)
                s_tt = sigma_from_rho(rho_tot_out, grid.g_cart)
            else:
                s_uu = s_dd = s_tt = None
            e_xc = xc.energy(rho_out_s[0], rho_out_s[1], vol, s_uu, s_dd, s_tt)
            e_loc = local_energy(rho_g_out, vloc_g, vol)
            e_nl = sum(nonlocal_energy(becps_s[sp], _stack_dij(system),
                                       occ_s[sp], system.kweights)
                       for sp in range(nspin))
            e_ew = ewald_energy(system.positions, system.charges, grid.cell)
            energies = EnergyBreakdown(
                kinetic=e_kin, hartree=e_h, xc=e_xc, local=e_loc,
                nonlocal_=e_nl, ewald=e_ew, smearing=entropy_term,
            )
        e_free = float(energies.free_energy)

        def to_mix_basis(chans):
            vecs = [r_to_g(c.to(CDTYPE)).reshape(-1)[mask_flat] for c in chans]
            if nspin == 1:
                return vecs[0]
            return torch.cat([vecs[0] + vecs[1], vecs[0] - vecs[1]])  # (total, mag)

        rho_in_vec = to_mix_basis(rho_s)
        rho_out_vec = to_mix_basis(rho_out_s)
        if nspin == 2:  # only the TOTAL is conserved; its G=0 residual must vanish
            tot_res = rho_out_vec[0] - rho_in_vec[0]
            assert torch.isfinite(rho_out_vec).all(), "density diverged (NaN/Inf)"
            assert tot_res.abs() < 1e-8, "total G=0 residual nonzero"
        res_norm = float(torch.linalg.norm(rho_out_vec - rho_in_vec)) * vol
        de = abs(e_free - e_free_prev) if e_free_prev is not None else float("inf")
        history.append({"iter": it, "free_energy": e_free, "dE": de, "res": res_norm})
        if verbose:
            mag = ""
            if nspin == 2:
                m = float((rho_out_s[0] - rho_out_s[1]).mean()) * vol
                mag = f"   m = {m:+.4f} muB"
            print(f"  SCF {it:3d}  F = {e_free:+.10f} eV   dE = {de:.3e}   "
                  f"|drho| = {res_norm:.3e}{mag}")

        if de < etol and res_norm < rhotol and tol_eff <= diago_tol * 1.01:
            converged = True
            rho_s = rho_out_s
            break

        e_free_prev = e_free
        mixed = mixer.step(rho_in_vec, rho_out_vec)
        ng = mixed.shape[0] // nspin
        if nspin == 1:
            chan_vecs = [mixed]
        else:  # back from (total, mag) to channels
            tot, mag = mixed[:ng], mixed[ng:]
            chan_vecs = [(tot + mag) / 2.0, (tot - mag) / 2.0]
        new_rho_s = []
        for vec in chan_vecs:
            rho_g_new = torch.zeros(grid.n_points, dtype=CDTYPE, device=device)
            rho_g_new[mask_flat] = vec
            new_rho_s.append(
                (torch.fft.ifftn(rho_g_new.reshape(grid.shape) * grid.n_points,
                                 dim=(-3, -2, -1))).real
            )
        rho_s = new_rho_s

    rho_tot_final = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    if nspin == 1:
        return SCFResult(
            converged=converged, n_iter=it, energies=energies, fermi=mu,
            eigenvalues=eigs_s[0], occupations=occ_s[0], coeffs=coeffs_list_s[0],
            rho=rho_tot_final, v_eff=veff_s[0], system=system, history=history,
        )
    m_density = rho_s[0] - rho_s[1]
    return SCFResult(
        converged=converged, n_iter=it, energies=energies, fermi=mu,
        eigenvalues=torch.stack(eigs_s), occupations=torch.stack(occ_s),
        coeffs=coeffs_list_s, rho=rho_tot_final, v_eff=torch.stack(veff_s),
        system=system, history=history, nspin=2, rho_spin=rho_s,
        mag_total=float(m_density.mean()) * vol,
        mag_abs=float(m_density.abs().mean()) * vol,
    )


def _stack_dij(system: System) -> torch.Tensor:
    """Block-diagonal dij over all atoms (identical across k — take from k=0)."""
    return system.proj_data[0].dij_full
