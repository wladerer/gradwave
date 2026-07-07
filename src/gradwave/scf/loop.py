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

    grid = build_fft_grid(cell, ecut, equal_dims=sym is not None, shape_override=fft_shape)
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
    eigenvalues: torch.Tensor  # (nk, nb) [eV]
    occupations: torch.Tensor  # (nk, nb) in [0,2]
    coeffs: list  # [(nb, npw_k) complex, detached]
    rho: torch.Tensor  # (n1,n2,n3) [e/Å³]
    v_eff: torch.Tensor  # (n1,n2,n3) [eV] — for band-structure solves
    system: System
    history: list = field(default_factory=list)


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
) -> SCFResult:
    grid, spheres = system.grid, system.spheres
    vol = grid.volume
    nk, nb = len(spheres), system.nbands
    if kerker is None:
        # auto policy: metals always; insulators once the cell is large
        # enough that long-wavelength charge sloshing dominates mixing —
        # the sloshing amplification goes like 4πe²χ/G²_min, so switch on
        # when the smallest nonzero |G| drops below ~0.8 Å⁻¹ (L ≳ 8 Å).
        g2_nonzero = grid.g2.reshape(-1)
        g2_min = float(g2_nonzero[g2_nonzero > 1e-12].min())
        kerker = (smearing != "none") or (g2_min < 0.64)

    rho = sad_density(grid, system.positions, system.species_of_atom, system.upfs,
                      system.n_electrons)

    mask_flat = grid.dens_mask.reshape(-1)
    g2_vec = grid.g2.reshape(-1)[mask_flat]
    mixer = PulayMixer(g2_vec, alpha=mixing_alpha, history=mixing_history, kerker=kerker)

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
    coeffs_b = torch.zeros(nk, nb, bk.npw_max, dtype=CDTYPE, device=device)
    coeffs_b[:, torch.arange(nb), torch.arange(nb)] = 1.0

    e_free_prev, converged, history = None, False, []
    eigs = torch.zeros(nk, nb, dtype=RDTYPE, device=device)
    occ = torch.zeros(nk, nb, dtype=RDTYPE, device=device)
    mu, entropy_term = 0.0, torch.zeros((), dtype=RDTYPE, device=device)
    v_eff = torch.zeros(grid.shape, dtype=RDTYPE, device=device)

    for it in range(1, max_iter + 1):
        rho_g_box = r_to_g(rho.to(CDTYPE))
        v_h_r = (
            torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2), dim=(-3, -2, -1))
            * grid.n_points
        ).real
        v_xc_r, _ = vxc_potential(xc, rho, grid)
        v_eff = v_h_r + v_xc_r + vloc_r

        # adaptive diagonalization tolerance (QE-style): loose while the
        # density is far from self-consistent, tightening with the residual
        if it == 1:
            tol_eff = max(diago_tol, 1e-3)
        else:
            tol_eff = max(diago_tol, min(1e-3, 0.03 * history[-1]["res"]))
        h = BatchedHamiltonian(bk, grid.shape, v_eff, projs_b)
        dav = davidson_batched(h.apply, coeffs_b, bk.t, bk.mask, tol=tol_eff)
        eigs, coeffs_b = dav.eigenvalues, dav.eigenvectors

        if smearing == "none":
            occ = fixed_occupations(eigs, system.n_electrons)
            mu = float(eigs[:, int(system.n_electrons // 2) - 1].max())
            entropy_term = torch.zeros((), dtype=RDTYPE, device=device)
        else:
            scheme = SCHEMES[smearing]
            mu = float(find_fermi(eigs, system.kweights, scheme, width, system.n_electrons))
            # NB: bare torch.tensor(mu) would be float32 and shift N_e by ~1e-7
            occ, s = occupations_and_entropy(
                eigs, torch.tensor(mu, dtype=RDTYPE, device=device), scheme, width
            )
            entropy_term = -width * (2.0 * system.kweights[:, None] * s).sum()

        rho_out = density_b(coeffs_b, occ, system.kweights, bk, grid.shape, vol)
        if system.rho_symmetrizer is not None:
            # IBZ-weighted density is not symmetric; project back onto the
            # invariant subspace (exact — see symmetry.py conventions)
            rho_sym_g = system.rho_symmetrizer.apply(r_to_g(rho_out.to(CDTYPE)))
            rho_out = (
                torch.fft.ifftn(rho_sym_g * grid.n_points, dim=(-3, -2, -1))
            ).real

        # energy at (orbitals, rho_out); per-k trimmed views for the energy assembly
        coeffs = [coeffs_b[ik, :, : int(bk.npw[ik])] for ik in range(nk)]
        bb = becp_b(projs_b, coeffs_b)
        becps = [bb[ik] for ik in range(nk)]
        energies = total_energy(
            coeffs_per_k=coeffs, occ=occ, kweights=system.kweights, spheres=spheres,
            grid=grid, rho=rho_out, positions=system.positions, charges=system.charges,
            species_index=system.species_index, vloc_tables=system.vloc_tables,
            becp_per_k=becps, dij_full=_stack_dij(system), xc=xc,
            entropy_term=entropy_term,
        )
        e_free = float(energies.free_energy)

        rho_in_vec = r_to_g(rho.to(CDTYPE)).reshape(-1)[mask_flat]
        rho_out_vec = r_to_g(rho_out.to(CDTYPE)).reshape(-1)[mask_flat]
        res_norm = float(torch.linalg.norm(rho_out_vec - rho_in_vec)) * vol
        de = abs(e_free - e_free_prev) if e_free_prev is not None else float("inf")
        history.append({"iter": it, "free_energy": e_free, "dE": de, "res": res_norm})
        if verbose:
            print(f"  SCF {it:3d}  F = {e_free:+.10f} eV   dE = {de:.3e}   |drho| = {res_norm:.3e}")

        if de < etol and res_norm < rhotol and tol_eff <= diago_tol * 1.01:
            converged = True
            rho = rho_out
            break

        e_free_prev = e_free
        mixed = mixer.step(rho_in_vec, rho_out_vec)
        rho_g_new = torch.zeros(grid.n_points, dtype=CDTYPE, device=device)
        rho_g_new[mask_flat] = mixed
        rho = (
            torch.fft.ifftn(rho_g_new.reshape(grid.shape) * grid.n_points, dim=(-3, -2, -1))
        ).real

    return SCFResult(
        converged=converged, n_iter=it, energies=energies, fermi=mu,
        eigenvalues=eigs, occupations=occ, coeffs=coeffs, rho=rho, v_eff=v_eff,
        system=system, history=history,
    )


def _stack_dij(system: System) -> torch.Tensor:
    """Block-diagonal dij over all atoms (identical across k — take from k=0)."""
    return system.proj_data[0].dij_full
