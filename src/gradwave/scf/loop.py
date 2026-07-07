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

from dataclasses import dataclass, field

import numpy as np
import torch

from gradwave.constants import HBAR2_2M
from gradwave.core.density import density_from_orbitals, sigma_from_rho
from gradwave.core.energies.hartree import hartree_potential_g
from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.energies.total import EnergyBreakdown, total_energy
from gradwave.core.fftbox import r_to_g
from gradwave.core.hamiltonian import HamiltonianK, becp, build_projector_data, projectors
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
from gradwave.solvers.davidson import davidson


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
) -> System:
    grid = build_fft_grid(cell, ecut)
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

    return System(
        grid=grid,
        spheres=spheres,
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
        kerker = smearing != "none"

    rho = sad_density(grid, system.positions, system.species_of_atom, system.upfs,
                      system.n_electrons)

    mask_flat = grid.dens_mask.reshape(-1)
    g2_vec = grid.g2.reshape(-1)[mask_flat]
    mixer = PulayMixer(g2_vec, alpha=mixing_alpha, history=mixing_history, kerker=kerker)

    # frozen projector matrices (positions fixed during SCF)
    projs = [projectors(pd, system.positions) for pd in system.proj_data]
    vloc_g = local_potential_g(
        system.positions, system.species_index, system.vloc_tables, grid.g_cart, vol
    )
    vloc_r = (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real

    # initial orbitals: lowest-kinetic plane waves (sphere ordering is by |k+G|²)
    coeffs = []
    for sph in spheres:
        c = torch.zeros(nb, sph.npw, dtype=CDTYPE)
        c[torch.arange(nb), torch.arange(nb)] = 1.0
        coeffs.append(c)

    e_free_prev, converged, history = None, False, []
    eigs = torch.zeros(nk, nb, dtype=RDTYPE)
    occ = torch.zeros(nk, nb, dtype=RDTYPE)
    mu, entropy_term = 0.0, torch.zeros((), dtype=RDTYPE)
    v_eff = torch.zeros(grid.shape, dtype=RDTYPE)

    for it in range(1, max_iter + 1):
        rho_g_box = r_to_g(rho.to(CDTYPE))
        v_h_r = (
            torch.fft.ifftn(hartree_potential_g(rho_g_box, grid.g2), dim=(-3, -2, -1))
            * grid.n_points
        ).real
        v_xc_r, _ = vxc_potential(xc, rho, grid)
        v_eff = v_h_r + v_xc_r + vloc_r

        for ik, sph in enumerate(spheres):
            h = HamiltonianK(sph, grid.shape, v_eff, system.proj_data[ik], projs[ik])
            t_g = HBAR2_2M * sph.kpg2
            res = davidson(h.apply, coeffs[ik], t_g, tol=diago_tol)
            eigs[ik], coeffs[ik] = res.eigenvalues, res.eigenvectors

        if smearing == "none":
            occ = fixed_occupations(eigs, system.n_electrons)
            mu = float(eigs[:, int(system.n_electrons // 2) - 1].max())
            entropy_term = torch.zeros((), dtype=RDTYPE)
        else:
            scheme = SCHEMES[smearing]
            mu = float(find_fermi(eigs, system.kweights, scheme, width, system.n_electrons))
            # NB: bare torch.tensor(mu) would be float32 and shift N_e by ~1e-7
            occ, s = occupations_and_entropy(eigs, torch.tensor(mu, dtype=RDTYPE), scheme, width)
            entropy_term = -width * (2.0 * system.kweights[:, None] * s).sum()

        rho_out = density_from_orbitals(coeffs, occ, system.kweights, spheres, grid.shape, vol)

        # energy at (orbitals, rho_out)
        becps = [becp(projs[ik], coeffs[ik]) for ik in range(nk)]
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

        if de < etol and res_norm < rhotol:
            converged = True
            rho = rho_out
            break

        e_free_prev = e_free
        mixed = mixer.step(rho_in_vec, rho_out_vec)
        rho_g_new = torch.zeros(grid.n_points, dtype=CDTYPE)
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
