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
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from gradwave.constants import RY_EV
from gradwave.core import opcount
from gradwave.core.batch import BatchedK
from gradwave.core.density import sigma_from_rho
from gradwave.core.energies.esm import esm_energy, esm_mode_of, esm_potential
from gradwave.core.energies.ewald import ewald_energy
from gradwave.core.energies.hartree import hartree_potential_r
from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.energies.total import EnergyBreakdown, total_energy
from gradwave.core.fftbox import g_to_r_box, r_to_g
from gradwave.core.hamiltonian import ProjectorData, build_projector_data
from gradwave.core.hubbard import HubbardData, HubbardManifold
from gradwave.core.xc.base import XCFunctional
from gradwave.core.xc.spin import SpinXC
from gradwave.dtypes import CDTYPE, CDTYPE_LOW, RDTYPE, RDTYPE_LOW
from gradwave.grids import FFTGrid, GSphere, build_fft_grid, build_gsphere
from gradwave.pseudo.kb import beta_form_factors
from gradwave.pseudo.upf import UPFData
from gradwave.scf.common import (
    MP_CROSSOVER,
    SCHEMES,
    adaptive_diago_tol,
    assemble_pw_energies,
    constant_mu_occupations,
    convergence_gate,
    hubbard_u_ramp_scale,
    mix_hubbard_occ,
    record_iteration,
    shared_fermi_occupations,
    spin_sigmas,
    spin_xc_energy,
    symmetrize_rho,
    validate_hubbard_conv,
    warm_start_densities,
)
from gradwave.scf.guess import sad_density
from gradwave.scf.layout import MixLayout
from gradwave.scf.learned_precond import MultipoleKerkerPrecond
from gradwave.scf.local_tf import LocalTFPrecond
from gradwave.scf.mixing import BroydenMixer, JohnsonMixer, PulayMixer
from gradwave.scf.setup_common import (
    _unique_shells,
    build_core_density,
    build_symmetrizer_and_kpoints,
    build_vloc_tables,
    coupled_axes,
    default_nbands,
    find_symmetry_groups,
)
from gradwave.scf.uspp_setup import USPPSystem

if TYPE_CHECKING:
    # postscf.hybrid imports scf() back from this module — a real (not merely
    # stylistic) circular import at runtime, so this type is annotation-only.
    # Same annotation-only story as MultiKFockExchange above: scf() takes an
    # optional distributed context, but gradwave.distributed itself never
    # imports this module at runtime (it only names System/DistKContext in
    # type comments), so importing it eagerly here would be a needless
    # top-level dependency on torch.distributed for every non-distributed run.
    from gradwave.distributed import DistKContext
    from gradwave.postscf.hybrid import MultiKFockExchange

    # Annotation-only cross-module types for System's fields below. No import
    # cycle here (symmetry.py and scf/alchemical.py's own top-level imports
    # never reach back into this module), but every concrete use of these
    # classes at runtime in this file already goes through its own
    # function-local import (e.g. CollinearMagneticSymmetrizer at its
    # isinstance-check call sites below), so this stays annotation-only too.
    from gradwave.scf.alchemical import AlchemicalSpec, SubstitutionSpec
    from gradwave.scf.recorder import SCFRecorder
    from gradwave.symmetry import (
        CollinearMagneticSymmetrizer,
        MagneticSymmetrizer,
        RhoSymmetrizer,
        SpaceGroup,
    )

logger = logging.getLogger(__name__)


@dataclass
class System:
    """Frozen per-geometry setup (Layer B product)."""

    grid: FFTGrid
    spheres: list[GSphere]
    kweights: torch.Tensor
    positions: torch.Tensor  # (na,3) Å, detached
    species_of_atom: list[int]
    upfs: list[UPFData]
    charges: torch.Tensor  # (na,) Z_val
    species_index: torch.Tensor
    vloc_tables: torch.Tensor  # (nspecies, n1,n2,n3) [eV·Å³], G=0 = alpha-Z
    proj_data: list[ProjectorData]  # per-k ProjectorData
    n_electrons: float
    nbands: int
    ecut: float = 0.0  # eV — needed to build additional G-spheres (band paths)
    batch: BatchedK | None = None  # the padded k-batched tensors
    sym: SpaceGroup | None = None  # set when IBZ reduction is active, else None
    rho_symmetrizer: RhoSymmetrizer | CollinearMagneticSymmetrizer | MagneticSymmetrizer | None = (
        None  # paired with sym; the magnetic variants only when magmoms is set
    )
    so_beta_tables: list[torch.Tensor] | None = None  # FR pseudos: per-species (nk, nchan, npw_max)
    is_fr: bool = False  # fully-relativistic pseudos (spinor SCF only)
    rho_core: torch.Tensor | None = None  # NLCC core density on the grid [e/Å³]
    vloc_atom: torch.Tensor | None = None  # (na,n1,n2,n3) per-atom local table;
    # the alchemical composition channel sets a lambda-blended table here so
    # V_loc stays differentiable in composition (scf/alchemical.py)
    alchemical: AlchemicalSpec | SubstitutionSpec | None = None  # endpoint spec
    # for the composition gradient: AlchemicalSpec (whole-cell A→B,
    # setup_alchemical_system / alchemical_energy_gradient) or SubstitutionSpec
    # (heterogeneous, setup_alchemical_substitution / alchemical_gap_gradient)

    def to(self, device: str) -> System:
        """Copy with every tensor moved to `device` (setup stays CPU/numpy-built)."""

        def mv(obj, fields):
            return dataclasses.replace(obj, **{f: getattr(obj, f).to(device) for f in fields})

        grid = mv(self.grid, ["g_cart", "g2", "dens_mask"])
        spheres = [mv(s, ["k_cart", "miller", "kpg", "kpg2", "flat_idx"]) for s in self.spheres]
        proj_data = [
            mv(pd, ["atom_index", "f_ylm_phase_free", "kpg", "dij_full"]) for pd in self.proj_data
        ]
        batch = mv(
            self.batch,
            [
                "npw",
                "mask",
                "flat_idx",
                "kpg",
                "t",
                "proj_phase_free",
                "proj_atom_index",
                "dij_full",
            ],
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
            so_beta_tables=(
                [t.to(device) for t in self.so_beta_tables]
                if self.so_beta_tables is not None
                else None
            ),
            rho_core=self.rho_core.to(device) if self.rho_core is not None else None,
            vloc_atom=self.vloc_atom.to(device) if self.vloc_atom is not None else None,
            alchemical=self.alchemical,  # endpoint spec stays on the build device
        )


def setup_system(
    cell: np.ndarray,
    positions: np.ndarray,  # (na,3) Cartesian Å
    species_of_atom: list[int],
    upfs: list[UPFData],
    ecut: float,
    # tuple[int, ...] (not the tighter tuple[int, int, int]) because callers
    # often build these via a length-3 generator expression (e.g.
    # api.py's `tuple(... for i in range(3))`), which a type checker can't
    # statically narrow to a fixed-length tuple even though it always is one.
    kmesh: tuple[int, ...] = (1, 1, 1),
    kshift: tuple[int, ...] = (0, 0, 0),
    nbands: int | None = None,
    use_symmetry: bool = False,
    symprec: float = 1e-6,
    fft_shape: list[int] | tuple[int, ...] | None = None,
    time_reversal: bool = True,  # False for noncollinear/SOC (TR flips m)
    magmoms: np.ndarray | None = None,  # (na, 3) moment directions → magnetic (Shubnikov) symmetry
    collinear_magnetic: bool = False,  # collinear nspin=2 FM/AFM Shubnikov fold
    tot_charge: float = 0.0,  # net cell charge q [e]: n_electrons = ΣZ − q, with
    # a compensating uniform jellium background (see io/... charged-cell handling)
) -> System:
    """use_symmetry: reduce k to the IBZ and symmetrize ρ each SCF step.
    Requires an unshifted (Γ-centered) mesh — shifted meshes fall back to
    time-reversal-only reduction. M4's implicit backward requires
    use_symmetry=False (a perturbation breaks the crystal symmetry).

    magmoms (with use_symmetry=True) switches to the MAGNETIC space group of
    that moment configuration: k folds into the magnetic IBZ (unitary ops as
    W⁻ᵀ, anti-unitary g·T ops as −W⁻ᵀ). For the collinear (ρ↑, ρ↓) path the
    plain per-channel k → −k time reversal is added on top (H_σ real ⇒ no spin
    swap), reclaiming the −k fold that the magnetic group omits when its
    sublattice-swap op is inversion·T (corundum R-3̄c) rather than a translation
    (bcc Cr); the spinor path leaves it out (TR flips m⃗ there). Directions are
    what matter — magnitudes only distinguish zero from nonzero and same from
    different.

    Two magnetic representations share that k-fold:
    - the spinor (ρ, m⃗) symmetrizer (default) — only scf_noncollinear consumes it;
    - the collinear (ρ↑, ρ↓) symmetrizer (collinear_magnetic=True) — the nspin=2
      FM/AFM path, consumed by scf. This reclaims the magnetic-group k-reduction
      for collinear antiferromagnets, which otherwise forfeit it and drop to a
      near-full time-reversal-only mesh. magmoms then carry the collinear
      sublattice pattern (e.g. [[0,0,+1],[0,0,-1]] for a 2-atom AFM); they must
      be collinear (all ∥ one axis) or the constructor rejects them.
    """
    cell = np.asarray(cell, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    sym = mag_sym = None
    if use_symmetry and tuple(kshift) == (0, 0, 0):
        sym, mag_sym = find_symmetry_groups(cell, positions, species_of_atom, symprec, magmoms)

    # equalize only symmetry-COUPLED axes (setup_common.coupled_axes)
    grid = build_fft_grid(
        cell, ecut, equal_dims=coupled_axes(sym, mag_sym), shape_override=fft_shape
    )
    # time_reversal=False for magnetic systems (k≢−k); for nonmagnetic runs
    # (incl. nonmagnetic + SOC, where Kramers keeps k≡−k) it stays True
    rho_symmetrizer, kfrac, kw = build_symmetrizer_and_kpoints(
        grid,
        cell,
        kmesh,
        kshift,
        sym,
        mag_sym,
        time_reversal,
        collinear_magnetic=collinear_magnetic,
        magmoms=magmoms,
    )
    spheres = [build_gsphere(grid, ecut, k) for k in kfrac]

    charges = torch.tensor([upfs[s].z_valence for s in species_of_atom], dtype=RDTYPE)
    # net cell charge q = tot_charge: remove q electrons (q>0 → cation/electron-
    # deficient). The G=0 electrostatics carry an implicit uniform −q jellium
    # background; the finite net-charge monopole self-energy is a post-SCF
    # correction (charged-cell / Makov-Payne), not part of the SCF total.
    n_electrons = float(charges.sum()) - float(tot_charge)
    if n_electrons <= 0:
        raise ValueError(f"tot_charge={tot_charge} leaves n_electrons={n_electrons} ≤ 0")
    if nbands is None:
        nbands = default_nbands(max(float(charges.sum()), n_electrons))

    # local potential tables on the dense box, per species (the NC setup
    # keeps the single-|G|-shell guard — see setup_common.build_vloc_tables)
    g_flat = np.sqrt(grid.g2.reshape(-1).numpy())
    uniq, inverse = _unique_shells(g_flat)
    vloc_tables = build_vloc_tables(upfs, uniq, inverse, grid.shape, guard_single_shell=True)

    # per-k projector data (scalar path); FR pseudos store raw F tables for
    # the spinor projector builder instead (scalar m-expansion is invalid).
    #
    # The projector radial transform F_i(|k+G|) depends only on the MAGNITUDE
    # |k+G|, and neighbouring k-points share most of their |k+G| shells. Pool
    # every |k+G| across the whole mesh, dedupe, and run each species' SBT once
    # on the unique shells; per-k tables are then a gather. This collapses the
    # nk independent radial-transform sweeps that dominated large-cell setup.
    is_fr = any(b.j is not None for u in upfs for b in u.betas)
    npw_list = [sph.npw for sph in spheres]
    npw_max = max(npw_list)
    q_per_k = [np.sqrt(sph.kpg2.numpy()) for sph in spheres]
    uniq_q, inv_q = _unique_shells(np.concatenate(q_per_k))
    ff_species = [beta_form_factors(upf, uniq_q) for upf in upfs]  # (nproj, n_uniq)
    offs = np.cumsum([0, *npw_list])
    proj_data = []
    so_tabs = (
        [torch.zeros(len(spheres), u.n_proj, npw_max, dtype=RDTYPE) for u in upfs]
        if is_fr
        else None
    )
    dij_species = [torch.as_tensor(upf.dij, dtype=RDTYPE) for upf in upfs]
    for ik, sph in enumerate(spheres):
        inv_k = inv_q[offs[ik] : offs[ik + 1]]
        beta_tables = [torch.as_tensor(ff[:, inv_k], dtype=RDTYPE) for ff in ff_species]
        if is_fr:
            # so_tabs is built as a real list exactly when is_fr (see the
            # ternary above); the two are set together, so it's never None here.
            assert so_tabs is not None
            for sp_i in range(len(upfs)):
                so_tabs[sp_i][ik, :, : sph.npw] = beta_tables[sp_i]
            beta_ls = [[] for _ in upfs]
            beta_tables = [t[:0] for t in beta_tables]
        else:
            beta_ls = [[b.l for b in upf.betas] for upf in upfs]
        proj_data.append(
            build_projector_data(
                sph, species_of_atom, beta_tables, beta_ls, dij_species, grid.volume
            )
        )

    # NLCC core density on the grid (frozen; enters XC only)
    rho_core = build_core_density(upfs, species_of_atom, positions, grid, uniq, inverse)

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
        so_beta_tables=so_tabs,
        is_fr=is_fr,
        rho_core=rho_core,
    )


def vxc_potential(
    xc: XCFunctional, rho: torch.Tensor, grid: FFTGrid, tau: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(v_xc(r) [eV], E_xc [eV]) via autograd — GGA divergence term included.

    For a meta-GGA, τ is passed as a held-fixed constant so this returns the
    multiplicative part v_xc = ∂e_xc/∂ρ|_{σ,τ}; the τ-response ∂e_xc/∂τ is a
    separate generalized-KS operator (see vtau_potential / core.metagga)."""
    rho_leaf = rho.detach().clone().requires_grad_(True)
    tau_c = None if tau is None else tau.detach()
    with torch.enable_grad():
        sigma = sigma_from_rho(rho_leaf, grid.g_cart) if xc.needs_gradient else None
        e_xc = xc.energy(rho_leaf, grid.volume, sigma, tau_c if xc.needs_tau else None)
        (v,) = torch.autograd.grad(e_xc, rho_leaf)
    return v * (grid.n_points / grid.volume), e_xc.detach()


def vtau_potential(
    xc: XCFunctional, rho: torch.Tensor, tau: torch.Tensor, grid: FFTGrid
) -> torch.Tensor:
    """v_τ(r) = ∂e_xc/∂τ|_{ρ,σ} [scaled], via autograd on a τ leaf.

    The scale (n_points/volume) undoes the (Ω/N) that `energy()` folds in, so
    the result is the pointwise energy-density derivative — exactly the field
    `core.metagga.metagga_tau_operator` consumes for −½∇·(v_τ∇ψ)."""
    rho_c = rho.detach()
    tau_leaf = tau.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        sigma = sigma_from_rho(rho_c, grid.g_cart) if xc.needs_gradient else None
        e_xc = xc.energy(rho_c, grid.volume, sigma, tau_leaf)
        # A needs_tau functional whose energy happens not to depend on τ (a τ-flat
        # limit) gives ∂e/∂τ = 0: e_xc then carries no grad_fn, so short-circuit to
        # a zero v_τ (inert operator) rather than letting grad() raise.
        if not e_xc.requires_grad:
            return torch.zeros_like(tau_leaf)
        (v,) = torch.autograd.grad(e_xc, tau_leaf, allow_unused=True)
    if v is None:
        return torch.zeros_like(tau_leaf)
    return v * (grid.n_points / grid.volume)


def _harris_foulkes_gap(
    system: System,
    xc: XCFunctional | SpinXC,
    rho_s_in: list[torch.Tensor],
    eigs_s: list[torch.Tensor],
    occ_s: list[torch.Tensor],
    e_free: float,
    e_ewald: torch.Tensor,
    entropy_term: torch.Tensor,
    nspin: int,
) -> float:
    """Harris-Foulkes minus Kohn-Sham free energy at this iteration [eV].

    E_HF[ρ_in] = Σ f ε − E_H[ρ_in] − ∫v_xc[ρ_in]ρ_in + E_xc[ρ_in] + E_ewald
    − σS, with the eigenvalues of H[ρ_in] (this iteration's solve). Both
    energies equal the self-consistent energy at the fixed point and differ at
    second order in the residual with opposite-signed leading terms, so the gap
    is the zero-machinery bracket of the SCF convergence error (docs/ideas.md's
    error-budget section). Everything here is already computed or one grid
    pass; the caller guards the orbital-dependent terms (Hubbard/Fock/meta-GGA)
    whose double counting this plain form omits.
    """
    from gradwave.core.energies.hartree import hartree_energy

    grid = system.grid
    cell = grid.volume / grid.n_points
    e_band = sum(
        float((system.kweights[:, None] * occ_s[sp] * eigs_s[sp]).sum())
        for sp in range(nspin)
    )
    rho_tot_in = rho_s_in[0] if nspin == 1 else rho_s_in[0] + rho_s_in[1]
    e_h_in = float(hartree_energy(r_to_g(rho_tot_in.to(CDTYPE)), grid.g2, grid.volume))
    core = system.rho_core
    if nspin == 1:
        assert isinstance(xc, XCFunctional)
        rho_xc = rho_tot_in if core is None else rho_tot_in + core
        v_xc, e_xc_in = vxc_potential(xc, rho_xc, grid)
        int_vxc = float((v_xc * rho_tot_in).sum()) * cell
    else:
        assert isinstance(xc, SpinXC)
        c2 = None if core is None else 0.5 * core
        ru = rho_s_in[0] if c2 is None else rho_s_in[0] + c2
        rd = rho_s_in[1] if c2 is None else rho_s_in[1] + c2
        vu, vd, e_xc_in = vxc_spin_potential(xc, ru, rd, grid)
        int_vxc = (float((vu * rho_s_in[0]).sum())
                   + float((vd * rho_s_in[1]).sum())) * cell
    e_hf = (e_band - e_h_in - int_vxc + float(e_xc_in) + float(e_ewald)
            + float(entropy_term))
    return e_hf - e_free


@dataclass
class SCFResult:
    converged: bool
    n_iter: int
    energies: EnergyBreakdown
    fermi: float
    eigenvalues: torch.Tensor  # (nk, nb) [eV]; (nspin, nk, nb) when nspin=2
    occupations: torch.Tensor  # (nk, nb) in [0,2]; (nspin, nk, nb) in [0,1] for spin
    coeffs: list[torch.Tensor] | list[list[torch.Tensor]]  # [(nb, npw_k)] per k;
    # list-of-lists [spin][k] when nspin=2
    rho: torch.Tensor  # TOTAL density (n1,n2,n3) [e/Å³]
    v_eff: torch.Tensor  # (n1,n2,n3) [eV]; (nspin,n1,n2,n3) when nspin=2
    system: System
    history: list[dict[str, int | float]] = field(default_factory=list)
    nspin: int = 1
    rho_spin: list[torch.Tensor] | None = None  # [ρ↑, ρ↓] when nspin=2
    mag_total: float = 0.0  # ∫(ρ↑−ρ↓) dr [μB]
    mag_abs: float = 0.0  # ∫|ρ↑−ρ↓| dr [μB]
    hub_occ: list[list[torch.Tensor]] | None = None  # DFT+U per-spin occupation matrices [σ][site]
    drho_scf: torch.Tensor | None = None  # last self-consistency residual ρ_out−ρ_in
    # (total density) for the SCF-error estimate
    formalism: str = "nc"  # result-type tag shared by all four SCF drivers
    kerker_used: bool | None = None  # resolved Kerker on/off (auto → concrete)
    recorder: Any = None  # scf.recorder.SCFRecorder — per-iteration flight recorder
    smearing: str = "none"  # smearing scheme name; the metallic χ₀ response needs
    width: float = 0.1  # it plus the width σ [eV] to form d(occ)/dε at the Fermi level
    boundary: str = "periodic"  # electrostatic BC the SCF ran with (periodic |
    # open_z | open_z_metal); forces/stress read it to add the ESM contribution
    esm_bias: float = 0.0  # applied capacitor bias [V] (open_z_metal); forces read it
    n_electrons: float = 0.0  # electron count (floats under constant-µ / target_mu);
    # the grand potential is Ω = free_energy − fermi·n_electrons


# A warm-start source for scf(): either a converged SCFResult, or the plainer
# dict/SimpleNamespace-like view checkpoint.as_start_from() hands back (the two
# shapes used at different points along the pipeline; see _seed_density/_seed_orbitals).
_StartFrom = (
    dict[str, "System | int | torch.Tensor | None"]
    | dict[str, "System | int | torch.Tensor | list[torch.Tensor] | None"]
    | SCFResult
    | None
)


def vxc_spin_potential(
    xc: SpinXC,
    rho_up: torch.Tensor,
    rho_dn: torch.Tensor,
    grid: FFTGrid,
    tau_s: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(v↑, v↓, E_xc) via autograd on a SpinXC — GGA terms included.

    For a meta-GGA, tau_s = [τ↑, τ↓] is passed as held-fixed constants so this
    returns the multiplicative v_xcσ = ∂e/∂ρ_σ|_{σ,τ}; the τ-response is the
    separate per-channel operator (see vtau_spin_potential)."""
    ru = rho_up.detach().clone().requires_grad_(True)
    rd = rho_dn.detach().clone().requires_grad_(True)
    tu, td = (None, None) if tau_s is None else (tau_s[0].detach(), tau_s[1].detach())
    with torch.enable_grad():
        s_uu, s_dd, s_tot = spin_sigmas(ru, rd, xc, grid.g_cart)
        e_xc = xc.energy(
            ru,
            rd,
            grid.volume,
            s_uu,
            s_dd,
            s_tot,
            tu if xc.needs_tau else None,
            td if xc.needs_tau else None,
        )
        vu, vd = torch.autograd.grad(e_xc, (ru, rd))
    scale = grid.n_points / grid.volume
    return vu * scale, vd * scale, e_xc.detach()


def vtau_spin_potential(xc, rho_up, rho_dn, tau_up, tau_dn, grid):
    """(v_τ↑, v_τ↓) [scaled] via autograd on the per-channel τ leaves — the
    fields core.metagga.metagga_tau_operator consumes for each spin's −½∇·(v_τ∇ψ)."""
    ru, rd = rho_up.detach(), rho_dn.detach()
    tu = tau_up.detach().clone().requires_grad_(True)
    td = tau_dn.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        s_uu, s_dd, s_tot = spin_sigmas(ru, rd, xc, grid.g_cart)
        e_xc = xc.energy(ru, rd, grid.volume, s_uu, s_dd, s_tot, tu, td)
        if not e_xc.requires_grad:  # τ-flat spin meta-GGA → zero, inert operators
            zeros = torch.zeros_like(tu)
            return zeros, zeros.clone()
        vu, vd = torch.autograd.grad(e_xc, (tu, td), allow_unused=True)
    scale = grid.n_points / grid.volume
    vu = torch.zeros_like(tu) if vu is None else vu * scale
    vd = torch.zeros_like(td) if vd is None else vd * scale
    return vu, vd


def local_potential_r(system: System, vloc_g: torch.Tensor | None = None) -> torch.Tensor:
    """v_loc(r) on the dense grid [eV] — the SCF's local-potential path."""
    grid = system.grid
    if vloc_g is None:
        vloc_g = local_potential_g(
            system.positions,
            system.species_index,
            system.vloc_tables,
            grid.g_cart,
            grid.volume,
            vloc_atom=system.vloc_atom,
        )
    return g_to_r_box(vloc_g, real=True)


def effective_potentials(
    system: System | USPPSystem,
    xc: XCFunctional | SpinXC,
    rho_s: list[torch.Tensor],
    vloc_r: torch.Tensor,
    tau: torch.Tensor | list[torch.Tensor] | None = None,
    boundary: str = "periodic",
    esm_bias: float = 0.0,
) -> list[torch.Tensor]:
    """Per-spin v_eff(r) from per-channel densities — THE assembly the SCF
    iterates with. A standalone function (not inlined in the loop) so the
    off-stationarity E↔H consistency gate can test the exact potential the
    solver applies (tests/unit/test_energy_hamiltonian_consistency.py).

    `tau` (meta-GGA only) is the current kinetic-energy density passed to the
    local v_xc as a held-fixed constant — a grid for nspin=1, a [τ↑, τ↓] list for
    nspin=2. The τ-response operator −½∇·(v_τ∇ψ) is NOT part of v_eff — it is
    applied separately in the H-apply."""
    grid = system.grid
    nspin = len(rho_s)
    rho_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    # Real density → real Hartree potential in one rfft round trip (half the
    # transform work of the full-complex fftn/ifftn; bit-exact to it).
    v_h_r = hartree_potential_r(rho_tot, grid.g2)
    # Open-boundary (ESM): add the open-minus-periodic electrostatic potential
    # δΔE/δρ. Spin-independent (acts on the total charge); zero when periodic.
    esm_mode = esm_mode_of(boundary)
    if esm_mode is not None:
        v_h_r = v_h_r + esm_potential(rho_tot, system.positions, system.charges,
                                      grid, mode=esm_mode, bias=esm_bias)
    core = system.rho_core
    if nspin == 1:
        # nspin=1 is always dispatched with a collinear XCFunctional (the
        # spin-family classes are only ever used on the nspin=2 branch below,
        # via vxc_spin_potential) and tau, if given, is the single grid field.
        assert isinstance(xc, XCFunctional)
        assert tau is None or isinstance(tau, torch.Tensor)
        v_xc_r, _ = vxc_potential(xc, rho_tot if core is None else rho_tot + core, grid, tau=tau)
        return [v_h_r + v_xc_r + vloc_r]
    # Any SpinXC subclass is valid here (LSDA_PW92/SpinPBE/LearnableSpinX/
    # SpinR2SCAN/...) -- vxc_spin_potential is fully generic over the base
    # class (duck-typed on .energy()/.needs_tau), so narrowing to specific
    # concrete classes would wrongly reject real, registered functionals
    # (e.g. SPIN_XC_REGISTRY["r2scan"] -> SpinR2SCAN).
    assert isinstance(xc, SpinXC)
    # tau is a per-spin [τ↑, τ↓] list on this (nspin=2) branch -- never the
    # bare grid Tensor the nspin=1 branch above takes. isinstance narrowing
    # against `list` doesn't cleanly separate the Tensor arm of tau's static
    # union (a Tensor is not provably disjoint from `list` to the checker),
    # so state the real contract with a cast instead of an isinstance assert.
    tau_s = cast("list[torch.Tensor] | None", tau)
    cu2 = None if core is None else 0.5 * core
    v_up, v_dn, _ = vxc_spin_potential(
        xc,
        rho_s[0] if cu2 is None else rho_s[0] + cu2,
        rho_s[1] if cu2 is None else rho_s[1] + cu2,
        grid,
        tau_s=tau_s,
    )
    return [v_h_r + v_up + vloc_r, v_h_r + v_dn + vloc_r]


def resolve_atom_moments(
    start_mag: list[float] | None,
    species_of_atom: list[int],
    n_species: int,
    *,
    default,
) -> list[float]:
    """Per-atom moment fractions from start_mag: one entry per atom
    (AFM/ferrimagnetic) or one per species (broadcast to its atoms); `default`
    seeds None. Raise on a length matching neither. (Canonical rule: a length
    matching the atom count is read per-atom.)"""
    na = len(species_of_atom)
    if start_mag is None:
        return [default] * na
    if len(start_mag) == na:
        return [float(m) for m in start_mag]
    if len(start_mag) == n_species:
        return [float(start_mag[sp]) for sp in species_of_atom]
    raise ValueError("start_mag must have one entry per atom or per species")


def _seed_density(
    system: System,
    nspin: int,
    start_from: _StartFrom,
    start_mag: list[float] | None,
    grid: FFTGrid,
    vol: float,
) -> list[torch.Tensor]:
    """Initial per-spin density: warm-start from a previous state (volume-
    rescaled so the electron count is conserved), else SAD — spin-split by
    start_mag for nspin=2."""
    if start_from is not None:
        return warm_start_densities(start_from, nspin, grid, vol, system.positions.device)
    if nspin == 1:
        return [
            sad_density(
                grid, system.positions, system.species_of_atom, system.upfs, system.n_electrons
            )
        ]
    na = len(system.species_of_atom)
    nspecies = len(system.upfs)
    mags_at = resolve_atom_moments(start_mag, system.species_of_atom, nspecies, default=0.5)
    mags_by_sp = {}
    for a, sp in enumerate(system.species_of_atom):
        mags_by_sp.setdefault(sp, set()).add(round(mags_at[a], 12))
    uniform_per_species = all(len(v) == 1 for v in mags_by_sp.values())
    # A CollinearMagneticSymmetrizer is BUILT for exactly this case (its magnetic
    # group already encodes the sublattice pattern), so it is exempt: the plain
    # RhoSymmetrizer is not — it would fold the two spin sublattices together.
    from gradwave.symmetry import CollinearMagneticSymmetrizer

    if (
        system.rho_symmetrizer is not None
        and not uniform_per_species
        and not isinstance(system.rho_symmetrizer, CollinearMagneticSymmetrizer)
    ):
        raise ValueError(
            "non-uniform per-atom moments break the chemical space group "
            "(magnetic group is smaller) — build the system with "
            "use_symmetry=False, or collinear_magnetic=True + magmoms for the "
            "magnetic (Shubnikov) k-fold, for AFM/ferrimagnetic configurations"
        )
    n_up = sum(float(system.charges[a]) * (1 + mags_at[a]) / 2 for a in range(na))
    n_dn = system.n_electrons - n_up
    return [
        sad_density(
            grid,
            system.positions,
            system.species_of_atom,
            system.upfs,
            n_up,
            atom_scale=[(1 + m) / 2 for m in mags_at],
        ),
        sad_density(
            grid,
            system.positions,
            system.species_of_atom,
            system.upfs,
            n_dn,
            atom_scale=[(1 - m) / 2 for m in mags_at],
        ),
    ]


def _seed_orbitals(
    nk: int,
    nb: int,
    bk: BatchedK,
    nspin: int,
    device: torch.device,
    start_from: _StartFrom,
) -> list[torch.Tensor]:
    """Initial per-spin orbital guess: an identity block of the lowest-|k+G|²
    plane waves, overwritten by shape-compatible previous orbitals (the QE
    wfc-extrapolation analogue) when start_from carries them."""
    c0 = torch.zeros(nk, nb, bk.npw_max, dtype=CDTYPE, device=device)
    c0[:, torch.arange(nb), torch.arange(nb)] = 1.0
    coeffs_b_s = [c0.clone() for _ in range(nspin)]
    if start_from is not None:
        # _StartFrom's dict variants type "coeffs" as (at most) a flat
        # list[Tensor], but a warm-started nspin=2 run actually stores a
        # list-of-lists [spin][k] (see SCFResult.coeffs) -- a real gap in
        # _StartFrom's value union (same shared debt as the dict-literal
        # "site" records flagged elsewhere), not something to paper over with
        # a narrower annotation here. cast past it, like postscf/_kb.py does
        # for its own untypeable dict payload.
        prev_c = cast(
            "list[Any] | None",
            start_from.get("coeffs")
            if isinstance(start_from, dict)
            else getattr(start_from, "coeffs", None),
        )
        if prev_c is not None:
            chans = [prev_c] if nspin == 1 else list(prev_c)
            compat = len(chans) == nspin and all(
                len(ch) == nk
                and all(
                    ch[ik].shape[0] >= nb and ch[ik].shape[1] == int(bk.npw[ik]) for ik in range(nk)
                )
                for ch in chans
            )
            if compat:
                for sp, ch in enumerate(chans):
                    for ik in range(nk):
                        coeffs_b_s[sp][ik, :, : int(bk.npw[ik])] = ch[ik][:nb].to(
                            device=device, dtype=CDTYPE
                        )
    return coeffs_b_s


def _validate_scf_args(
    system: System,
    nspin: int,
    eigensolver: str,
    smearing: str,
    mixing_scheme: str,
    precond: str,
    tot_magnetization: float | None = None,
):
    """Reject unsupported argument combinations up front, before any work."""
    if nspin not in (1, 2):
        raise ValueError(
            "nspin must be 1 or 2 (noncollinear spin uses scf_noncollinear, the spinor SCF)"
        )
    from gradwave.solvers.registry import available, is_registered

    if not is_registered(eigensolver):
        raise ValueError(f"unknown eigensolver {eigensolver!r}; registered: {available()}")
    if nspin == 2 and smearing == "none" and tot_magnetization is None:
        raise ValueError(
            "nspin=2 without smearing requires tot_magnetization "
            "(fixed spin moment); otherwise pass a smearing"
        )
    if system.is_fr:
        raise ValueError(
            "fully-relativistic pseudos require the spinor SCF "
            "(scf_noncollinear) — SOC has no collinear representation"
        )
    if hasattr(system.rho_symmetrizer, "apply_m"):
        raise ValueError(
            "system was built with SPINOR magnetic symmetry "
            "(magmoms without collinear_magnetic) — only "
            "scf_noncollinear consumes it; for collinear nspin=2 "
            "rebuild with collinear_magnetic=True"
        )
    from gradwave.symmetry import CollinearMagneticSymmetrizer

    if isinstance(system.rho_symmetrizer, CollinearMagneticSymmetrizer) and nspin != 2:
        raise ValueError(
            "a collinear magnetic (Shubnikov) system folds the two "
            "spin channels — it requires nspin=2"
        )
    if mixing_scheme not in ("pulay", "broyden", "johnson"):
        raise ValueError("mixing_scheme must be 'pulay', 'broyden', or 'johnson'")
    if precond not in ("kerker", "local_tf"):
        raise ValueError("precond must be 'kerker' or 'local_tf'")


def _resolve_mixing_scheme(mixing_scheme: str | None, nspin: int) -> str:
    """Magnetic-aware mixing-scheme default. None → johnson for collinear-spin
    (nspin==2) systems, pulay otherwise; an explicit scheme always wins.

    The magnetization channel near the Stoner instability — not charge sloshing
    — limits convergence on magnetic metals (Phase-2 residual instrumentation),
    and johnson's normalized multisecant update is the robust scheme there
    (Fe FM 15→14 vs pulay; never worse than pulay on the Si/Al non-magnetic
    anchors). Gating on nspin==2 keeps non-magnetic systems on pulay."""
    if mixing_scheme is not None:
        return mixing_scheme
    return "johnson" if nspin == 2 else "pulay"


def _resolve_kerker(kerker: bool | None, smearing: str, grid: FFTGrid) -> bool:
    """Kerker on/off. An explicit setting wins; otherwise the auto policy turns
    it on for metals always and for insulators once the smallest nonzero |G|
    drops below ~0.8 Å⁻¹ (cell ≳ 8 Å), where long-wavelength charge sloshing —
    amplified like 4πe²χ/G²_min — starts to dominate mixing."""
    if kerker is not None:
        return kerker
    g2_nonzero = grid.g2.reshape(-1)
    g2_min = float(g2_nonzero[g2_nonzero > 1e-12].min())
    return (smearing != "none") or (g2_min < 0.64)


def _build_mixer_precond(
    grid: FFTGrid,
    nspin: int,
    layout: MixLayout,
    mixing_scheme: str,
    mixing_alpha: float,
    mixing_history: int,
    kerker: bool,
    precond: str,
    precond_op: MultipoleKerkerPrecond | None,
    float_charge: bool = False,
) -> tuple[PulayMixer | BroydenMixer | JohnsonMixer, LocalTFPrecond | None]:
    """Construct the density mixer and resolve its preconditioner.

    (mixer, tf_precond) genuinely range over all 6 combinations here --
    `mixing_scheme` (pulay/broyden/johnson) and `precond` (kerker/local_tf) are
    independent knobs, so e.g. (JohnsonMixer, LocalTFPrecond) is reachable too,
    not just the 4 pairings a previous, narrower return-type annotation
    enumerated (an inaccuracy in the annotation, not the mixer/precond
    construction logic below, which has always allowed any combination).

    Returns (mixer, tf_precond); tf_precond is the LocalTFPrecond needing a
    per-iteration set_density(), or None. Kerker and any grid-block precond act
    on the density-total block only for nspin=2, preserving per-channel counts.
    """
    _MixerCls = {"pulay": PulayMixer, "broyden": BroydenMixer, "johnson": JohnsonMixer}[
        mixing_scheme
    ]
    # Johnson saturates at history 12 on FM Ni (mixing.py); honour an explicit
    # override but lift the default-8 up to 12 for it.
    hist = 12 if mixing_scheme == "johnson" and mixing_history == 8 else mixing_history
    mixer = _MixerCls(
        layout.g2_full,
        alpha=mixing_alpha,
        history=hist,
        kerker=kerker,
        # constant-µ lets the charge (G=0) float, so the mixer must mix it and
        # NOT assert charge conservation.
        check_g0=(nspin == 1) and not float_charge,
        kerker_mask=layout.kerker_mask if nspin == 2 else None,
    )

    tf_precond = None
    if precond_op is not None:
        # a caller-supplied operator (learned radial filter). Static across
        # iterations, so unlike local_tf it needs no per-step set_density. By
        # default it acts on the density-total block; an operator declaring
        # acts_on="grid" (BlockPrecond) spans every nspin grid block, so a
        # magnetization-channel filter reaches the (total, mag) pair.
        mixer.precond_op = precond_op
        if nspin == 2:
            spans_grid = getattr(precond_op, "acts_on", "total") == "grid"
            mixer.precond_slice = slice(0, nspin * layout.ng if spans_grid else layout.ng)
        else:
            mixer.precond_slice = None
    elif precond == "local_tf":
        # position-dependent TF screening on the density-total block; capped at
        # the bare-Kerker q0 so a bulk metal is unchanged and only the vacuum is
        # unscreened. set_density() is called with the current n(r) each iter.
        from gradwave.scf.local_tf import LocalTFPrecond

        tf_precond = LocalTFPrecond(grid.g2, grid.shape, layout.mask, q0_max=mixer.q0)
        mixer.precond_op = tf_precond
        mixer.precond_slice = slice(0, layout.ng) if nspin == 2 else None
    return mixer, tf_precond


def _solve_bands(
    veff_sp: torch.Tensor,
    coeffs_sp: torch.Tensor,
    bk: BatchedK,
    grid_shape: tuple[int, int, int],
    projs_b: torch.Tensor,
    hub: HubbardData | None,
    hub_q: torch.Tensor | None,
    n_hub_sp: list[torch.Tensor] | None,
    hub_alpha: list[float] | None,
    fock_apply_sp: Callable[[torch.Tensor], torch.Tensor] | None,
    metagga_apply_sp: Callable[[torch.Tensor], torch.Tensor] | None,
    eigensolver: str,
    tol_eff: float,
    use_low: bool,
    cdtype: torch.dtype,
    t_solve: torch.Tensor,
    device: torch.device,
    u_scale: float = 1.0,
    coarse=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigensolve one spin channel of the NC standard problem H x = ε x.

    Builds H (KB nonlocal + optional Hubbard V_U), composes the additive Fock
    and meta-GGA operators onto the H-apply, diagonalizes, and returns
    (eigenvalues [RDTYPE], coeffs [CDTYPE]). fock_apply_sp / metagga_apply_sp are
    this spin's already-indexed operators (or None); each is captured by DEFAULT
    ARGUMENT so the composed closure binds this operator, not a later one.

    ``u_scale`` (default 1.0) linearly scales the Hubbard V_U D-matrix for the
    U-ramp; the rigid manifold probe alpha is added AFTER scaling so the ramp
    never touches the linear-response probe.

    ``coarse`` (a ``core.batch.CoarseDraft`` or None) enables the certified
    coarse-box local apply: V_eff is truncated onto the coarse box and the
    draft operator is used ONLY when the truncation's ℓ1 tail bound is
    ≤ 0.1·tol_eff (Eisenstat-Walker-style — an error slaved to the tolerance
    the solve already tolerates). Once the adaptive schedule tightens past
    the bound, this falls back to the exact operator automatically, so the
    converged fixed point is untouched.
    """
    from gradwave.core.batch import BatchedHamiltonian, coarse_draft_veff
    from gradwave.solvers.registry import get as get_solver

    hub_dij = None
    if hub is not None:
        from gradwave.core.hubbard import hubbard_dmatrix

        # n_hub_sp is passed as None only when hub is None (see the call site
        # in scf()'s main loop), so it's always real here.
        assert n_hub_sp is not None
        dij = hubbard_dmatrix(n_hub_sp, hub.sites, hub.nproj, device)
        if u_scale != 1.0:  # U-ramp: V_U is linear in U_eff (see hubbard_u_ramp_scale)
            dij = dij * u_scale
        if hub_alpha is not None:  # rigid manifold probe α·I (linear response)
            for si, s in enumerate(hub.sites):
                st, dim = s["start"], s["dim"]
                dij[st : st + dim, st : st + dim] += hub_alpha[si] * torch.eye(
                    dim, dtype=CDTYPE, device=device
                )
        # apply convention wants D^T; D is Hermitian so D^T = conj(D)
        hub_dij = dij.conj()
    smooth = None
    if coarse is not None:
        sm, eps = coarse_draft_veff(veff_sp, coarse)
        if eps <= 0.1 * tol_eff:
            smooth = sm
            logger.debug("coarse draft active: eps=%.3e tol_eff=%.3e", eps, tol_eff)
    h = BatchedHamiltonian(
        bk, grid_shape, veff_sp, projs_b, hub_q=hub_q, hub_dij=hub_dij, smooth=smooth
    )
    apply = h.apply
    if fock_apply_sp is not None:

        def apply(c, _base=h.apply, _f=fock_apply_sp):
            return _base(c) + _f(c)

    if metagga_apply_sp is not None:

        def apply(c, _base=apply, _m=metagga_apply_sp):
            return _base(c) + _m(c)

    dav = get_solver(eigensolver)(
        apply, coeffs_sp.to(cdtype), t_solve, bk.mask, tol=tol_eff, nbands=coeffs_sp.shape[1]
    )
    eigenvalues = dav.eigenvalues.to(RDTYPE)
    c = dav.eigenvectors.to(CDTYPE)
    if use_low:
        # fp32 leaves ‖ψ‖ accurate only to ~1e-6; renormalize in fp64 so the
        # density's electron count (ρ at G=0) is conserved to the mixer's
        # tolerance (off-diagonal overlaps don't touch G=0)
        c = c / torch.linalg.norm(c, dim=-1, keepdim=True).clamp_min(1e-30)
    return eigenvalues, c


def _bootstrap_tau(
    xc: XCFunctional | SpinXC,
    coeffs_b_s: list[torch.Tensor],
    system: System,
    nspin: int,
    nk: int,
    nb: int,
    bk: BatchedK,
    grid: FFTGrid,
    vol: float,
    device: torch.device,
) -> list[torch.Tensor] | None:
    """Seed the per-spin kinetic-energy density τ from the initial orbitals so
    iteration 1 has a valid τ for the τ-dependent v_xc (refined immediately).
    Returns the per-spin tau_list, or None for non-meta-GGA functionals."""
    if not xc.needs_tau:
        return None
    from gradwave.core.metagga import tau_b

    # bootstrap occupations: nspin=1 fills 2 e⁻/band, nspin=2 fills 1 e⁻/band
    f0 = 2.0 if nspin == 1 else 1.0
    nocc = max(int(round(system.n_electrons / (2 if nspin == 1 else nspin))), 1)
    occ0 = torch.zeros(nk, nb, dtype=RDTYPE, device=device)
    occ0[:, :nocc] = f0
    return [
        tau_b(coeffs_b_s[sp], occ0, system.kweights, bk, grid.shape, vol) for sp in range(nspin)
    ]


def _build_metagga_apply(
    xc: XCFunctional | SpinXC,
    rho_s: list[torch.Tensor],
    rho_tot: torch.Tensor,
    tau_list: list[torch.Tensor] | None,
    system: System,
    nspin: int,
    bk: BatchedK,
    grid: FFTGrid,
) -> list[Callable[[torch.Tensor], torch.Tensor]] | None:
    """Per-spin meta-GGA generalized-KS operator −½∇·(v_τσ∇ψ_σ), or None when the
    functional is not τ-dependent. v_τσ = ∂e_xc/∂τ_σ from the current (ρ, τ); each
    spin's operator is captured by DEFAULT ARGUMENT so the spin solve binds this
    v_τ, not a later one."""
    if not xc.needs_tau:
        return None
    from gradwave.core.metagga import metagga_tau_operator

    # tau_list is bootstrapped/rebuilt in lockstep with xc.needs_tau by the
    # caller (_bootstrap_tau before the loop, then the `if xc.needs_tau:`
    # rebuild each iteration — see scf()) so it's never None here.
    assert tau_list is not None
    if nspin == 1:
        assert isinstance(xc, XCFunctional)
        rho_for_xc = rho_tot if system.rho_core is None else rho_tot + system.rho_core
        v_tau_s = [vtau_potential(xc, rho_for_xc, tau_list[0], grid)]
    else:
        cu2 = None if system.rho_core is None else 0.5 * system.rho_core
        r_u = rho_s[0] if cu2 is None else rho_s[0] + cu2
        r_d = rho_s[1] if cu2 is None else rho_s[1] + cu2
        v_tau_s = list(vtau_spin_potential(xc, r_u, r_d, tau_list[0], tau_list[1], grid))
    return [
        (lambda c, _v=v_tau_s[sp]: metagga_tau_operator(c, _v, bk, grid.shape))
        for sp in range(nspin)
    ]


def _assemble_scf_energies(
    system: System,
    xc: XCFunctional | SpinXC,
    grid: FFTGrid,
    vol: float,
    spheres: list[GSphere],
    nk: int,
    nspin: int,
    coeffs_b_s: list[torch.Tensor],
    occ_s: list[torch.Tensor],
    rho_tot_out: torch.Tensor,
    rho_out_s: list[torch.Tensor],
    tau_list: list[torch.Tensor] | None,
    entropy_term: torch.Tensor,
    e_ewald: torch.Tensor,
    vloc_g: torch.Tensor,
    e_hub: torch.Tensor,
    e_fock: torch.Tensor,
    projs_b: torch.Tensor,
    boundary: str = "periodic",
    esm_bias: float = 0.0,
) -> tuple[EnergyBreakdown, list[list[torch.Tensor]]]:
    """Total-energy breakdown at (current orbitals, mixed-out density), plus the
    per-k trimmed coeff views that flow to SCFResult. Returns (energies,
    coeffs_list_s). Under no_grad — the energies are detached scalars (the
    differentiable energy is rebuilt in postscf/forces.py), so no autograd graph
    is threaded here.

    npw from the CPU-side spheres (int(bk.npw[ik]) is a host sync per k); ONE
    becp over the whole batch, then per-k views (calling becp_b inside the per-k
    comprehension recomputed the full-batch contraction nk times).
    """
    from gradwave.core.batch import becp_b

    coeffs_list_s = [
        [coeffs_b_s[sp][ik, :, : system.spheres[ik].npw] for ik in range(nk)] for sp in range(nspin)
    ]
    becps_s = []
    for sp in range(nspin):
        b_all = becp_b(projs_b, coeffs_b_s[sp])
        becps_s.append([b_all[ik] for ik in range(nk)])
    if nspin == 1:
        # nspin=1 is always dispatched with a collinear XCFunctional (mirrors
        # effective_potentials' own nspin-based dispatch above in this file).
        assert isinstance(xc, XCFunctional)
        energies = total_energy(
            coeffs_per_k=coeffs_list_s[0],
            occ=occ_s[0],
            kweights=system.kweights,
            spheres=spheres,
            grid=grid,
            rho=rho_tot_out,
            positions=system.positions,
            charges=system.charges,
            species_index=system.species_index,
            vloc_tables=system.vloc_tables,
            becp_per_k=becps_s[0],
            dij_full=_stack_dij(system),
            xc=xc,
            entropy_term=entropy_term,
            rho_core=system.rho_core,
            tau=(tau_list[0] if tau_list else None),
            e_ewald=e_ewald,
            vloc_g=vloc_g,
        )
        energies.hubbard = e_hub
        energies.fock = e_fock
    else:
        rho_g_out = r_to_g(rho_tot_out.to(CDTYPE))
        energies = assemble_pw_energies(
            coeffs_list_s,
            occ_s,
            system.kweights,
            spheres,
            grid,
            vol,
            rho_g_out,
            spin_xc_energy(xc, rho_out_s, system.rho_core, vol, grid.g_cart, tau_s=tau_list),
            vloc_g,
            becps_s,
            _stack_dij(system),
            system.positions,
            system.charges,
            entropy_term,
            nspin,
            e_hub=float(e_hub),
            e_ewald=e_ewald,
        )
        energies.fock = e_fock
    esm_mode = esm_mode_of(boundary)
    if esm_mode is not None:
        # ΔE = open-minus-periodic electrostatic correction (both spins act on
        # the total charge). Detached like the rest of this no_grad breakdown;
        # the differentiable copy for forces is rebuilt in postscf/forces.py.
        energies.esm = esm_energy(rho_tot_out, system.positions, system.charges,
                                  grid, mode=esm_mode, bias=esm_bias)
    return energies, coeffs_list_s


def _hubbard_occ_update(
    hub: HubbardData | None,
    hub_q: torch.Tensor | None,
    coeffs_b_s: list[torch.Tensor],
    occ_s: list[torch.Tensor],
    system: System,
    nspin: int,
    device: torch.device,
    dist_ctx: "DistKContext | None" = None,
    n_hub_prev: list[list[torch.Tensor]] | None = None,
    occ_mix: float = 1.0,
    u_scale: float = 1.0,
) -> tuple[list[list[torch.Tensor]], torch.Tensor] | tuple[None, torch.Tensor]:
    """Refresh the DFT+U per-spin occupation matrices from the fresh orbitals and
    return (n_hub_s, e_hub). No Hubbard manifold → (None, 0). For nspin=1 the
    [0,2] occupation splits into two equal spin channels.

    ``occ_mix`` (β, default 1.0) damps the returned matrices against the previous
    iteration's ``n_hub_prev`` — ``n = (1-β)·n_prev + β·n_new`` — the
    occupation-matrix mixing that contracts the large-U-on-metal flip-flop. The
    energy ``e_hub`` is always evaluated at the FRESH ``n_new`` (the occupation of
    the current orbitals), not the mixed carry-forward; the two coincide at the
    fixed point. ``u_scale`` (default 1.0) scales ``e_hub`` for the U-ramp, in
    lockstep with the D-matrix scaling in ``_solve_bands``. β=1.0 AND u_scale=1.0
    reproduce today's numbers bit-for-bit.

    Under ``dist_ctx`` (distributed k-point-sharded SCF), ``system``/``occ_s``/
    ``coeffs_b_s``/``hub_q`` are all THIS RANK's local k-shard, so
    ``occupation_matrices``' k-weighted sum below is only a PARTIAL sum over
    the local shard — exactly like ``core.batch.density_b``'s partial density.
    ``n_hub`` is therefore ``all_reduce``-SUMmed across ranks to the full-mesh
    value before ``e_hub`` is computed. Unlike kinetic/nonlocal energy, e_hub
    is NOT itself k-extensive-linear — ``hubbard_energy`` is a NONLINEAR
    (Tr[n(1−n)]) function of n_hub, so summing each rank's LOCAL e_hub would
    be wrong (Tr[(nA+nB)(1−nA−nB)] ≠ Tr[nA(1−nA)] + Tr[nB(1−nB)]). e_hub is
    instead recomputed from the already-reduced, full-mesh n_hub_s."""
    e_hub = torch.zeros((), dtype=RDTYPE, device=device)
    if hub is None:
        return None, e_hub
    from gradwave.core.hubbard import hubbard_energy, hubbard_occ_and_energy, occupation_matrices

    assert hub_q is not None  # set together with hub (scf()'s `if hubbard:` block)
    n_hub_s, e_hub = hubbard_occ_and_energy(
        lambda isp, w: occupation_matrices(hub_q, coeffs_b_s[isp], w, system.kweights, hub.sites),
        occ_s,
        hub.sites,
        nspin,
    )
    if dist_ctx is not None:
        from gradwave.distributed import all_reduce_

        n_hub_s = [[all_reduce_(m, dist_ctx) for m in mats] for mats in n_hub_s]
        # e_hub above was computed from this rank's PARTIAL (local-shard)
        # n_hub_s — recompute from the just-reduced, full-mesh matrices,
        # mirroring hubbard_occ_and_energy's own e_hub formula (nspin=2:
        # sum of the two channels' Dudarev traces; nspin=1: half-filled
        # single channel doubled).
        if nspin == 2:
            e_hub = torch.zeros((), dtype=RDTYPE, device=device)
            for mats in n_hub_s:
                e_hub = e_hub + hubbard_energy(mats, hub.sites)
        else:
            e_hub = 2.0 * hubbard_energy(n_hub_s[0], hub.sites)
    if nspin == 1:
        n_hub_s = [n_hub_s[0], n_hub_s[0]]  # loop returns BOTH spin channels
    # U-ramp: E_U is linear in U_eff, so scaling the full-U energy by u_scale
    # matches the ramped-U D-matrix built in _solve_bands this same iteration.
    if u_scale != 1.0:
        e_hub = e_hub * u_scale
    # occupation-matrix damping for the NEXT iteration's V_U (energy stays fresh)
    n_hub_s = mix_hubbard_occ(n_hub_prev, n_hub_s, occ_mix)
    return n_hub_s, e_hub


def _scf_residual_and_record(
    layout: MixLayout,
    rho_s: list[torch.Tensor],
    rho_out_s: list[torch.Tensor],
    rho_tot: torch.Tensor,
    rho_tot_out: torch.Tensor,
    mixer_hook: Callable[[int, torch.Tensor, torch.Tensor], None] | None,
    it: int,
    e_free: float,
    e_free_prev: float | None,
    t_it: float,
    history: list[dict[str, int | float]],
    nspin: int,
    vol: float,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, float]:
    """Pack the in/out densities, run the mixer_hook probe, validate nspin=2
    total-charge conservation, and record the iteration. Returns (rho_in_vec,
    rho_out_vec, res_norm, drho_scf, de). drho_scf is the PRE-mixing real-space
    total residual kept for the post-SCF convergence-error estimate."""
    rho_in_vec = layout.pack(rho_s)
    rho_out_vec = layout.pack(rho_out_s)
    if mixer_hook is not None:
        mixer_hook(it, rho_in_vec, rho_out_vec)
    if nspin == 2:  # only the TOTAL is conserved; its G=0 residual must vanish
        if not torch.isfinite(rho_out_vec).all():
            raise RuntimeError("density diverged (NaN/Inf)")
        if (rho_out_vec[0] - rho_in_vec[0]).abs() >= 1e-8:
            raise ValueError("total G=0 residual nonzero")
    res_norm = float(torch.linalg.norm(rho_out_vec - rho_in_vec)) * vol
    drho_scf = rho_tot_out - rho_tot
    de = record_iteration(history, it, e_free, e_free_prev, res_norm, t_it)
    logger.debug("SCF iter %d: F=%+.10f eV  dE=%.3e  |drho|=%.3e", it, e_free, de, res_norm)
    if verbose:
        mag = ""
        if nspin == 2:
            m = float((rho_out_s[0] - rho_out_s[1]).mean()) * vol
            mag = f"   m = {m:+.4f} muB"
        print(
            f"  SCF {it:3d}  F = {e_free:+.10f} eV   "
            f"dE = {(0.0 if de == float('inf') else de):.3e}   "
            f"|drho| = {res_norm:.3e}   {history[-1]['t']:5.2f}s{mag}"
        )
    return rho_in_vec, rho_out_vec, res_norm, drho_scf, de


def _fermi_occupations(eigs_s, system, smearing, width, nspin, device, *,
                       target_mu, tot_magnetization, dist_ctx, kweights_global):
    """Fermi level + occupations from the current eigenvalues, returning
    ``(occ_s, mu, entropy_term, n_float, eigs_global_s, occ_global_s)`` where
    ``occ_s`` is THIS rank's slice and ``eigs_global_s`` / ``occ_global_s`` are
    the full-mesh gathered eigenvalues/occupations (equal to the local arrays
    off the distributed path). The caller carries the globals forward so the
    final reassembled SCFResult reports full-mesh eigenvalues/occupations rather
    than one rank's shard.

    Three regimes: constant-µ (``target_mu`` set), the default shared-Fermi
    search, and the k-point-sharded distributed path — where the Fermi level
    depends on eigenvalues from EVERY k-point, so the eigenvalues are gathered
    for the search and only the local shard's occupations are sliced back."""
    if dist_ctx is not None:
        from gradwave.distributed import gather_cat

        eigs_global_s = [gather_cat(eigs_s[sp], dist_ctx) for sp in range(nspin)]
        if target_mu is not None:
            occ_global_s, mu, entropy_term, n_float = constant_mu_occupations(
                eigs_global_s, kweights_global, smearing, width, target_mu,
                nspin, device)
        else:
            occ_global_s, mu, entropy_term = shared_fermi_occupations(
                eigs_global_s, kweights_global, smearing, width,
                system.n_electrons, nspin, device,
                tot_magnetization=tot_magnetization)
            n_float = float(system.n_electrons)
        occ_s = [occ_global_s[sp][dist_ctx.k_start : dist_ctx.k_end]
                 for sp in range(nspin)]
        return occ_s, mu, entropy_term, n_float, eigs_global_s, occ_global_s
    if target_mu is not None:
        occ_s, mu, entropy_term, n_float = constant_mu_occupations(
            eigs_s, system.kweights, smearing, width, target_mu, nspin, device)
        return occ_s, mu, entropy_term, n_float, eigs_s, occ_s
    occ_s, mu, entropy_term = shared_fermi_occupations(
        eigs_s, system.kweights, smearing, width, system.n_electrons, nspin,
        device, tot_magnetization=tot_magnetization)
    return occ_s, mu, entropy_term, float(system.n_electrons), eigs_s, occ_s


def _output_density(coeffs_b_s, occ_s, system, bk, grid, vol, nspin, *,
                    dist_ctx, collinear_mag):
    """Output density ρ_out from the fresh orbitals: per-spin ``density_b``, a
    distributed all-reduce that completes the k-sum across shards, then
    symmetrization. A collinear magnetic (Shubnikov) system folds ρ↑/ρ↓ JOINTLY
    (anti-unitary ops swap the spin channels, so they cannot be symmetrized
    separately); otherwise each channel folds independently. Returns
    ``(rho_out_s, rho_tot_out)``."""
    from gradwave.core.batch import density_b

    rho_raw_s = [
        density_b(coeffs_b_s[sp], occ_s[sp], system.kweights, bk, grid.shape, vol)
        for sp in range(nspin)
    ]
    if dist_ctx is not None:
        from gradwave.distributed import all_reduce_

        # each rank's density_b summed its own k-shard; the reduce completes the
        # sum over the full mesh (core/batch.py's einsum, extended across ranks)
        rho_raw_s = [all_reduce_(r, dist_ctx) for r in rho_raw_s]
    if collinear_mag:
        from gradwave.scf.common import symmetrize_rho_pair

        rho_out_s = list(symmetrize_rho_pair(
            system.rho_symmetrizer, rho_raw_s[0], rho_raw_s[1], grid))
    else:
        rho_out_s = [symmetrize_rho(system.rho_symmetrizer, r, grid)
                     for r in rho_raw_s]
    rho_tot_out = rho_out_s[0] if nspin == 1 else rho_out_s[0] + rho_out_s[1]
    return rho_out_s, rho_tot_out


def _apply_spin_precond(mixer, system, coeffs_list_s, eigs_s, mu, smearing,
                        width, rho_out_s, xc, dist_ctx, ng):
    """Set the mixer's Stoner spin preconditioner on the magnetization channel
    from the current orbitals (arXiv:2606.26693): a rank-r Woodbury Newton step
    for the Stoner-expansive spin mode, applied to the m-channel residual before
    mixing. Residual-space, so the fixed point is unchanged. Opt-in / off by
    default — near-identity in the insulating limit but wash-to-unstable on FM
    metals with sparse Fermi sampling (see test_nc_spin_precond_convergence),
    never auto-enabled. Sets ``mixer.extra_precond`` (``None`` in the insulating
    limit, where ``build_stoner_precond`` returns nothing)."""
    from gradwave.scf.spin_precond import build_stoner_precond

    sp = build_stoner_precond(
        system, coeffs_list_s, eigs_s, mu, SCHEMES[smearing], width,
        rho_out_s[0] + rho_out_s[1], rho_out_s[0] - rho_out_s[1], xc,
        dist_ctx=dist_ctx)
    if sp is None:
        mixer.extra_precond = None
        return

    def _spin_pc(rvec, _sp=sp, _ng=ng):
        out = rvec.clone()
        out[_ng : 2 * _ng] = _sp.apply(rvec[_ng : 2 * _ng])
        return out

    mixer.extra_precond = _spin_pc


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
    mixing_scheme: str | None = None,  # pulay | broyden | johnson. None →
    # magnetic-aware auto: johnson for nspin==2 (the spin channel near the
    # Stoner instability limits magnetic convergence — Fe FM 15→14, and johnson
    # is the robust scheme there; see mixing.py + the Phase-2 sweep), pulay
    # otherwise. An explicit scheme always wins.
    kerker: bool | None = None,
    diago_tol: float = 1e-9,
    verbose: bool = True,
    nspin: int = 1,
    start_mag: list[float] | None = None,  # initial moment fractions: per-species OR per-atom
    tot_magnetization: float | None = None,  # fix the spin moment M=N↑−N↓ (nspin=2, no smearing);
    # per-channel integer occupations instead of a shared-μ fill — see
    # shared_fermi_occupations. None (default) uses smearing to find the moment.
    mixed_precision: bool = False,  # opt-in fp32 draft (see note at resolution below)
    eigensolver: str = "davidson",  # davidson | chebyshev (NC standard problem only)
    precond: str = "kerker",  # kerker | local_tf (position-dependent TF screening)
    spin_precond: bool = False,  # Stoner m-channel preconditioner (smeared nspin=2
    # only; scf/spin_precond.py) — the physics-informed damping of the Stoner-
    # expansive magnetization mode, applied to the residual before mixing. No-op
    # otherwise; mirrors scf_uspp's spin_precond. Not input-reachable (kwarg only,
    # matching the collinear USPP path — scf.magnetic.spin_precond feeds only the
    # spinor scf_noncollinear driver).
    precond_op: MultipoleKerkerPrecond | None = None,  # callable r→P·r on the density-total
    # block, overriding kerker/local_tf (e.g. a fitted scf.learned_precond.MultipoleKerkerPrecond);
    # must leave the G=0 component untouched and needs no per-iteration state
    mixer_hook: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,  # research probe:
    # called (it, rho_in_vec, rho_out_vec) each step before mixing, e.g. to capture the residual
    # history a preconditioner fit consumes (see scf.learned_precond.response_from_residuals)
    hubbard: list[HubbardManifold] | None = None,  # list[core.hubbard.HubbardManifold] — Dudarev +U
    hub_occ_mix: float = 1.0,  # DFT+U occupation-matrix damping β in (0,1]: n_hub carried into the
    # next iteration's V_U is (1-β)·n_prev + β·n_new. 1.0 (default) = today's raw one-step lag,
    # bit-for-bit. β~0.3 contracts the large-U-on-metal occupation flip-flop (see the manual)
    hub_u_ramp_iters: int = 0,  # DFT+U linear U ramp: U_eff climbs 1/N→full over the first N
    # iterations, then holds; convergence is blocked until it completes so the final energy is at
    # full U. 0 (default) = off. Pairs with hub_occ_mix for stiff large-U metallic +U systems.
    hub_alpha: list[float] | None = None,  # per-site rigid manifold potential α [eV], lin-response
    start_from: _StartFrom = None,  # previous SCFResult (or checkpoint view) on the SAME FFT grid
    fock: MultiKFockExchange | None = None,  # optional orbital-dep. operator (hybrid Fock exchange)
    dist_ctx: "DistKContext | None" = None,  # k-point-sharded distributed SCF (see
    # gradwave.distributed): `system` is THIS RANK's local k-shard; None (default) runs
    # the ordinary single-process path, byte-for-byte unchanged.
    recorder: "SCFRecorder | None" = None,  # per-iteration flight recorder (scf.recorder);
    # None (default) builds a fresh cheap-path recorder — detached, off the autograd graph
    energy_metric: bool = False,  # opt-in energy-metric convergence gate: converge on the
    # residual's exact second-order energy error 1/2<r|K_Hxc|r> < entol instead of
    # rhotol (etol and the stale-solve guard unchanged). The honest criterion for
    # metallic magnets, whose magnetization-channel density residual floors above any
    # reachable rhotol while the (Hartree-dominated) energy error settles far below it.
    # False (default) leaves the density gate bit-for-bit unchanged and costs nothing.
    entol: float = 1e-6,  # eV, the energy-error threshold for energy_metric (see docs)
    boundary: str = "periodic",  # periodic | open_z | open_z_metal — open-boundary
    # (ESM) electrostatics in the z direction (slab geometry; c ⊥ a,b). open_z =
    # vacuum both sides; open_z_metal = metal Dirichlet planes (a capacitor). Adds
    # the differentiable ESM correction to v_eff and the total energy
    # (core/energies/esm.py); no dipole correction, box-independent surfaces.
    esm_bias: float = 0.0,  # applied capacitor bias [V] for boundary="open_z_metal"
    target_mu: float | None = None,  # constant-potential (grand-canonical) SCF: hold
    # the Fermi level µ [eV] fixed and let the electron count N float. Requires a
    # smearing scheme and boundary="open_z_metal" (the plates source/sink the charge).
    # None (default) runs the ordinary fixed-N SCF.
) -> SCFResult:
    # `fock`, when given, adds an orbital-dependent operator to the Hamiltonian
    # each SCF step (a hybrid functional's Fock exchange). It must expose
    # `rebuild(coeffs_b_s, occ_s, system) -> (apply_delta_s, e_fock)`, returning a
    # per-spin list of callables (nk,nb,npw)->(nk,nb,npw) added to h.apply and the
    # exchange energy scalar. Like DFT+U, the operator lags one iteration (built
    # from the previous step's orbitals) and converges as the density does; the
    # matching semilocal-exchange down-scaling lives in the passed-in `xc`.
    grid, spheres = system.grid, system.spheres
    vol = grid.volume
    nk, nb = len(spheres), system.nbands
    mixing_scheme = _resolve_mixing_scheme(mixing_scheme, nspin)
    _validate_scf_args(
        system, nspin, eigensolver, smearing, mixing_scheme, precond, tot_magnetization
    )
    if target_mu is not None:
        # Constant-potential (grand-canonical) SCF: the electron count floats to
        # hold µ, so the cell is charged — the metal plates must source/sink the
        # charge (a vacuum ESM or a periodic cell cannot), and the Fermi edge must
        # be broadened by a smearing.
        if boundary != "open_z_metal":
            raise ValueError(
                "target_mu (constant-µ) requires boundary='open_z_metal' — the "
                "capacitor plates carry the floating counter-charge")
        if smearing == "none":
            raise ValueError("target_mu (constant-µ) requires a smearing scheme")
    if hubbard:
        validate_hubbard_conv(hub_occ_mix, hub_u_ramp_iters)
    if dist_ctx is not None:
        if fock is not None:
            raise NotImplementedError(
                "distributed (dist_ctx) SCF does not yet support hybrid Fock "
                "exchange — it couples orbitals across k in ways beyond the "
                "density/energy reduction implemented here (see "
                "gradwave.distributed's module docstring). DFT+U IS "
                "supported: the Hubbard occupation matrix is a k-extensive "
                "sum reduced the same way as the density (see "
                "_hubbard_occ_update)."
            )
        # IBZ symmetry needs no special handling here: the rho_symmetrizer is
        # k-set-independent and is applied to the density AFTER the cross-rank
        # all_reduce below, so every rank symmetrizes the identical global
        # density (see gradwave.distributed's module docstring).
        if start_from is not None:
            # A relax/EOS warm start hands the previous, reassembled FULL-mesh
            # result here; `system` is only this rank's k-shard, so slice the
            # per-k orbital seed down to [k_start, k_end) or the seed's k-count
            # check silently cold-starts every orbital (see shard_start_from).
            from gradwave.distributed import shard_start_from

            start_from = shard_start_from(start_from, dist_ctx)
    kerker = _resolve_kerker(kerker, smearing, grid)

    rho_s = _seed_density(system, nspin, start_from, start_mag, grid, vol)

    # MixLayout owns the packed-vector structure (density-sphere channels in
    # the (total, magnetization) basis; no becsum blocks on the NC path).
    # Spin mixing runs in that basis: Kerker damps long-wavelength charge
    # sloshing on the TOTAL block only — applying it to both channels would
    # freeze the per-channel electron counts (the G=0 Kerker zero forbids
    # charge transfer between spin channels)
    layout = MixLayout(grid, nspin, [])
    mixer, tf_precond = _build_mixer_precond(
        grid,
        nspin,
        layout,
        mixing_scheme,
        mixing_alpha,
        mixing_history,
        kerker,
        precond,
        precond_op,
        float_charge=target_mu is not None,
    )

    # flight recorder (scf.recorder): cheap per-iteration diagnostics, always
    # collected in memory, detached and off the autograd graph. Defaults on; a
    # caller may pass a preconfigured recorder. Noncollinear is out of scope.
    if recorder is None:
        from gradwave.scf.recorder import SCFRecorder

        seed_mom = (
            float((rho_s[0] - rho_s[1]).abs().mean()) * vol if nspin == 2 else 0.0
        )
        recorder = SCFRecorder(grid.g2, nspin=nspin, seed_moment=seed_mom)
    recorder.set_mixing(
        scheme=mixing_scheme, alpha=mixing_alpha, kerker=bool(kerker), history=mixing_history
    )
    # per-iteration primitive-op tally (FFT/eigh/H-apply) for the recorder; the
    # loop snapshots the delta each outer iteration and restores the default-off
    # state on exit (finally, below).
    opcount.enable()

    from gradwave.core.batch import projectors_b

    device = system.positions.device
    bk = system.batch
    # System.batch is Optional only to let System.to()/tests build a partial
    # instance; every System that reaches scf() came from setup_system(),
    # which always fills it via build_batched() (never returns None).
    assert bk is not None
    # Certified coarse-box local apply for the loose-tolerance iterations
    # (see CoarseDraft / _solve_bands' `coarse`). Default OFF: measured on
    # diamond Si (ecut 12 Ry, 18³→14³ box), the certification never passes —
    # the ℓ1 tail bound is ~15 Ha and even the ACTUAL per-band perturbation
    # ‖(H_draft−H)ψ‖ is 3.7e-3 Ha on converged states, vs the 0.1·tol_eff =
    # 1e-4 Ha it must clear (tol_eff's first-iteration value is 1e-3). The
    # error is set by V_eff's near-core Fourier tail, which does not shrink
    # with system size, so a 0.75 box can never certify; a box big enough to
    # certify saves too little volume to pay. Kept behind the hatch
    # (GRADWAVE_COARSE_DRAFT=on) for larger-box experiments only.
    coarse_cd = None
    if os.environ.get("GRADWAVE_COARSE_DRAFT", "off").strip().lower() == "on":
        from gradwave.core.batch import build_coarse_draft

        coarse_cd = build_coarse_draft(bk, grid.shape)
    # Opt-in, NOT auto: it drafts only the *early* Davidson iterations in fp32
    # and re-polishes in fp64 below MP_CROSSOVER, so it pays only when those
    # early solves are compute-bound. The RTX 3050 battery
    # (benchmarks/solver_battery/results/mixed_precision/{,cuda/}) shows the win
    # is Davidson-only and size-dependent: insulators win at every size
    # (Si2/MgO/Si16 1.16-1.34×), but metals REGRESS while small and launch-bound
    # (Al/Cu/Fe 0.76-0.97×) and only cross into a win once the grid is large
    # enough to be compute-bound (Cu8 supercell 1.23×, -5 iters). LOBPCG never
    # wins (its fp64 polish dominates); best case tops out ~1.35× because the
    # polish keeps most work fp64. Callers enable it per system.
    mp_crossover = MP_CROSSOVER  # fp64 once the diago tolerance drops below this

    # frozen projector matrices (positions fixed during SCF)
    projs_b = projectors_b(bk, system.positions)

    # DFT+U: frozen atomic-orbital projectors; the per-spin occupation matrices
    # are recomputed from the orbitals each iteration (like the density) and
    # lag one step into V_U — they converge as the density does.
    hub = hub_q = None
    n_hub_s = None
    if hubbard:
        from gradwave.core.hubbard import build_hubbard_projectors, hubbard_projectors

        hub = build_hubbard_projectors(system, hubbard)
        hub_q = hubbard_projectors(hub, system.positions)  # phased (positions fixed)
        # _hubbard_occ_update always returns BOTH spin channels (nspin=1 splits
        # [0,2] into two equal halves), so seed the lagged/damping-target matrix
        # with two channels too — otherwise the occ_mix zip length-mismatches on
        # the first iteration for nspin=1. Only n_hub_s[0] is read for nspin=1.
        n_hub_s = [
            [torch.zeros(s["dim"], s["dim"], dtype=CDTYPE, device=device) for s in hub.sites]
            for _ in range(2)
        ]

    vloc_g = local_potential_g(
        system.positions,
        system.species_index,
        system.vloc_tables,
        grid.g_cart,
        vol,
        vloc_atom=system.vloc_atom,
    )
    vloc_r = local_potential_r(system, vloc_g)

    # E_ewald depends only on the (frozen) ionic positions — constant across the
    # SCF loop, so build it once here and thread it into the per-iteration energy
    # assembly instead of rebuilding the image/G lists and the (na,na,nR) pair
    # tensor every step.
    e_ewald = ewald_energy(system.positions, system.charges, grid.cell)

    # initial orbitals: lowest-kinetic plane waves, reusing previous orbitals
    # (QE wfc-extrapolation analogue) when start_from carries compatible ones
    coeffs_b_s = _seed_orbitals(nk, nb, bk, nspin, device, start_from)

    e_free_prev, converged, history = None, False, []
    # Bound before the loop purely so ty can see these names as always defined
    # after it (the loop runs `for it in range(1, max_iter + 1)`, and max_iter
    # is always >= 1 in practice -- never surfaced as a user knob below 1, see
    # SCFParams/scf() callers); the placeholders are overwritten on the loop's
    # first pass every real invocation.
    it = 0
    e_free = 0.0
    de = float("nan")
    res_norm = float("nan")
    drho_scf: torch.Tensor | None = None
    energies: EnergyBreakdown | None = None
    coeffs_list_s: list[list[torch.Tensor]] | None = None
    eigs_s = [torch.zeros(nk, nb, dtype=RDTYPE, device=device) for _ in range(nspin)]
    occ_s = [torch.zeros(nk, nb, dtype=RDTYPE, device=device) for _ in range(nspin)]
    # Global (full-mesh) eigenvalues/occupations under dist_ctx — gathered each
    # iteration below, then substituted into the returned SCFResult so it looks
    # like an ordinary full-mesh run (see the post-loop dist_ctx block). Bound
    # here (same "always defined after the loop" reasoning as it/e_free/etc.
    # above) since they are only reassigned inside `if dist_ctx is not None`.
    eigs_global_s = eigs_s
    occ_global_s = occ_s
    mu, entropy_term = 0.0, torch.zeros((), dtype=RDTYPE, device=device)
    n_float = float(system.n_electrons)  # floats each iteration under target_mu
    veff_s = [torch.zeros(grid.shape, dtype=RDTYPE, device=device) for _ in range(nspin)]

    # hybrid Fock exchange: the operator lags one iteration (built from the
    # previous step's orbitals), like the DFT+U occupation matrices above.
    fock_apply_s = None
    e_fock = torch.zeros((), dtype=RDTYPE, device=device)

    # meta-GGA: the per-channel kinetic-energy density τ_σ and its generalized-KS
    # operator −½∇·(v_τσ∇ψ_σ). Like Fock/DFT+U, τ is rebuilt from the orbitals
    # each iteration and lags one step. Bootstrap τ from the seed orbitals
    # (rough, refined immediately) so iteration 1 has a valid τ for the
    # τ-dependent v_xc; the energy each iteration uses the current orbitals' τ.
    # tau_list is per-spin (length nspin); the nspin=1 potential/energy sites
    # take tau_list[0], the nspin=2 sites take the whole list.
    tau_list = _bootstrap_tau(xc, coeffs_b_s, system, nspin, nk, nb, bk, grid, vol, device)

    from gradwave.symmetry import CollinearMagneticSymmetrizer

    collinear_mag = isinstance(system.rho_symmetrizer, CollinearMagneticSymmetrizer)

    # Static across the loop (the mesh doesn't change), so gathered once here
    # rather than every iteration alongside the eigenvalues.
    kweights_global = system.kweights
    if dist_ctx is not None:
        from gradwave.distributed import gather_cat

        kweights_global = gather_cat(system.kweights, dist_ctx)

    if verbose:
        _ec = getattr(system, "ecut", None)
        _ecs = f"ecut {_ec / RY_EV:.0f} Ry · " if _ec else ""
        print(
            f"SCF  {len(system.positions)} atoms · {len(system.kweights)} k(IBZ)"
            f" · {system.nbands} bands · {_ecs}grid "
            f"{'×'.join(str(n) for n in grid.shape)} · nspin {nspin}"
            f" · {system.kweights.device}",
            flush=True,
        )

    for it in range(1, max_iter + 1):
        t_it = time.perf_counter()
        _op_prev = opcount.snapshot()      # per-iteration primitive-op baseline
        _t_eig_s = 0.0                      # eigensolve wall this iteration
        rho_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
        if tf_precond is not None:
            tf_precond.set_density(rho_tot)
        tau_arg = None if tau_list is None else (tau_list[0] if nspin == 1 else tau_list)
        veff_s = effective_potentials(system, xc, rho_s, vloc_r, tau=tau_arg,
                                      boundary=boundary, esm_bias=esm_bias)

        # meta-GGA generalized-KS operator: v_τσ = ∂e_xc/∂τ_σ from the current
        # (ρ, τ), applied additively as −½∇·(v_τσ∇ψ_σ) per spin in the H-apply.
        metagga_apply_s = _build_metagga_apply(
            xc, rho_s, rho_tot, tau_list, system, nspin, bk, grid
        )

        # adaptive diagonalization tolerance, quadratic schedule (see
        # common.adaptive_diago_tol). Warm starts skip the loose first solve
        # (it would floor the density residual at eigensolver noise), but NOT
        # all the way to diago_tol: after an ionic move the seed orbitals are
        # stale and one full-precision Davidson against the new H is slower
        # than letting the schedule tighten from 1e-6 (measured on diamond
        # relax: 61 s tight vs 47 s baseline)
        tol_eff = adaptive_diago_tol(
            it,
            history,
            diago_tol,
            system.n_electrons,
            schedule="quadratic",
            first_tol=1e-3 if start_from is None else 1e-6,
        )
        use_low = mixed_precision and tol_eff > mp_crossover
        cdtype = CDTYPE_LOW if use_low else CDTYPE
        t_solve = bk.t.to(RDTYPE_LOW) if use_low else bk.t
        # DFT+U U-ramp factor for THIS iteration; 1.0 when the ramp is off. The
        # same u_scale scales the V_U D-matrix (here) and the E_U energy
        # (_hubbard_occ_update below), so energy and potential stay at one U.
        u_scale = hubbard_u_ramp_scale(it, hub_u_ramp_iters)
        _t_eig0 = time.perf_counter()
        for sp in range(nspin):
            fock_sp = fock_apply_s[sp] if fock_apply_s is not None else None
            mgga_sp = metagga_apply_s[sp] if metagga_apply_s is not None else None
            n_hub_sp = None
            if hub is not None:
                # n_hub_s is set together with hub (the `if hubbard:` block
                # above, and refreshed in lockstep by _hubbard_occ_update
                # below), so it's never None when hub isn't.
                assert n_hub_s is not None
                n_hub_sp = n_hub_s[sp]
            eigs_s[sp], coeffs_b_s[sp] = _solve_bands(
                veff_s[sp],
                coeffs_b_s[sp],
                bk,
                grid.shape,
                projs_b,
                hub,
                hub_q,
                n_hub_sp,
                hub_alpha,
                fock_sp,
                mgga_sp,
                eigensolver,
                tol_eff,
                use_low,
                cdtype,
                t_solve,
                device,
                u_scale,
                coarse=coarse_cd,
            )
        _t_eig_s = time.perf_counter() - _t_eig0

        occ_s, mu, entropy_term, n_float, eigs_global_s, occ_global_s = _fermi_occupations(
            eigs_s, system, smearing, width, nspin, device,
            target_mu=target_mu, tot_magnetization=tot_magnetization,
            dist_ctx=dist_ctx, kweights_global=kweights_global)

        # hybrid Fock: rebuild the exchange operator from the fresh orbitals
        # (used next iteration) and its energy (used in this iteration's total).
        if fock is not None:
            fock_apply_s, e_fock = fock.rebuild(coeffs_b_s, occ_s, system)

        # DFT+U occupation matrices from the fresh orbitals; E_U (Dudarev).
        # n_hub_s on entry is the PREVIOUS iteration's matrix (damping target);
        # the update returns the mixed carry-forward and the fresh-U-scaled E_U.
        n_hub_s, e_hub = _hubbard_occ_update(
            hub, hub_q, coeffs_b_s, occ_s, system, nspin, device, dist_ctx,
            n_hub_prev=n_hub_s, occ_mix=hub_occ_mix, u_scale=u_scale,
        )

        rho_out_s, rho_tot_out = _output_density(
            coeffs_b_s, occ_s, system, bk, grid, vol, nspin,
            dist_ctx=dist_ctx, collinear_mag=collinear_mag)

        # meta-GGA: rebuild τ_σ from the fresh orbitals — this iteration's energy
        # uses it, and it lags into next iteration's v_τ (like the Fock and DFT+U
        # rebuilds above). No symmetrization: τ is a scalar orbital field that
        # inherits the crystal symmetry through the density path.
        if xc.needs_tau:
            from gradwave.core.metagga import tau_b

            tau_list = [
                tau_b(coeffs_b_s[sp], occ_s[sp], system.kweights, bk, grid.shape, vol)
                for sp in range(nspin)
            ]

        # energy at (orbitals, rho_out); the per-k trimmed coeff views are reused
        # for the SCFResult on the final iteration.
        energies, coeffs_list_s = _assemble_scf_energies(
            system,
            xc,
            grid,
            vol,
            spheres,
            nk,
            nspin,
            coeffs_b_s,
            occ_s,
            rho_tot_out,
            rho_out_s,
            tau_list,
            entropy_term,
            e_ewald,
            vloc_g,
            e_hub,
            e_fock,
            projs_b,
            boundary,
            esm_bias,
        )
        if dist_ctx is not None:
            from gradwave.distributed import all_reduce_

            # Kinetic and nonlocal (projector) energy are sums over k, computed
            # above from this rank's local shard only. Every other term
            # (Hartree, XC, local pseudopotential, Ewald, entropy) is a
            # function of the ALREADY-global density/eigenvalues, so it is
            # identical on every rank without further communication.
            energies.kinetic = all_reduce_(energies.kinetic, dist_ctx)
            energies.nonlocal_ = all_reduce_(energies.nonlocal_, dist_ctx)
        e_free = float(energies.free_energy)

        rho_in_vec, rho_out_vec, res_norm, drho_scf, de = _scf_residual_and_record(
            layout,
            rho_s,
            rho_out_s,
            rho_tot,
            rho_tot_out,
            mixer_hook,
            it,
            e_free,
            e_free_prev,
            t_it,
            history,
            nspin,
            vol,
            verbose,
        )

        # energy-metric gate (opt-in): the residual's exact second-order energy
        # error 1/2<r|K_Hxc|r>, per-channel (charge/magnetization). Computed only
        # when selected, so the default density-gate path is bit-for-bit unchanged
        # and pays nothing (one f_xc HVP per iteration otherwise). The
        # Harris-Foulkes/KS gap rides along as the zero-machinery bracket of the
        # same error, skipped when an orbital-dependent term (Hubbard/Fock/
        # meta-GGA) would need extra double-counting terms.
        e_metric = e_metric_charge = e_metric_mag = e_hf_gap = None
        if energy_metric:
            from gradwave.postscf._response import kernel_energy_error

            r_s = [rho_out_s[sp] - rho_s[sp] for sp in range(nspin)]
            e_metric, e_metric_charge, e_metric_mag = kernel_energy_error(
                grid, xc, r_s, rho_s, system.rho_core, nspin
            )
            if hub is None and fock is None and not xc.needs_tau:
                e_hf_gap = _harris_foulkes_gap(
                    system, xc, rho_s, eigs_s, occ_s, e_free, e_ewald,
                    entropy_term, nspin
                )

        # flight recorder: cheap detached per-iteration metrics. drho_scf is the
        # total real-space residual already formed above; history[-1]["t"] is the
        # loop's own timing for this iteration (no extra sync).
        recorder.record(
            it=it,
            free_energy=e_free,
            dE=de,
            res_norm=res_norm,
            t_iter=float(history[-1]["t"]),
            drho_r=drho_scf,
            eigs=eigs_s,
            fermi=mu,
            entropy=float(entropy_term),
            mag_abs=(float((rho_out_s[0] - rho_out_s[1]).abs().mean()) * vol)
            if nspin == 2
            else None,
            e_metric=e_metric,
            e_metric_charge=e_metric_charge,
            e_metric_mag=e_metric_mag,
            e_hf_gap=e_hf_gap,
            subspace_size=mixer.subspace_size,
            op_counts=opcount.since(_op_prev),
            t_eig_s=_t_eig_s,
        )

        # Block convergence until the U-ramp reaches full U (u_scale==1.0), so
        # the reported final energy is never at a partial, ramped U_eff.
        ramp_done = hub_u_ramp_iters <= 0 or it >= hub_u_ramp_iters
        if ramp_done and convergence_gate(de, res_norm, tol_eff, etol, rhotol, diago_tol,
                                          energy_error=e_metric, entol=entol):
            converged = True
            rho_s = rho_out_s
            break

        e_free_prev = e_free
        if spin_precond and nspin == 2 and smearing != "none":
            _apply_spin_precond(mixer, system, coeffs_list_s, eigs_s, mu,
                                smearing, width, rho_out_s, xc, dist_ctx, layout.ng)
        # (total, mag) → per-channel r-space densities (MixLayout.unpack)
        rho_s, _ = layout.unpack(mixer.step(rho_in_vec, rho_out_vec))

    # Band-count guard (collinear nspin=2). default_nbands sizes the per-channel
    # band count from the PARAMAGNETIC ceil(N/2) with no moment dependence, so a
    # sizable spin moment can leave the majority channel with more occupied
    # states than bands — the Fermi solver then cannot place all majority
    # electrons and the electron count / free energy come out wrong. Occupation
    # left in the highest band is the direct symptom (occ_s is per-state in
    # [0,1] on the nspin=2 path); warn to raise nbands. Cheap: one reduction.
    if nspin == 2:
        top_occ = max(float(o[:, -1].max()) for o in occ_s)
        if top_occ > 1e-3:
            logger.warning(
                "collinear nspin=2: the highest band (of %d) carries occupation "
                "%.3g — the majority spin channel is not fully accommodated, so "
                "the converged electron count / free energy may be wrong. "
                "default_nbands sizes bands from the paramagnetic ceil(N/2) with "
                "no moment dependence; pass an explicit larger nbands covering "
                "ceil((N+|M|)/2), e.g. setup_system(..., nbands=...).",
                nb, top_occ,
            )

    if not converged:
        logger.warning(
            "SCF did NOT converge in %d iterations: F=%+.10f eV, dE=%.3e, "
            "|drho|=%.3e (etol=%.1e, rhotol=%.1e)",
            it,
            e_free,
            de,
            res_norm,
            etol,
            rhotol,
        )
    if verbose:
        _tag = "converged" if converged else "NOT CONVERGED"
        _extra = ""
        if nspin == 2:
            _md = rho_s[0] - rho_s[1]
            _extra = f" · m = {float(_md.mean()) * vol:+.4f} muB"
        _fm = "" if mu is None else f" · Fermi = {mu:.4f} eV"
        print(f"SCF {_tag} in {it} iterations · F = {e_free:+.10f} eV{_fm}{_extra}", flush=True)

    opcount.disable()   # restore the default-off tally state after the run
    rho_tot_final = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    # The loop above always runs (max_iter >= 1 in practice) so these are real
    # values from the last iteration by the time we get here, never the
    # pre-loop placeholders.
    assert energies is not None and coeffs_list_s is not None
    if dist_ctx is not None:
        from gradwave.distributed import gather_list_cat

        # Reassemble a normal, full-mesh SCFResult: gather this rank's local
        # per-k coefficients into the global per-k list, and substitute the
        # GLOBAL eigenvalues/occupations (already gathered above) and the
        # ORIGINAL unsharded system — a caller sees the same shapes/content a
        # single-process run on the full mesh would have produced.
        coeffs_list_s = [gather_list_cat(coeffs_list_s[sp], dist_ctx) for sp in range(nspin)]
        eigs_s = eigs_global_s
        occ_s = occ_global_s
        # DistKContext.full_system is System | USPPSystem (shared with the
        # USPP driver's dist_ctx) -- this rank's dist_ctx was built by
        # shard_system, so it's always a System here.
        system = cast("System", dist_ctx.full_system)
    if nspin == 1:
        return SCFResult(
            converged=converged,
            n_iter=it,
            energies=energies,
            fermi=mu,
            eigenvalues=eigs_s[0],
            occupations=occ_s[0],
            coeffs=coeffs_list_s[0],
            rho=rho_tot_final,
            v_eff=veff_s[0],
            system=system,
            history=history,
            hub_occ=n_hub_s,
            drho_scf=drho_scf,
            kerker_used=bool(kerker),
            recorder=recorder,
            smearing=smearing,
            width=width,
            boundary=boundary,
            esm_bias=esm_bias,
            n_electrons=n_float,
        )
    m_density = rho_s[0] - rho_s[1]
    return SCFResult(
        converged=converged,
        n_iter=it,
        energies=energies,
        fermi=mu,
        eigenvalues=torch.stack(eigs_s),
        occupations=torch.stack(occ_s),
        coeffs=coeffs_list_s,
        rho=rho_tot_final,
        v_eff=torch.stack(veff_s),
        system=system,
        history=history,
        nspin=2,
        rho_spin=rho_s,
        mag_total=float(m_density.mean()) * vol,
        mag_abs=float(m_density.abs().mean()) * vol,
        hub_occ=n_hub_s,
        drho_scf=drho_scf,
        kerker_used=bool(kerker),
        recorder=recorder,
        smearing=smearing,
        width=width,
        boundary=boundary,
        esm_bias=esm_bias,
        n_electrons=n_float,
    )


def _stack_dij(system: System) -> torch.Tensor:
    """Block-diagonal dij over all atoms (identical across k — take from k=0)."""
    return system.proj_data[0].dij_full
