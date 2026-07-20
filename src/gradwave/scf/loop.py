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
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from gradwave.core.density import sigma_from_rho
from gradwave.core.energies.hartree import hartree_potential_g
from gradwave.core.energies.local_pp import local_potential_g
from gradwave.core.energies.total import EnergyBreakdown, total_energy
from gradwave.core.fftbox import r_to_g
from gradwave.core.hamiltonian import build_projector_data
from gradwave.core.xc.base import XCFunctional
from gradwave.dtypes import CDTYPE, CDTYPE_LOW, RDTYPE, RDTYPE_LOW
from gradwave.grids import build_fft_grid, build_gsphere
from gradwave.pseudo.kb import beta_form_factors
from gradwave.pseudo.upf import UPFData
from gradwave.scf.common import (
    MP_CROSSOVER,
    adaptive_diago_tol,
    assemble_pw_energies,
    convergence_gate,
    record_iteration,
    shared_fermi_occupations,
    spin_sigmas,
    spin_xc_energy,
    symmetrize_rho,
    warm_start_densities,
)
from gradwave.scf.guess import sad_density
from gradwave.scf.layout import MixLayout
from gradwave.scf.mixing import PulayMixer
from gradwave.scf.setup_common import (
    _unique_shells,
    build_core_density,
    build_symmetrizer_and_kpoints,
    build_vloc_tables,
    coupled_axes,
    default_nbands,
    find_symmetry_groups,
)


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
    so_beta_tables: list | None = None  # FR pseudos: per-species (nk, nchan, npw_max)
    is_fr: bool = False  # fully-relativistic pseudos (spinor SCF only)
    rho_core: torch.Tensor | None = None  # NLCC core density on the grid [e/Å³]

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
            so_beta_tables=(
                [t.to(device) for t in self.so_beta_tables]
                if self.so_beta_tables is not None else None
            ),
            rho_core=self.rho_core.to(device) if self.rho_core is not None else None,
        )


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
    time_reversal: bool = True,  # False for noncollinear/SOC (TR flips m)
    magmoms=None,  # (na, 3) moment directions → magnetic (Shubnikov) symmetry
) -> System:
    """use_symmetry: reduce k to the IBZ and symmetrize ρ each SCF step.
    Requires an unshifted (Γ-centered) mesh — shifted meshes fall back to
    time-reversal-only reduction. M4's implicit backward requires
    use_symmetry=False (a perturbation breaks the crystal symmetry).

    magmoms (with use_symmetry=True) switches to the MAGNETIC space group of
    that moment configuration: k folds into the magnetic IBZ (unitary ops as
    W⁻ᵀ, anti-unitary g·T ops as −W⁻ᵀ — time_reversal is ignored, the group
    decides) and (ρ, m⃗) are symmetrized over the full Shubnikov group each
    step. Only scf_noncollinear consumes such a system; the collinear loops
    reject it. Directions are what matter — magnitudes only distinguish
    zero from nonzero and same from different.
    """
    cell = np.asarray(cell, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    sym = mag_sym = None
    if use_symmetry and tuple(kshift) == (0, 0, 0):
        sym, mag_sym = find_symmetry_groups(cell, positions, species_of_atom,
                                            symprec, magmoms)

    # equalize only symmetry-COUPLED axes (setup_common.coupled_axes)
    grid = build_fft_grid(cell, ecut, equal_dims=coupled_axes(sym, mag_sym),
                          shape_override=fft_shape)
    # time_reversal=False for magnetic systems (k≢−k); for nonmagnetic runs
    # (incl. nonmagnetic + SOC, where Kramers keeps k≡−k) it stays True
    rho_symmetrizer, kfrac, kw = build_symmetrizer_and_kpoints(
        grid, cell, kmesh, kshift, sym, mag_sym, time_reversal)
    spheres = [build_gsphere(grid, ecut, k) for k in kfrac]

    charges = torch.tensor([upfs[s].z_valence for s in species_of_atom], dtype=RDTYPE)
    n_electrons = float(charges.sum())
    if nbands is None:
        nbands = default_nbands(n_electrons)

    # local potential tables on the dense box, per species (the NC setup
    # keeps the single-|G|-shell guard — see setup_common.build_vloc_tables)
    g_flat = np.sqrt(grid.g2.reshape(-1).numpy())
    uniq, inverse = _unique_shells(g_flat)
    vloc_tables = build_vloc_tables(upfs, uniq, inverse, grid.shape,
                                    guard_single_shell=True)

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
    so_tabs = [torch.zeros(len(spheres), u.n_proj, npw_max, dtype=RDTYPE) for u in upfs] \
        if is_fr else None
    dij_species = [torch.as_tensor(upf.dij, dtype=RDTYPE) for upf in upfs]
    for ik, sph in enumerate(spheres):
        inv_k = inv_q[offs[ik]:offs[ik + 1]]
        beta_tables = [
            torch.as_tensor(ff[:, inv_k], dtype=RDTYPE) for ff in ff_species
        ]
        if is_fr:
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
    rho_core = build_core_density(upfs, species_of_atom, positions, grid,
                                  uniq, inverse)

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
    hub_occ: list | None = None  # DFT+U per-spin occupation matrices [σ][site]
    drho_scf: torch.Tensor | None = None  # last self-consistency residual ρ_out−ρ_in
                                          # (total density) for the SCF-error estimate
    formalism: str = "nc"  # result-type tag shared by all four SCF drivers


def vxc_spin_potential(xc, rho_up, rho_dn, grid):
    """(v↑, v↓, E_xc) via autograd on a SpinXC — GGA terms included."""
    ru = rho_up.detach().clone().requires_grad_(True)
    rd = rho_dn.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        s_uu, s_dd, s_tot = spin_sigmas(ru, rd, xc, grid.g_cart)
        e_xc = xc.energy(ru, rd, grid.volume, s_uu, s_dd, s_tot)
        vu, vd = torch.autograd.grad(e_xc, (ru, rd))
    scale = grid.n_points / grid.volume
    return vu * scale, vd * scale, e_xc.detach()


def local_potential_r(system, vloc_g: torch.Tensor | None = None) -> torch.Tensor:
    """v_loc(r) on the dense grid [eV] — the SCF's local-potential path."""
    grid = system.grid
    if vloc_g is None:
        vloc_g = local_potential_g(system.positions, system.species_index,
                                   system.vloc_tables, grid.g_cart, grid.volume)
    return (torch.fft.ifftn(vloc_g, dim=(-3, -2, -1)) * grid.n_points).real


def effective_potentials(system, xc, rho_s: list, vloc_r: torch.Tensor) -> list:
    """Per-spin v_eff(r) from per-channel densities — THE assembly the SCF
    iterates with. A standalone function (not inlined in the loop) so the
    off-stationarity E↔H consistency gate can test the exact potential the
    solver applies (tests/unit/test_energy_hamiltonian_consistency.py)."""
    grid = system.grid
    nspin = len(rho_s)
    rho_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    v_h_r = (
        torch.fft.ifftn(hartree_potential_g(r_to_g(rho_tot.to(CDTYPE)), grid.g2),
                        dim=(-3, -2, -1))
        * grid.n_points
    ).real
    core = system.rho_core
    if nspin == 1:
        v_xc_r, _ = vxc_potential(xc, rho_tot if core is None else rho_tot + core, grid)
        return [v_h_r + v_xc_r + vloc_r]
    cu2 = None if core is None else 0.5 * core
    v_up, v_dn, _ = vxc_spin_potential(
        xc,
        rho_s[0] if core is None else rho_s[0] + cu2,
        rho_s[1] if core is None else rho_s[1] + cu2,
        grid,
    )
    return [v_h_r + v_up + vloc_r, v_h_r + v_dn + vloc_r]


def _seed_density(system, nspin, start_from, start_mag, grid, vol):
    """Initial per-spin density: warm-start from a previous state (volume-
    rescaled so the electron count is conserved), else SAD — spin-split by
    start_mag for nspin=2."""
    if start_from is not None:
        return warm_start_densities(start_from, nspin, grid, vol,
                                    system.positions.device)
    if nspin == 1:
        return [sad_density(grid, system.positions, system.species_of_atom,
                            system.upfs, system.n_electrons)]
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
    return [
        sad_density(grid, system.positions, system.species_of_atom, system.upfs,
                    n_up, atom_scale=[(1 + m) / 2 for m in mags_at]),
        sad_density(grid, system.positions, system.species_of_atom, system.upfs,
                    n_dn, atom_scale=[(1 - m) / 2 for m in mags_at]),
    ]


def _seed_orbitals(nk, nb, bk, nspin, device, start_from):
    """Initial per-spin orbital guess: an identity block of the lowest-|k+G|²
    plane waves, overwritten by shape-compatible previous orbitals (the QE
    wfc-extrapolation analogue) when start_from carries them."""
    c0 = torch.zeros(nk, nb, bk.npw_max, dtype=CDTYPE, device=device)
    c0[:, torch.arange(nb), torch.arange(nb)] = 1.0
    coeffs_b_s = [c0.clone() for _ in range(nspin)]
    if start_from is not None:
        prev_c = (start_from.get("coeffs") if isinstance(start_from, dict)
                  else getattr(start_from, "coeffs", None))
        if prev_c is not None:
            chans = [prev_c] if nspin == 1 else list(prev_c)
            compat = len(chans) == nspin and all(
                len(ch) == nk and all(
                    ch[ik].shape[0] >= nb
                    and ch[ik].shape[1] == int(bk.npw[ik])
                    for ik in range(nk))
                for ch in chans)
            if compat:
                for sp, ch in enumerate(chans):
                    for ik in range(nk):
                        coeffs_b_s[sp][ik, :, : int(bk.npw[ik])] = (
                            ch[ik][:nb].to(device=device, dtype=CDTYPE))
    return coeffs_b_s


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
    mixed_precision: bool = False,  # opt-in fp32 draft (see note at resolution below)
    eigensolver: str = "davidson",  # davidson | chebyshev (NC standard problem only)
    precond: str = "kerker",  # kerker | local_tf (position-dependent TF screening)
    hubbard=None,  # list[core.hubbard.HubbardManifold] — Dudarev DFT+U corrections
    hub_alpha=None,  # per-site rigid manifold potential α [eV] — linear-response probe
    start_from=None,  # previous SCFResult (or checkpoint view) on the SAME FFT grid
    fock=None,  # optional orbital-dependent operator (hybrid Fock exchange); see below
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
    if nspin not in (1, 2):
        raise ValueError("nspin must be 1 or 2 (noncollinear spin uses "
                         "scf_noncollinear, the spinor SCF)")
    if eigensolver not in ("davidson", "chebyshev"):
        raise ValueError("eigensolver must be 'davidson' or 'chebyshev'")
    if nspin == 2 and smearing == "none":
        raise ValueError("nspin=2 requires smearing (fixed magnetic occupations ambiguous)")
    if system.is_fr:
        raise ValueError("fully-relativistic pseudos require the spinor SCF "
                         "(scf_noncollinear) — SOC has no collinear representation")
    if hasattr(system.rho_symmetrizer, "apply_m"):
        raise ValueError("system was built with magnetic symmetry (magmoms=...) — "
                         "only scf_noncollinear consumes it (anti-unitary ops would "
                         "mis-fold collinear spin channels); rebuild without magmoms")
    if kerker is None:
        # auto policy: metals always; insulators once the cell is large
        # enough that long-wavelength charge sloshing dominates mixing —
        # the sloshing amplification goes like 4πe²χ/G²_min, so switch on
        # when the smallest nonzero |G| drops below ~0.8 Å⁻¹ (L ≳ 8 Å).
        g2_nonzero = grid.g2.reshape(-1)
        g2_min = float(g2_nonzero[g2_nonzero > 1e-12].min())
        kerker = (smearing != "none") or (g2_min < 0.64)

    rho_s = _seed_density(system, nspin, start_from, start_mag, grid, vol)

    # MixLayout owns the packed-vector structure (density-sphere channels in
    # the (total, magnetization) basis; no becsum blocks on the NC path).
    # Spin mixing runs in that basis: Kerker damps long-wavelength charge
    # sloshing on the TOTAL block only — applying it to both channels would
    # freeze the per-channel electron counts (the G=0 Kerker zero forbids
    # charge transfer between spin channels)
    layout = MixLayout(grid, nspin, [])
    mixer = PulayMixer(layout.g2_full, alpha=mixing_alpha,
                       history=mixing_history, kerker=kerker,
                       check_g0=nspin == 1,
                       kerker_mask=layout.kerker_mask if nspin == 2 else None)

    if precond not in ("kerker", "local_tf"):
        raise ValueError("precond must be 'kerker' or 'local_tf'")
    tf_precond = None
    if precond == "local_tf":
        # position-dependent TF screening on the density-total block; capped at
        # the bare-Kerker q0 so a bulk metal is unchanged and only the vacuum is
        # unscreened. set_density() is called with the current n(r) each iter.
        from gradwave.scf.local_tf import LocalTFPrecond
        tf_precond = LocalTFPrecond(grid.g2, grid.shape, layout.mask,
                                    q0_max=mixer.q0)
        mixer.precond_op = tf_precond
        mixer.precond_slice = slice(0, layout.ng) if nspin == 2 else None

    from gradwave.core.batch import BatchedHamiltonian, becp_b, density_b, projectors_b
    from gradwave.solvers.davidson import davidson_batched

    device = system.positions.device
    bk = system.batch
    # Opt-in, NOT auto: benchmarking (RTX 3050) showed the fp32 draft is
    # situational — a clear win only for moderate-grid, many-k, smeared/SOC
    # systems (e.g. GaAs, 1.45×), but a REGRESSION for fixed-occupation
    # insulators (Si8 0.35×: the fp32 draft inflates SCF iterations ~50%) and
    # neutral for metals / very large grids. Callers enable it per system.
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
        n_hub_s = [[torch.zeros(s["dim"], s["dim"], dtype=CDTYPE, device=device)
                    for s in hub.sites] for _ in range(nspin)]

    vloc_g = local_potential_g(
        system.positions, system.species_index, system.vloc_tables, grid.g_cart, vol
    )
    vloc_r = local_potential_r(system, vloc_g)

    # initial orbitals: lowest-kinetic plane waves, reusing previous orbitals
    # (QE wfc-extrapolation analogue) when start_from carries compatible ones
    coeffs_b_s = _seed_orbitals(nk, nb, bk, nspin, device, start_from)

    e_free_prev, converged, history = None, False, []
    eigs_s = [torch.zeros(nk, nb, dtype=RDTYPE, device=device) for _ in range(nspin)]
    occ_s = [torch.zeros(nk, nb, dtype=RDTYPE, device=device) for _ in range(nspin)]
    mu, entropy_term = 0.0, torch.zeros((), dtype=RDTYPE, device=device)
    veff_s = [torch.zeros(grid.shape, dtype=RDTYPE, device=device) for _ in range(nspin)]

    # hybrid Fock exchange: the operator lags one iteration (built from the
    # previous step's orbitals), like the DFT+U occupation matrices above.
    fock_apply_s = None
    e_fock = torch.zeros((), dtype=RDTYPE, device=device)

    def symmetrize(r_out):
        return symmetrize_rho(system.rho_symmetrizer, r_out, grid)

    for it in range(1, max_iter + 1):
        t_it = time.perf_counter()
        rho_tot = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
        if tf_precond is not None:
            tf_precond.set_density(rho_tot)
        veff_s = effective_potentials(system, xc, rho_s, vloc_r)

        # adaptive diagonalization tolerance, quadratic schedule (see
        # common.adaptive_diago_tol). Warm starts skip the loose first solve
        # (it would floor the density residual at eigensolver noise), but NOT
        # all the way to diago_tol: after an ionic move the seed orbitals are
        # stale and one full-precision Davidson against the new H is slower
        # than letting the schedule tighten from 1e-6 (measured on diamond
        # relax: 61 s tight vs 47 s baseline)
        tol_eff = adaptive_diago_tol(
            it, history, diago_tol, system.n_electrons, schedule="quadratic",
            first_tol=1e-3 if start_from is None else 1e-6)
        use_low = mixed_precision and tol_eff > mp_crossover
        cdtype = CDTYPE_LOW if use_low else CDTYPE
        t_solve = bk.t.to(RDTYPE_LOW) if use_low else bk.t
        for sp in range(nspin):
            hub_dij = None
            if hub is not None:
                from gradwave.core.hubbard import hubbard_dmatrix
                dij = hubbard_dmatrix(n_hub_s[sp], hub.sites, hub.nproj, device)
                if hub_alpha is not None:  # rigid manifold probe α·I (linear response)
                    for si, s in enumerate(hub.sites):
                        st, dim = s["start"], s["dim"]
                        dij[st:st + dim, st:st + dim] += hub_alpha[si] * torch.eye(
                            dim, dtype=CDTYPE, device=device)
                # apply convention wants D^T; D is Hermitian so D^T = conj(D)
                hub_dij = dij.conj()
            h = BatchedHamiltonian(bk, grid.shape, veff_s[sp], projs_b,
                                   hub_q=hub_q, hub_dij=hub_dij)
            apply = h.apply
            if fock_apply_s is not None and fock_apply_s[sp] is not None:
                _fa = fock_apply_s[sp]
                def apply(c, _base=h.apply, _f=_fa):
                    return _base(c) + _f(c)
            if eigensolver == "chebyshev":
                from gradwave.solvers.chebyshev import chebyshev_filtered_batched
                dav = chebyshev_filtered_batched(
                    apply, coeffs_b_s[sp].to(cdtype), t_solve, bk.mask,
                    tol=tol_eff)
            else:
                dav = davidson_batched(apply, coeffs_b_s[sp].to(cdtype),
                                       t_solve, bk.mask, tol=tol_eff)
            eigs_s[sp] = dav.eigenvalues.to(RDTYPE)
            c = dav.eigenvectors.to(CDTYPE)
            if use_low:
                # fp32 leaves ‖ψ‖ accurate only to ~1e-6; renormalize in fp64
                # so the density's electron count (ρ at G=0) is conserved to
                # the mixer's tolerance (off-diagonal overlaps don't touch G=0)
                c = c / torch.linalg.norm(c, dim=-1, keepdim=True).clamp_min(1e-30)
            coeffs_b_s[sp] = c

        occ_s, mu, entropy_term = shared_fermi_occupations(
            eigs_s, system.kweights, smearing, width, system.n_electrons,
            nspin, device)

        # hybrid Fock: rebuild the exchange operator from the fresh orbitals
        # (used next iteration) and its energy (used in this iteration's total).
        if fock is not None:
            fock_apply_s, e_fock = fock.rebuild(coeffs_b_s, occ_s, system)

        # DFT+U occupation matrices from the fresh orbitals; E_U (Dudarev).
        # occ_s is per-spin f∈[0,1] when nspin=2; for nspin=1 the [0,2]
        # occupation splits into two equal spin channels.
        e_hub = torch.zeros((), dtype=RDTYPE, device=device)
        if hub is not None:
            from gradwave.core.hubbard import hubbard_energy, occupation_matrices
            if nspin == 2:
                for sp in range(nspin):
                    n_hub_s[sp] = occupation_matrices(
                        hub_q, coeffs_b_s[sp], occ_s[sp], system.kweights, hub.sites)
                e_hub = sum(hubbard_energy(n_hub_s[sp], hub.sites) for sp in range(nspin))
            else:
                n_half = occupation_matrices(
                    hub_q, coeffs_b_s[0], 0.5 * occ_s[0], system.kweights, hub.sites)
                n_hub_s = [n_half, n_half]
                e_hub = 2.0 * hubbard_energy(n_half, hub.sites)

        rho_out_s = [
            symmetrize(density_b(coeffs_b_s[sp], occ_s[sp], system.kweights,
                                 bk, grid.shape, vol))
            for sp in range(nspin)
        ]
        rho_tot_out = rho_out_s[0] if nspin == 1 else rho_out_s[0] + rho_out_s[1]

        # energy at (orbitals, rho_out); per-k trimmed views for the assembly.
        # npw from the CPU-side spheres (int(bk.npw[ik]) is a host sync per k
        # per iteration — the probe counted 36/iteration); ONE becp over the
        # whole batch, then per-k views (calling becp_b inside the per-k
        # comprehension recomputed the full-batch contraction nk times)
        coeffs_list_s = [
            [coeffs_b_s[sp][ik, :, : system.spheres[ik].npw]
             for ik in range(nk)]
            for sp in range(nspin)
        ]
        becps_s = []
        for sp in range(nspin):
            b_all = becp_b(projs_b, coeffs_b_s[sp])
            becps_s.append([b_all[ik] for ik in range(nk)])
        if nspin == 1:
            energies = total_energy(
                coeffs_per_k=coeffs_list_s[0], occ=occ_s[0], kweights=system.kweights,
                spheres=spheres, grid=grid, rho=rho_tot_out, positions=system.positions,
                charges=system.charges, species_index=system.species_index,
                vloc_tables=system.vloc_tables, becp_per_k=becps_s[0],
                dij_full=_stack_dij(system), xc=xc, entropy_term=entropy_term,
                rho_core=system.rho_core,
            )
            energies.hubbard = e_hub
            energies.fock = e_fock
        else:
            rho_g_out = r_to_g(rho_tot_out.to(CDTYPE))
            energies = assemble_pw_energies(
                coeffs_list_s, occ_s, system.kweights, spheres, grid, vol,
                rho_g_out,
                spin_xc_energy(xc, rho_out_s, system.rho_core, vol,
                               grid.g_cart),
                vloc_g, becps_s, _stack_dij(system), system.positions,
                system.charges, entropy_term, nspin, e_hub=e_hub)
            energies.fock = e_fock
        e_free = float(energies.free_energy)

        rho_in_vec = layout.pack(rho_s)
        rho_out_vec = layout.pack(rho_out_s)
        if nspin == 2:  # only the TOTAL is conserved; its G=0 residual must vanish
            tot_res = rho_out_vec[0] - rho_in_vec[0]
            if not torch.isfinite(rho_out_vec).all():
                raise RuntimeError("density diverged (NaN/Inf)")
            if tot_res.abs() >= 1e-8:
                raise ValueError("total G=0 residual nonzero")
        res_norm = float(torch.linalg.norm(rho_out_vec - rho_in_vec)) * vol
        # keep the real-space total-density residual of the last step for the
        # post-SCF convergence-error estimate (ρ_out − ρ_in at this iteration)
        drho_scf = rho_tot_out - rho_tot
        de = record_iteration(history, it, e_free, e_free_prev, res_norm, t_it)
        if verbose:
            mag = ""
            if nspin == 2:
                m = float((rho_out_s[0] - rho_out_s[1]).mean()) * vol
                mag = f"   m = {m:+.4f} muB"
            print(f"  SCF {it:3d}  F = {e_free:+.10f} eV   dE = {de:.3e}   "
                  f"|drho| = {res_norm:.3e}{mag}")

        if convergence_gate(de, res_norm, tol_eff, etol, rhotol, diago_tol):
            converged = True
            rho_s = rho_out_s
            break

        e_free_prev = e_free
        # (total, mag) → per-channel r-space densities (MixLayout.unpack)
        rho_s, _ = layout.unpack(mixer.step(rho_in_vec, rho_out_vec))

    rho_tot_final = rho_s[0] if nspin == 1 else rho_s[0] + rho_s[1]
    if nspin == 1:
        return SCFResult(
            converged=converged, n_iter=it, energies=energies, fermi=mu,
            eigenvalues=eigs_s[0], occupations=occ_s[0], coeffs=coeffs_list_s[0],
            rho=rho_tot_final, v_eff=veff_s[0], system=system, history=history,
            hub_occ=n_hub_s, drho_scf=drho_scf,
        )
    m_density = rho_s[0] - rho_s[1]
    return SCFResult(
        converged=converged, n_iter=it, energies=energies, fermi=mu,
        eigenvalues=torch.stack(eigs_s), occupations=torch.stack(occ_s),
        coeffs=coeffs_list_s, rho=rho_tot_final, v_eff=torch.stack(veff_s),
        system=system, history=history, nspin=2, rho_spin=rho_s,
        mag_total=float(m_density.mean()) * vol,
        mag_abs=float(m_density.abs().mean()) * vol,
        hub_occ=n_hub_s, drho_scf=drho_scf,
    )


def _stack_dij(system: System) -> torch.Tensor:
    """Block-diagonal dij over all atoms (identical across k — take from k=0)."""
    return system.proj_data[0].dij_full
