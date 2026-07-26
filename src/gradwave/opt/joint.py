"""Joint geometry + electronic direct minimization (research prototype, #122).

Instead of nesting an SCF inside every BFGS geometry step (~10 full SCFs per
relaxation), descend simultaneously on (strain, positions, orbital
coefficients): the KS total energy is an explicit, autograd-differentiable
function of all three, and one L-BFGS run drives them together.

    E(ε, s, Z) with  a(ε) = a₀(1+ε)ᵀ,  τ = s·a(ε)  (s fractional),
    C_k = Löwdin(Z_k · P_k)  (orthonormal occupied orbitals per k),
    ρ(r) = Σ_k w_k Σ_b f |ψ_kb(r)|² / Ω.

Scope (deliberate, prototype):
- norm-conserving, nspin=1, fixed integer occupations (insulators only —
  smeared/Aufbau occupations reintroduce level-crossing discontinuities);
- fixed-basis strain parametrization, exactly the Nielsen–Martin convention
  postscf/stress.py uses (integer Miller labels frozen, G = m·B(ε)); an outer
  rebuild loop re-freezes the G-sphere at the relaxed cell so the answer does
  not inherit the initial cell's basis;
- ``use_symmetry=False`` systems (no density symmetrizer on the graph).

Orthonormality is enforced INSIDE the energy via Löwdin (S^{-1/2}) so the
optimizer variables Z are unconstrained; a per-G Teter-style diagonal
preconditioner P_k absorbs the kinetic-energy spread that otherwise makes the
coefficient block badly conditioned (condition ~ ecut/gap).

Only occupied bands are carried (the occupied-subspace energy is invariant
under band rotations, so degenerate/near-crossing *occupied* Ritz values are
harmless — the insulator gap is what protects the subspace itself).

H-application accounting: one "H-apply" = Ĥ acting on one band vector
(≈ 2 sphere↔grid FFTs + one projector contraction). The SCF/Davidson side is
counted exactly by ``count_h_applies`` (patches ``BatchedHamiltonian.apply``).
One joint closure (energy + backward) costs, per band and k, one ψ FFT for ρ
plus its adjoint in backward, i.e. ≈ 1 H-apply per (band, k); the Hartree/XC/
local pieces are band-independent (a handful of dense-box FFTs, amortized).
``JointResult.h_equiv`` uses that 1:1 rate and adds the seed-SCF's exact count.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from gradwave.constants import E2, HBAR2_2M
from gradwave.core.fftbox import g_to_r, r_to_g
from gradwave.core.xc.base import XCFunctional
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.postscf._strain import (
    ewald_strained,
    kinetic_band,
    local_pp_energy,
    nlcc_core_strained,
    strained_dens_sphere,
    strained_kpg,
    strained_phases,
    strained_projector_cols,
)
from gradwave.pseudo.radial_torch import RadialTables
from gradwave.scf.loop import System, scf, setup_system

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- accounting
class HApplyCounter:
    """Counts Hamiltonian applications, in band-vector units."""

    def __init__(self):
        self.count = 0


@contextlib.contextmanager
def count_h_applies():
    """Patch ``BatchedHamiltonian.apply`` to count band-vector applications.

    Yields an ``HApplyCounter`` whose ``count`` is Σ over calls of nk·nb of the
    applied block — the exact number of single-band H·ψ products the
    eigensolvers performed inside the ``with`` block.
    """
    from gradwave.core.batch import BatchedHamiltonian

    counter = HApplyCounter()
    orig = BatchedHamiltonian.apply

    def counting_apply(self, c):
        counter.count += c.shape[0] * c.shape[1]
        return orig(self, c)

    BatchedHamiltonian.apply = counting_apply
    try:
        yield counter
    finally:
        BatchedHamiltonian.apply = orig


# ------------------------------------------------------------- orbital param
def lowdin(z: torch.Tensor) -> torch.Tensor:
    """Orthonormalize the rows of z via Cholesky: C = L⁻¹Z with ZZᴴ = LLᴴ.

    NOT the symmetric (eigh-based) Löwdin frame — deliberately. The energy is
    invariant under which orthonormal frame of span(Z) is returned, but
    autograd through ``eigh`` carries 1/(λᵢ−λⱼ) factors that go NaN the moment
    ZZᴴ has (near-)repeated eigenvalues — which symmetry-degenerate bands
    produce *exactly*, killing the first backward pass (observed on Si at the
    2×2×2 zone-boundary k-points). Cholesky's backward is smooth for any SPD
    matrix regardless of its spectrum; a trace-scaled jitter keeps it SPD when
    a line-search trial drives Z toward rank deficiency.
    """
    s = z @ z.conj().transpose(-2, -1)
    n = s.shape[-1]
    jitter = 1e-13 * torch.diagonal(s.detach(), dim1=-2, dim2=-1).mean().real
    eye = torch.eye(n, dtype=s.dtype, device=s.device)
    ell = torch.linalg.cholesky(s + jitter * eye)
    return torch.linalg.solve_triangular(ell, z, upper=False)


def teter_precond(kpg2: torch.Tensor, ekin_ref: float) -> torch.Tensor:
    """Diagonal coefficient preconditioner p(G) = 1/(1 + T_G/T_ref).

    Multiplying the raw variables columnwise (C = Löwdin(Z·p)) makes L-BFGS's
    implicit identity metric on Z behave like the Teter–Payne–Allan
    preconditioned metric on C, flattening the kinetic-energy spread of the
    plane-wave basis (the main conditioning obstacle to direct minimization).
    """
    return 1.0 / (1.0 + HBAR2_2M * kpg2 / max(ekin_ref, 1e-6))


def _coeffs_from_z(z_params, precond, npws):
    """Orthonormal per-k coefficient list from the raw (real-view) leaves."""
    out = []
    for z, p, npw in zip(z_params, precond, npws, strict=True):
        c = torch.view_as_complex(z) * p[None, :npw]
        out.append(lowdin(c))
    return out


# --------------------------------------------------------------- energy fn
def joint_energy(
    system: System,
    xc: XCFunctional,
    tabs: list[RadialTables],
    eps: torch.Tensor,       # (3,3) strain (symmetrized inside)
    frac: torch.Tensor,      # (na,3) fractional positions
    coeffs: list[torch.Tensor],  # per-k (n_occ, npw_k), orthonormal rows
    occ: torch.Tensor,       # (nk, n_occ) fixed occupations
) -> torch.Tensor:
    """KS total energy [eV] on the joint (ε, s, C) autograd graph.

    Mirrors ``postscf.stress._energy_strained`` (fixed-basis strain, integer
    Miller labels, differentiable radial transforms) but with the electronic
    state LIVE: ρ is rebuilt from ``coeffs`` on the graph and positions enter
    as fractional coordinates riding the strained cell. At ε=0, reference
    fractional positions, and converged orbitals this reproduces the SCF
    total energy (tested to ~1e-6 eV).
    """
    grid = system.grid
    dev = system.positions.device
    shape = grid.shape
    omega0 = grid.volume
    kw = system.kweights

    eps_s = 0.5 * (eps + eps.transpose(-2, -1))
    f_map = torch.eye(3, dtype=RDTYPE, device=dev) + eps_s
    a0 = torch.as_tensor(grid.cell, dtype=RDTYPE, device=dev)
    a_e = a0 @ f_map.T
    b_e = 2.0 * math.pi * torch.linalg.inv(a_e).T
    omega = torch.linalg.det(a_e)
    omega = omega * torch.sign(omega.detach())
    pos_e = frac @ a_e

    mask, m_box, g_sph, _g2, is_g0, q_sph, inv_g2 = strained_dens_sphere(
        grid, b_e, dev)

    # density from the live orbitals, in the fixed-coefficient normalization
    # ρ̃(G)·Ω₀ (electron counts — strain enters only through the 1/Ω scaling)
    rho0 = None
    for ik, sph in enumerate(system.spheres):
        psi = g_to_r(coeffs[ik], sph.flat_idx, shape)  # (nb, n1,n2,n3)
        w = (kw[ik] * occ[ik]).to(RDTYPE)
        contrib = torch.einsum("b,bxyz->xyz", w, psi.real**2 + psi.imag**2)
        rho0 = contrib if rho0 is None else rho0 + contrib
    rho0 = rho0 / omega0  # e/Å³ at the reference cell

    rho_t = (r_to_g(rho0.to(CDTYPE)) * omega0).reshape(-1)[mask]

    # Hartree (G=0 excluded)
    e_h = 0.5 * 4.0 * math.pi * E2 / omega * ((rho_t.abs() ** 2) * inv_g2).sum()

    # local pseudopotential (G=0 carries alpha-Z)
    phases = strained_phases(g_sph, pos_e)
    e_loc = local_pp_energy(tabs, system.species_of_atom, phases,
                            rho_t / omega.to(rho_t.dtype), q_sph, is_g0)

    # XC (density values scale as 1/detJ; NLCC core rebuilt on the graph)
    rho_xc = rho0 * (omega0 / omega)
    if system.rho_core is not None:
        rho_xc = rho_xc + nlcc_core_strained(
            tabs, system.species_of_atom, phases, q_sph, omega, grid, mask)
    sigma = None
    if xc.needs_gradient:
        from gradwave.core.density import sigma_from_rho

        g_box = (m_box @ b_e).reshape(*shape, 3)
        sigma = sigma_from_rho(rho_xc, g_box)
    if xc.needs_tau:
        raise NotImplementedError("meta-GGA joint minimization (τ rebuild)")
    e_xc = xc.energy(rho_xc, omega, sigma, None)

    # kinetic + nonlocal per k on the strained (k+G)
    e_kin = torch.zeros((), dtype=RDTYPE, device=dev)
    e_nl = torch.zeros((), dtype=RDTYPE, device=dev)
    lmax = max((b.l for u in system.upfs for b in u.betas), default=0)
    for ik, sph in enumerate(system.spheres):
        kpg, kpg2 = strained_kpg(sph, b_e)
        c = coeffs[ik]
        o = occ[ik]
        e_kin = e_kin + (kw[ik] * o * kinetic_band(c, kpg2)).sum()
        pd = system.proj_data[ik]
        if pd.dij_full.shape[0] == 0:
            continue
        p = strained_projector_cols(tabs, system.species_of_atom,
                                    pd.atom_index, lmax, kpg, kpg2, omega,
                                    pos_e)
        b_ovl = c @ p.conj().T
        quad = torch.einsum("bi,ij,bj->b", b_ovl.conj(),
                            pd.dij_full.to(b_ovl.dtype), b_ovl).real
        e_nl = e_nl + (kw[ik] * o * quad).sum()

    e_ew = ewald_strained(pos_e, system.charges, a_e, b_e, omega, grid.cell)

    return e_kin + e_h + e_xc + e_loc + e_nl + e_ew


# --------------------------------------------------------- metal free energy
def joint_free_energy(
    system: System,
    xc: XCFunctional,
    tabs: list[RadialTables],
    eps: torch.Tensor,
    frac: torch.Tensor,
    coeffs: list[torch.Tensor],   # per-k (n_bands, npw_k), orthonormal rows
    smearing: str,
    width: float,
    *,
    lmax: int,
    n_inner: int = 5,
    broaden: float | None = None,
):
    """Mermin free energy F = E − σS on the joint (ε, s, C) graph, with
    occupations from a subspace diagonalisation (metals; see opt/_metals.py).

    The occupation solve converges (ρ, V_eff, μ) in a short DETACHED
    self-consistent loop, then rebuilds the subspace Hamiltonian ONCE on the
    LIVE (coeffs, frac) graph and diagonalises it with the degeneracy-robust
    ``eigh`` (:class:`~gradwave.opt._metals.RobustEigh`, Lorentzian-broadened
    ``1/(λ_i−λ_j)`` with ε = smearing width). Keeping the subspace rotation and
    Fermi occupations LIVE makes the force the full Mermin free-energy force,
    closing the frozen-occupation bias (#129), while the base-cell H_sub keeps
    ``dF/dε = Ω·stress`` exact (fixed occupations, no explicit strain term).
    Returns ``(F, occ, mu, eigs)``.
    """
    from gradwave.opt._metals import robust_subspace_occupations

    rot, occ, entropy, eigs, mu = robust_subspace_occupations(
        system, xc, tabs, coeffs, frac, smearing, width, system.n_electrons,
        lmax=lmax, n_inner=n_inner, broaden=broaden)
    # rotate the live orbitals onto the (live) Ritz basis: C' = rotᵀ · C
    coeffs_rot = [torch.einsum("ai,ag->ig", rot[ik], coeffs[ik])
                  for ik in range(len(coeffs))]
    e = joint_energy(system, xc, tabs, eps, frac, coeffs_rot, occ)
    # Grand-potential form: with the occupations LIVE in λ, E − σS alone carries
    # the residual ∂(E−σS)/∂f_i = μ per state, and the live electron count
    # N = Σ_k w_k Σ_i f_i(λ_i) is unconstrained — a spurious particle-number
    # force μ·∂N/∂(positions, orbitals) would contaminate the gradient. The
    # Legendre term −μ(N − N_e) is zero in VALUE at the Fermi solution (Σf = N_e
    # by construction of μ) but its gradient exactly cancels that drift — the
    # actual Mermin variational principle is stationary in this form.
    ne_live = (occ * system.kweights[:, None]).sum()
    mu_t = torch.tensor(mu, dtype=ne_live.dtype, device=ne_live.device)
    f = e + entropy - mu_t * (ne_live - system.n_electrons)
    return f, occ.detach(), mu, eigs


# ------------------------------------------------------------ orbital seeds
def _transfer_coeffs(old_spheres, new_spheres, coeffs):
    """Map per-k coefficients between G-spheres by matching Miller indices.

    Used across basis rebuilds (the plane-wave count changes with the cell);
    unmatched columns are zero-filled and the caller's Löwdin restores
    orthonormality. k ordering must match (same mesh, same reduction).
    """
    out = []
    for sph_o, sph_n, c in zip(old_spheres, new_spheres, coeffs, strict=True):
        lut = {tuple(m): i for i, m in enumerate(sph_o.miller.tolist())}
        idx_new, idx_old = [], []
        for j, m in enumerate(sph_n.miller.tolist()):
            i = lut.get(tuple(m))
            if i is not None:
                idx_new.append(j)
                idx_old.append(i)
        c_new = torch.zeros(c.shape[0], sph_n.npw, dtype=CDTYPE,
                            device=c.device)
        c_new[:, idx_new] = c[:, idx_old]
        out.append(c_new)
    return out


# ------------------------------------------------------------------ driver
@dataclass
class JointResult:
    converged: bool
    energy: float            # eV — joint functional at the final point
    cell: np.ndarray         # (3,3) Å, relaxed
    positions: np.ndarray    # (na,3) Cartesian Å, relaxed
    coeffs: list             # per-k orthonormal occupied orbitals (final basis)
    system: System           # system at the final (rebuilt) cell
    n_closures: int = 0      # energy+gradient evaluations
    h_seed: int = 0          # exact H-applies spent in seed/rebuild SCF passes
    h_equiv: int = 0         # h_seed + n_closures · nk · n_occ (see module doc)
    n_cycles: int = 0        # basis rebuild cycles used
    fmax: float = 0.0        # max |force| at exit [eV/Å]
    smax: float = 0.0        # max |σ| at exit [eV/Å³]
    history: list = field(default_factory=list)  # (closure, E) trace


def _grad_metrics(eps_p, frac_p, z_params, a_e_np, omega0):
    """(fmax [eV/Å], smax [eV/Å³], max coeff grad) from the leaves' grads."""
    if frac_p.grad is None:
        return float("inf"), float("inf"), float("inf")
    # pos = frac @ a_e  →  dE/dpos = dE/dfrac · a_e⁻ᵀ ... force = −dE/dpos
    de_dpos = frac_p.grad.detach().cpu().numpy() @ np.linalg.inv(a_e_np).T
    de_dpos = de_dpos - de_dpos.mean(axis=0, keepdims=True)
    fmax = float(np.abs(de_dpos).max())
    if eps_p.grad is None:  # fix_cell — strain is not a leaf
        smax = 0.0
    else:
        g = eps_p.grad.detach()
        smax = float((0.5 * (g + g.T) / omega0).abs().max())
    cmax = max(float(z.grad.detach().abs().max()) for z in z_params)
    return fmax, smax, cmax


def joint_relax(
    cell: np.ndarray,
    positions: np.ndarray,     # (na,3) Cartesian Å
    species_of_atom: list[int],
    upfs: list,
    xc: XCFunctional,
    *,
    ecut: float,
    kmesh=(1, 1, 1),
    n_occ: int | None = None,
    smearing: str = "none",    # "none" (insulator) | gaussian | fermi-dirac | mp1 | cold
    width: float = 0.1,        # smearing width σ [eV] (metals only)
    n_bands: int | None = None,  # bands carried for metals (default: SCF nbands)
    n_inner: int = 6,          # detached occupation fixed-point iters (metals)
    broaden: float | None = None,  # robust-eigh Lorentzian ε [eV]; default = width
    fmax: float = 0.005,       # eV/Å
    smax: float = 5e-4,        # eV/Å³ (≈ 0.08 GPa)
    ctol: float = 5e-4,        # max |dE/dZ| [eV]
    max_closures: int = 800,
    lbfgs_chunk: int = 25,
    history_size: int = 120,
    seed_scf_iters: int = 3,
    max_rebuilds: int = 3,
    rebuild_tol: float = 2e-4,  # ‖ε‖_max below which no further rebuild
    fix_cell: bool = False,
    device: str = "cpu",
    verbose: bool = False,
) -> JointResult:
    """Jointly relax (cell, positions, orbitals) of an NC insulator or metal.

    Outer loop: freeze the G-sphere at the current cell, L-BFGS the joint
    energy over (ε, fractional positions, preconditioned orbital variables),
    then rebuild the basis at the relaxed cell and repeat until the residual
    strain per cycle is below ``rebuild_tol``. Orbitals cross rebuilds by
    Miller-index transfer; the first cycle seeds them with ``seed_scf_iters``
    loose SCF iterations (counted, exactly, in ``h_seed``).

    ``smearing="none"`` (default): fixed 2 e⁻/band occupations, ``n_occ`` =
    N_e/2, minimising the KS total energy — insulators only.

    ``smearing`` set to a scheme (gaussian, fermi-dirac, mp1, cold): the METAL
    path. ``n_bands ≥ N_e/2`` orbitals are carried, occupations come from a
    subspace diagonalisation fed to the same Fermi/entropy machinery the SCF
    uses (opt/_metals.py), and the object minimised is the Mermin free energy
    ``F = E − σS`` via ``joint_free_energy`` — a SINGLE consistent objective
    over (ε, positions, orbitals) descended by one persistent L-BFGS, exactly
    like the insulator path. The subspace rotation and Fermi occupations stay
    LIVE through the degeneracy-robust ``eigh`` backend
    (:class:`~gradwave.opt._metals.RobustEigh`), so the gradient is the full
    Mermin free-energy gradient — closing the frozen-occupation force bias
    (#129). Works with LDA or PBE ``xc``.
    """
    metal = smearing != "none"
    cell = np.asarray(cell, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    coeffs_init = None
    prev_spheres = None
    h_seed = 0
    n_closures = 0
    history: list = []

    for cycle in range(max_rebuilds + 1):
        system = setup_system(
            cell=cell, positions=positions, species_of_atom=species_of_atom,
            upfs=upfs, ecut=ecut, kmesh=kmesh, use_symmetry=False,
        ).to(device)
        nk = len(system.spheres)
        lmax = max((b.l for u in system.upfs for b in u.betas), default=0)
        if metal:
            n_carry = n_bands if n_bands is not None else int(system.nbands)
            occ = None  # occupations are variational (computed each closure)
        else:
            if n_occ is None:
                if abs(system.n_electrons / 2 - round(system.n_electrons / 2)) > 1e-9:
                    raise ValueError("odd electron count — not an insulator at "
                                     "fixed occupations; pass n_occ explicitly")
                n_occ = int(round(system.n_electrons / 2))
            n_carry = n_occ
            occ = torch.full((nk, n_occ), 2.0, dtype=RDTYPE, device=device)
        tabs = [RadialTables(u, device=device) for u in system.upfs]

        # ---- orbital seed: loose SCF on cycle 0, Miller transfer afterwards
        if coeffs_init is None:
            with count_h_applies() as counter:
                res = scf(system, xc, smearing=smearing, width=width,
                          max_iter=seed_scf_iters, verbose=False,
                          etol=0.0, rhotol=0.0, diago_tol=1e-4)
            h_seed += counter.count
            coeffs_init = [lowdin(c[:n_carry].to(CDTYPE)) for c in res.coeffs]
        else:
            coeffs_init = [lowdin(c) for c in
                           _transfer_coeffs(prev_spheres, system.spheres,
                                            coeffs_init)]

        # ---- preconditioner from the seed's mean band kinetic energy
        ekin_ref = float(np.mean([
            kinetic_band(coeffs_init[ik], system.spheres[ik].kpg2)
            .mean().item() for ik in range(nk)]))
        precond = [teter_precond(sph.kpg2, ekin_ref).to(RDTYPE)
                   for sph in system.spheres]

        # ---- leaves: strain (3,3), fractional positions, raw orbital vars
        a0_np = np.asarray(system.grid.cell, dtype=np.float64)
        frac0 = positions @ np.linalg.inv(a0_np)
        eps_p = torch.zeros(3, 3, dtype=RDTYPE, device=device,
                            requires_grad=not fix_cell)
        frac_p = torch.tensor(frac0, dtype=RDTYPE, device=device,
                              requires_grad=True)
        z_params = []
        for ik in range(nk):
            z0 = coeffs_init[ik] / precond[ik][None, :].to(CDTYPE)
            z_params.append(torch.view_as_real(z0.contiguous())
                            .clone().requires_grad_(True))
        npws = [sph.npw for sph in system.spheres]

        leaves = ([frac_p] if fix_cell else [eps_p, frac_p]) + z_params
        chunk = lbfgs_chunk

        def make_opt(_leaves=leaves, _chunk=chunk):
            return torch.optim.LBFGS(
                _leaves, lr=1.0, max_iter=_chunk,
                history_size=history_size, line_search_fn="strong_wolfe",
                tolerance_grad=0.0, tolerance_change=0.0)

        opt_holder = [make_opt()]

        # loop-scoped state bound by DEFAULT ARGUMENT (each cycle's closure
        # must see its own system/leaves, and B023 flags late binding).
        #
        # The metal objective is the Mermin free energy from ``joint_free_energy``
        # with the subspace rotation/occupations LIVE through the degeneracy-robust
        # eigh (opt/_metals.RobustEigh): a single consistent function of
        # (ε, positions, orbitals), so — unlike the earlier frozen-occupation
        # block-coordinate scheme — one L-BFGS descends it exactly as the
        # insulator path descends E, and the force is bias-free (#129). The
        # curvature memory is kept while chunks make progress and reset when one
        # stalls (metal curvature turns over as the Fermi ensemble shifts; stale
        # pairs then poison strong-Wolfe and freeze the step at zero).
        def closure(_oh=opt_holder, _z=z_params, _p=precond, _n=npws, _sys=system,
                    _tabs=tabs, _eps=eps_p, _frac=frac_p, _occ=occ, _lmax=lmax):
            nonlocal n_closures
            _oh[0].zero_grad()
            # volume-collapse guard: a strong-Wolfe trial step along strain is
            # unbounded and det(1+ε) → 0 sends 1/Ω terms to overflow → NaN
            # grads → permanently NaN parameters. Return a finite penalty that
            # pushes ε back instead; the line search backtracks off it.
            eps_s = 0.5 * (_eps + _eps.transpose(-2, -1))
            detj = torch.linalg.det(
                torch.eye(3, dtype=RDTYPE, device=_eps.device) + eps_s)
            if float(detj.detach()) < 0.2 or float(detj.detach()) > 5.0:
                e = 1e3 * ((detj - 1.0) ** 2 + (eps_s ** 2).sum() + 1.0)
            elif metal:
                coeffs = _coeffs_from_z(_z, _p, _n)
                e = joint_free_energy(_sys, xc, _tabs, _eps, _frac, coeffs,
                                      smearing, width, lmax=_lmax,
                                      n_inner=n_inner, broaden=broaden)[0]
            else:
                coeffs = _coeffs_from_z(_z, _p, _n)
                e = joint_energy(_sys, xc, _tabs, _eps, _frac, coeffs, _occ)
            e.backward()
            n_closures += 1
            history.append((n_closures, float(e.detach())))
            return e

        converged_inner = False
        e_prev = float("inf")
        stalled_once = False
        while n_closures < max_closures:
            opt_holder[0].step(closure)
            e_now = history[-1][1]
            a_e_np = a0_np @ (np.eye(3)
                              + eps_p.detach().cpu().numpy()).T
            f_now, s_now, c_now = _grad_metrics(
                eps_p, frac_p, z_params, a_e_np, system.grid.volume)
            # the metal orbital gradient carries the occupation/rotation response
            # channels, whose broadened-eigh backward and detached inner solve
            # leave a small residual near stationarity; relax the orbital-gradient
            # gate for metals (physical convergence is carried by fmax/smax,
            # which are unaffected). Insulator gate unchanged.
            ctol_eff = max(ctol, 2.5e-3) if metal else ctol
            if verbose:
                print(f"  joint cycle {cycle} closures {n_closures:4d}  "
                      f"E = {e_now:+.8f} eV  fmax {f_now:.2e}  "
                      f"smax {s_now:.2e}  cmax {c_now:.2e}", flush=True)
            if (f_now < fmax and c_now < ctol_eff
                    and (fix_cell or s_now < smax)):
                converged_inner = True
                break
            # stall recovery: a chunk that lowers E by < 1e-9 eV means the
            # strong-Wolfe search rejected every trial step — on metals the
            # curvature pairs go stale as the Fermi ensemble turns over, and a
            # poisoned direction freezes the step at zero. Recovery mirrors what
            # a basis-rebuild cycle does (which observably un-sticks the run)
            # without the re-setup: re-canonicalize the orbital leaves
            # (Z ← C/p with a fresh Teter reference — L-BFGS steps let Z drift
            # in Löwdin's null directions, whose flat/noisy curvature poisons
            # the memory) and start a fresh optimizer. Two stalls in a row —
            # a full recovery that still cannot move — means the residual
            # gradient is at the noise floor of the occupation solve: stop.
            if abs(e_now - e_prev) < 1e-9:
                if stalled_once:
                    break
                stalled_once = True
                with torch.no_grad():
                    coeffs_now = _coeffs_from_z(z_params, precond, npws)
                    ekin_now = float(np.mean([
                        kinetic_band(coeffs_now[ik], system.spheres[ik].kpg2)
                        .mean().item() for ik in range(nk)]))
                    for ik in range(nk):
                        precond[ik] = teter_precond(
                            system.spheres[ik].kpg2, ekin_now).to(RDTYPE)
                        z_new = coeffs_now[ik] / precond[ik][None, :].to(CDTYPE)
                        z_params[ik].copy_(torch.view_as_real(
                            z_new.contiguous()))
                opt_holder[0] = make_opt()
            else:
                stalled_once = False
            e_prev = e_now

        # ---- apply the relaxed strain/positions; decide on a rebuild
        eps_np = 0.5 * (eps_p.detach().cpu().numpy()
                        + eps_p.detach().cpu().numpy().T)
        cell = a0_np @ (np.eye(3) + eps_np).T
        frac_np = frac_p.detach().cpu().numpy()
        positions = frac_np @ cell
        coeffs_init = [c.detach() for c in
                       _coeffs_from_z(z_params, precond, npws)]
        prev_spheres = system.spheres
        strain_step = float(np.abs(eps_np).max())
        logger.info("joint cycle %d: %d closures, max|eps|=%.2e, "
                    "converged=%s", cycle, n_closures, strain_step,
                    converged_inner)
        if (fix_cell or strain_step < rebuild_tol) and converged_inner:
            break
        if n_closures >= max_closures:
            break

    f_final, s_final, _ = _grad_metrics(
        eps_p, frac_p, z_params, cell, system.grid.volume)
    return JointResult(
        converged=converged_inner,
        energy=float(history[-1][1]) if history else float("nan"),
        cell=cell, positions=positions, coeffs=coeffs_init, system=system,
        n_closures=n_closures, h_seed=h_seed,
        h_equiv=h_seed + n_closures * nk * n_carry,
        n_cycles=cycle + 1, fmax=f_final, smax=s_final, history=history,
    )
