"""Stress tensor via autograd through a strain parameterization (Layer A entry).

At SCF convergence the free energy is stationary in (ψ, f), so the fixed-basis
(Nielsen–Martin) stress is the partial derivative of the energy expression at
fixed plane-wave coefficients under a homogeneous strain ε:

    r → (1+ε) r,   a_i → (1+ε) a_i,   τ → (1+ε) τ,
    G_m = m·B → m·B(ε)  (integer Miller labels m fixed),
    Ω → det(1+ε) Ω,     ρ(G) → ρ(G)·Ω₀/Ω(ε)   (fixed coefficients c),

    σ_αβ = (1/Ω) ∂E/∂ε_αβ            (P = −tr σ / 3)

QE's analytic stress uses the same fixed-basis convention, so the two agree
at identical ecut/k/pseudopotentials (the shared basis-set incompleteness —
"Pulay stress" — is not corrected by either code).

Every ε-dependent quantity is rebuilt from integers (Miller indices, k_frac,
image counts) on the autograd graph; radial form factors go through the
differentiable spherical Bessel transforms in pseudo/radial_torch.py. The
smearing entropy has no explicit ε-dependence at fixed occupations and is
omitted. The ε = 0 energy of this expression must reproduce the SCF
breakdown to ~1e-9 eV — tested, and worth asserting when debugging.

Sign convention: σ as returned is +(1/Ω)∂E/∂ε (tension positive); QE prints
the negative of the pressure-like part the same way, so the comparison in the
tests is direct. Units: eV/Å³; stress_kbar() converts.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gradwave.constants import E2
from gradwave.dtypes import RDTYPE
from gradwave.postscf._strain import (
    box_millers,
    ewald_strained,
    kinetic_band,
    local_pp_energy,
    nlcc_core_strained,
    strain_cell,
    strained_dens_sphere,
    strained_kpg,
    strained_phases,
    strained_projector_cols,
)
from gradwave.pseudo.radial_torch import RadialTables

# Backward-compatible private aliases (paw_stress historically imported these
# from here; the implementations now live in postscf._strain).
_box_millers = box_millers
_ewald_strained = ewald_strained

EV_A3_TO_KBAR = 1602.176634  # 1 eV/Å³ = 160.2176634 GPa


def stress_kbar(sigma: torch.Tensor) -> torch.Tensor:
    return sigma * EV_A3_TO_KBAR


def _sigma(rho_box: torch.Tensor, g_box: torch.Tensor) -> torch.Tensor:
    """σ = |∇ρ|² on the strained grid (deferred sigma_from_rho import)."""
    from gradwave.core.density import sigma_from_rho

    return sigma_from_rho(rho_box, g_box)


def stress(res, xc, symmetrize: bool = True) -> torch.Tensor:
    """σ_αβ = (1/Ω) ∂E/∂ε_αβ at the converged SCF point, (3,3) [eV/Å³].

    Collinear nspin=2 is supported (the kinetic/nonlocal sums run per spin
    channel and E_xc uses the per-spin densities, mirroring the SCF energy's
    spin assembly). Scalar-relativistic, no +U (the Hubbard strain term is not
    implemented). NLCC is handled (the core density is rebuilt on the graph).
    """
    system = res.system
    if getattr(system, "is_fr", False):
        raise NotImplementedError("stress for fully-relativistic pseudos not implemented yet")
    if getattr(res, "hub_occ", None) is not None:
        raise NotImplementedError("stress with DFT+U not implemented yet")

    dev = system.positions.device
    eps = torch.zeros(3, 3, dtype=torch.float64, device=dev, requires_grad=True)
    e = _energy_strained(res, xc, eps)
    (grad,) = torch.autograd.grad(e, eps)

    omega0 = system.grid.volume
    sigma = 0.5 * (grad + grad.T) / omega0
    if symmetrize and system.sym is not None:
        sigma = symmetrize_stress(sigma, system.sym, system.grid.cell)
    return sigma


def symmetrize_stress(sigma: torch.Tensor, sg, cell: np.ndarray) -> torch.Tensor:
    """σ ← (1/N) Σ_op S σ Sᵀ with S the Cartesian rotation of each op."""
    a_t = np.asarray(cell, dtype=float).T
    acc = torch.zeros_like(sigma)
    for w_mat in sg.rotations:
        s = torch.as_tensor(
            a_t @ w_mat @ np.linalg.inv(a_t), dtype=sigma.dtype, device=sigma.device
        )
        acc = acc + s @ sigma @ s.T
    return acc / sg.n_ops


def _energy_strained(
    res, xc, eps: torch.Tensor, *, rho=None, coeffs=None, spheres=None
) -> torch.Tensor:
    """The KS energy as a function of strain at fixed coefficients/occupations.

    Also usable with a plain (non-leaf) eps for finite-difference checks.

    The optional ``rho`` (grid density), ``coeffs`` (per-k, on ``spheres``) and
    ``spheres`` override the converged, detached electronic state. They are used
    WITHOUT detaching, so a caller can carry an extra autograd graph through them
    (e.g. a density-matrix perturbation for a discretization-error estimate).
    When omitted the converged detached state is used, i.e. the plain stress.
    """
    system = res.system
    grid = system.grid
    dev = system.positions.device
    rdt = RDTYPE

    nspin = getattr(res, "nspin", 1)
    rho = res.rho.detach() if rho is None else rho  # total ρ↑+ρ↓
    spheres = system.spheres if spheres is None else spheres

    # Normalize orbitals/occupations to a leading spin axis so one code path
    # covers both nspin values (the spin sum assemble_pw_energies uses for the
    # SCF energy, so the analytic stress matches its strain derivative). The
    # optional ``coeffs`` override (stress_error's discretization path) is
    # nspin=1 and carries its own graph, so it is not detached here.
    if coeffs is not None:
        coeffs_s = [coeffs]
    else:
        coeffs_s = (res.coeffs if nspin == 2 else [res.coeffs])
        coeffs_s = [[c.detach() for c in cs] for cs in coeffs_s]
    occ = res.occupations.detach()
    occ_s = occ if nspin == 2 else occ[None]
    rho_spin = [r.detach() for r in res.rho_spin] if nspin == 2 else None

    _f_map, a_e, b_e, omega0, omega, pos_e = strain_cell(
        grid, system.positions, eps)

    # dense-box Miller indices, restricted to the density sphere
    shape = grid.shape
    mask, m_box, g_sph, _g2_sph, is_g0, q_sph, inv_g2 = strained_dens_sphere(
        grid, b_e, dev)

    # fixed density coefficients: ρ̃(G) = ρ(G)·Ω₀  [e]
    from gradwave.core.fftbox import r_to_g

    rho_t = (r_to_g(rho.to(torch.complex128)) * omega0).reshape(-1)[mask]
    rho_g = rho_t / omega.to(rho_t.dtype)

    # ---- Hartree (G=0 excluded)
    e_h = 0.5 * 4.0 * math.pi * E2 / omega * ((rho_t.abs() ** 2) * inv_g2).sum()

    # ---- local pseudopotential (G=0 carries alpha-Z)
    tabs = [RadialTables(u, device=dev) for u in system.upfs]
    phases = strained_phases(g_sph, pos_e)
    e_loc = local_pp_energy(tabs, system.species_of_atom, phases, rho_g,
                            q_sph, is_g0)

    kw = system.kweights

    # ---- XC (density values scale as 1/detJ; NLCC core rebuilt on the graph).
    # nspin=2 uses the per-spin densities and the SpinXC (ρ↑, ρ↓, volume, σ…)
    # signature — the NLCC core is split half/half into the channels, exactly
    # as scf.common.spin_xc_energy assembles the SCF E_xc, so the ε=0 strained
    # energy still reproduces the SCF total.
    #
    # meta-GGA τ is rebuilt on the strain graph (per spin for nspin=2) because
    # unlike ρ (which only scales as 1/Ω) τ = ½Σf|∇ψ|² also picks up the
    # strained (k+G) in ∇ψ — the explicit strain dependence a GGA has not. This
    # is what makes the meta-GGA stress genuinely different from forces.
    core_e = None
    if system.rho_core is not None:
        core_e = nlcc_core_strained(tabs, system.species_of_atom,
                                    phases, q_sph, omega, grid, mask)
    g_box = None
    if xc.needs_gradient:
        g_box = (m_box @ b_e).reshape(*shape, 3)
    if nspin == 1:
        rho_xc = rho * (omega0 / omega)
        if core_e is not None:
            rho_xc = rho_xc + core_e
        sigma_xc = _sigma(rho_xc, g_box) if xc.needs_gradient else None
        tau_xc = (_tau_strained(coeffs_s[0], spheres, b_e, omega, occ_s[0], kw,
                                shape) if xc.needs_tau else None)
        e_xc = xc.energy(rho_xc, omega, sigma_xc, tau_xc)
    else:
        c2 = 0.0 if core_e is None else 0.5 * core_e
        r_u = rho_spin[0] * (omega0 / omega) + c2
        r_d = rho_spin[1] * (omega0 / omega) + c2
        if xc.needs_gradient:
            s_uu, s_dd, s_tt = (_sigma(r_u, g_box), _sigma(r_d, g_box),
                                _sigma(r_u + r_d, g_box))
        else:
            s_uu = s_dd = s_tt = None
        if xc.needs_tau:
            tau_u = _tau_strained(coeffs_s[0], spheres, b_e, omega, occ_s[0],
                                  kw, shape)
            tau_d = _tau_strained(coeffs_s[1], spheres, b_e, omega, occ_s[1],
                                  kw, shape)
        else:
            tau_u = tau_d = None
        e_xc = xc.energy(r_u, r_d, omega, s_uu, s_dd, s_tt, tau_u, tau_d)

    # ---- kinetic + nonlocal, per k (strained k+G from integer Miller + k_frac),
    # summed over spin channels. The strained projector cols are geometry-only
    # (spin-independent), so they are built once per k and reused across spins.
    e_kin = torch.zeros((), dtype=rdt, device=dev)
    e_nl = torch.zeros((), dtype=rdt, device=dev)
    lmax = max((b.l for u in system.upfs for b in u.betas), default=0)
    for ik, sph in enumerate(spheres):
        kpg, kpg2 = strained_kpg(sph, b_e)
        pd = system.proj_data[ik]
        p = None
        if pd.dij_full.shape[0] != 0:
            p = strained_projector_cols(tabs, system.species_of_atom,
                                        pd.atom_index, lmax, kpg, kpg2, omega,
                                        pos_e)
        for sp in range(nspin):
            c = coeffs_s[sp][ik]
            o = occ_s[sp]
            band = kinetic_band(c, kpg2)
            e_kin = e_kin + (kw[ik] * o[ik, : c.shape[0]] * band).sum()
            if p is None:
                continue
            b_ovl = c @ p.conj().T  # (nb, nproj_tot)
            quad = torch.einsum(
                "bi,ij,bj->b", b_ovl.conj(), pd.dij_full.to(b_ovl.dtype), b_ovl
            ).real
            e_nl = e_nl + (kw[ik] * o[ik, : c.shape[0]] * quad).sum()

    # ---- Ewald (integer image/G sets fixed at ε=0, vectors strained)
    e_ew = ewald_strained(pos_e, system.charges, a_e, b_e, omega, grid.cell)

    return e_kin + e_h + e_xc + e_loc + e_nl + e_ew


def _tau_strained(coeffs, spheres, b_e, omega, occ, kw, shape):
    """τ(ε) = ½ Σ_k w_k Σ_n f |∇ψ|² on the strained cell, at fixed coefficients.

    ∇ψ uses the strained (k+G) (strained_kpg), so autograd carries τ's explicit
    strain dependence — the piece the GGA σ path (a function of ρ, which only
    scales as 1/Ω) does not have. Reduces to core.metagga.tau_b at ε=0, so the
    ε=0 strained energy still reproduces the SCF energy for a meta-GGA."""
    from gradwave.core.fftbox import g_to_r

    tau = None
    for ik, sph in enumerate(spheres):
        kpg, _ = strained_kpg(sph, b_e)  # (npw, 3)
        c = coeffs[ik]  # (nb, npw)
        w = kw[ik] * occ[ik, : c.shape[0]]  # (nb,)
        grad2 = None
        for d in range(3):
            gd = (1j * kpg[:, d])[None, :] * c  # i(k+G)_d c  → ∂_d ψ
            psid = g_to_r(gd, sph.flat_idx, shape)  # (nb, n1, n2, n3)
            term = psid.real ** 2 + psid.imag ** 2
            grad2 = term if grad2 is None else grad2 + term
        contrib = 0.5 * torch.einsum("b,bxyz->xyz", w.to(grad2.dtype), grad2)
        tau = contrib if tau is None else tau + contrib
    return tau / omega
