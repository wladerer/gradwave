"""Blocks genuinely shared between the NC and USPP/PAW SCF loops
(refactor stage 4, deliberately minimal — full loop unification is
deferred until the S=1 overhead is measured AND NC maintenance hurts).
"""

from __future__ import annotations

import time

import torch

from gradwave.core.density import sigma_from_rho
from gradwave.core.fftbox import r_to_g
from gradwave.core.occupations import (
    SCHEMES,
    find_fermi,
    fixed_occupations,
    occupations_and_entropy,
)
from gradwave.dtypes import CDTYPE, RDTYPE

# Mixed precision: the fp32 draft solves run while the adaptive diago
# tolerance is above this; below it every solve is fp64. One constant for all
# drivers (each used to carry its own copy of the same 1e-5).
MP_CROSSOVER = 1e-5


def record_iteration(history, it, e_free, e_free_prev, res_norm, t_it):
    """Append the per-iteration record shared by all four SCF drivers and
    return dE (inf on the first iteration). The schema
    {"iter", "free_energy", "dE", "res", "t"} is consumed by the adaptive
    tolerance schedule, the trust-region logic, and post-SCF diagnostics."""
    de = abs(e_free - e_free_prev) if e_free_prev is not None else float("inf")
    history.append({"iter": it, "free_energy": e_free, "dE": de,
                    "res": res_norm, "t": time.perf_counter() - t_it})
    return de


def convergence_gate(de, res_norm, tol_eff, etol, rhotol, diago_tol):
    """The strict convergence gate shared by all four SCF drivers: energy
    settled AND density residual down AND the adaptive diago tolerance at its
    floor (a loose-solve iteration can fake a small dE/res pair)."""
    return de < etol and res_norm < rhotol and tol_eff <= diago_tol * 1.01


def adaptive_diago_tol(it, history, diago_tol, n_electrons, *, schedule,
                       first_tol=1e-3):
    """Adaptive diagonalization tolerance (QE-style): loose while the density
    is far from self-consistent, tightening with the previous residual.

    schedule="quadratic" (QE's ethr ~ dr2/nelec/10): the collinear drivers'
    choice — a linear schedule floors each iteration's density residual at
    the eigensolver noise, so the tail converges at the schedule's pace
    instead of the mixer's.
    schedule="linear" (0.03·res): the spinor drivers' choice.
    The collinear/spinor divergence is HISTORICAL — preserved exactly here;
    do not unify the schedules without re-measuring both loop families.

    first_tol is the it==1 tolerance target (drivers pass a tighter value on
    warm starts); it is floored at diago_tol like every other iteration."""
    if it == 1:
        return max(diago_tol, first_tol)
    r_prev = history[-1]["res"]
    if schedule == "quadratic":
        return max(diago_tol, min(1e-3, 0.1 * r_prev * r_prev / n_electrons))
    if schedule != "linear":
        raise ValueError("schedule must be 'quadratic' or 'linear'")
    return max(diago_tol, min(1e-3, 0.03 * r_prev))


def symmetrize_rho(rho_symmetrizer, r_out, grid):
    """Round-trip a real-space density through the symmetry averager (via G).

    Returns ``r_out`` unchanged when there is no symmetrizer. Shared by the
    collinear, USPP/PAW, and noncollinear loops, which all applied this exact
    FFT → apply → iFFT round-trip inline.
    """
    if rho_symmetrizer is None:
        return r_out
    sym_g = rho_symmetrizer.apply(r_to_g(r_out.to(CDTYPE)))
    return torch.fft.ifftn(sym_g * grid.n_points, dim=(-3, -2, -1)).real


def spin_sigmas(r_u, r_d, xc, g_cart):
    """(σ_uu, σ_dd, σ_tot) for a GGA, or (None, None, None) for an LDA.

    σ_tot uses the total density r_u + r_d. Callers that need autograd through
    the densities supply leaf tensors and wrap the call in enable_grad.
    """
    if not xc.needs_gradient:
        return None, None, None
    return (
        sigma_from_rho(r_u, g_cart),
        sigma_from_rho(r_d, g_cart),
        sigma_from_rho(r_u + r_d, g_cart),
    )


def warm_start_densities(start_from, nspin, grid, vol, dev):
    """Validated per-spin densities from a previous SCF state: requires the
    SAME FFT grid and spin count. ρ carries a 1/Ω normalization, so the
    channels are rescaled by the volume ratio and the electron count is
    exactly conserved on the new cell. Accepts a result object (attribute
    access — SCFResult/USPPResult) or a checkpoint dict view."""
    def _prev(key, default=None):
        return (start_from.get(key, default)
                if isinstance(start_from, dict)
                else getattr(start_from, key, default))

    prev_grid = _prev("system").grid
    if tuple(prev_grid.shape) != tuple(grid.shape):
        raise ValueError("start_from requires the same FFT grid "
                         f"({tuple(prev_grid.shape)} vs {tuple(grid.shape)})")
    if int(_prev("nspin", 1) or 1) != nspin:
        raise ValueError("start_from nspin mismatch")
    chg = float(prev_grid.volume) / float(vol)
    if nspin == 1:
        return [_prev("rho").detach().to(dev) * chg]
    return [r.detach().to(dev) * chg for r in _prev("rho_spin")]


def spin_xc_energy(xc, rho_out_s, rho_core, vol, g_cart):
    """E_xc for the collinear nspin=2 energy assembly: the NLCC core is split
    half/half into the spin channels; GGA sigmas via spin_sigmas."""
    c2 = 0.0 if rho_core is None else 0.5 * rho_core
    r_u, r_d = rho_out_s[0] + c2, rho_out_s[1] + c2
    s_uu, s_dd, s_tt = spin_sigmas(r_u, r_d, xc, g_cart)
    return xc.energy(r_u, r_d, vol, s_uu, s_dd, s_tt)


def assemble_pw_energies(coeffs_s, occ_s, kweights, spheres, grid, vol,
                         rho_g_out, e_xc, vloc_g, becps_s, dij_full,
                         positions, charges, entropy_term, nspin,
                         e_hub=0.0, e_onec=None):
    """The plane-wave EnergyBreakdown assembly shared by the collinear
    drivers: per-spin kinetic and nonlocal sums (with the BARE D for
    USPP/PAW), total-density Hartree/local, caller-supplied E_xc. e_onec=None
    leaves the PAW one-center field at its (zero) default for the NC path."""
    from gradwave.core.energies.ewald import ewald_energy
    from gradwave.core.energies.hartree import hartree_energy
    from gradwave.core.energies.kinetic import kinetic_energy
    from gradwave.core.energies.local_pp import local_energy
    from gradwave.core.energies.nl_pp import nonlocal_energy
    from gradwave.core.energies.total import EnergyBreakdown

    extra = {} if e_onec is None else {"onecenter": e_onec}
    return EnergyBreakdown(
        kinetic=sum(kinetic_energy(coeffs_s[sp], occ_s[sp], kweights, spheres)
                    for sp in range(nspin)),
        hartree=hartree_energy(rho_g_out, grid.g2, vol),
        xc=e_xc,
        local=local_energy(rho_g_out, vloc_g, vol),
        nonlocal_=sum(nonlocal_energy(becps_s[sp], dij_full, occ_s[sp],
                                      kweights)
                      for sp in range(nspin)),
        ewald=ewald_energy(positions, charges, grid.cell),
        smearing=entropy_term,
        hubbard=e_hub,
        **extra,
    )


def shared_fermi_occupations(eigs_s, kweights, smearing, width, n_electrons,
                             nspin, device):
    """Occupations, Fermi level, and entropy term for per-spin eigenvalue
    stacks with a SHARED Fermi level (both spin channels fill from one μ;
    the spin degeneracy g = 2 for nspin=1, 1 per channel otherwise).

    Returns (occ_s per spin, mu float, entropy_term tensor). smearing
    "none" gives fixed occupations (nspin=1 only — a spin system needs a
    shared Fermi level to exchange charge between channels)."""
    g_spin = 2 if nspin == 1 else 1
    if smearing == "none":
        if nspin != 1:
            raise ValueError("nspin=2 requires smearing (shared Fermi level)")
        occ_s = [fixed_occupations(eigs_s[0], n_electrons)]
        mu = float(eigs_s[0][:, int(n_electrons // 2) - 1].max())
        entropy_term = torch.zeros((), dtype=RDTYPE, device=device)
        return occ_s, mu, entropy_term
    scheme = SCHEMES[smearing]
    eigs_cat = torch.cat(eigs_s, dim=0)  # (nspin·nk, nb)
    kw_cat = torch.cat([kweights] * nspin)
    mu = float(find_fermi(eigs_cat, kw_cat, scheme, width, n_electrons,
                          degeneracy=g_spin))
    # NB: bare torch.tensor(mu) would be float32 and shift N_e by ~1e-7
    mu_t = torch.tensor(mu, dtype=RDTYPE, device=device)
    occ_s, ent = [], torch.zeros((), dtype=RDTYPE, device=device)
    for isp in range(nspin):
        o, s_ent = occupations_and_entropy(eigs_s[isp], mu_t, scheme, width,
                                           degeneracy=g_spin)
        occ_s.append(o)
        ent = ent - width * (g_spin * kweights[:, None] * s_ent).sum()
    return occ_s, mu, ent
