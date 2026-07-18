"""Off-stationarity E↔H consistency gates (Tier-0 verification).

The KS Hamiltonian is BY DEFINITION the Wirtinger derivative of the
total-energy functional: H ψ_nk = ∂E/∂(w_k f_nk ψ*_nk), with
v_eff = δ(E_H + E_xc + E_loc)/δρ. An inconsistency between the assembled E
and the H the SCF iterates is invisible at a self-consistent stationary
point (first-order errors vanish there) and can be invisible against QE
too when both codes share the error — the PAW ddd lesson (see
docs/verification.md). At a RANDOM non-stationary state nothing cancels.

These tests build the total energy exactly as scf.loop does (density_b,
becp_b, total_energy / the spin assembly), differentiate it with autograd
at random orbital coefficients, and demand the gradient equal the SCF's
own BatchedHamiltonian applied to those coefficients:

    grad_c E == 2 · w_k · f_nk · (H c)_nk      (torch grad = 2·∂E/∂c̄)

to near machine precision. A companion test checks the closed-form
potential expressions (Hartree, local) against autograd of the matching
energy expressions — two independent implementations of the same
functional derivative.

The identities hold at ANY cutoff/grid (they are properties of the
discretized functional, not of converged physics), so tiny, fast systems
are enough. Geometries are rattled/low-symmetry on purpose: symmetric
fixtures let error terms cancel by symmetry (the O₂-vs-Si lesson).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.batch import BatchedHamiltonian, becp_b, density_b, projectors_b
from gradwave.core.energies.hartree import hartree_energy, hartree_potential_g
from gradwave.core.energies.kinetic import kinetic_energy
from gradwave.core.energies.local_pp import local_energy, local_potential_g
from gradwave.core.energies.nl_pp import nonlocal_energy
from gradwave.core.energies.total import total_energy
from gradwave.core.fftbox import r_to_g
from gradwave.dtypes import CDTYPE, RDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.common import spin_sigmas
from gradwave.scf.loop import (
    _stack_dij,
    effective_potentials,
    local_potential_r,
    setup_system,
)

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994

torch.set_num_threads(4)


def _si2_rattled(ecut_ry=12.0, kmesh=(2, 1, 1)):
    a = 5.43
    lattice = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    # rattled off the diamond site — P1, no cancellations by symmetry
    positions = np.array([[0.0, 0.0, 0.0], [1.45, 1.27, 1.41]])
    si = parse_upf(FIX / "pseudos" / "Si_ONCV_PBE-1.2.upf")
    return setup_system(lattice, positions, [0, 0], [si],
                        ecut=ecut_ry * RY, kmesh=kmesh)


def _ni_box(ecut_ry=15.0):
    """One NLCC atom (PD_Ni carries a core density) in a box, off-center."""
    ni = parse_upf(FIX / "pseudos" / "PD_Ni_PBE.upf")
    return setup_system(8.0 * np.eye(3), np.array([[0.7, 0.4, 0.2]]), [0], [ni],
                        ecut=ecut_ry * RY, kmesh=(1, 1, 1))


def _random_coeffs(system, nb, seed):
    """Normalized random bands, padded (nk, nb, npw_max) — NOT an eigenstate."""
    bk = system.batch
    gen = torch.Generator().manual_seed(seed)
    c = torch.randn(bk.nk, nb, bk.npw_max, generator=gen, dtype=RDTYPE) \
        + 1j * torch.randn(bk.nk, nb, bk.npw_max, generator=gen, dtype=RDTYPE)
    # smooth envelope so the density is well scaled, then normalize per band
    c = c.to(CDTYPE) / (1.0 + bk.t)[:, None, :]
    c = c * bk.mask[:, None, :]
    return c / torch.linalg.norm(c, dim=-1, keepdim=True)


def _occ(system, nb, values):
    """Fixed fractional occupations, deliberately non-uniform across bands/k."""
    nk = system.batch.nk
    occ = torch.tensor(values, dtype=RDTYPE)[None, :].repeat(nk, 1)
    if nk > 1:  # vary across k too
        occ[1:] = occ[1:].flip(dims=(1,))
    return occ


def _eh_gap_nspin1(system, xc, nb, occ_values, seed=0):
    """max |grad E − 2 w f Hc| / max |2 w f Hc| over the masked region."""
    grid, bk, spheres = system.grid, system.batch, system.spheres
    nk = bk.nk
    kw = system.kweights
    occ = _occ(system, nb, occ_values)
    projs = projectors_b(bk, system.positions)
    dij = _stack_dij(system)

    c = _random_coeffs(system, nb, seed).requires_grad_(True)
    trimmed = [c[ik, :, : int(bk.npw[ik])] for ik in range(nk)]
    rho = density_b(c, occ, kw, bk, grid.shape, grid.volume)
    eb = total_energy(
        coeffs_per_k=trimmed, occ=occ, kweights=kw, spheres=spheres, grid=grid,
        rho=rho, positions=system.positions, charges=system.charges,
        species_index=system.species_index, vloc_tables=system.vloc_tables,
        becp_per_k=[becp_b(projs, c)[ik] for ik in range(nk)], dij_full=dij,
        xc=xc, rho_core=system.rho_core,
    )
    (g,) = torch.autograd.grad(eb.total, c)

    with torch.no_grad():
        veff = effective_potentials(system, xc, [rho.detach()],
                                    local_potential_r(system))[0]
        h = BatchedHamiltonian(bk, grid.shape, veff, projs)
        expected = 2.0 * kw[:, None, None] * occ[:, :, None] * h.apply(c.detach())
    mask = bk.mask[:, None, :]
    gap = ((g - expected) * mask).abs().max()
    return float(gap / expected.abs().max())


def test_grad_energy_equals_hamiltonian_lda():
    from gradwave.core.xc.lda_pw92 import LDA_PW92
    system = _si2_rattled()
    assert _eh_gap_nspin1(system, LDA_PW92(), nb=5,
                          occ_values=[2.0, 2.0, 2.0, 1.2, 0.4]) < 1e-10


def test_grad_energy_equals_hamiltonian_pbe():
    from gradwave.core.xc.pbe import PBE
    system = _si2_rattled()
    assert _eh_gap_nspin1(system, PBE(), nb=5,
                          occ_values=[2.0, 2.0, 2.0, 1.2, 0.4]) < 1e-10


def test_grad_energy_equals_hamiltonian_nlcc():
    """NLCC: ρ_core shifts the XC argument in E and in v_xc — consistently."""
    from gradwave.core.xc.pbe import PBE
    system = _ni_box()
    assert system.rho_core is not None
    assert _eh_gap_nspin1(system, PBE(), nb=8,
                          occ_values=[2.0] * 5 + [1.5, 0.8, 0.3]) < 1e-10


def test_grad_energy_equals_hamiltonian_spin():
    """nspin=2: per-channel identity with the loop's spin energy assembly."""
    from gradwave.core.xc.spin import SpinPBE
    xc = SpinPBE()
    system = _si2_rattled()
    grid, bk, spheres = system.grid, system.batch, system.spheres
    nk, nb = bk.nk, 5
    kw = system.kweights
    occ_s = [_occ(system, nb, [1.0, 1.0, 1.0, 0.6, 0.2]),
             _occ(system, nb, [1.0, 1.0, 0.7, 0.3, 0.1])]
    projs = projectors_b(bk, system.positions)
    dij = _stack_dij(system)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume)

    cs = [_random_coeffs(system, nb, seed).requires_grad_(True) for seed in (1, 2)]
    rho_s = [density_b(cs[sp], occ_s[sp], kw, bk, grid.shape, grid.volume)
             for sp in range(2)]
    rho_tot = rho_s[0] + rho_s[1]
    rho_g = r_to_g(rho_tot.to(CDTYPE))
    # mirror of scf.loop's nspin=2 energy assembly
    e_kin = sum(
        kinetic_energy([cs[sp][ik, :, : int(bk.npw[ik])] for ik in range(nk)],
                       occ_s[sp], kw, spheres) for sp in range(2))
    s_uu, s_dd, s_tt = spin_sigmas(rho_s[0], rho_s[1], xc, grid.g_cart)
    e_xc = xc.energy(rho_s[0], rho_s[1], grid.volume, s_uu, s_dd, s_tt)
    e_h = hartree_energy(rho_g, grid.g2, grid.volume)
    e_loc = local_energy(rho_g, vloc_g, grid.volume)
    e_nl = sum(
        nonlocal_energy([becp_b(projs, cs[sp])[ik] for ik in range(nk)],
                        dij, occ_s[sp], kw) for sp in range(2))
    e = e_kin + e_xc + e_h + e_loc + e_nl
    grads = torch.autograd.grad(e, cs)

    with torch.no_grad():
        veff_s = effective_potentials(system, xc,
                                      [r.detach() for r in rho_s],
                                      local_potential_r(system, vloc_g))
        mask = bk.mask[:, None, :]
        for sp in range(2):
            h = BatchedHamiltonian(bk, grid.shape, veff_s[sp], projs)
            expected = (2.0 * kw[:, None, None] * occ_s[sp][:, :, None]
                        * h.apply(cs[sp].detach()))
            gap = ((grads[sp] - expected) * mask).abs().max()
            assert float(gap / expected.abs().max()) < 1e-10, f"spin {sp}"


def test_potentials_equal_autograd_of_energies():
    """Closed-form v_H, v_loc vs autograd of E_H, E_loc — two independent
    implementations of the same functional derivative must agree."""
    system = _si2_rattled(kmesh=(1, 1, 1))
    grid = system.grid
    occ = _occ(system, 4, [2.0, 2.0, 1.5, 0.5])
    rho0 = density_b(_random_coeffs(system, 4, 3), occ, system.kweights,
                     system.batch, grid.shape, grid.volume)
    vloc_g = local_potential_g(system.positions, system.species_index,
                               system.vloc_tables, grid.g_cart, grid.volume)

    rho = rho0.detach().requires_grad_(True)
    e = (hartree_energy(r_to_g(rho.to(CDTYPE)), grid.g2, grid.volume)
         + local_energy(r_to_g(rho.to(CDTYPE)), vloc_g, grid.volume))
    (g,) = torch.autograd.grad(e, rho)
    v_from_e = g * (grid.n_points / grid.volume)

    v_h = (torch.fft.ifftn(hartree_potential_g(r_to_g(rho0.to(CDTYPE)), grid.g2),
                           dim=(-3, -2, -1)) * grid.n_points).real
    v_closed = v_h + local_potential_r(system, vloc_g)
    scale = max(float(v_closed.abs().max()), 1.0)
    assert float((v_from_e - v_closed).abs().max()) / scale < 1e-10
