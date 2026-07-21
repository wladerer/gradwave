"""Self-consistent meta-GGA SCF at Γ — the τ generalized-KS operator in the loop.

No meta-GGA reference (SCAN/r2SCAN) fixture exists yet, so these gates are
intrinsic rather than vs-QE:

  * a meta-GGA whose energy_density ignores τ (needs_tau=True but τ-flat) must
    reproduce the PBE SCF bit-for-bit — v_τ = ∂e/∂τ = 0, the operator vanishes,
    and no existing path moves. This pins the plumbing and the gating.
  * a genuine τ-dependent functional (PBE + λ∫τ) must converge, shift the energy
    off PBE, and — the real correctness gate — satisfy the stationary-energy
    (Hellmann–Feynman) identity dE_total/dλ = ∫τ on the converged state. That
    holds only if the τ operator wired into H is exactly ∂E_xc/∂ψ*, i.e. the
    generalized-KS scheme is variationally self-consistent at SCF scale.
  * the same two gates on the nspin=2 spin path (fcc Al), where the per-channel
    operator −½∇·(v_τσ∇ψ_σ) acts on each spin's orbitals.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.metagga import tau_b
from gradwave.core.xc.pbe import PBE
from gradwave.core.xc.spin import SpinPBE
from gradwave.dtypes import CDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo, si_fcc

FIX = Path(__file__).parents[1] / "fixtures" / "qe"


class _MetaFlat(PBE):
    """needs_tau=True but the energy ignores τ — a meta-GGA with no τ response."""

    needs_tau = True

    def energy_density(self, rho, sigma=None, tau=None):
        return super().energy_density(rho, sigma)


class _MetaLinearTau(PBE):
    """PBE plus a linear-in-τ term λ·τ. ∂e/∂τ = λ (a constant v_τ), so the
    operator is λ·(−½∇²): a valid Hermitian perturbation of the kinetic term
    with the exact known explicit derivative ∂E_xc/∂λ = ∫τ."""

    needs_tau = True

    def __init__(self, lam: float = 0.0):
        super().__init__()
        self.lam = lam

    def energy_density(self, rho, sigma=None, tau=None):
        e = super().energy_density(rho, sigma)
        if tau is not None and self.lam != 0.0:
            e = e + self.lam * tau
        return e


class _SpinMetaFlat(SpinPBE):
    """needs_tau=True SpinXC whose energy ignores τ↑/τ↓ — spin-PBE limit."""

    needs_tau = True

    def energy_density(self, rho_up, rho_dn, sigma_uu=None, sigma_dd=None,
                       sigma_tot=None, tau_up=None, tau_dn=None):
        return super().energy_density(rho_up, rho_dn, sigma_uu, sigma_dd, sigma_tot)


class _SpinMetaLinearTau(SpinPBE):
    """Spin-PBE plus λ·(τ↑ + τ↓): a per-channel constant v_τ = λ."""

    needs_tau = True

    def __init__(self, lam: float = 0.0):
        super().__init__()
        self.lam = lam

    def energy_density(self, rho_up, rho_dn, sigma_uu=None, sigma_dd=None,
                       sigma_tot=None, tau_up=None, tau_dn=None):
        e = super().energy_density(rho_up, rho_dn, sigma_uu, sigma_dd, sigma_tot)
        if tau_up is not None and self.lam != 0.0:
            e = e + self.lam * (tau_up + tau_dn)
        return e


def _system():
    cell, pos = si_fcc()
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    return setup_system(cell, pos, [0, 0], [upf], ecut=18 * RY, kmesh=(1, 1, 1), nbands=8)


def _al_spin_system():
    FCC = np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    al = parse_upf(FIX / "pseudos" / "Al_ONCV_PBE-1.2.upf")
    return setup_system(4.05 / 2 * FCC, np.zeros((1, 3)), [0], [al],
                        ecut=20 * RY, kmesh=(2, 2, 2), nbands=10)


def _integrated_tau(res) -> float:
    """∫τ dr [eV·... ] on a converged result, rebuilding batched coeffs."""
    system = res.system
    bk, grid = system.batch, system.grid
    nk = len(system.spheres)
    nb = res.occupations.shape[1]
    m = bk.npw_max
    coeffs = torch.zeros(nk, nb, m, dtype=CDTYPE)
    for ik in range(nk):
        npw = system.spheres[ik].npw
        coeffs[ik, :, :npw] = res.coeffs[ik]
    tau = tau_b(coeffs, res.occupations, system.kweights, bk, grid.shape, grid.volume)
    return float(tau.sum()) * grid.volume / grid.n_points


@pytest.fixture(scope="module")
def pbe_ref():
    res = scf(_system(), PBE(), smearing="none", etol=1e-9, rhotol=1e-8, verbose=False)
    assert res.converged
    return res


def test_tau_flat_metagga_reduces_to_pbe(pbe_ref):
    """A τ-flat meta-GGA is the PBE SCF bit-for-bit (the τ operator is inert)."""
    res = scf(_system(), _MetaFlat(), smearing="none", etol=1e-9, rhotol=1e-8,
              verbose=False)
    assert res.converged
    assert abs(float(res.energies.free_energy)
               - float(pbe_ref.energies.free_energy)) < 1e-9


def test_linear_tau_metagga_converges_and_shifts(pbe_ref):
    res = scf(_system(), _MetaLinearTau(lam=0.05), smearing="none", etol=1e-9,
              rhotol=1e-8, verbose=False, max_iter=100)
    assert res.converged
    # the τ term is a real perturbation: the energy moves off PBE
    assert abs(float(res.energies.free_energy)
               - float(pbe_ref.energies.free_energy)) > 1e-3


@pytest.mark.slow
def test_stationary_energy_derivative(pbe_ref):
    """dE_total/dλ = ∫τ on the converged state (stationary-energy theorem).

    The implicit density-response term vanishes at self-consistency, so the total
    energy's λ-derivative is the explicit ∂E_xc/∂λ = ∫τ. Verified by a finite
    difference of re-converged SCF energies against ∫τ built from the orbitals."""
    lam, h = 0.05, 2e-3
    res = scf(_system(), _MetaLinearTau(lam=lam), smearing="none", etol=1e-10,
              rhotol=1e-9, verbose=False, max_iter=120)
    assert res.converged
    int_tau = _integrated_tau(res)

    def e_at(lm):
        r = scf(_system(), _MetaLinearTau(lam=lm), smearing="none", etol=1e-10,
                rhotol=1e-9, verbose=False, max_iter=120)
        assert r.converged
        return float(r.energies.free_energy)

    fd = (e_at(lam + h) - e_at(lam - h)) / (2 * h)
    assert abs(fd - int_tau) < 1e-3 * max(abs(int_tau), 1.0)


@pytest.fixture(scope="module")
def spin_pbe_ref():
    res = scf(_al_spin_system(), SpinPBE(), smearing="gaussian", width=0.1,
              etol=1e-10, rhotol=1e-9, verbose=False, nspin=2, start_mag=[0.0])
    assert res.converged
    return res


@pytest.mark.slow
def test_spin_tau_flat_reduces_to_spin_pbe(spin_pbe_ref):
    """A τ-flat spin meta-GGA is the spin-PBE nspin=2 SCF bit-for-bit."""
    res = scf(_al_spin_system(), _SpinMetaFlat(), smearing="gaussian", width=0.1,
              etol=1e-10, rhotol=1e-9, verbose=False, nspin=2, start_mag=[0.0])
    assert res.converged
    assert abs(float(res.energies.free_energy)
               - float(spin_pbe_ref.energies.free_energy)) < 1e-8


@pytest.mark.slow
def test_spin_linear_tau_converges_and_shifts(spin_pbe_ref):
    """A genuine per-channel τ term converges on the nspin=2 path and moves the
    energy — the per-spin generalized-KS operator is acting on both channels."""
    res = scf(_al_spin_system(), _SpinMetaLinearTau(lam=0.05), smearing="gaussian",
              width=0.1, etol=1e-10, rhotol=1e-9, verbose=False, nspin=2,
              start_mag=[0.0], max_iter=100)
    assert res.converged
    assert abs(float(res.energies.free_energy)
               - float(spin_pbe_ref.energies.free_energy)) > 1e-3
