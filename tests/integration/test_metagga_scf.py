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
  * nspin=2 meta-GGA is guarded off until the spin path lands.
"""

import pytest
import torch

from gradwave.core.metagga import tau_b
from gradwave.core.xc.pbe import PBE
from gradwave.dtypes import CDTYPE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import RY, pseudo, si_fcc


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


def _system():
    cell, pos = si_fcc()
    upf = parse_upf(pseudo("Si_ONCV_PBE-1.2.upf"))
    return setup_system(cell, pos, [0, 0], [upf], ecut=18 * RY, kmesh=(1, 1, 1), nbands=8)


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


def test_nspin2_metagga_guarded():
    with pytest.raises(NotImplementedError):
        scf(_system(), _MetaFlat(), smearing="gaussian", width=0.1, nspin=2,
            verbose=False, max_iter=1)
