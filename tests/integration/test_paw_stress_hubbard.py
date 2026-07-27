"""Fixed-basis USPP/PAW stress with DFT+U (Dudarev) — the S-dressed unblock.

The PAW/USPP stress path now adds the strain derivative of the Dudarev E_U
= Σ_{I,σ} (U−J)/2 Tr[n^{Iσ}(1−n^{Iσ})]. Unlike the norm-conserving +U stress
(postscf/_strain.hubbard_energy_strained), the atomic-orbital projections that
build n^{Iσ} are S-dressed — Sφ = φ + Σ_ij|β_i⟩q_ij⟨β_j|φ⟩ — so the strain
enters through BOTH the φ form factor / Y_lm / phase AND the augmentation β
projectors inside S. All of it rides the same autograd strain graph the base
PAW stress uses (postscf/paw_stress._energy_strained_uspp), so the +U stress is
the analytic strain derivative of the same energy expression by construction.

The oracle only checks a mathematical identity (autograd == FD of the exact
same energy expression, at fixed converged coefficients), so accuracy vs QE is
irrelevant here — the system is deliberately small (Γ-only, modest ecut) to
keep this a fast Tier-0/1 check; one SCF is shared by both tests.

Two checks, mirroring tests/integration/test_stress_hubbard.py (the NC analogue)
and the PAW derivative tests:

- self-oracle: on a rattled, low-symmetry Si PAW cell with a Hubbard U on the
  Si p manifold, the +U contribution to the ε=0 strained energy reproduces the
  SCF's E_U, and the analytic +U stress equals a central finite difference of
  the same strained energy w.r.t. strain.
- U=0 inertness: with a U=0 manifold the S-dressed +U machinery runs but adds
  exactly zero, so the stress is bit-for-bit the pre-existing no-+U tensor.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_hubbard import HubbardManifold, hubbard_sites
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"

# Rattled, low-symmetry 2-atom Si cell: a full anisotropic stress tensor.
SI_CELL = 5.43 / 2 * np.array([
    [0.02, 1.01, 0.99],
    [1.00, 0.01, 1.02],
    [1.01, 0.98, 0.00],
])
SI_FRAC = np.array([[0.015, -0.010, 0.020], [0.240, 0.255, 0.245]])
SI_POS = SI_FRAC @ SI_CELL

# +U on the Si p manifold. Unphysical, but the check is the pure mathematical
# identity σ = (1/Ω) dE_tot/dε, so any manifold with a real strain dependence
# exercises the S-dressed Hubbard strain term.
MAN = [HubbardManifold(species=0, l=1, u=3.0, j=0.0)]


@pytest.fixture(scope="module")
def si_paw_hubbard():
    """One small Γ-only USPP/PAW +U SCF, shared by both tests below."""
    torch.set_num_threads(4)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    system = setup_uspp(SI_CELL, SI_POS, [0, 0], [paw], ecut=20 * RY,
                        kmesh=(1, 1, 1), ecutrho=80 * RY, use_symmetry=False)
    res = scf_uspp(system, PBE(), smearing="gaussian", width=0.05,
                   hubbard=MAN, etol=1e-9, rhotol=1e-8, verbose=False,
                   max_iter=60)
    assert res.converged
    assert abs(float(res.energies.hubbard)) > 1e-3  # E_U genuinely nonzero
    return res


@pytest.mark.slow
def test_paw_stress_hubbard_autograd_vs_fd(si_paw_hubbard):
    """+U PAW stress vs central finite difference of the (+U) strained energy."""
    from gradwave.postscf.paw_stress import _energy_strained_uspp, stress_uspp

    res = si_paw_hubbard
    xc = PBE()
    # The ε=0 strained expression omits the constant one-center energy (only its
    # strain derivative is carried, via ddd·becsum), so it does NOT equal the SCF
    # total — the PAW derivative tests never assert that. Instead isolate the +U
    # contribution: at ε=0 the S-dressed strained occupation matrices reduce to
    # the SCF's S-metric ones, so E_U(ε=0) must reproduce res.energies.hubbard.
    z = torch.zeros(3, 3, dtype=torch.float64)
    eu0 = (float(_energy_strained_uspp(res, xc, z))
           - float(_energy_strained_uspp(
               dataclasses.replace(res, hub_sites=None), xc, z)))
    assert abs(eu0 - float(res.energies.hubbard)) < 1e-5, (
        eu0, float(res.energies.hubbard))

    sig = stress_uspp(res, xc, symmetrize=False).cpu().numpy()
    d = 1e-6

    def e_strained(ep):
        return float(_energy_strained_uspp(res, xc, ep))

    def fd_component(i, j):
        ep = torch.zeros(3, 3, dtype=torch.float64)
        ep[i, j] = d
        return (e_strained(ep) - e_strained(-ep)) / (2 * d)

    omega = res.system.grid.volume
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (2, 1), (0, 2)]:
        fd_sym = 0.5 * (fd_component(i, j) + fd_component(j, i)) / omega
        rel = abs(sig[i, j] - fd_sym) / max(abs(fd_sym), 1e-8)
        assert rel < 1e-5 or abs(sig[i, j] - fd_sym) < 1e-7, (
            i, j, sig[i, j], fd_sym, rel)


@pytest.mark.slow
def test_paw_stress_hubbard_u0_is_inert(si_paw_hubbard):
    """A U=0 manifold exercises the whole S-dressed +U stress path (radial SBT,
    Y_lm, S-dressing, occupation matrices) but contributes exactly zero, so the
    stress is bit-for-bit the no-+U PAW stress on the same converged state."""
    from gradwave.postscf.paw_stress import stress_uspp

    res = si_paw_hubbard
    xc = PBE()
    sig_no_u = stress_uspp(dataclasses.replace(res, hub_sites=None), xc,
                           symmetrize=False).cpu().numpy()
    sites_u0 = hubbard_sites(res.system,
                             [HubbardManifold(species=0, l=1, u=0.0, j=0.0)])
    sig_u0 = stress_uspp(dataclasses.replace(res, hub_sites=sites_u0), xc,
                         symmetrize=False).cpu().numpy()
    assert np.array_equal(sig_no_u, sig_u0), np.abs(sig_no_u - sig_u0).max()
