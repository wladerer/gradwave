"""Smeared fixed-spin-moment (FSM) SCF: tot_magnetization WITH smearing.

The smeared FSM mode solves two per-channel Fermi levels (μ↑ for
N↑=(N_e+M)/2, μ↓ for N↓) so the moment is pinned while the Fermi edges stay
broadened — the mode E(M) fixed-spin-moment curves need on metals. The
physics gate: along the curve ∂F/∂M = (μ↑−μ↓)/2, so at M = m₀ (the
unconstrained moment) the FSM run must reproduce the unconstrained SCF's
free energy and give μ↑ ≈ μ↓. See scf.common.fsm_smeared_occupations.
"""

import numpy as np
import pytest
import torch

from gradwave.core.xc.spin import SpinPBE
from gradwave.pseudo.upf import parse_upf
from gradwave.scf.loop import scf, setup_system
from tests.helpers import PSEUDOS, RY, si_fcc

HA = 27.211386245988  # eV


def _fe_bcc():
    """1-atom primitive bcc Fe, deliberately coarse (fast-tier budget): the
    consistency gate below compares an FSM run against the unconstrained run
    on the SAME settings, so absolute convergence quality cancels."""
    a = 2.87
    cell = 0.5 * a * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])
    pos = np.zeros((1, 3))
    fe = parse_upf(str(PSEUDOS / "Fe_ONCV_PBE-1.2.upf"))
    return setup_system(cell, pos, [0], [fe], ecut=30 * RY,
                        kmesh=(3, 3, 3), nbands=14)


def test_fsm_smeared_consistency_gate_fe():
    """Unconstrained smeared SCF gives moment m₀; FSM at M=m₀ reproduces the
    free energy to ~1e-6 Ha, pins the moment exactly, and gives μ↑ ≈ μ↓
    (∂F/∂M = 0 at the free minimum).

    Fermi-Dirac smearing on purpose: its counting function is strictly
    monotone, so each channel's Fermi level is unique and the consistency
    property is EXACT at any mesh (measured here: dF ~ 5e-12 eV, μ↑−μ↓ ~ 1e-7,
    FSM converges in 2 iterations from the free state). mp1's negative kernel
    makes N_σ(μ) locally non-monotone on coarse meshes — the free solution's
    μ↓ can sit on a decreasing branch, an unstable root under the per-channel
    constraint (measured on this 3³ cell: the FSM fixed point then lands
    6.7e-4 eV above the free run). On production meshes (14³) the channel DOS
    is dense enough that mp1 behaves; the E(M) campaign checks that with its
    argmin-vs-unconstrained sanity gate rather than this unit-style test."""
    torch.set_num_threads(4)
    kw = dict(nspin=2, smearing="fermi-dirac", width=0.1, start_mag=[0.5],
              mixing_scheme="pulay", max_iter=120, etol=1e-10, rhotol=1e-9,
              verbose=False)
    r_free = scf(_fe_bcc(), SpinPBE(), **kw)
    assert r_free.converged
    assert r_free.fermi_spin is None  # unconstrained path untouched
    m0 = float(r_free.mag_total)
    assert m0 > 1.0  # sanity: the coarse cell is still ferromagnetic

    r_fsm = scf(_fe_bcc(), SpinPBE(), tot_magnetization=m0,
                start_from=r_free, **kw)
    assert r_fsm.converged
    # (a) the moment is pinned to M (occupation-level exact; the real-space
    # integral matches to FFT/quadrature precision)
    assert abs(float(r_fsm.mag_total) - m0) < 1e-6
    # (b) same free energy as the unconstrained run (~1e-6 Ha)
    e_free = float(r_free.energies.free_energy)
    e_fsm = float(r_fsm.energies.free_energy)
    assert abs(e_fsm - e_free) < 2e-6 * HA
    # (c) two Fermi levels reported, coinciding at the free minimum
    assert r_fsm.fermi_spin is not None
    mu_up, mu_dn = r_fsm.fermi_spin
    assert abs(mu_up - mu_dn) < 1e-3  # eV; ∂F/∂M = (μ↑−μ↓)/2 = 0 at M=m₀
    assert r_fsm.fermi == pytest.approx(0.5 * (mu_up + mu_dn), abs=1e-12)
    assert abs(r_fsm.fermi - r_free.fermi) < 1e-3


def test_fsm_smeared_zero_moment_matches_unconstrained_si():
    """Nonmagnetic Si: FSM at M=0 with smearing must equal the unconstrained
    smeared nspin=2 run (identical channels ⇒ μ↑ = μ↓ = shared μ exactly)."""
    torch.set_num_threads(4)
    cell, pos = si_fcc()

    def make():
        si = parse_upf(str(PSEUDOS / "Si_ONCV_PBE-1.2.upf"))
        return setup_system(cell, pos, [0, 0], [si], ecut=15 * RY,
                            kmesh=(2, 2, 2), nbands=12)

    kw = dict(nspin=2, smearing="gaussian", width=0.2, start_mag=[0.0, 0.0],
              max_iter=80, etol=1e-10, rhotol=1e-9, verbose=False)
    r1 = scf(make(), SpinPBE(), **kw)
    r0 = scf(make(), SpinPBE(), tot_magnetization=0.0, **kw)
    assert r1.converged and r0.converged
    assert abs(float(r0.energies.free_energy)
               - float(r1.energies.free_energy)) < 1e-7
    assert abs(float(r0.mag_total)) < 1e-8
    mu_up, mu_dn = r0.fermi_spin
    assert mu_up == pytest.approx(mu_dn, abs=1e-8)
    assert mu_up == pytest.approx(r1.fermi, abs=1e-6)
