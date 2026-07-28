"""DFT+U (Dudarev) through the USPP/PAW position/Berry response and
hessian_column — the S-dressed +U unblock (issue #147, tranche 3).

The position response (``uspp_position.position_density_response``) and the
analytic Hessian column (``uspp_position.hessian_column`` — the machinery
``postscf.phonons.gamma_hessian`` folds into Γ phonons) now thread the
occupation-matrix channel n^{Iσ} of DFT+U. Like the +U PAW *stress* (#156),
the atomic-orbital projections are S-dressed (Sφ = φ + Σ|β⟩q⟨β|φ⟩), but here
the strain graph is replaced by the atomic-DISPLACEMENT graph: Sφ moves with
its atom, so the bare position perturbation gains δH_U = |∂(Sφ)⟩D_U⟨Sφ| + h.c.,
and the self-consistent response carries the Dudarev kernel δD_U = −(U−J)·herm(δn)
through the same apply_chi0/k_hub adjoint machinery.

Self-oracles (all mathematical identities on a deliberately small Γ-only PAW +U
system — analytic response == finite difference of the SAME converged-state
code, so accuracy vs QE is irrelevant; one SCF triple is shared):

- position response: analytic dρ*/dτ and dbecsum*/dτ vs central FD of fully
  converged +U SCF re-runs (mirrors ``test_position_density_response_vs_scf_fd``).
- hessian_column: the analytic +U mixed column vs central FD of the +U-aware
  analytic forces ``paw_forces.forces_uspp`` (mirrors
  ``test_hessian_column_vs_fd_of_forces``); forces_uspp already reads the +U
  force off ``hubbard_e_channel``, so this cross-validates the two independent
  +U second-derivative paths.
- U=0 inertness: with a U=0 manifold the whole S-dressed +U response machinery
  runs (dsphi jvp, bare δn, hub fixed-point channel) but adds exactly zero, so
  hessian_column matches the non-+U path essentially bit-for-bit.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.paw_forces import forces_uspp
from gradwave.postscf.uspp_position import (
    hessian_column,
    position_density_response,
)
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_hubbard import HubbardManifold
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
# rattled, low-symmetry Si dimer: no exact band degeneracy, every column entry
# nonzero (the P1 regime the sibling column check uses)
POS0 = np.array([[0.02, -0.01, 0.015], [1.3575, 1.36, 1.35]])
# unphysical +U on the Si p manifold — the check is the pure identity
# (analytic == FD), so any manifold with a real τ dependence exercises the path
MAN = [HubbardManifold(species=0, l=1, u=3.0, j=0.0)]
A, ALPHA, H = 1, 0, 2e-3  # displaced atom / Cartesian / central-difference step


def _build(pos, shape=None):
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    return setup_uspp(CELL, pos, [0, 0], [paw], ecut=15 * RY, kmesh=(2, 2, 2),
                      ecutrho=60 * RY, use_symmetry=False, fft_shape=shape)


def _scf(pos, shape=None, prev=None, man=MAN):
    r = scf_uspp(_build(pos, shape), PBE(), etol=1e-12, rhotol=1e-10,
                 verbose=False, max_iter=150, hubbard=man, start_from=prev)
    assert r["converged"]
    return r


@pytest.fixture(scope="module")
def si_paw_u_triple():
    """One +U SCF at POS0 plus the two ±H displaced +U SCFs (atom A, dir ALPHA),
    shared by the position-response and hessian FD checks."""
    torch.set_num_threads(4)
    base = _scf(POS0)
    assert abs(float(base.energies.hubbard)) > 1e-2  # E_U genuinely nonzero
    shape = tuple(base["system"].grid.shape)
    pp, pm = POS0.copy(), POS0.copy()
    pp[A, ALPHA] += H
    pm[A, ALPHA] -= H
    disp_p = _scf(pp, shape=shape, prev=base)
    disp_m = _scf(pm, shape=shape, prev=base)
    return base, disp_p, disp_m


@pytest.mark.slow
def test_position_density_response_hubbard_vs_scf_fd(si_paw_u_triple):
    """Self-consistent +U dρ*/dτ and dbecsum*/dτ vs central FD of +U SCF re-runs."""
    base, rp, rm = si_paw_u_triple
    drho_an, dbec_an, _ = position_density_response(base, PBE(), A, ALPHA)

    drho_fd = (rp["rho"] - rm["rho"]) / (2 * H)
    rel = float((drho_an - drho_fd).norm() / drho_fd.norm())
    assert rel < 2e-4, f"+U self-consistent drho rel {rel:.2e}"
    for i in range(2):
        dfd = (rp["rho_ij_atoms"][i] - rm["rho_ij_atoms"][i]) / (2 * H)
        relb = float((dbec_an[i] - dfd).abs().max()
                     / dfd.abs().max().clamp_min(1e-30))
        assert relb < 2e-4, f"+U self-consistent dbec[{i}] rel {relb:.2e}"


@pytest.mark.slow
def test_hessian_column_hubbard_vs_fd_of_forces(si_paw_u_triple):
    """The analytic +U Hessian column vs central FD of the +U-aware forces."""
    base, rp, rm = si_paw_u_triple
    col_an = hessian_column(base, PBE(), A, ALPHA)
    fp = forces_uspp(rp, PBE(), remove_net=False)
    fm = forces_uspp(rm, PBE(), remove_net=False)
    col_fd = -(fp - fm) / (2 * H)
    rel = float((col_an - col_fd).norm() / col_fd.norm())
    assert rel < 2e-4, f"+U hessian column rel {rel:.2e}"


@pytest.mark.slow
def test_hessian_column_u0_is_inert():
    """A U=0 manifold drives the entire S-dressed +U response (dsphi jvp, bare
    δn, hub fixed-point channel, in-graph E_U) but contributes exactly zero, so
    hessian_column reproduces the non-+U column essentially bit-for-bit."""
    torch.set_num_threads(4)
    res_u0 = _scf(POS0, man=[HubbardManifold(species=0, l=1, u=0.0, j=0.0)])
    col_u0 = hessian_column(res_u0, PBE(), A, ALPHA)
    # strip the (conditional) Hubbard keys → the pre-existing no-+U path
    col_plain = hessian_column(
        dataclasses.replace(res_u0, hub_occ=None, hub_sites=None),
        PBE(), A, ALPHA)
    rel = float((col_u0 - col_plain).norm()
                / col_plain.norm().clamp_min(1e-30))
    assert rel < 1e-8, f"U=0 +U hessian not inert: rel {rel:.2e}"
