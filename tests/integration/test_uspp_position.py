"""Analytic position response through the USPP/PAW SCF (task #71).

Stage 1 — the bare map derivative ∂F/∂τ at fixed input: S-metric window
perturbation theory (c_mn with the δS numerator, −½⟨n|δS|n⟩ diagonal),
the δS-corrected Sternheimer complement, moving KB/aug projector phases,
δv_loc, and the NLCC core motion f_xc·∂ρ_core/∂τ (omitting the core
term costs 18% — found by this gate). Validated against central FD of
_scf_iteration at displaced positions: observed 2.4e-5 relative (the h²
truncation floor).

Stage 2 — the self-consistent response δx = (1 − χ̃K)⁻¹ δx_bare via the
Newton-finisher fixed point. Validated against central FD of fully
converged SCF re-runs: observed 3.0e-5 relative, 11 Anderson iterations.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.xc.pbe import PBE
from gradwave.postscf.uspp_position import (
    bare_position_derivative,
    hessian_column,
    position_density_response,
)
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_loop import _build_iter_ops, _scf_iteration

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
RY = 13.605693122994
CELL = 5.43 / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
POS0 = np.array([[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]])


def _build(paw, pos, shape=None):
    return setup_uspp(CELL, pos, [0, 0], [paw], ecut=15 * RY,
                      kmesh=(2, 2, 2), ecutrho=60 * RY, fft_shape=shape)


@pytest.mark.slow
def test_bare_position_derivative_vs_raw_map_fd():
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    res = scf_uspp(_build(paw, POS0), PBE(), etol=1e-12, rhotol=1e-10,
                   verbose=False, max_iter=80)
    assert res["converged"]
    shape = tuple(res["system"].grid.shape)
    rho_star = res["rho"].detach().clone()
    bec_star = [m.detach().clone() for m in res["rho_ij_atoms"]]

    def raw_map_at(pos):
        ops = _build_iter_ops(_build(paw, pos, shape=shape), PBE(),
                              nspin=1, smearing="none", width=0.0,
                              batched=True)
        step = _scf_iteration(ops, [rho_star], [bec_star],
                              [[None] * ops.nk], [None], None, 1e-11, 0)
        return step["rho_out_s"][0], step["rho_ij_s"][0]

    a, alpha = 1, 0
    drho_an, dbec_an = bare_position_derivative(res, PBE(), a, alpha)
    h = 2e-3
    pp, pm = POS0.copy(), POS0.copy()
    pp[a, alpha] += h
    pm[a, alpha] -= h
    rp, bp = raw_map_at(pp)
    rm, bm = raw_map_at(pm)
    drho_fd = (rp - rm) / (2 * h)
    rel = float((drho_an - drho_fd).norm() / drho_fd.norm())
    assert rel < 2e-4, f"bare drho rel {rel:.2e}"
    for i in range(2):
        dfd = (bp[i] - bm[i]) / (2 * h)
        relb = float((dbec_an[i] - dfd).abs().max()
                     / dfd.abs().max().clamp_min(1e-30))
        assert relb < 2e-4, f"bare dbec[{i}] rel {relb:.2e}"


@pytest.mark.slow
def test_position_density_response_vs_scf_fd():
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")

    def scf_at(pos, shape=None, prev=None):
        r = scf_uspp(_build(paw, pos, shape=shape), PBE(), etol=1e-12,
                     rhotol=1e-10, verbose=False, max_iter=80,
                     start_from=prev)
        assert r["converged"]
        return r

    res = scf_at(POS0)
    shape = tuple(res["system"].grid.shape)
    a, alpha = 1, 0
    drho_an, dbec_an, _ = position_density_response(res, PBE(), a, alpha)

    h = 2e-3
    pp, pm = POS0.copy(), POS0.copy()
    pp[a, alpha] += h
    pm[a, alpha] -= h
    rp = scf_at(pp, shape=shape, prev=res)
    rm = scf_at(pm, shape=shape, prev=res)
    drho_fd = (rp["rho"] - rm["rho"]) / (2 * h)
    rel = float((drho_an - drho_fd).norm() / drho_fd.norm())
    assert rel < 2e-4, f"self-consistent drho rel {rel:.2e}"
    bp, bm = rp["rho_ij_atoms"], rm["rho_ij_atoms"]
    for i in range(2):
        dfd = (bp[i] - bm[i]) / (2 * h)
        relb = float((dbec_an[i] - dfd).abs().max()
                     / dfd.abs().max().clamp_min(1e-30))
        assert relb < 2e-4, f"self-consistent dbec[{i}] rel {relb:.2e}"


@pytest.mark.slow
def test_hessian_column_vs_fd_of_forces():
    """Stage 3: one analytic Hessian column (mixed second derivative
    through the self-consistent response, contracted through the force
    graph) vs central FD of the validated analytic forces. Observed
    2.0e-5 relative (the h² floor). Uses a displaced geometry so every
    column entry is nonzero."""
    from gradwave.postscf.paw_forces import forces_uspp

    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    pos0 = np.array([[0.02, -0.01, 0.015], [1.3575, 1.36, 1.35]])

    def scf_at(pos, shape=None, prev=None):
        r = scf_uspp(_build(paw, pos, shape=shape), PBE(), etol=1e-12,
                     rhotol=1e-10, verbose=False, max_iter=80,
                     start_from=prev)
        assert r["converged"]
        return r

    res = scf_at(pos0)
    shape = tuple(res["system"].grid.shape)
    a, alpha = 1, 0
    col_an = hessian_column(res, PBE(), a, alpha)

    h = 2e-3
    pp, pm = pos0.copy(), pos0.copy()
    pp[a, alpha] += h
    pm[a, alpha] -= h
    fp = forces_uspp(scf_at(pp, shape=shape, prev=res), PBE(),
                     remove_net=False)
    fm = forces_uspp(scf_at(pm, shape=shape, prev=res), PBE(),
                     remove_net=False)
    col_fd = -(fp - fm) / (2 * h)
    rel = float((col_an - col_fd).norm() / col_fd.norm())
    assert rel < 2e-4, f"hessian column rel {rel:.2e}"
