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
from gradwave.core.xc.spin import SpinPBE
from gradwave.postscf.uspp_position import (
    bare_position_derivative,
    hessian_column,
    position_density_response,
)
from gradwave.pseudo.upf_paw import parse_upf_paw
from gradwave.scf.uspp import scf_uspp, setup_uspp
from gradwave.scf.uspp_loop import _build_iter_ops, _scf_iteration
from tests.helpers import RY

FIX = Path(__file__).parents[1] / "fixtures" / "qe"
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


@pytest.mark.slow
def test_gamma_hessian_symmetry_reconstruction():
    """HessianSymmetry on real columns: ideal diamond Si needs ONE
    irreducible displacement; a second column computed directly must
    match its symmetry reconstruction. This gates the rotation/atom-map
    conventions against the actual response physics (the unit test only
    checks internal consistency on synthetic matrices)."""
    from gradwave.postscf.phonons import HessianSymmetry

    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    # 20³, NOT the auto 18³: the diamond glide translation (¼,¼,¼) must
    # land on grid points or the XC-quadrature eggbox breaks the group
    # invariance of the discretized energy surface itself (measured:
    # 2.6e-2 column anisotropy at 18³ — see gamma_hessian's guard)
    res = scf_uspp(_build(paw, POS0, shape=(20, 20, 20)), PBE(),
                   etol=1e-12, rhotol=1e-10, verbose=False, max_iter=80)
    assert res["converged"]
    system = res["system"]
    hs = HessianSymmetry(np.asarray(system.grid.cell),
                         system.positions.detach().cpu().numpy(),
                         list(system.species_of_atom))
    assert len(hs.displacements) == 1

    cols = [hessian_column(res, PBE(), a, alpha)
            for a, alpha in hs.displacements]
    h_rec = hs.reconstruct(cols)

    # a column NOT in the irreducible set, computed directly
    col_direct = hessian_column(res, PBE(), 1, 2).numpy()
    # measured 4.1e-5 at 20³ — the residual XC-quadrature eggbox at this
    # coarse grid, not the reconstruction (18³ sits at 5.2e-2; the same
    # anisotropy is in the directly computed 6-column Hessian)
    rel = (np.linalg.norm(h_rec[:, :, 1, 2] - col_direct)
           / np.linalg.norm(col_direct))
    assert rel < 2e-4, f"symmetry-reconstructed column rel {rel:.2e}"


def _fold_nonmagnetic_nspin2(r1):
    """Build a fixed-occupation nspin=2 USPPResult from a converged nspin=1
    one by splitting every spin-resolved quantity equally between the two
    channels (ρ^σ = ρ/2, becsum^σ = becsum/2, occ^σ = occ/2, same orbitals and
    eigenvalues). For a nonmagnetic system this IS an exact converged nspin=2
    fixed point (V↑ = V↓ because ρ↑ = ρ↓), so the analytic response identity
    holds on it. It is the only route to a fixed-occupation nspin=2 PAW state:
    scf_uspp has no fixed-spin-moment mode, and the position response requires
    smearing='none' (the metals gate), so an actual nspin=2 SCF run cannot
    produce a valid input here. The stacked (2, nk, nb) occupation/eigenvalue
    layout mirrors scf_uspp's own nspin=2 result assembly."""
    r2 = dict(r1)
    r2["nspin"] = 2
    r2["rho_spin"] = [r1["rho"] * 0.5, r1["rho"] * 0.5]
    r2["rho_ij_atoms"] = [[m * 0.5 for m in r1["rho_ij_atoms"]],
                          [m * 0.5 for m in r1["rho_ij_atoms"]]]
    r2["coeffs"] = [list(r1["coeffs"]), list(r1["coeffs"])]
    r2["occupations"] = torch.stack(
        [r1["occupations"] * 0.5, r1["occupations"] * 0.5])
    r2["eigenvalues"] = torch.stack([r1["eigenvalues"], r1["eigenvalues"]])
    return r2


@pytest.mark.standard
def test_nspin2_nonmagnetic_limit_matches_nspin1():
    """Self-oracle for the nspin=2 position-response unblock: on a nonmagnetic
    PAW Si dimer the nspin=2 position density response and Hessian column must
    reproduce the nspin=1 result. This is an exact algebraic identity (the two
    spin channels are equal, so the per-spin threading — halved occupations,
    per-spin frozen operators, spin-summed density, the SpinXC/spin-f_xc core
    path, spin_sigma_triple in hessian_column — must collapse to the collinear
    path); any factor-2 spin-folding error would miss it by orders. No FD and
    no second SCF: one nspin=1 SCF is shared, folded into a nonmagnetic nspin=2
    state (see _fold_nonmagnetic_nspin2)."""
    torch.set_num_threads(8)
    paw = parse_upf_paw(FIX / "pseudos" / "Si.pbe-n-kjpaw_psl.1.0.0.UPF")
    # displaced, low-symmetry dimer so every column entry is nonzero
    pos0 = np.array([[0.02, -0.01, 0.015], [1.3575, 1.36, 1.35]])
    res1 = scf_uspp(setup_uspp(CELL, pos0, [0, 0], [paw], ecut=12 * RY,
                               kmesh=(2, 2, 2), ecutrho=48 * RY),
                    PBE(), etol=1e-11, rhotol=1e-9, verbose=False, max_iter=80)
    assert res1["converged"]
    res2 = _fold_nonmagnetic_nspin2(res1)
    a, alpha = 1, 0

    # bare map derivative: nspin=2 spin-summed δρ == nspin=1 δρ
    br1, _ = bare_position_derivative(res1, PBE(), a, alpha)
    br2, _ = bare_position_derivative(res2, SpinPBE(), a, alpha)
    rel_bare = float((br1 - (br2[0] + br2[1])).norm() / br1.norm())
    assert rel_bare < 1e-9, f"bare drho rel {rel_bare:.2e}"

    # self-consistent position response: spin-summed δρ* and δbec* == nspin=1
    drho1, dbec1, _ = position_density_response(res1, PBE(), a, alpha)
    drho2, dbec2, _ = position_density_response(res2, SpinPBE(), a, alpha)
    rel_rho = float((drho1 - (drho2[0] + drho2[1])).norm() / drho1.norm())
    assert rel_rho < 1e-9, f"self-consistent drho rel {rel_rho:.2e}"
    for i in range(2):
        dtot = dbec2[0][i] + dbec2[1][i]
        relb = float((dbec1[i] - dtot).abs().max()
                     / dbec1[i].abs().max().clamp_min(1e-30))
        assert relb < 1e-9, f"self-consistent dbec[{i}] rel {relb:.2e}"

    # Hessian column: d²E_total is spin-summed, so it matches directly
    col1 = hessian_column(res1, PBE(), a, alpha)
    col2 = hessian_column(res2, SpinPBE(), a, alpha)
    rel_col = float((col1 - col2).norm() / col1.norm())
    assert rel_col < 1e-9, f"hessian column rel {rel_col:.2e}"
