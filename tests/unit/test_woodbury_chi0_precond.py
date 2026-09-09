"""Unit tests for the cheap subspace-χ₀ Woodbury preconditioner (M1).

Two independent checks:

1. ``test_woodbury_identity`` — the Woodbury algebra in ``WoodburyPrecond``
   reproduces a dense ``(1 − M)⁻¹`` for a random low-rank M. Pure linear
   algebra, fast tier.
2. ``test_subspace_factors_match_matrixfree`` — the low-rank factorisation the
   builder extracts from a converged reference reproduces the matrix-free
   subspace χ₀ (``apply_chi0_subspace``) applied after K_Hxc, i.e. the implied
   M_ρ = χ₀_sub·K_Hxc of the frozen operator equals the direct apply to
   round-off. Needs one small SCF → standard tier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gradwave.core.fftbox import r_to_g
from gradwave.dtypes import CDTYPE
from gradwave.scf.implicit import apply_k_hxc
from gradwave.scf.subspace_chi0 import (
    WoodburyPrecond,
    apply_chi0_subspace,
    build_woodbury_subspace,
)

RY = 13.605693122994
_PSEUDO = Path(__file__).resolve().parents[1] / "fixtures" / "qe" / "pseudos"


def test_woodbury_identity():
    """(1 − M)⁻¹ via WoodburyPrecond == dense solve, random low-rank M."""
    torch.manual_seed(0)
    ng, ncol, vol = 20, 5, 1.7
    u = torch.randn(ncol, ng, dtype=CDTYPE)
    w = torch.randn(ncol, ng, dtype=CDTYPE)
    c = torch.randn(ncol, dtype=torch.float64) * 0.1  # small → (1−M) well-posed

    pre = WoodburyPrecond(u, w, c, vol, nspin=1, ng=ng, g0_idx=0)
    # M[g,g'] = Σ_p u[p,g] (vol c_p) conj(w[p,g'])  (the bare-G Woodbury operator)
    m = torch.einsum("pg,p,ph->gh", u, (vol * c).to(CDTYPE), w.conj())
    r = torch.randn(ng, dtype=CDTYPE)
    dense = torch.linalg.solve(torch.eye(ng, dtype=CDTYPE) - m, r)
    got = pre._apply_updn(r)
    assert torch.allclose(got, dense, atol=1e-10, rtol=1e-8)


def test_woodbury_identity_nspin2():
    """nspin=2 analogue of ``test_woodbury_identity`` with a NONZERO low-rank M.

    The full ``WoodburyPrecond.__call__`` for nspin=2 is a composition:
    convert the mixer residual (tot,mag) → (up,dn), apply (1 − M)⁻¹ on the
    stacked 2·ng (up,dn) vector, convert back, then pin the total-channel G=0.
    The trivial W=0 test cannot see a transposed/mis-indexed spin block because
    (1 − M)⁻¹ = 1 collapses the middle. Here M ≠ 0 with the up- and dn-blocks
    deliberately DIFFERENT, so the (tot,mag)↔(up,dn) wrapping is exercised, and
    the result is checked against a dense reference built from the SAME layout.
    """
    torch.manual_seed(3)
    ng, ncol, vol = 6, 4, 1.3
    nvec = 2 * ng
    # low-rank columns live in the stacked (up,dn) space (nvec = 2·ng); make the
    # up/dn halves of every column different so the spin wrapping is non-trivial.
    u = torch.randn(ncol, nvec, dtype=CDTYPE)
    w = torch.randn(ncol, nvec, dtype=CDTYPE)
    c = torch.randn(ncol, dtype=torch.float64) * 0.1  # small → (1−M) well-posed
    g0 = 2  # G=0 slot inside the ng-length total channel

    pre = WoodburyPrecond(u, w, c, vol, nspin=2, ng=ng, g0_idx=g0)

    # dense operator in the SAME (tot,mag) basis the SCF hands over:
    #   updn = T_fwd @ x         updn[:ng]=(tot+mag)/2, updn[ng:]=(tot-mag)/2
    #   y    = (1 − M)⁻¹ updn    M[g,h] = Σ_p u[p,g] (vol c_p) conj(w[p,h])
    #   res  = T_back @ y        res[:ng]=up+dn, res[ng:]=up-dn
    #   res[g0] = 0              (total-channel charge conservation)
    eye_ng = torch.eye(ng, dtype=CDTYPE)
    t_fwd = 0.5 * torch.cat([torch.cat([eye_ng, eye_ng], 1),
                             torch.cat([eye_ng, -eye_ng], 1)], 0)
    t_back = torch.cat([torch.cat([eye_ng, eye_ng], 1),
                        torch.cat([eye_ng, -eye_ng], 1)], 0)
    m = torch.einsum("pg,p,ph->gh", u, (vol * c).to(CDTYPE), w.conj())

    x = torch.randn(nvec, dtype=CDTYPE)  # mixer residual ρ(G), (tot,mag), complex

    def dense_ref(t_fwd_, t_back_):
        updn = t_fwd_ @ x
        y = torch.linalg.solve(torch.eye(nvec, dtype=CDTYPE) - m, updn)
        res = t_back_ @ y
        res[g0] = 0.0
        return res

    ref = dense_ref(t_fwd, t_back)
    got = pre(x)
    assert torch.allclose(got, ref, atol=1e-10, rtol=1e-8)

    # The test has teeth: a mis-indexed spin block — up/dn halves of the (tot,mag)
    # transform swapped — gives a materially different answer, so a transposed
    # spin block in the real apply WOULD be caught (not silently absorbed).
    t_fwd_bad = 0.5 * torch.cat([torch.cat([eye_ng, -eye_ng], 1),
                                 torch.cat([eye_ng, eye_ng], 1)], 0)
    ref_bad = dense_ref(t_fwd_bad, t_back)
    assert not torch.allclose(got, ref_bad, atol=1e-6)


def test_woodbury_totmag_roundtrip():
    """nspin=2 (tot,mag)↔(up,dn) transform is FFT-free and self-inverse when
    the operator is identity (cvals → 0 ⇒ WoodburyPrecond ≈ 1)."""
    torch.manual_seed(1)
    ng = 8
    u = torch.randn(2, 2 * ng, dtype=CDTYPE)
    w = torch.zeros(2, 2 * ng, dtype=CDTYPE)  # W=0 ⇒ M=0 ⇒ (1−M)⁻¹ = 1
    c = torch.ones(2, dtype=torch.float64)
    pre = WoodburyPrecond(u, w, c, 1.0, nspin=2, ng=ng, g0_idx=0)
    r = torch.randn(2 * ng, dtype=CDTYPE)  # mixer residual ρ(G) is complex
    r[0] = 0.0  # total-channel G=0 is pinned (charge conservation), as in the SCF
    assert torch.allclose(pre(r), r, atol=1e-12)


def _tiny_al_metal():
    """Small fcc-Al bulk, smeared (a metal) — cheap enough for a unit SCF."""

    from gradwave.core.xc.pbe import PBE
    from gradwave.pseudo.upf import parse_upf
    from gradwave.scf.loop import scf, setup_system

    upf = parse_upf(_PSEUDO / "Al_ONCV_PBE-1.2.upf")
    a = 4.05
    cell = a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])
    system = setup_system(cell, np.array([[0.0, 0, 0]]), [0], [upf],
                          ecut=18.0 * RY, kmesh=(2, 2, 2), nbands=10,
                          use_symmetry=False)
    xc = PBE()
    res = scf(system, xc, smearing="gaussian", width=0.1, nspin=1,
              mixing_scheme="pulay", precond="local_tf", mixing_alpha=0.7,
              etol=1e-9, rhotol=1e-8, max_iter=80, verbose=False)
    return res, xc


@pytest.mark.standard
def test_subspace_factors_match_matrixfree():
    """The frozen low-rank M_ρ = χ₀_sub·K_Hxc equals the direct matrix-free
    apply on a random real potential — certifies the codensity assembly, the
    δμ term, the volume convention, and the K_Hxc coupling."""
    torch.set_num_threads(4)
    res, xc = _tiny_al_metal()
    assert res.converged
    pre = build_woodbury_subspace(res, xc, pair_cut=1e-9, max_cols=4096)
    assert pre is not None and pre.n_col > 0

    grid = res.system.grid
    mask = grid.dens_mask.reshape(-1)
    vol = float(grid.volume)
    torch.manual_seed(2)
    # A physical density residual: band-limited to the density sphere (the only
    # thing the mixer ever hands the preconditioner). A full-spectrum random
    # field would carry high-G noise the density-sphere projection legitimately
    # drops but the matrix-free product-then-FFT folds back in — an artefact of
    # the probe, not the operator.
    from gradwave.core.fftbox import g_to_r_box
    wg = torch.zeros(grid.n_points, dtype=CDTYPE)
    wg[mask] = r_to_g(torch.randn(grid.shape, dtype=torch.float64).to(CDTYPE)
                      ).reshape(-1)[mask]
    w = g_to_r_box(wg.reshape(grid.shape), real=True)

    # direct: χ₀_sub(K_Hxc w)
    z = apply_k_hxc(res, xc, w)
    target = apply_chi0_subspace(res, z)  # grid field
    target_sph = r_to_g(target.to(CDTYPE)).reshape(-1)[mask]

    # frozen operator: M_ρ ŵ = Σ_p u_p (vol c_p) ⟨w_p, ŵ⟩  (nspin=1: nvec = ng)
    w_sph = r_to_g(w.to(CDTYPE)).reshape(-1)[mask]
    proj = torch.einsum("ag,g->a", pre._w.conj(), w_sph)
    m_apply = torch.einsum("ag,a->g", pre._u,
                           (vol * pre.cvals).to(CDTYPE) * proj)

    rel = float((m_apply - target_sph).abs().max()
                / target_sph.abs().max().clamp_min(1e-30))
    assert rel < 1e-7, f"low-rank M_ρ vs matrix-free: rel={rel:.2e}"
